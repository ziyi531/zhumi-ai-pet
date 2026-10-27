"""HTTP 服务 + 心跳调度。

从 ai_pet/ 目录运行：python -m uvicorn server:app --port 8600
设备平时连 /ws（WebSocket，协议见 PROTOCOL.md）：对话、摸头、状态同步、主动搭话实时推送。
/api/poll 等 HTTP 接口保留给以后省电的“定时醒来拉取”模式和调试面板。
"""
import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import asr
import brain
import clock
import cloud
import config
import db
import link
import memory_admin
import relay
import tts
import websearch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

scheduler = BackgroundScheduler()


@asynccontextmanager
async def lifespan(app):
    db.init()
    if clock.offset_seconds():  # 调试页快进过、没拨回来：大脑会活在“将来”（真机上被问几点答错过）
        logging.getLogger("pet").warning("虚拟时钟比真实时间快 %.1f 小时（调试快进留下的）；"
                                         "关掉大脑后运行 python tools/fix_clock.py 拨回来", clock.offset_seconds() / 3600)
    if not db.get_kv("born_at"):
        ts = clock.fmt(clock.now())
        db.set_kv("born_at", ts)
        db.set_kv("last_interaction", ts)
    scheduler.add_job(brain.tick, "interval", minutes=config.TICK_MINUTES, id="tick",
                      max_instances=1, coalesce=True)
    scheduler.start()
    threading.Thread(target=asr.warmup, daemon=True).start()
    threading.Thread(target=websearch.warmup, daemon=True).start()
    threading.Thread(target=tts.warmup, daemon=True).start()  # 拉起团子的声音服务（tts_service.py）
    pusher = asyncio.create_task(link.pusher())
    relay.tunnel.start()  # 出门在外连回家：SSH 反向隧道接到 Lightsail
    yield
    relay.tunnel.stop()
    pusher.cancel()
    scheduler.shutdown(wait=False)
    tts.shutdown()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=config.BASE_DIR / "static"), name="static")


@app.middleware("http")
async def lan_only(request: Request, call_next):
    """从公网（隧道）进来的 HTTP 请求一律拒绝：记忆、日记、导出备份只给家里局域网看。公网只能连 /ws。"""
    if relay.via_relay(request.scope):
        return JSONResponse({"detail": "只能在家里的局域网访问"}, status_code=403)
    return await call_next(request)


class ChatIn(BaseModel):
    text: str


class LogItem(BaseModel):
    kind: str
    count: int = 1


class LogIn(BaseModel):
    events: list[LogItem]


class ClockIn(BaseModel):
    advance_hours: float | None = None
    to: str | None = None  # "HH:MM"


@app.get("/")
def index():
    """网页版团子（八条风表情 + 聊天）。"""
    return RedirectResponse("/static/pet.html")


@app.get("/debug")
def debug():
    """调试用假设备：状态条、快进时钟、立即心跳。"""
    return RedirectResponse("/static/device.html")


@app.get("/voices")
def voices_page():
    """团子声音的试听页（Qwen3-TTS 捏声音的过程和候选）。"""
    return RedirectResponse("/static/qwen3tts.html")


@app.websocket("/ws")
async def device_socket(ws: WebSocket):
    await link.handle(ws)


@app.post("/api/chat")
def chat(body: ChatIn):
    text = body.text.strip()[:500]
    if not text:
        raise HTTPException(400, "空消息")
    return brain.chat(text)


@app.get("/api/poll")
def poll():
    """设备醒来拉取：待播事件 + 权威状态（设备用它校准本地的状态影子）。"""
    now = clock.now()
    events = brain.take_events()
    keep = ("id", "kind", "text", "face", "vibe", "voice")
    return {"time": clock.fmt(now), "events": [{k: e[k] for k in keep} for e in events],
            "state": brain.current_state(now).to_dict()}


@app.post("/api/device_log")
def device_log(body: LogIn):
    """设备上报本地发生的事，比如离线期间被摸了几次。"""
    for e in body.events:
        brain.record_device_event(e.kind, e.count)
    return {"ok": True}


@app.get("/api/state")
def get_state():
    return brain.status()


@app.post("/api/clock")
def set_clock(body: ClockIn):
    try:
        if body.to:
            clock.set_time(body.to)
        elif body.advance_hours:
            clock.advance(body.advance_hours)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"time": clock.fmt(clock.now())}


@app.post("/api/tick")
def tick_now():
    return brain.tick()


@app.get("/memory")
def memory_page():
    """记忆管理页：团子记得的事、日记、性格数值、自我认识，可以改、删、导出。"""
    return RedirectResponse("/static/memory.html")


class MemoryEditIn(BaseModel):
    text: str | None = None
    kind: str | None = None
    importance: int | None = None


class TraitsIn(BaseModel):
    values: dict[str, float]


class SelfViewIn(BaseModel):
    text: str


@app.get("/api/memories")
def memories_overview():
    return memory_admin.overview(clock.now())


@app.patch("/api/memories/{mid}")
def memory_edit(mid: int, body: MemoryEditIn):
    try:
        row = memory_admin.edit(mid, body.text, body.kind, body.importance)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    if not row:
        raise HTTPException(404, "没有这条记忆")
    return row


@app.delete("/api/memories/{mid}")
def memory_delete(mid: int):
    if not db.delete_memory(mid):
        raise HTTPException(404, "没有这条记忆")
    return {"ok": True}


@app.delete("/api/profile/{key}")
def profile_delete(key: str):
    if not db.delete_profile(key):
        raise HTTPException(404, "没有这一项")
    return {"ok": True}


@app.delete("/api/followups/{fid}")
def followup_cancel(fid: int):
    db.close_followup(fid, "cancelled", clock.fmt(clock.now()))
    return {"ok": True}


@app.delete("/api/diary/{day}")
def diary_delete(day: str):
    if not db.delete_diary(day):
        raise HTTPException(404, "没有这天的日记")
    return {"ok": True}


@app.put("/api/self_view")
def self_view_set(body: SelfViewIn):
    text = body.text.strip()[: config.SELF_VIEW_MAX_CHARS]
    db.set_kv("self_view", text)
    return {"self_view": text}


@app.put("/api/traits")
def traits_set(body: TraitsIn):
    return {"traits": memory_admin.set_traits(body.values, clock.now())}


@app.get("/api/export")
def export_backup():
    """完整备份（JSON 文件下载）。只在本机访问，不上传任何地方。"""
    data = memory_admin.export(clock.now())
    name = f"tuanzi-backup-{datetime.now():%Y%m%d-%H%M}.json"
    return Response(json.dumps(data, ensure_ascii=False, indent=1), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.post("/api/nightly")
def nightly_now():
    """调试用：马上做一次夜间整理（合并、淡忘、写日记、修订自我认识），平时团子睡觉时自己会做。"""
    return brain.run_nightly()


@app.get("/api/health")
def health():
    return {"ollama": brain.client.is_up(config.LOCAL_PROBE_S), "model": brain.client.model, "asr": asr.available(),
            "tts": tts.available(), "cloud_standin": cloud.standin_available()}
