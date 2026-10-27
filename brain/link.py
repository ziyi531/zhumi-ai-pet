"""设备连接：大脑和设备（网页团子、ESP32）之间的 WebSocket 协议，格式见 PROTOCOL.md。

- 设备说话、摸头都从这里进来；回复、状态同步、主动搭话从这里推出去。
- 主动搭话由心跳（brain.tick）写进 events 表，这里的 pusher 每隔几秒取出来推给所有在线设备；
  没有设备在线时事件留在表里，设备连上后马上补推（太旧的会作废）。
- 原来的 HTTP 接口（/api/poll 等）保留，给以后省电的“定时醒来拉取”模式和调试面板用。
- 团子的声音：设备在 hello 里声明 audio_out 的，say 之后大脑把合成的语音流式推过去
  （tts start → 二进制帧 → tts stop）：网页团子要 pcm16，ESP32 要 opus（60ms 一包）；没声明的只收文字。
- 录音：网页团子发 pcm16，ESP32 发 opus（listen start 里的 format），大脑解码成 PCM 再识别。
- 出门在外的设备经 Lightsail 中转、SSH 隧道进来（relay.py），hello 里必须带设备密钥 token。
"""
import asyncio
import json
import logging
import time

from fastapi import WebSocket

import asr
import brain
import clock
import config
import opus
import persona
import relay
import tts

AUDIO_FORMATS = ("pcm16", "opus")

PROTO = 1
log = logging.getLogger("pet.link")


class Client:
    def __init__(self, device: str, audio_out: str | None):
        self.device = device
        self.audio_out = audio_out  # hello 里声明的声音格式（pcm16 / opus）：会播大脑下发的声音
        self.audio = audio_out is not None
        self.hush = 0       # 主人开口时加一：正在推的那段声音对这台设备就不推了
        self.turns = asyncio.Lock()  # 同一台设备的对话按顺序一轮一轮来（对话在后台任务里跑）


class Hub:
    def __init__(self):
        self.clients: dict[WebSocket, Client] = {}
        self.seq = 0         # 每段要说的话的编号，say 和 tts 用它对上
        self.tasks = set()   # 后台推声音的任务（留引用，免得被回收）
        self._lock, self._lock_loop = None, None

    async def send(self, ws: WebSocket, msg: dict):
        await ws.send_text(json.dumps(msg, ensure_ascii=False))

    async def try_send(self, ws: WebSocket, msg: dict) -> bool:
        try:
            await self.send(ws, msg)
            return True
        except Exception:
            self.clients.pop(ws, None)
            return False

    async def broadcast(self, msg: dict):
        for ws in list(self.clients):
            await self.try_send(ws, msg)

    def next_id(self) -> int:
        self.seq += 1
        return self.seq

    def voice_lock(self) -> asyncio.Lock:
        """同一时间只推一段声音（声音服务也是一句一句合成）。锁跟着事件循环走，自测里每个 TestClient 一个循环。"""
        loop = asyncio.get_running_loop()
        if self._lock_loop is not loop:
            self._lock, self._lock_loop = asyncio.Lock(), loop
        return self._lock

    def spawn(self, coro):
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)


hub = Hub()


def state_msg(kind: str = "state") -> dict:
    now = clock.now()
    return {"type": kind, "proto": PROTO, "time": clock.fmt(now), "state": brain.current_state(now).to_dict()}


def say_msg(item: dict, proactive: bool, sid: int | None = None) -> dict:
    """sid 不为空表示这段话的声音随后推过来（tts 消息带同样的 id）。"""
    msg = {"type": "say", "text": item["text"], "face": item.get("face") or "idle",
           "vibe": item.get("vibe") or "tap", "voice": item.get("voice") or "normal", "proactive": proactive,
           "audio": sid is not None}
    if sid is not None:
        msg["id"] = sid
    if item.get("tools"):
        msg["tools"] = item["tools"]  # 这次回答前用过的工具 [{"tool", "detail"}]
    if item.get("searched"):
        msg["searched"] = item["searched"]  # 这次回答前上网查过的词（旧字段）
    if item.get("brain"):
        msg["brain"] = item["brain"]  # "cloud"：本地大脑没开，这句是云端代答的
    if item.get("end"):
        msg["end"] = True  # 主人道别了：连续对话到这里结束，设备说完不再接着听
    return msg


async def voice_id(item: dict, listeners) -> int | None:
    """这段话要不要出声：有能播的设备、有能念的字、声音服务在。要的话分配一个编号。"""
    if not listeners or not tts.speech_text(item["text"]):
        return None
    if not await asyncio.to_thread(tts.available):
        return None
    return hub.next_id()


async def speak(targets, item: dict, sid: int):
    """合成一次，推给 targets 里所有设备：tts start → 二进制帧（PCM16 或 Opus 包）→ tts stop。"""
    text, mood = tts.speech_text(item["text"]), tts.mood_for(item)
    async with hub.voice_lock():
        live = {ws: hub.clients[ws].hush for ws in targets if ws in hub.clients}
        fmt = {ws: hub.clients[ws].audio_out for ws in live}
        enc = opus.Encoder(tts.SAMPLE_RATE) if "opus" in fmt.values() else None  # 要 Opus 的设备共用一个编码器

        def alive(ws):
            c = hub.clients.get(ws)
            return c is not None and c.hush == live[ws]

        for ws in list(live):
            start = {"type": "tts", "state": "start", "id": sid, "format": fmt[ws], "rate": tts.SAMPLE_RATE}
            if fmt[ws] == "opus":
                start["frame_ms"] = opus.FRAME_MS
            if not await hub.try_send(ws, start):
                live.pop(ws)
        if not live:
            return

        async def push(pcm: bytes, packets: list[bytes]):
            for ws in list(live):
                if not alive(ws):
                    continue
                try:
                    if fmt[ws] == "opus":
                        for p in packets:
                            await ws.send_bytes(p)
                    elif pcm:
                        await ws.send_bytes(pcm)
                except Exception:
                    live.pop(ws)
                    hub.clients.pop(ws, None)
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        flag = {"stop": False}  # 合成线程看这个：设备都走了/都被打断了就别合成了

        def work():
            err = None
            try:
                for pcm in tts.stream(text, mood, cancelled=lambda: flag["stop"]):
                    loop.call_soon_threadsafe(q.put_nowait, pcm)
            except Exception as e:
                log.warning("声音合成失败：%s", e)
                err = str(e) or type(e).__name__
            loop.call_soon_threadsafe(q.put_nowait, ("end", err))

        worker = asyncio.create_task(asyncio.to_thread(work))
        sent, err = 0, None
        while True:
            pcm = await q.get()
            if isinstance(pcm, tuple):
                err = pcm[1]
                break
            await push(pcm, enc.encode(pcm) if enc else [])
            sent += len(pcm)
            flag["stop"] = not any(alive(ws) for ws in live)
        await worker
        if enc and sent:
            await push(b"", enc.flush())  # 最后不够 60ms 的一截补静音发出去
        for ws in live:
            stop = {"type": "tts", "state": "stop", "id": sid}
            if not alive(ws):
                stop["cancelled"] = True
            elif err and not sent:
                stop["error"] = "声音没合成出来"
            await hub.try_send(ws, stop)
        log.info("说（%s，%.1f 秒）：%s", mood, sent / 2 / tts.SAMPLE_RATE, text)


class Listen:
    """一次“按住说话”：listen start 之后收到的二进制帧都是录音，listen stop 时整段识别。
    Opus 录音（ESP32）一帧一个包，收到就解码成 PCM 攒起来。"""

    def __init__(self):
        self.active = False
        self.rate = 16000
        self.buf = bytearray()
        self.decoder = None

    def start(self, fmt: str, rate: int):
        self.active, self.buf, self.rate = True, bytearray(), rate
        self.decoder = opus.Decoder(rate) if fmt == "opus" else None

    def feed(self, data: bytes):
        if not self.active or len(self.buf) >= self.max_bytes():
            return
        pcm = self.decoder.decode(data) if self.decoder else data
        self.buf.extend(pcm[: self.max_bytes() - len(self.buf)])

    def max_bytes(self):
        return int(config.ASR_MAX_SECONDS * self.rate * 2)


async def do_chat(ws: WebSocket, text: str, tone: str = ""):
    """一轮对话。在后台任务里跑：大脑想得再久（比如冷启动加载模型要近一分钟），
    handle() 的接收循环也照常读消息——否则 uvicorn 收不到心跳回应，四五十秒就把连接断了。"""
    client = hub.clients.get(ws)
    if not client:
        return
    async with client.turns:
        if not await hub.try_send(ws, {"type": "thinking"}):
            return
        loop = asyncio.get_running_loop()

        def on_tool(name, detail):  # 在大脑的线程里被调用：告诉设备“在网上查…”“在问云端…”
            msg = {"type": "thinking", "tool": name, "detail": detail}
            if name == "web_search":
                msg["search"] = detail  # 旧字段，先留着
            asyncio.run_coroutine_threadsafe(hub.try_send(ws, msg), loop)

        reply = await asyncio.to_thread(brain.chat, text, tone, on_tool)
        sid = await voice_id(reply, [ws] if client.audio else [])
        if not await hub.try_send(ws, say_msg(reply, proactive=False, sid=sid)):
            return
        await hub.try_send(ws, state_msg())
        if sid is not None:
            hub.spawn(speak([ws], reply, sid))  # 后台推声音，这期间设备说话（打断）也能马上收到


async def finish_listen(ws: WebSocket, pcm: bytes, rate: int):
    """录音结束：识别 → 把听到的话告诉设备 → 交给大脑。在后台任务里跑。"""
    seconds = len(pcm) / 2 / rate
    if not asr.available():
        await hub.try_send(ws, {"type": "heard", "text": "", "error": "本地语音识别还没装好"})
        return
    if seconds < config.ASR_MIN_SECONDS:
        await hub.try_send(ws, {"type": "heard", "text": ""})
        return
    heard = await asyncio.to_thread(asr.transcribe_pcm16, pcm, rate)
    tone = "、".join(t for t in (heard.emotion, heard.event) if t)
    # 听错的名字改回来（“独蜜”→“猪咪”）；只认出“.”“The.”这种：当没听清（多半是环境声）
    text = brain.meaningful(persona.fix_asr(heard.text))
    log.info("听到（%.1f 秒%s）：%s%s", seconds, f"，{tone}" if tone else "", heard.text,
             "（当没听清）" if heard.text and not text else "")
    await hub.try_send(ws, {"type": "heard", "text": text, "emotion": heard.emotion, "event": heard.event})
    if text:
        await do_chat(ws, text[:500], tone)


async def handle(ws: WebSocket):
    await ws.accept()
    listen = Listen()
    try:
        while True:
            frame = await ws.receive()
            if frame["type"] == "websocket.disconnect":
                break
            if frame.get("bytes") is not None:
                listen.feed(frame["bytes"])  # 二进制帧 = 录音数据（PCM16 或一个 Opus 包，看 listen start 的 format）
                continue
            try:
                msg = json.loads(frame.get("text") or "")
                kind = msg.get("type")
            except (ValueError, AttributeError):
                await hub.send(ws, {"type": "error", "message": "不是合法的 JSON"})
                continue

            if kind == "hello":
                if relay.via_relay(ws.scope) and not relay.token_ok(msg.get("token")):
                    log.warning("公网来的连接密钥不对，断开")
                    await hub.send(ws, {"type": "error", "message": "密钥不对"})
                    await ws.close(code=4401)
                    break
                device = str(msg.get("device") or "unknown")[:20]
                audio_out = msg.get("audio_out")
                if audio_out not in AUDIO_FORMATS or (audio_out == "opus" and not opus.available()):
                    audio_out = None  # 不认识的格式、或者没装 av 编不了 Opus：只发文字
                hub.clients[ws] = Client(device, audio_out)
                log.info("设备上线：%s%s（在线 %d 个）", device, f"，能出声（{audio_out}）" if audio_out else "",
                         len(hub.clients))
                await hub.send(ws, state_msg("welcome"))
                await push_pending()  # 离线期间攒下的主动搭话
            elif ws not in hub.clients:
                await hub.send(ws, {"type": "error", "message": "先发 hello"})
            elif kind == "chat":
                text = str(msg.get("text") or "").strip()[:500]
                if not text:
                    await hub.send(ws, {"type": "error", "message": "空消息"})
                    continue
                hub.spawn(do_chat(ws, text))
            elif kind == "listen":
                state = msg.get("state")
                if state == "start":
                    hub.clients[ws].hush += 1  # 主人一开口，团子就闭嘴听
                    fmt = msg.get("format") or "pcm16"
                    if fmt not in AUDIO_FORMATS or (fmt == "opus" and not opus.available()):
                        await hub.send(ws, {"type": "error", "message": f"不支持的录音格式：{fmt}"})
                        continue
                    listen.start(fmt, int(msg.get("rate") or 16000))
                elif state == "stop" and listen.active:
                    pcm, listen.buf, listen.active = bytes(listen.buf), bytearray(), False
                    hub.spawn(finish_listen(ws, pcm, listen.rate))
                elif state == "cancel":
                    listen.active, listen.buf = False, bytearray()
            elif kind == "hush":
                hub.clients[ws].hush += 1  # 主人摸头让团子别说了：正在推的这段声音不推了（tts stop 带 cancelled）
            elif kind == "pat":
                count = max(1, min(int(msg.get("count") or 1), 100))
                await asyncio.to_thread(brain.record_device_event, "pat", count)
                await hub.send(ws, state_msg())
            elif kind == "ping":
                await hub.send(ws, {"type": "pong"})
            else:
                await hub.send(ws, {"type": "error", "message": f"不认识的消息类型：{kind}"})
    finally:
        client = hub.clients.pop(ws, None)
        if client:
            log.info("设备离线：%s（在线 %d 个）", client.device, len(hub.clients))


async def push_pending():
    if not hub.clients:
        return
    for event in await asyncio.to_thread(brain.take_events):
        listeners = [ws for ws, c in hub.clients.items() if c.audio]
        sid = await voice_id(event, listeners)
        for ws in list(hub.clients):
            await hub.try_send(ws, say_msg(event, proactive=True, sid=sid if ws in listeners else None))
        if sid is not None:
            hub.spawn(speak(listeners, event, sid))


async def pusher():
    """后台任务：有设备在线时，把主动搭话推出去，并定期同步状态。"""
    last_state = time.monotonic()
    while True:
        await asyncio.sleep(config.PUSH_INTERVAL_S)
        try:
            await asyncio.to_thread(brain.fire_reminders)  # 设备不在线也照样到点（事件留着，连上就播）
            await push_pending()
            if hub.clients and time.monotonic() - last_state >= config.STATE_PUSH_S:
                last_state = time.monotonic()
                await hub.broadcast(state_msg())
        except Exception:
            log.exception("推送失败")
