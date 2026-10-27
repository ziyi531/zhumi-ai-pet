"""团子的嗓子：Qwen3-TTS 合成服务，单独一个进程，跑在加速版环境 .venv-fastq 里。

大脑（server.py）启动时会自动拉起它；单独调试时也可以手动运行（在 ai_pet/ 目录）：
  .venv-fastq\\Scripts\\python.exe tts_service.py

只监听本机：
  GET  /health  -> {"ready": true, "voice": "小仓鼠", "sample_rate": 24000}
  POST /speak   {"text": "...", "mood": "normal|happy|pout|sad|sleepy"}
                -> 流式返回 PCM16（24kHz、单声道、小端），边合成边吐，第一段约 0.4 秒就到

声音和语气参数都在 voice/voice.json，换声音只改那里。情绪由大脑按表情决定，这里只照做。
和大脑分成两个进程，是因为加速版要锁 transformers==5.15.1、CUDA 版 torch，和大脑的环境冲突。
"""
import asyncio
import json
import logging
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from faster_qwen3_tts import FasterQwen3TTS
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent
VOICE_DIR = BASE_DIR / "voice"
MODEL_DIR = BASE_DIR / "models" / "qwen3tts"
PORT = int(os.environ.get("PET_TTS_PORT", "8601"))

CHUNK_STEPS = 6          # 每 6 帧（12Hz，约 0.5 秒）吐一段
MAX_PAUSE_S = 0.5        # 句中停顿最长保留这么久（“……”有时会让模型停好几秒）
SENTENCE_GAP_S = 0.18    # 句与句之间补一点停顿
SILENT_RMS = 0.004       # 低于这个算静音（约 -48 dBFS）

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("pet.tts_service")

voice = json.loads((VOICE_DIR / "voice.json").read_text(encoding="utf-8"))
REF_AUDIO = str(VOICE_DIR / voice["ref_audio"])
model = None
sample_rate = 24000
gpu_lock = threading.Lock()  # 一张卡一次只合成一句


def hanzi(text):
    return len(re.findall(r"[一-鿿A-Za-z0-9]", text))


def sentences(text):
    """按句号、问号、感叹号、换行切开；太短的并进下一句（“嗯？”单独合成容易怪）。"""
    parts = [p.strip() for p in re.split(r"(?<=[。！？!?；;\n])", text) if p.strip()]
    out, buf = [], ""
    for p in parts:
        buf += p
        if hanzi(buf) >= 6:
            out.append(buf)
            buf = ""
    if buf:
        if out and hanzi(buf) < 3:
            out[-1] += buf
        else:
            out.append(buf)
    return [s for s in out if hanzi(s) > 0]


def gated(chunks, stop):
    """去掉句首句尾的静音、把太长的句中停顿截短。chunks 是 float32 数组流。"""
    pending, started = [], False
    for c in chunks:
        if stop.is_set():
            return
        silent = len(c) == 0 or float(np.sqrt(np.mean(c * c))) < SILENT_RMS
        if silent:
            if started:
                pending.append(c)
            continue
        if pending:
            pause = np.concatenate(pending)[: int(MAX_PAUSE_S * sample_rate)]
            pending = []
            yield pause
        started = True
        yield c


def synth(text, mood, stop):
    """逐句流式合成，产出 PCM16 字节。带防跑飞：按字数限制长度，超长就截断。"""
    instruct = voice["calm"] + voice["moods"].get(mood, "")
    for i, s in enumerate(sentences(text)):
        n = hanzi(s)
        limit = int((n / 2.0 + 2) * sample_rate)  # 最慢按 2 字/秒再加 2 秒
        gen = model.generate_voice_clone_streaming(
            text=s, language="Chinese", ref_audio=REF_AUDIO, ref_text=voice["ref_text"],
            temperature=voice["temperature"], instruct=instruct or None,
            max_new_tokens=int(12 * (n / 2.0 + 2)), chunk_size=CHUNK_STEPS)
        try:
            if i:
                yield np.zeros(int(SENTENCE_GAP_S * sample_rate), dtype=np.int16).tobytes()
            total = 0
            for c in gated((np.asarray(c, dtype=np.float32) for c, _, _ in gen), stop):
                total += len(c)
                yield (np.clip(c, -1, 1) * 32767).astype("<i2").tobytes()
                if total > limit:
                    log.warning("这句太长了，截断：%s", s)
                    break
        finally:
            gen.close()
        if stop.is_set():
            return


app = FastAPI()


class SpeakIn(BaseModel):
    text: str
    mood: str = "normal"


@app.get("/health")
def health():
    mb = lambda v: round(v / 2**20)  # noqa: E731
    return {"ready": model is not None, "voice": voice["name"], "sample_rate": sample_rate,
            "gpu_reserved_mb": mb(torch.cuda.memory_reserved()), "gpu_peak_mb": mb(torch.cuda.max_memory_reserved())}


@app.post("/speak")
async def speak(req: SpeakIn):
    if model is None:
        raise HTTPException(503, "模型还在加载")
    text = req.text.strip()[:300]
    if not sentences(text):
        raise HTTPException(400, "没有能念的字")
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()

    def work():
        t0 = time.perf_counter()
        first = None
        try:
            with gpu_lock:
                for pcm in synth(text, req.mood, stop):
                    first = first or time.perf_counter() - t0
                    loop.call_soon_threadsafe(q.put_nowait, pcm)
            log.info("[%s] %s（首段 %.2fs，共 %.2fs）", req.mood, text, first or -1, time.perf_counter() - t0)
        except Exception:
            log.exception("合成失败：%s", text)
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)

    threading.Thread(target=work, daemon=True).start()

    async def body():
        try:
            while (pcm := await q.get()) is not None:
                yield pcm
        finally:
            stop.set()  # 大脑那边断开（比如主人开口打断）就别再合成了

    return StreamingResponse(body(), media_type="application/octet-stream",
                             headers={"X-Sample-Rate": str(sample_rate)})


class CpuEmbedding(torch.nn.Module):
    """文字词表（15 万 × 2048，约 620MB）放到 CPU 上查。

    它只在每句开头把台词和语气提示的几十个字查一次（不在 CUDA Graph 的逐帧循环里），挪到 CPU 几乎不影响速度，
    却能给 Ollama 的大脑腾出 0.6GB 显存——8GB 的卡上两边都整个放在显卡上会超，超了就被挤进内存，慢几十倍。
    """

    def __init__(self, emb: torch.nn.Embedding):
        super().__init__()
        self.table = emb.weight.detach().to("cpu")
        self.device = emb.weight.device

    def forward(self, ids):
        return self.table[ids.to("cpu")].to(self.device)


def load():
    global model, sample_rate
    t = time.perf_counter()
    m = FasterQwen3TTS.from_pretrained(str(MODEL_DIR / voice["model"]), max_seq_len=1024)
    talker = m.model.model.talker
    talker.model.text_embedding = CpuEmbedding(talker.model.text_embedding)
    torch.cuda.empty_cache()
    m.warmup(prefill_len=100)
    sample_rate = getattr(m, "sample_rate", None) or 24000
    # 每种情绪先完整合成一句：抽好参考音特征（会被缓存），CUDA Graph 也热一遍，第一次说话就不卡
    for mood in voice["moods"]:
        for _ in synth_warm(m, "嗯，我在呢。", mood):
            pass
    torch.cuda.empty_cache()  # 预热时的临时缓冲还给显卡
    model = m
    log.info("团子的声音（%s）准备好了，用时 %.1fs，显存峰值 %.2f GB、常驻 %.2f GB", voice["name"], time.perf_counter() - t,
             torch.cuda.max_memory_reserved() / 2**30, torch.cuda.memory_reserved() / 2**30)


def synth_warm(m, text, mood):
    instruct = voice["calm"] + voice["moods"].get(mood, "")
    yield from m.generate_voice_clone_streaming(
        text=text, language="Chinese", ref_audio=REF_AUDIO, ref_text=voice["ref_text"],
        temperature=voice["temperature"], instruct=instruct or None, max_new_tokens=60, chunk_size=CHUNK_STEPS)


if __name__ == "__main__":
    threading.Thread(target=load, daemon=True).start()  # 先把端口开起来，/health 能报“还在加载”
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
