"""团子的声音：大脑这边的语音合成客户端。

真正合成在 tts_service.py（Qwen3-TTS，单独进程，跑在 .venv-fastq 里，只监听本机）；
大脑启动时自动把它拉起来，退出时关掉。这里负责：
  - 按表情和作息决定用什么情绪说（代码决定，LLM 不参与）；
  - 把回复里不该念的东西去掉（动作描写、表情符号）；
  - 流式取回 PCM16，交给 link.py 推给设备。
服务没装好、没起来时 available() 为 False，设备自己用文字/本地语音兜底。
"""
import logging
import os
import re
import subprocess
import threading
import time

import requests

import config

log = logging.getLogger("pet.tts")

SAMPLE_RATE = 24000
# 表情 → 说话的情绪（voice/voice.json 里每种情绪对应一句很淡的语气提示）
MOOD_BY_FACE = {"happy": "happy", "shy": "happy", "pout": "pout", "sad": "sad", "lonely": "sad",
                "sleepy": "sleepy", "sleep": "sleepy"}

_proc = None
_ready = False
_checked_at = 0.0  # 上次探测服务的时间：没起来时别每次都去连（Windows 连没开的端口要卡 2 秒）
_lock = threading.Lock()


def mood_for(item: dict) -> str:
    """深夜犯困的回复一律用犯困的语气，其余看表情。"""
    if item.get("voice") == "sleepy":
        return "sleepy"
    return MOOD_BY_FACE.get(item.get("face") or "", "normal")


def speech_text(text: str) -> str:
    """要念出来的部分：去掉（蹭蹭）这类动作描写、表情符号和 markdown 记号。"""
    t = re.sub(r"[（(][^（）()]{0,12}[）)]", "", text)
    t = re.sub(r"[\U0001F000-\U0001FAFF☀-➿️]", "", t)
    t = re.sub(r"[*#_`~]", "", t)
    return t.strip() if re.search(r"[一-鿿A-Za-z0-9]", t) else ""


def installed() -> bool:
    return config.TTS_PYTHON.exists() and (config.BASE_DIR / "voice" / "voice.json").exists()


def _health():
    try:
        return requests.get(f"{config.TTS_URL}/health", timeout=1).json()
    except Exception:
        return None


def available() -> bool:
    """服务起来并且模型加载好了。没起来时最多每 10 秒探测一次。"""
    global _ready, _checked_at
    if not _ready and time.monotonic() - _checked_at > 10:
        _checked_at = time.monotonic()
        h = _health()
        _ready = bool(h and h.get("ready"))
    return _ready


def warmup():
    """大脑启动时调用：服务没在跑就拉起来，等它把模型加载好（第一次大约 20 秒）。"""
    global _proc, _ready
    h = _health()
    if h and h.get("ready"):
        _ready = True
        log.info("团子的声音服务已经在跑了")
        return
    if not config.TTS_AUTOSTART:
        return
    if not installed():
        log.warning("没找到语音合成环境（%s），团子先用浏览器自带的语音", config.TTS_PYTHON)
        return
    with _lock:
        if _proc is None or _proc.poll() is not None:
            config.DATA_DIR.mkdir(exist_ok=True)
            logf = open(config.DATA_DIR / "tts_service.log", "ab")
            _proc = subprocess.Popen([str(config.TTS_PYTHON), "tts_service.py"], cwd=config.BASE_DIR,
                                     stdout=logf, stderr=subprocess.STDOUT,
                                     env={**os.environ, "PYTHONIOENCODING": "utf-8"},  # 日志用 UTF-8，别按 GBK 写
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            log.info("拉起团子的声音服务（pid %d）", _proc.pid)
    deadline = time.monotonic() + config.TTS_START_TIMEOUT_S
    while time.monotonic() < deadline:
        if _proc.poll() is not None:
            log.error("声音服务退出了（代码 %s），看 data/tts_service.log", _proc.returncode)
            return
        h = _health()
        if h and h.get("ready"):
            _ready = True
            log.info("团子的声音准备好了：%s", h.get("voice"))
            return
        time.sleep(1)
    log.error("声音服务 %d 秒还没准备好，看 data/tts_service.log", config.TTS_START_TIMEOUT_S)


def shutdown():
    global _proc, _ready
    if _proc and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _proc.kill()
    _proc, _ready = None, False


def stream(text: str, mood: str, cancelled=lambda: False):
    """流式合成：逐段产出 PCM16 字节（24kHz 单声道小端，每段偶数字节）。失败时抛异常。"""
    global _ready
    rest = b""
    try:
        with requests.post(f"{config.TTS_URL}/speak", json={"text": text, "mood": mood},
                           stream=True, timeout=(3, 30)) as r:
            r.raise_for_status()
            for chunk in r.iter_content(config.TTS_CHUNK_BYTES):
                if cancelled():
                    return
                data, rest = rest + chunk, b""
                if len(data) % 2:
                    data, rest = data[:-1], data[-1:]
                if data:
                    yield data
    except requests.ConnectionError:
        _ready = False  # 服务没了，下次 available() 重新探测
        raise
