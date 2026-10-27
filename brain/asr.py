"""本地语音识别：sherpa-onnx + SenseVoice-Small（int8 量化），纯 CPU，不联网。

模型放在 models/sensevoice/（model.int8.onnx + tokens.txt），下载方法见 README。
SenseVoice 除了文字还会给出说话的情绪和声音事件（笑、哭），我们把它当作语气提示交给大脑。
"""
import logging
import re
import threading
from dataclasses import dataclass

import numpy as np

import config

log = logging.getLogger("pet.asr")

EMOTIONS = {"HAPPY": "开心", "SAD": "难过", "ANGRY": "生气", "FEARFUL": "害怕",
            "DISGUSTED": "嫌弃", "SURPRISED": "惊讶"}
EVENTS = {"Laughter": "在笑", "Cry": "在哭"}

_lock = threading.Lock()
_recognizer = None


@dataclass
class Heard:
    text: str
    emotion: str = ""  # 中文情绪词，没有明显情绪时为空
    event: str = ""    # 在笑 / 在哭


def model_files():
    d = config.ASR_MODEL_DIR
    return d / "model.int8.onnx", d / "tokens.txt"


def available() -> bool:
    return all(p.exists() for p in model_files())


def _get():
    global _recognizer
    if _recognizer is None:
        import sherpa_onnx  # 延迟导入：没装语音识别时，文字聊天照常能用

        model, tokens = model_files()
        _recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model), tokens=str(tokens), num_threads=config.ASR_THREADS,
            language=config.ASR_LANGUAGE, use_itn=True)  # use_itn：加标点、把“一百”写成 100
        log.info("语音识别模型已加载：%s", model.parent)
    return _recognizer


def warmup():
    """服务启动时在后台先把模型加载好，免得第一次说话要多等一两秒。"""
    if not available():
        log.warning("没找到语音识别模型（%s），按住说话不可用，文字聊天不受影响", config.ASR_MODEL_DIR)
        return
    with _lock:
        _get()


def _tag(s) -> str:
    m = re.search(r"<\|(\w+)\|>", s or "")
    return m.group(1) if m else (s or "")


def transcribe_pcm16(pcm: bytes, sample_rate: int = 16000) -> Heard:
    """识别一段 16 位小端单声道 PCM。"""
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    with _lock:
        rec = _get()
        stream = rec.create_stream()
        stream.accept_waveform(sample_rate, samples)
        rec.decode_stream(stream)
        r = stream.result
    return Heard(text=r.text.strip(),
                 emotion=EMOTIONS.get(_tag(getattr(r, "emotion", "")), ""),
                 event=EVENTS.get(_tag(getattr(r, "event", "")), ""))
