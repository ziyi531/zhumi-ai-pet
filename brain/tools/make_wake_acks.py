"""生成唤醒应答语音包：被“Hi，喵喵”叫醒时猪咪随口应一句（“在呢～”“怎么啦？”……）。

用猪咪自己的声音（声音服务 tts_service.py 要开着，大脑开着时就在）合成，去掉首尾静音，
存成 Ogg Opus（24kHz 单声道、60ms 一帧，和大脑推给板子的声音同一种格式），放进固件 main/assets/。
固件编译时把它们嵌进去（main/CMakeLists.txt 的 EMBED_FILES），叫醒时随机放一句——不用等大脑，离线也能应。

用法（在 ai_pet/ 目录）：python tools/make_wake_acks.py
"""
import sys
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tts  # noqa: E402

ACKS = [("在呢～", "happy"), ("怎么啦？", "normal"), ("嗯？", "normal"), ("我在呢。", "happy")]
OUT_DIR = Path(r"C:\Users\76012\esp\ai_pet_fw\main\assets")
RATE = tts.SAMPLE_RATE


def synth(text, mood) -> np.ndarray:
    pcm = b"".join(tts.stream(text, mood))
    return np.frombuffer(pcm, dtype=np.int16)


def trim(a: np.ndarray, threshold=500, pad_ms=40) -> np.ndarray:
    """去掉首尾静音，前后各留一点；最后 30ms 淡出，免得“啪”一声。"""
    loud = np.flatnonzero(np.abs(a.astype(np.int32)) > threshold)
    if not len(loud):
        return a
    pad = RATE * pad_ms // 1000
    a = a[max(0, loud[0] - pad): loud[-1] + pad].copy()
    fade = min(len(a), RATE * 30 // 1000)
    a[-fade:] = (a[-fade:] * np.linspace(1, 0, fade)).astype(np.int16)
    return a


def write_ogg(a: np.ndarray, path: Path):
    with av.open(str(path), "w", format="ogg") as out:
        s = out.add_stream("libopus", rate=RATE)
        s.layout = "mono"
        s.codec_context.bit_rate = 32000
        s.codec_context.options = {"frame_duration": "60", "application": "voip"}
        frame = av.AudioFrame.from_ndarray(a.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = RATE
        for p in s.encode(frame):
            out.mux(p)
        for p in s.encode(None):
            out.mux(p)


def main():
    if not tts.available():
        sys.exit("声音服务没开（先启动大脑）")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, (text, mood) in enumerate(ACKS):
        a = trim(synth(text, mood))
        path = OUT_DIR / f"wake_ack_{i}.ogg"
        write_ogg(a, path)
        print(f"{path.name}: {text}（{mood}）{len(a) / RATE:.2f} 秒，{path.stat().st_size} 字节")


if __name__ == "__main__":
    main()
