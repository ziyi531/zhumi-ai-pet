"""Opus 编解码（ESP32 用）：用 PyAV 自带的 libopus，不用另装 DLL。

ESP32 的语音模块（小智）本来就收发 Opus：录音 16kHz、团子的声音 24kHz，都是单声道、一包 60ms。
网页团子照旧用 PCM（局域网里带宽不成问题，浏览器也省得解码）。
"""
import logging

log = logging.getLogger("pet.opus")
FRAME_MS = 60

try:
    import av
    import numpy as np
except ImportError:  # 没装 av：ESP32 就只收文字，不出声、不能说话
    av = None


def available() -> bool:
    return av is not None


class Encoder:
    """PCM16 → Opus 包。PCM 可以随便切块喂进来，凑够 60ms 出一包；最后 flush 把不够一包的补静音发出去。"""

    def __init__(self, rate: int):
        self.rate = rate
        self.frame = rate * FRAME_MS // 1000
        self.ctx = av.CodecContext.create("libopus", "w")
        self.ctx.sample_rate, self.ctx.layout, self.ctx.format = rate, "mono", "s16"
        self.ctx.bit_rate = 32000
        self.ctx.options = {"frame_duration": str(FRAME_MS), "application": "voip"}
        self.ctx.open()
        self.buf = bytearray()
        self.pts = 0

    def _frame(self, pcm: bytes):
        f = av.AudioFrame.from_ndarray(np.frombuffer(pcm, dtype=np.int16).reshape(1, -1), format="s16", layout="mono")
        f.sample_rate, f.pts = self.rate, self.pts
        self.pts += self.frame
        return f

    def encode(self, pcm: bytes) -> list[bytes]:
        self.buf.extend(pcm)
        out, step = [], self.frame * 2
        while len(self.buf) >= step:
            out += [bytes(p) for p in self.ctx.encode(self._frame(bytes(self.buf[:step])))]
            del self.buf[:step]
        return out

    def flush(self) -> list[bytes]:
        out = []
        if self.buf:
            out += self.encode(b"\0" * (self.frame * 2 - len(self.buf)))
        out += [bytes(p) for p in self.ctx.encode(None)]
        return out


class Decoder:
    """Opus 包 → PCM16（单声道、指定采样率）。一个包解出来就是 60ms。"""

    def __init__(self, rate: int):
        self.rate = rate
        self.ctx = av.CodecContext.create("libopus", "r")
        self.ctx.sample_rate, self.ctx.layout = rate, "mono"
        self.resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)

    def decode(self, packet: bytes) -> bytes:
        try:
            frames = self.ctx.decode(av.Packet(packet))
        except Exception as e:  # 丢包、坏包：跳过这一包，别让整段录音作废
            log.debug("Opus 包解不开：%s", e)
            return b""
        out = bytearray()
        for f in frames:
            for r in self.resampler.resample(f):
                out += r.to_ndarray().astype(np.int16).tobytes()
        return bytes(out)
