"""Ollama 客户端：普通对话、流式对话、JSON 输出。（复制自 bill_analyzer/llm.py）"""
import json
import logging
import re
import subprocess
import threading
import time

import requests

import config

log = logging.getLogger("pet.llm")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def strip_think(text: str) -> str:
    """Qwen3 等模型会输出 <think>...</think> 思考过程，这里去掉。"""
    text = THINK_RE.sub("", text)
    if "<think>" in text:  # 没闭合的思考块
        text = text.split("<think>")[0]
    return text.strip()


class OllamaClient:
    def __init__(self, url=config.OLLAMA_URL, model=config.DEFAULT_MODEL, num_ctx=config.NUM_CTX):
        self.url = url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self._caps = None  # 模型能力（/api/show 的 capabilities），第一次用时查
        self._num_gpu = None  # 这次加载算好的显卡层数（模型被卸载后重新算）
        self._fit_lock = threading.Lock()

    # ---------- 放几层进显卡 ----------
    def num_gpu(self):
        """config.NUM_GPU 是数字就用它；是 "fit" 就在模型加载前按显卡此刻的空余现算，留出余量。

        8GB 卡上大脑要和声音服务、Windows 桌面挤在一起，桌面（dwm）的占用还会慢慢涨（实测 200MB → 744MB），
        固定层数迟早溢出：一溢出 Windows 就来回搬显存，大脑卡到 100% 出不来字。
        模型一直加载着时沿用同一个数（换数 Ollama 会重新加载）；被卸载后下一句再重新算。
        """
        if config.NUM_GPU != "fit":
            return config.NUM_GPU
        with self._fit_lock:
            loaded = self._loaded_vram_mb()
            if loaded is None or self._num_gpu is None:
                self._num_gpu = fit_layers(gpu_used_mb(), loaded or 0)
                log.info("大脑放 %s 层进显卡（按此刻空余显存算）", self._num_gpu)
            return self._num_gpu

    def _loaded_vram_mb(self):
        """我们的模型此刻在显卡上占多少 MB；没加载返回 None。"""
        try:
            for m in requests.get(f"{self.url}/api/ps", timeout=3).json().get("models") or []:
                if m.get("name") == self.model or m.get("model") == self.model:
                    return m.get("size_vram", 0) / 2**20
        except Exception:
            pass
        return None

    def capabilities(self):
        if self._caps is None:
            try:
                r = requests.post(f"{self.url}/api/show", json={"model": self.model}, timeout=10)
                r.raise_for_status()
                self._caps = set(r.json().get("capabilities") or [])
            except Exception:
                return set()  # 查不到先不缓存，下次再试
        return self._caps

    # ---------- 基础 ----------
    def list_models(self):
        r = requests.get(f"{self.url}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def is_up(self, connect_timeout=None):
        """connect_timeout：只等这么久连上（Windows 上端口没开要 2 秒才报“拒绝连接”）。"""
        try:
            if connect_timeout:
                requests.get(f"{self.url}/api/tags", timeout=(connect_timeout, 5)).raise_for_status()
            else:
                self.list_models()
            return True
        except Exception:
            return False

    def _payload(self, messages, stream, temperature, json_mode):
        p = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "keep_alive": config.KEEP_ALIVE,
            "options": {"temperature": temperature, "num_ctx": self.num_ctx},
        }
        n = self.num_gpu()
        if n is not None:
            p["options"]["num_gpu"] = n  # 放进显卡的层数，给声音服务和桌面留位置
        if "thinking" in self.capabilities():
            p["think"] = config.THINK  # Qwen3.5 这类会“先想一想”的模型：宠物聊天不用想，想了慢好几倍
        if json_mode:
            p["format"] = "json"
        return p

    # ---------- 非流式 ----------
    def _post_chat(self, payload):
        """模型要现加载时，显卡偶尔报 CUDA 错误、Ollama 返回 500（真机实测，第二次加载就好了）：等一下再试一次。"""
        r = requests.post(f"{self.url}/api/chat", json=payload, timeout=config.REQUEST_TIMEOUT)
        if r.status_code >= 500:
            log.warning("Ollama 返回 %d（多半是加载模型失败），%.0f 秒后再试一次", r.status_code, config.LOAD_RETRY_S)
            time.sleep(config.LOAD_RETRY_S)
            r = requests.post(f"{self.url}/api/chat", json=payload, timeout=config.REQUEST_TIMEOUT)
        r.raise_for_status()
        return r

    def chat(self, messages, temperature=0.2, json_mode=False) -> str:
        r = self._post_chat(self._payload(messages, False, temperature, json_mode))
        return strip_think(r.json()["message"]["content"])

    def chat_json(self, messages, temperature=0.0):
        """要求模型输出 JSON，并尽量容错地解析。解析失败返回 None。"""
        text = self.chat(messages, temperature=temperature, json_mode=True)
        return parse_json_loose(text)

    def chat_tools(self, messages, tools, temperature=0.7) -> dict:
        """带工具的对话（Ollama 原生 tools）：返回 assistant 消息，可能有 tool_calls。"""
        p = self._payload(messages, False, temperature, False)
        p["tools"] = tools
        msg = self._post_chat(p).json()["message"]
        msg["content"] = strip_think(msg.get("content") or "")
        return msg

    # ---------- 流式 ----------
    def chat_stream(self, messages, temperature=0.3):
        r = requests.post(
            f"{self.url}/api/chat",
            json=self._payload(messages, True, temperature, False),
            stream=True,
            timeout=config.REQUEST_TIMEOUT,
        )
        r.raise_for_status()

        def pieces():
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                yield data.get("message", {}).get("content", "")
                if data.get("done"):
                    break

        yield from filter_think_stream(pieces())


def gpu_used_mb() -> float:
    """整张卡此刻已用显存（MB）；查不到返回 0。"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        return float(out.strip().splitlines()[0])
    except Exception:
        return 0.0


def fit_layers(used_mb: float, ours_mb: float = 0) -> int:
    """按显卡空余算能放几层。used_mb 是整卡已用，ours_mb 是我们模型已经占着的（重算时当作可用）。"""
    f = config.GPU_FIT
    free = f["total_mb"] - f["reserved_mb"] - (used_mb - ours_mb) - f["margin_mb"]
    n = int((free - f["base_mb"]) // f["per_layer_mb"])
    return max(0, min(f["layers"], n))


def filter_think_stream(pieces):
    """在流式输出中过滤掉 <think>...</think>。"""
    buf, in_think, started = "", False, False
    for piece in pieces:
        buf += piece
        out = ""
        while True:
            if in_think:
                i = buf.find("</think>")
                if i == -1:
                    buf = buf[-8:]
                    break
                buf, in_think = buf[i + 8:], False
                continue
            i = buf.find("<think>")
            if i != -1:
                out += buf[:i]
                buf, in_think = buf[i + 7:], True
                continue
            k = buf.rfind("<", max(0, len(buf) - 7))
            if k != -1 and "<think>".startswith(buf[k:]):
                out, buf = out + buf[:k], buf[k:]
            else:
                out, buf = out + buf, ""
            break
        if out:
            if not started:
                out = out.lstrip()
                started = bool(out)
            if out:
                yield out
    if buf and not in_think:
        yield buf


def parse_json_loose(text):
    text = strip_think(text or "")
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"[\{\[].*[\}\]]", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None
