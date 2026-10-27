"""实测：Ollama 的 Qwen 大脑和 Qwen3-TTS 0.6B（faster-qwen3-tts 加速版）同时放在显卡上，
显存够不够、两边各自多快、团子开口要等多久。

依次：
  1. 让 Ollama 加载大脑模型，记显存，测一轮对话速度（基线）；
  2. 加载 0.6B Base（CUDA Graph 加速），用 qwen3tts_trial.py 捏好的参考音流式合成几句台词，
     记显存、首段音频延迟和 RTF；
  3. TTS 模型留在显卡上，再测一轮对话速度（看有没有因为显存不够被挤到内存里变慢）。

用加速版的独立环境运行（在 ai_pet/ 目录下，Ollama 要先开着）：
  .venv-fastq\\Scripts\\python.exe tools\\qwen3tts_bench.py [--voice 声音id] [--no-llm]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

TRIAL = ROOT / "static" / "qwen3tts"
LINES = [("normal", "下午说眼睛酸，现在好点了吗？"), ("happy", "你回来啦！我等你好久了～"),
         ("pout", "哼，这次就先原谅你吧。"), ("sleepy", "已经很晚了哦，你也快去睡吧……晚安。")]


def gpu_used_mb():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    return int(out.strip().splitlines()[0])


def ollama_chat():
    """一轮团子风格的短对话，返回 (生成速度 token/s, 总用时 s)。"""
    t = time.perf_counter()
    r = requests.post(f"{config.OLLAMA_URL}/api/chat", json={
        "model": config.DEFAULT_MODEL, "stream": False, "keep_alive": "30m",
        "options": {"num_ctx": config.NUM_CTX, "temperature": 0.7},
        "messages": [{"role": "system", "content": "你是团子，寄居在电路板里的小灵体，黏人、好奇，每次只说 1~3 句话。"},
                     {"role": "user", "content": "我今天改了一下午 bug，眼睛好酸。"}]}, timeout=300)
    r.raise_for_status()
    j = r.json()
    return round(j["eval_count"] / (j["eval_duration"] / 1e9), 1), round(time.perf_counter() - t, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", help="用哪个候选声音的参考音（默认 manifest 里第一个）")
    ap.add_argument("--no-llm", action="store_true", help="不加载 Ollama，只测 TTS")
    args = ap.parse_args()
    manifest = json.loads((TRIAL / "manifest.json").read_text(encoding="utf-8"))
    vid = args.voice or manifest["order"][0]
    refs = {r["mood"]: r for r in manifest["voices"][vid]["refs"]}
    res = {"voice": vid, "gpu_idle_mb": gpu_used_mb()}

    if not args.no_llm:
        requests.post(f"{config.OLLAMA_URL}/api/generate", json={"model": config.DEFAULT_MODEL, "keep_alive": "30m"},
                      timeout=300).raise_for_status()
        ollama_chat()  # 热身，让 KV 缓存按 num_ctx 分配好
        res["gpu_llm_mb"] = gpu_used_mb()
        res["llm_tok_s_alone"], res["llm_reply_s_alone"] = ollama_chat()

    from faster_qwen3_tts import FasterQwen3TTS
    t = time.perf_counter()
    model = FasterQwen3TTS.from_pretrained(str(ROOT / "models" / "qwen3tts" / "Qwen3-TTS-12Hz-0.6B-Base"))
    res["tts_load_s"] = round(time.perf_counter() - t, 1)
    t = time.perf_counter()
    model.warmup(prefill_len=100)
    res["tts_warmup_s"] = round(time.perf_counter() - t, 1)

    def speak(mood, text):
        ref = refs[mood]
        t0 = time.perf_counter()
        first, total, sr = None, 0, 24000
        for chunk, sr, _ in model.generate_voice_clone_streaming(
                text=text, language="Chinese", ref_audio=str(TRIAL / ref["file"]), ref_text=ref["text"], chunk_size=8):
            if first is None:
                first = time.perf_counter() - t0
            total += len(chunk)
        return first, time.perf_counter() - t0, total / sr

    for mood, text in LINES:  # 热身：每个参考音先走一遍（第一次要抽取参考特征）
        speak(mood, text)
    res["gpu_llm_tts_mb"] = gpu_used_mb()

    cost = dur = 0.0
    ttfa = []
    for mood, text in LINES:
        first, s, d = speak(mood, text)
        cost += s
        dur += d
        ttfa.append(round(first, 2))
    res["tts_first_audio_s"] = ttfa
    res["tts_rtf"] = round(cost / dur, 2)  # 合成用时 / 音频时长，越小越快，< 1 就比实时快
    res["gpu_peak_mb"] = gpu_used_mb()
    if not args.no_llm:
        res["llm_tok_s_with_tts"], res["llm_reply_s_with_tts"] = ollama_chat()
    res["gpu_total_mb"] = torch.cuda.get_device_properties(0).total_memory // 2**20
    print(json.dumps(res, ensure_ascii=False, indent=1))
    (TRIAL / "bench.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
