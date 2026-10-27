"""在用户挑中的声音上微调“说话方式”，音色不变。

用户反馈（2026-10-19，小仓鼠）：情绪起伏太大，像朗读课文时刻意的抑扬顿挫；语速再慢一点点。
所以这里不再给每种情绪各“演”一段参考音，全部从那段平常语气的参考音克隆：
  - 平稳：只用平常参考音；
  - 慢一点：参考音先用 WSOLA 不变调放慢，克隆出来的话会跟着放慢（不对成品硬拉，避免失真）；
  - 淡淡情绪：再加一句很淡的语气提示（Base 模型的 instruct，ICL 模式下才比较听话）。
每句都算语速（字/秒，去掉首尾静音），写进 manifest，试听页显示。

用加速版的独立环境运行（在 ai_pet/ 目录下）：
  .venv-fastq\\Scripts\\python.exe tools\\qwen3tts_refine.py [--voice cangshu] [--slow 0.92]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from audiotsm import wsola
from audiotsm.io.array import ArrayReader, ArrayWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen3tts_trial import LINES, MODELS, MOODS, OUT, FasterQwen3TTS  # noqa: E402

# 所有“淡淡情绪”共用的前半句：压住朗读腔
CALM = "像平时随口和熟悉的人聊天，不像朗读也不像配音，语气起伏很小，语速稍慢、不慌不忙。"
SUBTLE = {
    "normal": "",
    "happy": "心情不错，语气里带一点点笑意。",
    "pout": "有一点点小傲娇，轻轻地嘟囔。",
    "sad": "有一点点失落，声音稍微轻一些。",
    "sleepy": "有点困了，声音软软的，稍微再慢一点。",
}


def stretch(wav, speed):
    """WSOLA 不变调变速，speed < 1 变慢。"""
    reader = ArrayReader(wav[np.newaxis, :].astype(np.float32))
    writer = ArrayWriter(channels=1)
    wsola(channels=1, speed=speed).run(reader, writer)
    return writer.data[0]


def hanzi(text):
    return len(re.findall(r"[一-鿿]", text))


def rate(path, text):
    """语速：汉字数 / 去掉首尾静音后的时长。"""
    y, sr = sf.read(path, dtype="float32")
    y, _ = librosa.effects.trim(y, top_db=35)
    return round(hanzi(text) / max(0.1, len(y) / sr), 2)


def trim(wav, sr, pad=0.12):
    """切掉首尾多余静音（“……”结尾时模型会拖好几秒），两头各留一点。"""
    _, (a, b) = librosa.effects.trim(wav, top_db=40)
    p = int(pad * sr)
    return wav[max(0, a - p):min(len(wav), b + p)]


def speak(model, text, ref, ref_text, temp, instruct, tries=3):
    """合成一句，带防跑飞：按字数限制最长 token，太长就换种子重来。返回 (wav, sr, 第几次成功)。"""
    n = hanzi(text)
    max_tokens = int(12 * (n / 2.0 + 2))  # 12Hz：最慢按 2 字/秒再加 2 秒
    limit_s = n / 2.5 + 1.5               # 正常说完不该超过这么长
    for k in range(tries):
        torch.manual_seed(1234 + k)
        wavs, sr = model.generate_voice_clone(text=text, language="Chinese", ref_audio=str(ref), ref_text=ref_text,
                                              temperature=temp, instruct=instruct, max_new_tokens=max_tokens)
        wav = trim(wavs[0], sr)
        if len(wav) / sr <= limit_s:
            return wav, sr, k + 1
        print(f"  第 {k + 1} 次太长（{len(wav) / sr:.1f}s > {limit_s:.1f}s），换种子重来", flush=True)
    return wav, sr, tries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="cangshu")
    ap.add_argument("--slow-ref", type=float, default=0.85, help="参考音放慢到原速的多少")
    ap.add_argument("--slow-out", type=float, default=0.94, help="“再慢一点”版本把合成结果再放慢到多少")
    args = ap.parse_args()
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    src = manifest["voices"][args.voice]
    ref_text = MOODS["normal"][1]
    ref_file = OUT / f"{args.voice}_ref_normal.wav"

    wav, sr = sf.read(ref_file, dtype="float32")
    slow_file = OUT / f"{args.voice}_slow_ref_normal.wav"
    sf.write(slow_file, stretch(wav, args.slow_ref), sr)

    # (id, 名字, 参考音, 采样温度, 是否加淡淡情绪, 成品放慢, 说明)
    pr, po = round((1 - args.slow_ref) * 100), round((1 - args.slow_out) * 100)
    variants = [
        (f"{args.voice}_slow", f"{src['name']} · 平稳 · 慢一点", slow_file, 0.75, False, None,
         f"所有台词都从同一段平常语气的参考音克隆（参考音先不变调放慢约 {pr}%），情绪只靠台词本身；采样更保守，起伏更小。"),
        (f"{args.voice}_slow_mood", f"{src['name']} · 平稳 · 慢一点 · 淡淡情绪", slow_file, 0.75, True, None,
         "在上一个的基础上，每句加一句很淡的语气提示：" + CALM),
        (f"{args.voice}_slow_mood_post", f"{src['name']} · 平稳 · 再慢一点 · 淡淡情绪", slow_file, 0.75, True, args.slow_out,
         f"和上一个一样，再把成品不变调放慢约 {po}%。"),
    ]

    model = FasterQwen3TTS.from_pretrained(str(MODELS / "Qwen3-TTS-12Hz-0.6B-Base"))
    model.warmup(prefill_len=100)
    batch = f"第三批 · {src['name']}改进（同一个音色）"
    for vid, name, ref, temp, mood_hint, post, how in variants:
        items = []
        for i, (mood, text) in enumerate(LINES):
            ins = (CALM + SUBTLE[mood]) if mood_hint else None
            wav, out_sr, tries = speak(model, text, ref, ref_text, temp, ins)
            if post:
                wav = stretch(wav, post)
            f = OUT / f"{vid}_line{i}.wav"
            sf.write(f, wav, out_sr)
            items.append({"mood": mood, "text": text, "file": f.name, "sec": round(len(wav) / out_sr, 1),
                          "rate": rate(f, text), "tries": tries})
            print(f"[{name}] {text}: {items[-1]['sec']}s，{items[-1]['rate']} 字/秒", flush=True)
        manifest["voices"][vid] = {
            "name": name, "batch": batch, "desc": how + f"音色同第二批的“{src['name']}”。",
            "refs": [{"mood": "normal", "tone": "放慢后的参考音" if ref == slow_file else "原参考音",
                      "text": ref_text, "file": Path(ref).name}],
            "lines": items,
        }

    # 同一批里上一轮的试验版本不要了
    current = {v[0] for v in variants}
    for old in [k for k, v in manifest["voices"].items() if v.get("batch") == batch and k not in current]:
        for it in manifest["voices"].pop(old)["lines"]:
            (OUT / it["file"]).unlink(missing_ok=True)
    manifest["order"] = [vid for vid in manifest["order"] if vid in manifest["voices"]]

    # 给所有已有台词补上语速，方便和原版对比
    for v in manifest["voices"].values():
        for it in v["lines"]:
            it.setdefault("rate", rate(OUT / it["file"], it["text"]))
    new = [v[0] for v in variants]
    manifest["order"] = new + [vid for vid in manifest["order"] if vid not in new]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
