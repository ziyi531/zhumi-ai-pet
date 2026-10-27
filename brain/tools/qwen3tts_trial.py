"""Qwen3-TTS 试听：先用文字“捏”出几种团子的候选声音，再用克隆模型合成团子常说的话。

流程和以后正式接入一样：
  1. 1.7B VoiceDesign：按“声音描述 + 此刻的情绪”各合成一段参考音（每种情绪一段，只做一次）；
  2. 0.6B Base：拿对应情绪的参考音克隆，合成任意台词（日常对话走这一步，显存小）。
用 faster-qwen3-tts（CUDA Graph 加速），和以后的 TTS 服务一致。顺便记录显存和速度。
输出到 static/qwen3tts/（wav + manifest.json），试听页是 /static/qwen3tts.html。
跑之前让 Ollama 先卸掉大脑模型，1.7B 和 7B 放不进同一张 8GB 卡。

用加速版的独立环境运行（在 ai_pet/ 目录下）：
  .venv-fastq\\Scripts\\python.exe tools\\qwen3tts_trial.py [--only 声音id,...]
"""
import argparse
import gc
import json
import subprocess
import time
from pathlib import Path

import soundfile as sf
import torch
from faster_qwen3_tts import FasterQwen3TTS

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models" / "qwen3tts"
OUT = ROOT / "static" / "qwen3tts"

# 候选声音：(id, 名字, 批次, 声音描述)。都是文字描述出来的原创音色，不模仿任何真人。
# 设备喇叭小、低音弱，所以都偏清亮。试听页按这里的顺序排，新的一批放前面。
VOICES = [
    # 第二批：用户要“更像小宠物、软糯少女感”
    ("nuomi", "糯米团子", "第二批 · 小宠物软糯", "十五六岁的软糯少女，声音又软又糯，像刚出炉的糯米团子，音色小巧圆润，像个小小的身体在说话，音调偏高，带一点奶气和鼻音，句尾常常软软地上扬拖长。"),
    ("maomi", "撒娇小猫", "第二批 · 小宠物软糯", "软糯的少女音，像一只黏人的小猫在撒娇蹭人，声音轻软，带一点气声和呼噜般的鼻音，音调偏高，说话慢悠悠的，尾音拖得软软长长。"),
    ("cangshu", "小仓鼠", "第二批 · 小宠物软糯", "声音软糯的少女，音色小小的、圆圆的，像一只小仓鼠在说话，奶声奶气但不幼稚，说话时好像嘴里含着糖，咬字有点软乎乎的，乖巧黏人。"),
    ("tuzi", "小白兔", "第二批 · 小宠物软糯", "温柔软糯的少女音，音调清甜偏高，声音轻轻的、有点怯生生，像一只小白兔，带轻微的鼻音和气声，尾音软绵绵的，让人很想保护她。"),
    ("lingti", "电路小灵体", "第二批 · 小宠物软糯", "一只寄居在电路板里的小灵体，声音是软糯的少女音，又带着小动物般的奶萌感，音色小巧圆润，音调偏高，带轻微鼻音和气声，句尾软软上扬，黏人、好奇，偶尔有点小傲娇。"),
    # 第一批
    ("ruanmeng", "软萌少女", "第一批", "十六七岁的少女，声音软糯甜美，音调偏高，说话带一点鼻音和撒娇的尾音，语速稍慢，亲切黏人。"),
    ("naisheng", "奶声小孩", "第一批", "七八岁的小女孩，奶声奶气，带点稚气，音调高，口齿清楚，说话活泼好奇。"),
    ("jingling", "中性小精灵", "第一批", "声音中性、清亮通透的小精灵，听不出是男孩还是女孩，音调偏高，轻快灵动，带点俏皮。"),
    ("yuanqi", "元气少年", "第一批", "十二三岁的男孩，声音清亮干净，元气活泼，说话轻快，偶尔带点小骄傲。"),
    ("lanmao", "慵懒猫系", "第一批", "年轻女孩，声音轻柔慵懒，带一点气声，说话慢悠悠的，像刚睡醒的小猫，温柔黏人。"),
]

# 情绪 → (追加到声音描述后的语气说明, 参考音台词)。以后由代码按表情选情绪，LLM 不参与。
MOODS = {
    "normal": ("语气自然亲切，像在和好朋友聊天。", "你好呀，我是团子，今天也要一起加油哦。"),
    "happy": ("此刻非常开心，带着笑意和撒娇的语气。", "嘿嘿，被你摸头了，好舒服呀。"),
    "pout": ("此刻有点傲娇，假装生气，嘴硬心软。", "哼，才不是想你了呢，我只是刚好醒着。"),
    "sad": ("此刻有点委屈和失落，声音低一些，带着鼻音。", "你都好久没理我了……我一个人好无聊。"),
    "sleepy": ("此刻非常困，声音迷迷糊糊，打着哈欠，语速很慢。", "哈欠……好困哦，我们一起去睡觉吧。"),
}

# 克隆合成的台词（和参考音不同，检验换句话还像不像同一个声音）
LINES = [
    ("normal", "下午说眼睛酸，现在好点了吗？"),
    ("normal", "……你还在忙吗？我探个头看看你。"),
    ("happy", "你回来啦！我等你好久了～"),
    ("happy", "嘿嘿，再摸摸我嘛～"),
    ("pout", "哼，这次就先原谅你吧。"),
    ("sad", "你是不是把我忘了呀……"),
    ("sleepy", "已经很晚了哦，你也快去睡吧……晚安。"),
]


def gpu_used_mb():
    """整张卡已用显存（含其它程序，比如 Ollama），单位 MB。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def load(name):
    t = time.perf_counter()
    model = FasterQwen3TTS.from_pretrained(str(MODELS / name))
    model.warmup(prefill_len=100)
    return model, round(time.perf_counter() - t, 1)


def timed(fn, **kw):
    torch.cuda.synchronize()
    t = time.perf_counter()
    wavs, sr = fn(**kw)
    torch.cuda.synchronize()
    return wavs[0], sr, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="只做这些声音（逗号分隔的 id）")
    args = ap.parse_args()
    voices = [v for v in VOICES if not args.only or v[0] in args.only.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.setdefault("voices", {})
    perf = {"gpu_before_mb": gpu_used_mb(), "engine": "faster-qwen3-tts"}

    # ---- 1. VoiceDesign：每种声音 × 每种情绪一段参考音 ----
    design, perf["design_load_s"] = load("Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    cost = dur = 0.0
    for vid, name, _, desc in voices:
        for mood, (tone, text) in MOODS.items():
            torch.manual_seed(1234)  # 同一声音的各情绪用同一个种子，音色更稳
            wav, sr, s = timed(design.generate_voice_design, text=text, language="Chinese", instruct=desc + tone)
            sf.write(OUT / f"{vid}_ref_{mood}.wav", wav, sr)
            cost += s
            dur += len(wav) / sr
            print(f"[design] {name} {mood}: {len(wav) / sr:.1f}s 音频，用时 {s:.1f}s", flush=True)
    perf["design_peak_mb"] = gpu_used_mb()
    perf["design_rtf"] = round(cost / dur, 2) if dur else None
    del design
    gc.collect()
    torch.cuda.empty_cache()

    # ---- 2. 0.6B Base：用对应情绪的参考音克隆，合成台词 ----
    base, perf["clone_load_s"] = load("Qwen3-TTS-12Hz-0.6B-Base")
    cost = dur = 0.0
    per = []
    for vid, name, batch, desc in voices:
        items = []
        for i, (mood, text) in enumerate(LINES):
            torch.manual_seed(1234)
            wav, sr, s = timed(base.generate_voice_clone, text=text, language="Chinese",
                               ref_audio=str(OUT / f"{vid}_ref_{mood}.wav"), ref_text=MOODS[mood][1])
            f = OUT / f"{vid}_line{i}.wav"
            sf.write(f, wav, sr)
            cost += s
            dur += len(wav) / sr
            per.append(s)
            items.append({"mood": mood, "text": text, "file": f.name, "sec": round(len(wav) / sr, 1), "cost": round(s, 1)})
            print(f"[clone] {name} {text}: {len(wav) / sr:.1f}s 音频，用时 {s:.1f}s", flush=True)
        manifest["voices"][vid] = {
            "name": name, "batch": batch, "desc": desc,
            "refs": [{"mood": m, "tone": MOODS[m][0], "text": MOODS[m][1], "file": f"{vid}_ref_{m}.wav"} for m in MOODS],
            "lines": items,
        }
    perf["clone_peak_mb"] = gpu_used_mb()
    perf["clone_rtf"] = round(cost / dur, 2) if dur else None
    perf["clone_avg_line_s"] = round(sum(per) / len(per), 1) if per else None  # 含每个参考音第一次抽特征
    perf["gpu"] = torch.cuda.get_device_name(0)
    manifest["perf"] = perf
    for vid, _, batch, _ in VOICES:  # 旧数据补上批次
        if vid in manifest["voices"]:
            manifest["voices"][vid]["batch"] = batch
    known = {v[0] for v in VOICES}  # qwen3tts_refine.py 生成的改进版不在 VOICES 里，保持在最前面
    manifest["order"] = ([vid for vid in manifest.get("order", []) if vid not in known]
                         + [v[0] for v in VOICES if v[0] in manifest["voices"]])
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(perf, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
