"""把八条风表情上方的大字预先渲染成 8 位透明度蒙版，生成 main/face/captions.h。

芯片上不带字体引擎，只贴这几张小图。改字或换字体后重新运行：
    python tools/gen_captions.py

注意：默认用 Windows 自带的“微软雅黑 Bold”，只适合自用；如果固件要分发，
换成开源字体（比如思源黑体 / Noto Sans SC），用 --font 指定。
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# 和网页预览 faces.html 里 bajo.captions 一致
CAPTIONS = {
    "listen": "嗯？", "curious": "嗯？？", "pout": "哼！", "sad": "呜呜…", "lonely": "在吗…",
    "sleepy": "好困", "sleep": "Zzz", "happy": "嘿嘿", "shy": "害羞", "peek": "偷看", "think": "让我想想",
}
FACE_ORDER = ["idle", "listen", "think", "happy", "shy", "pout", "sad", "lonely", "curious", "sleepy", "sleep", "peek"]
SIZE = 30
STROKE = 4  # 网页里是 7px 的描边线宽，也就是向外 3.5px

OUT = Path(__file__).resolve().parent.parent / "main" / "face" / "captions.h"


def render(font, text):
    # 以“水平居中、基线”为锚点，这样设备上只要知道中心点和基线位置
    l, t, r, b = font.getbbox(text, anchor="ms", stroke_width=STROKE)
    w, h = r - l, b - t
    outline = Image.new("L", (w, h), 0)
    fill = Image.new("L", (w, h), 0)
    ImageDraw.Draw(outline).text((-l, -t), text, font=font, fill=255, anchor="ms",
                                 stroke_width=STROKE, stroke_fill=255)
    ImageDraw.Draw(fill).text((-l, -t), text, font=font, fill=255, anchor="ms")
    return l, t, w, h, outline.tobytes(), fill.tobytes()


def c_array(name, data):
    rows = [", ".join(str(v) for v in data[i:i + 32]) for i in range(0, len(data), 32)]
    return f"static const uint8_t {name}[] = {{\n    " + ",\n    ".join(rows) + "\n};\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", default=r"C:\Windows\Fonts\msyhbd.ttc")
    args = ap.parse_args()
    font = ImageFont.truetype(args.font, SIZE)

    parts = ["// 由 tools/gen_captions.py 生成，不要手改。\n#pragma once\n#include <cstdint>\n\n"
             "namespace face {\n\n"
             "struct Caption {\n    int dx, dy, w, h;  // 相对“中心、基线”的左上角偏移和尺寸\n"
             "    const uint8_t* outline;\n    const uint8_t* fill;\n};\n\n"]
    entries = []
    for face in FACE_ORDER:
        text = CAPTIONS.get(face)
        if not text:
            entries.append("    {0, 0, 0, 0, nullptr, nullptr},  // " + face)
            continue
        l, t, w, h, outline, fill = render(font, text)
        parts.append(c_array(f"kCap_{face}_outline", outline))
        parts.append(c_array(f"kCap_{face}_fill", fill))
        entries.append(f"    {{{l}, {t}, {w}, {h}, kCap_{face}_outline, kCap_{face}_fill}},  // {face}: {text}")
    parts.append("// 下标和 FaceId 一致\nstatic const Caption kCaptions[] = {\n" + "\n".join(entries) + "\n};\n\n}  // namespace face\n")
    OUT.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB source)")


if __name__ == "__main__":
    main()
