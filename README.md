# Zhumi 猪咪 — a local-first AI voice pet

An ESP32-S3 desk pet with an animated face that you talk to by voice. A home PC is its brain: speech recognition, the LLM, its voice and its long-term memory all run locally. When you're away from home, it reaches the PC through a small relay server.

一个能语音聊天的桌面 AI 宠物：ESP32-S3 是身体（表情、触摸、收音、喇叭），家里的电脑是大脑（识别、大模型、声音、记忆），全部在本地跑。详细文档是中文，在 `brain/` 和 `firmware/` 各自的 README 里。

```
 ESP32-S3 board                         Home PC (brain/)
 ┌──────────────────────┐   WebSocket   ┌──────────────────────────────────────┐
 │ face renderer (C++)  │  Opus 16 kHz  │ SenseVoice ASR (sherpa-onnx)         │
 │ wake word "Hi,喵喵"  │ ────────────▶ │ Qwen3.5-9B on Ollama + tool calling  │
 │ AEC / VAD (ESP-SR)   │               │ memory: recall · forgetting · diary  │
 │ touch: pat / hold /  │ ◀──────────── │ Qwen3-TTS custom voice               │
 │        hush          │  Opus 24 kHz  │ rules: loneliness, follow-ups,       │
 └──────────────────────┘               │        reminders (all in code)       │
        │ away from home: wss           └──────────────────────────────────────┘
        └──▶ Lightsail Nginx ──SSH reverse tunnel──▶ PC
```

## Highlights

- **Voice loop**: wake word or hold-to-talk. Opus audio streams both ways. Multi-turn conversation ends on silence, on a goodbye, or on noise. About 2 s from the end of your sentence to the pet's speech, all local.
- **The LLM only words things.** Numbers and decisions live in code: when to reach out (loneliness, follow-ups, curiosity, cooldowns, quiet hours), reminders, date and math tools. The model only phrases the result.
- **Memory**: BGE-M3 recall with a keyword bonus and a forgetting curve. Nightly consolidation merges memories, writes a diary and revises the pet's self-view. Personality traits drift slowly over time.
- **Firmware** (ESP-IDF, C++): an anti-aliased SDF face rasterizer (468 → 22 ms per frame), audio built on the open-source xiaozhi-esp32 stack, home Wi-Fi / phone-hotspot switching, and a TLS relay with a per-device token.
- **Privacy**: conversations stay on the PC. Optional cloud use is narrow and switchable: web search sends only a cleaned query, weather sends only a city name, and when the local model is down Gemini can answer as a stand-in.
- **Tests**: `brain/selftest.py` has 28 end-to-end tests using fake LLM, TTS and cloud backends.

## Layout

| Path | What |
|---|---|
| `brain/` | FastAPI server, brain logic, memory, tools, TTS service, web pages (see `brain/README.md`, `PROTOCOL.md`, `PROJECT.md`) |
| `firmware/` | ESP-IDF project for the LCSC 立创实战派 ESP32-S3 board (see `firmware/README.md`) |

## Not included

- Personal data: the chat and memory database, API keys and the device token.
- Machine-specific settings: `brain/local_settings.py` (copy `local_settings.example.py`) and the firmware `sdkconfig` (Wi-Fi, server URL and token are set in `idf.py menuconfig` → AI Pet).
- Model weights: SenseVoice, Qwen3-TTS and the Ollama models. Download them as described in `brain/README.md`.
- Build outputs, ESP-IDF managed components and the Python virtual environment.

## Third-party code

`firmware/main/xiaozhi_audio/` is adapted from [xiaozhi-esp32](https://github.com/78/xiaozhi-esp32) (MIT, license file included in that folder). The brain uses `lunardate` (GPL-3.0) as a pip dependency for lunar holidays.
