# 团子：便携 AI 宠物 · 第一阶段（PC 端灵魂闭环）

家里 PC 是大脑（Ollama + Qwen），ESP32 是身体。这一阶段先在 PC 上跑通心跳、状态、记忆和主动搭话，用网页假设备代替硬件。

## 运行

在 `ai_pet/` 目录下运行：

```
pip install -r requirements.txt
python -m uvicorn server:app --port 8600      # 或者双击 run.bat
```

浏览器打开 http://localhost:8600 是网页版团子（八条风表情、摸头、聊天、主动搭话）；http://localhost:8600/debug 是调试用假设备（状态条、快进时钟、立即心跳）。表情引擎在 `static/face_engine.js`，表情预览页 `static/faces.html` 和网页团子共用它。需要 Ollama 已启动，并且准备好大脑模型（默认 `qwen3.5:9b-text`，可以用环境变量 `PET_MODEL` 修改）。Ollama 不可用时，页面和规则照常运行，对话会改用兜底文案。

大脑模型是 Qwen3.5-9B 去掉视觉模块的纯文字版（8GB 显卡要和团子的声音服务挤在一起，省一点是一点）：

```
ollama pull qwen3.5:9b
ollama show --modelfile qwen3.5:9b
```

把输出里**第一个** `FROM` 的路径（主模型）填进 `ollama/qwen3.5-9b-text.Modelfile`（第二个 `FROM` 是视觉模块，不要），然后：

```
ollama create qwen3.5:9b-text -f ollama\qwen3.5-9b-text.Modelfile
```

它默认只把 33 层里的 28 层放进显卡（`PET_NUM_GPU` 可改），不开“思考”（`PET_THINK=1` 可开），用 Ollama 原生工具调用上网查资料。换回旧模型：`PET_MODEL=qwen2.5:7b`。

不依赖 Ollama 的自测：`python selftest.py`（使用临时数据库）。

### 按住说话（本地语音识别，可选）

用 sherpa-onnx 跑 SenseVoice-Small（int8），纯 CPU，不联网。没装的话文字聊天照常能用。

```
pip install sherpa-onnx
```

再把模型的 `model.int8.onnx`（约 230MB）和 `tokens.txt` 放到 `models/sensevoice/`，来源：
https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17

浏览器只允许在 `localhost` 或 https 页面上用麦克风，所以按住说话要在跑服务的这台电脑上打开。

### 团子的声音（本地语音合成，可选，要 NVIDIA 显卡）

团子用自己的声音说话：Qwen3-TTS（阿里通义，Apache 协议），用 faster-qwen3-tts 加速（CUDA Graph）。声音是用一段文字描述“捏”出来的原创音色“小仓鼠”，不克隆任何真人；参考音和语气参数在 `voice/`（`voice.json` 写了怎么来的）。捏声音的过程和候选在 http://localhost:8600/voices 。

合成跑在单独的进程 `tts_service.py`（只监听本机 8601 端口），用独立环境 `.venv-fastq`，因为它要 CUDA 版 torch 和锁版本的 transformers。大脑启动时会自动拉起它（约 20 秒加载好），退出时一起关掉。装法：

```
python -m venv .venv-fastq
.venv-fastq\Scripts\python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv-fastq\Scripts\python -m pip install -r requirements-tts.txt
```

再把 https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base 下载到 `models/qwen3tts/Qwen3-TTS-12Hz-0.6B-Base/`（2.3GB）。以后想重新捏声音才需要 1.7B-VoiceDesign（4.2GB），见 `tools/qwen3tts_trial.py`。

- 说话的情绪由代码按表情决定（开心/傲娇/委屈/犯困/平常），每种只加一句很淡的语气提示，不夸张。
- 网页团子在 hello 里声明能播声音，大脑回复后把语音流式推过去，回复到达后约 0.4 秒开口；主人一按说话就停。
- 显存：声音服务约 2.5GB，和大脑（qwen3.5:9b-text 放 28 层，约 4.6GB）一起放进 8GB 卡。它把 3 亿参数的文字词表放在 CPU 上（省 0.6GB），否则两边加起来会超、被挤进内存后慢几十倍。
- 没装、没起来或合成失败时，网页团子退回电脑自带的本地中文语音。

### 联网搜索（可选）

团子遇到不确定的事（新游戏、新闻、人物、价格……）会先上网查，再用自己的口吻回答；闲聊、主人说自己的事都不查。实现是工具调用：大脑发起 `web_search` 调用（Ollama 原生 tools；旧的 qwen2.5 改用 JSON 里写 `{"search": "..."}`），代码去搜，把标题和摘要作为工具返回交给它再回答。

- 用 `ddgs`（`pip install ddgs`，已在 requirements.txt 里），轮流问几个公开搜索引擎，免注册、免 API key。
- **只有搜索词离开这台电脑**，对话、记忆、主人的资料都不发；搜索词里出现主人的称呼会先抹掉。网页团子会显示“上网查「…」中”，回答下面注明查了什么。
- 不想让它联网：启动前设环境变量 `PET_WEB_SEARCH=0`。
- 一次大约多等 2~5 秒（搜索 1~4 秒，新闻类更久）。天气这类动态网页只能搜到链接、拿不到具体数字。

### 小工具箱和问云端（toolbox.py、cloud.py）

团子遇到不好在本地处理的事，会调用小工具，再用自己的口吻回答。只有你在**提问或请它办事**时才把工具交给它（`brain.wants_tools`），“我吃了火锅”这种陈述连工具都看不到。

| 工具 | 干什么 | 什么会离开电脑 |
|---|---|---|
| 算日子 | 节日（含春节、中秋等农历节日）还有几天、某天星期几 | 无 |
| 计算器 | 算账、打折、百分比（只认算式，不执行代码） | 无 |
| 天气 | 现在和三天预报（Open-Meteo，免注册） | 城市名 |
| 上网查 | 新游戏、新闻、价格 | 搜索词 |
| 问云端 | 9B 答不好的难题：推理、讲原理、写长一点的东西、代码 | 一句不含个人信息的问题 |
| 定提醒 | “30 秒后提醒我……”“三点叫我……”：到点板子出声提醒 | 无 |

- **时钟**：调试页（`/debug`）的快进会给大脑的时钟加偏移量，存在数据库里、重启也不丢。2026-10-27 发现快了约 40 小时，被问几点时答错（它当时还无视提示词编了个“下午三点半”，现在问几点、几号时会把真实钟点直接交给它）。大脑启动时有偏移会在日志里警告；关掉大脑后运行 `python tools/fix_clock.py` 把时钟和库里记下的所有时间一起拨回来（先自动备份）。
- **定时提醒**（2026-10-28）：大脑调 `set_reminder`，时间由代码算（相对时间或钟点，最远 7 天），存进 `reminders` 表。`link.pusher` 每秒查一次，到点马上推给设备出声（固定句式“叮——主人，时间到啦，该……啦！”，不等大脑现想）。
  主人自己定的，所以不受免打扰、冷却、每天主动次数限制，也不占主动次数；大脑关着时到点的，开机后 2 小时内补响，再久就作废。
  以前没有这个工具时，它嘴上答应“30 秒后提醒你”，其实被记成了回访，还被推到第二天；现在嘴上答应却没调用会被提醒一次（`brain.SAYS_WILL`），定了提醒的那句也不再记回访。待响的提醒在 `/api/state` 的 `reminders` 里。

- 天气默认用你所在的城市：设 `PET_CITY=城市名`，或者跟团子聊天时说过住哪儿（记进画像）；都没有它会问你。`PET_WEATHER=0` 关掉。
- **问云端用 Gemini API**。注意 Google AI Pro 订阅**不含** API，API 单独计费；套餐每月送的 Google Cloud 额度可以抵扣。**一定要开通 Cloud 结算用付费层**：免费层的提问会被 Google 拿去改进产品。步骤：
  1. 在 Google AI Studio 生成 API key，并给它关联的 Cloud 项目开通结算。
  2. 把 key 写进 `data/gemini_api_key.txt`（只放 key 一行），或者设环境变量 `GEMINI_API_KEY`。
  3. 重启大脑。默认模型 `gemini-3.5-flash-lite`（免费层实测约 5 秒；`gemini-3.8-flash` 要 60 秒以上，宠物等不了；`PET_CLOUD_MODEL` 可改），每天最多 20 次（`PET_CLOUD_DAILY`），`PET_CLOUD=0` 关掉。
  - 也可以先用免费层（不开结算）：问的是一般知识没关系，但这些问题可能被 Google 拿去训练、人工审阅。
  - 记事本 / PowerShell 存的 UTF-8 文件开头会带 BOM，代码读的时候会去掉，不用管。
- 只发一句问题：大脑先改写成能单独看懂的一句，代码再抹掉你的称呼；对话历史、记忆、画像都不发。云端的回答交回团子，由它浓缩成 1~3 句说给你听。网页上会注明“问了云端”。

### 本地大脑没开时云端代答

服务开着、但 Ollama 没开时，团子不再只说“脑袋没连上”，而是由 Gemini 照同一套人设代答（每句约 1.5~2 秒）：

- **会发给云端**：团子的人设和此刻状态、性格描述、自我认识、主人画像、最近 6 句对话、这句话按关键词勾起的记忆（最多 3 条）。这是你同意过的；不想要就设 `PET_CLOUD_STANDIN=0`。
- 每次对话前先探一下 Ollama（最多等 0.5 秒），开着就照常走本地，不发云端。
- 代答时不用小工具，也当场提炼不了记忆（存记忆要算向量，向量模型也在 Ollama 里）。这几轮会先攒着（只记消息编号和指纹，不另存原话），本地大脑一回来——主人下次开口，或者下一次心跳探到 Ollama 开了——就按当时的时间补提炼成记忆和回访。中途让团子“忘掉”的那几句不会补。
- 想起往事只按共有的词找（“咪咪”能想起咪咪，“复习好累”想不起“考研”），比本地粗一些。
- 主动搭话的措辞也会代答。每天最多 300 次（`PET_CLOUD_STANDIN_DAILY`），和“问云端”的 20 次分开算；用完了团子就说脑袋没连上。
- 网页上会注明“本地大脑没开，云端代答”，顶上的状态也会显示“云端代答中”。
- 声音不受影响：声音服务是单独的进程，代答的话照样用“小仓鼠”念。声音服务没好（刚启动约半分钟在加载，或者挂了）时网页只显示文字，不会换成电脑自带的语音。

### 记忆管理（`/memory`）

浏览器打开 http://localhost:8600/memory（网页团子下方的“记忆”链接也能进）：

- 看、改、删团子记得的事（改文字会重算向量），按类别、关键字筛选，淡忘的也能看到；每条有“记得多清楚”（遗忘曲线）。
- 看和删日记、主人画像、等着回头关心的事；手动调性格数值和团子的自我认识（调试人设用）。
- “现在整理一次”：立刻跑一遍夜间整理。“导出备份”：下载 `tuanzi-backup-时间.json`（记忆、日记、画像、回访、聊天记录、性格，不含向量）。
- 也可以直接对团子说“把……忘掉”“别记着”“忘掉这件事”：相关的记忆、回访和那几句原话都会真删。只说“忘掉这件事”时指你上一句说的事。

删除都是真删，不留痕迹。

### 出门在外连回家（relay.py）

板子连手机热点时，经你的 Lightsail 服务器中转连回这台电脑：

```
板子 ──wss──▶ your-domain.example:443/pet/ws（服务器 Nginx：个人主页的 https 里只加了这一条路径）
               ▼ 服务器 127.0.0.1:18600
         SSH 反向隧道（这台电脑主动连出去，路由器不用开端口）
               ▼ 这台电脑 127.0.0.2:8600（大脑）
```

- 大脑启动时自动建隧道、断了自动重连。服务器的 SSH 登录名和私钥位置不写在代码里：写进 `local_settings.py`（不进 git，照 `local_settings.example.py` 抄）或设环境变量 `PET_RELAY_HOST` / `PET_RELAY_KEY`；都没配就不开隧道。日志里有“隧道已连”。
- 安全：隧道故意接到 `127.0.0.2`，大脑据此认出“从公网进来的”（伪造请求头骗不过）：这类连接的 HTTP 一律 403（记忆、日记、导出只给局域网），`/ws` 的 hello 必须带设备密钥。
- 设备密钥在 `data/device_token.txt`（第一次启动自动生成），固件 `sdkconfig` 的 `CONFIG_PET_DEVICE_TOKEN` 要和它一致。
- 板子先连局域网地址，连续 2 次连不上就换 `wss://your-domain.example:443/pet/ws`，那个也连不上再换回来。
  实测（2026-10-26）：拔掉 USB 用电池，经中转连上大脑聊天，开口速度和在家差不多。
- 服务器配置：`/etc/nginx/sites-available/cai-ziyi-portfolio` 里的 `location = /pet/ws`（改之前的备份是同目录 `.bak-before-pet`）。
  另有一份 `pet.your-domain.example` 的站点配置备用：域名解析在 Namecheap（不是 Lightsail），`pet` 的 A 记录加了但 Namecheap 一直没发布（区域序列号停在 7 月）。
  它哪天生效了，用 certbot 加证书后可以把固件地址换成 `wss://pet.your-domain.example:443/ws`，主页配置里那段就能删掉。

## 结构

| 文件 | 作用 |
|---|---|
| `state.py` | 纯函数：由时间戳推导精力、孤单、好奇 |
| `rules.py` | 纯函数：要不要主动开口（回访 > 孤单 > 好奇；受免打扰、冷却、被无视翻倍、每日上限约束） |
| `memory.py` | 对话后提炼记忆；回访时间由代码校验 |
| `recall.py` `traits.py` `nightly.py` | 情景记忆和遗忘曲线、慢慢长成的性格、夜间整理（合并、淡忘、日记、自我认识） |
| `memory_admin.py` | 记忆管理页的后端、“忘掉这件事”、导出备份 |
| `persona.py` | 人设、所有 prompt、回复截句和出戏检测 |
| `websearch.py` | 联网搜索（只发搜索词） |
| `toolbox.py` | 小工具箱：算日子、计算器、天气、上网查、问云端的定义和执行 |
| `cloud.py` | 问云端大模型（Gemini API），只发一句清洗过的问题，有开关和每日额度 |
| `brain.py` | `chat()` / `tick()` 编排 |
| `server.py` | FastAPI 接口和 APScheduler 心跳（每 15 分钟一次） |
| `asr.py` | 本地语音识别（SenseVoice），顺带听出说话的情绪，作为语气提示交给大脑 |
| `tts.py` | 团子的声音（客户端）：按表情选情绪、去掉不该念的字、拉起/关掉声音服务、流式取 PCM |
| `tts_service.py` | 声音服务（Qwen3-TTS，单独进程，`.venv-fastq`），参数在 `voice/voice.json` |
| `relay.py` | 出门连回家：SSH 反向隧道、设备密钥、认出从公网进来的连接 |
| `opus.py` | ESP32 的 Opus 录音解码、声音编码（PyAV 自带 libopus） |
| `link.py` | 设备连接（WebSocket `/ws`）：对话、摸头、状态同步、主动搭话实时推送，协议见 `PROTOCOL.md` |
| `clock.py` | 虚拟时钟（只能往前拨），用来快进测试 |
| `static/device.html` | 网页假设备，也是固件本地本能的 JS 原型 |

设计原则：**数值和决定都在代码里，LLM 只负责措辞。** 参数集中在 `config.py`。

## 设备协议

设备（网页团子、ESP32）平时连 WebSocket `/ws`，详见 [PROTOCOL.md](PROTOCOL.md)。真机要连时，服务改为 `--host 0.0.0.0`，固件里填 PC 的局域网 IP。

## HTTP 接口（以后的省电拉取模式 + 调试）

- `GET /api/poll`：醒来时拉取待播事件 `{text, face, vibe, voice}` 和权威状态（用来校准设备本地的状态影子）。
- `POST /api/device_log`：上报本地发生的事，例如 `{"events":[{"kind":"pat","count":3}]}`。
- `POST /api/chat`：`{"text": ...}`，返回 `{text, face, vibe, voice, state}`。
- 记忆管理：`GET /api/memories`（总览）、`PATCH`/`DELETE /api/memories/{id}`、`DELETE /api/profile/{key}`、`DELETE /api/followups/{id}`、`DELETE /api/diary/{day}`、`PUT /api/traits`、`PUT /api/self_view`、`POST /api/nightly`、`GET /api/export`。
- 调试用：`GET /api/state`、`POST /api/clock`（`{"to":"22:00"}` 或 `{"advance_hours":4}`）、`POST /api/tick`。

face 的取值：idle / listen / think / happy / shy / pout / sad / lonely / curious / sleepy / sleep / peek。vibe 的取值：tap / purr / nudge。

服务默认只监听本机。真机（ESP32）要连时，在 `ai_pet/` 目录里用 `python -m uvicorn server:app --host 0.0.0.0 --port 8600` 启动（在上一级目录要加 `--app-dir ai_pet`）（第一次 Windows 会问防火墙，选“专用网络”允许）。板子连的是 `ws://<pc-lan-ip>:8600/ws`，电脑 IP 变了要改固件重烧。

数据存放在 `data/pet.db`，不会外传。
