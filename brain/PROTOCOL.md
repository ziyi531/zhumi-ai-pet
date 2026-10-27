# 设备 ↔ 大脑 协议（v1）

设备（ESP32 固件 `ai_pet_fw/main/net/pet_link.cc`、网页团子 `static/pet.html`）和 PC 大脑（`link.py`）之间的协议。改协议时三处一起改，并更新 `selftest.py` 里的 `test_websocket_link`。

- 地址：家里 `ws://<PC 的局域网 IP>:8600/ws`；出门在外 `wss://your-domain.example:443/pet/ws`（Lightsail 中转，见 README 和 `relay.py`）。
  经中转进来的连接，hello 必须带 `token`（设备密钥，`data/device_token.txt`），否则回 `error`“密钥不对”并断开；HTTP 接口一律 403。
- 文本帧都是 JSON，每条消息都有 `type` 字段。
- 设备 → 大脑的二进制帧是录音数据：只在 `listen start` 和 `listen stop` 之间有效，格式看 listen start 的 `format`：
  - `pcm16`（网页团子）：16kHz、16 位小端、单声道 PCM，帧长随意；
  - `opus`（ESP32）：16kHz 单声道，一帧一个 Opus 包（60ms），大脑解码后识别。
- 大脑 → 设备的二进制帧是团子的声音：只在 `tts start` 和 `tts stop` 之间有效，只发给 hello 里声明了 `audio_out` 的设备，格式按声明来：
  - `pcm16`（网页团子）：24kHz、16 位小端、单声道 PCM，每帧 0.1 秒；
  - `opus`（ESP32）：24kHz 单声道，一帧一个 Opus 包（60ms，`tts start` 里有 `frame_ms`），最后一包不够 60ms 补静音。
  - 大脑端编解码用 PyAV 自带的 libopus（`opus.py`）；没装 av 时 `opus` 设备只收文字。
- 时间统一是大脑的虚拟时间 `"YYYY-MM-DD HH:MM:SS"`（调试面板快进时，设备也跟着走）。

## 设备 → 大脑

| type | 字段 | 说明 |
|---|---|---|
| `hello` | `device`（`esp32` / `web`）、`proto`，可选 `audio_out`（`pcm16` / `opus`）、`token`（经中转时必须） | 连上后第一条消息，没发 hello 之前其它消息都会被拒。带 `audio_out` 表示会播大脑下发的声音（网页 `pcm16`，固件 `opus`）；不带就只收文字 |
| `chat` | `text` | 主人打字说的话 |
| `listen` | `state`：`start` / `stop` / `cancel`；start 时带 `format`（`pcm16` / `opus`）和 `rate`（16000） | 说话：start 之后发二进制录音帧，stop 时大脑整段识别；cancel 丢弃这段录音。网页是按住说话；固件是说“你好小智”后开始，靠人声检测在说过话、停顿 1.2 秒后自动 stop（5 秒没开口 cancel，最长 12 秒）。最长 30 秒，短于 0.3 秒不识别。start 同时表示“主人开口了”：正在推给这台设备的声音会停下（`tts stop` 带 `cancelled`） |
| `hush` | | 别说了：正在推给这台设备的声音停下（`tts stop` 带 `cancelled`）。固件在团子说话时被摸头就发这个 |
| `pat` | `count` | 被摸了几次。设备本地先做出反应，再上报；离线时攒着，连上后一起报 |
| `ping` | | 应用层保活，回 `pong`（ESP32 用 WebSocket 协议层的 ping，可以不发） |

## 大脑 → 设备

| type | 字段 | 说明 |
|---|---|---|
| `welcome` | `proto`、`time`、`state` | 回应 hello；随后会补推离线期间攒下的主动搭话 |
| `state` | `time`、`state` | 权威状态，每次对话/摸头后发一次，在线时每 30 秒同步一次 |
| `heard` | `text`、`emotion`、`event`，失败时还有 `error` | 识别结果。`text` 为空表示没听清；`emotion` 是听出来的情绪（开心/难过/生气…），`event` 是在笑/在哭，大脑回复时会照顾到 |
| `thinking` | 可选 `tool`、`detail`（上网搜索时还有旧字段 `search`） | 收到 `chat` 或识别完语音，开始想了（设备换成思考表情）。大脑决定先用工具时会再发一条带 `tool`（`web_search` / `ask_cloud` / `weather` / `calculator` / `date_calc` / `set_reminder`）和 `detail`（搜索词、城市、算式……）的，设备可以显示“在网上查…”“在问云端…” |
| `say` | `text`、`face`、`vibe`、`voice`、`proactive`、`audio`，`audio=true` 时还有 `id`；主人道别时有 `end: true`（连续对话到此结束，设备说完不再接着听）；用过工具时有 `tools`（上网查过还有 `searched`） | 要说的话。`proactive=true` 表示它主动来找你（设备先探头再说；主人定的提醒到点也是这样推过来）。`audio=true` 表示这段话的声音随后推过来（`tts` 带同一个 `id`），设备播它；`false` 时设备自己决定念不念。`tools` 是这次回答前用过的工具 `[{"tool", "detail"}]`；`searched` 是其中上网查过的搜索词（旧字段） |
| `tts` | `state`：`start` / `stop`、`id`；start 带 `format`（`pcm16` / `opus`）、`rate`（24000），opus 还有 `frame_ms`（60）；stop 可能带 `cancelled`（被打断）或 `error`（没合成出来，设备改用自己的兜底） | 团子的声音。start 和 stop 之间的二进制帧就是这段话的音频，边收边播。同一时间只有一段在推，id 按顺序递增 |
| `pong` | | 回应 ping |
| `error` | `message` | 出错说明 |

`state` 对象：`energy`、`loneliness`、`curiosity`（0~1）、`phase`（`awake` / `sleepy` / `asleep`）、`sleepy_talk`（深夜说话要犯困）、`hours_alone`。

取值：
- `face`：`idle listen think happy shy pout sad lonely curious sleepy sleep peek`（和表情引擎一致）
- `vibe`：`tap purr nudge`
- `voice`：`normal sleepy`

## 设备端的基础表情规则

设备不说话时显示什么，由设备根据最近一次 `state` 自己决定（网页和固件规则相同）：

1. 没连上大脑：大脑时间 23:00~07:00 显示 `sleep`，其余时间 `lonely`（还没连上过就 `idle`）。
2. `phase=asleep` → `sleep`；`phase=sleepy` → `sleepy`。
3. `loneliness >= 0.8` → `lonely`。
4. 其余 → `idle`。

摸头、收到 `say`、`thinking` 等都是临时表情，结束后回到基础表情。

## 以后

- 省电的拉取模式：设备睡着时断开连接，定时醒来调 `GET /api/poll` 取事件，再睡回去。
