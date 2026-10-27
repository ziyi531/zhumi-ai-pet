"""全局配置：宠物的性格参数、主动搭话的规则都在这里调。"""
import os
from pathlib import Path

# ---- Ollama ----
# 用 127.0.0.1 而不是 localhost：Windows 上 localhost 先试 IPv6（::1），Ollama 只听 IPv4，每次请求白等 2 秒
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
# qwen3.5:9b 的纯文字版（ollama/qwen3.5-9b-text.Modelfile 生成）；2026-10 从 qwen2.5:7b 换过来
DEFAULT_MODEL = os.environ.get("PET_MODEL", "qwen3.5:9b-text")
# Ollama 默认上下文很短，会悄悄截断人设和记忆，这里显式给够
NUM_CTX = int(os.environ.get("PET_NUM_CTX", "4096"))
# 支持“思考模式”的模型（Qwen3.5 等）要不要先想再答：宠物聊天句子短，不想更快
THINK = os.environ.get("PET_THINK", "0") == "1"
# 放进显卡的层数：None 交给 Ollama 自己估；数字就固定；"fit" 在每次加载模型前按显卡此刻的空余现算。
# 9B 全放显卡（5.2GB）+ 声音服务（2.5GB）+ Windows 桌面超了 8GB 卡，Windows 会来回搬显存，Ollama 卡到出不来字。
# 原来固定 28 层，后来桌面（dwm）的显存从 200MB 涨到 744MB，28 层也溢出了，所以改成现算。
_num_gpu = os.environ.get("PET_NUM_GPU", "fit" if DEFAULT_MODEL == "qwen3.5:9b-text" else "")
NUM_GPU = int(_num_gpu) if _num_gpu.isdigit() else ("fit" if _num_gpu == "fit" else None)
# "fit" 的估算参数（qwen3.5:9b-text Q4_K_M，按实测反推：28 层 ≈ 4.55GB、30 层 ≈ 4.79GB、33 层 ≈ 5.2GB）
GPU_FIT = {"total_mb": 8188, "reserved_mb": 232, "margin_mb": 400, "base_mb": 1050, "per_layer_mb": 125, "layers": 33}
# 工具调用的方式：native = Ollama 原生 tools（qwen3.5 实测 10 问全对）；
# json = 让模型在 JSON 里写 {"search": ...}（qwen2.5:7b 用原生 tools 一次都调不成，只能用这个）
TOOL_MODE = os.environ.get("PET_TOOL_MODE", "native" if DEFAULT_MODEL.startswith("qwen3.5") else "json")
# 秒。冷启动加载 9B 约 50 秒，留够；再久多半是卡住了（比如显存溢出），与其让主人干等，不如放弃、老实说
REQUEST_TIMEOUT = 120
# 模型闲着多久才从显卡里卸掉（Ollama 默认 5 分钟：隔几分钟再聊就要重新加载 8 秒，还偶尔加载失败）
KEEP_ALIVE = os.environ.get("PET_KEEP_ALIVE", "30m")
LOAD_RETRY_S = 1.0  # 加载模型失败（Ollama 返回 500，显卡偶发 CUDA 错误）时等这么久再试一次

# ---- 数据存放位置 ----
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PORT = int(os.environ.get("PET_PORT", "8600"))  # 大脑服务的端口（uvicorn --port 要和它一致）

# 出门在外连回家（relay.py）：Lightsail 服务器中转。没配 HOST 或找不到密钥就不开隧道。
# 服务器地址和密钥位置不写在代码里：设环境变量，或写进 local_settings.py（不进 git，见文件末尾）
RELAY_HOST = os.environ.get("PET_RELAY_HOST", "")  # 比如 ubuntu@1.2.3.4
RELAY_KEY = Path(os.environ["PET_RELAY_KEY"]) if os.environ.get("PET_RELAY_KEY") else None  # SSH 私钥文件
RELAY_REMOTE_PORT = 18600        # 服务器上隧道的入口（只监听服务器自己的 127.0.0.1，Nginx 转发过来）
RELAY_LOCAL_HOST = "127.0.0.2"   # 隧道在这台电脑上接到哪：用单独的本机地址，大脑据此认出“从公网进来的”
DB_FILE = Path(os.environ.get("PET_DB", DATA_DIR / "pet.db"))

# ---- 宠物 ----
PET_NAME = "猪咪"
# 语音识别常把名字听错（SenseVoice 没见过“猪咪”这个词，2026-10-27 把“它叫猪咪”认成了“他叫独蜜”，
# 猪咪就以为主人叫独蜜）。识别结果和喂给模型的聊天记录里，这些写法一律改成正确的名字
ASR_FIXES = {"独蜜": "猪咪", "杜米": "猪咪", "猪米": "猪咪", "朱米": "猪咪", "珠米": "猪咪", "竹米": "猪咪", "猪蜜": "猪咪"}
PET_OLD_NAMES = ("团子",)  # 以前的名字（2026-10-26 改名）：旧记忆、日记里写的是这个，它得知道那就是自己
WAKE_WORD = "Hi，喵喵"       # 固件的唤醒词（乐鑫 WakeNet wn9_himiaomiao_tts；“猪咪猪咪”用命令词识别实测认不准）
HISTORY_TURNS = 8          # 对话时带上最近几条消息
HISTORY_GAP_MINUTES = 20   # 上一句隔了这么久，告诉大脑那是旧话题了
MAX_SENTENCES = 3          # 每次回复最多几句
MAX_REPLY_CHARS = 80
# 出现这些词说明模型“出戏”了，会重试一次
BANNED_WORDS = ["人工智能", "AI助手", "AI 助手", "语言模型", "智能助手", "虚拟助手"]
# 团子把自己说成 AI：任何时候都拦（查资料回答 AI 话题时，上面那些词可能就是话题本身，只拦这些）
SELF_AI_WORDS = ["作为AI", "作为一个AI", "我是AI", "我是一个AI", "我是人工智能", "作为人工智能", "我是语言模型",
                 "作为语言模型", "作为一个语言模型", "我是助手", "我是你的助手", "我是一个助手"]

# ---- 心跳 ----
TICK_MINUTES = 15

# ---- 设备连接（WebSocket，见 PROTOCOL.md）----
PUSH_INTERVAL_S = 1.0     # 多久检查一次到点的提醒、有没有主动事件要推给在线的设备
STATE_PUSH_S = 30.0       # 多久给在线设备同步一次状态

# ---- 本地语音识别（sherpa-onnx + SenseVoice-Small）----
ASR_MODEL_DIR = Path(os.environ.get("PET_ASR_DIR", BASE_DIR / "models" / "sensevoice"))
ASR_LANGUAGE = "auto"     # auto / zh / en / ja / ko / yue
ASR_THREADS = 2
ASR_MAX_SECONDS = 30      # 一次最多录多久，超出的部分丢掉
ASR_MIN_SECONDS = 0.3     # 太短的当作手滑，不识别

# ---- 团子的声音（Qwen3-TTS，tts_service.py 单独进程，声音参数在 voice/voice.json）----
TTS_URL = os.environ.get("PET_TTS_URL", "http://127.0.0.1:8601")
TTS_PYTHON = Path(os.environ.get("PET_TTS_PYTHON", BASE_DIR / ".venv-fastq" / "Scripts" / "python.exe"))
TTS_AUTOSTART = os.environ.get("PET_TTS_AUTOSTART", "1") == "1"  # 大脑启动时自动拉起声音服务
TTS_START_TIMEOUT_S = 180  # 第一次加载模型 + 预热大约 20 秒，冷启动硬盘慢时更久
TTS_CHUNK_BYTES = 4800     # 推给设备的每个二进制帧：0.1 秒 24kHz PCM16

# ---- 联网搜索（websearch.py）：团子唯一会“出门”的地方，只把搜索词发给搜索引擎，不带对话和记忆 ----
WEB_SEARCH = os.environ.get("PET_WEB_SEARCH", "1") == "1"
SEARCH_RESULTS = 5          # 每次取几条结果（标题 + 摘要）
SEARCH_TIMEOUT_S = 8
SEARCH_MAX_QUERY_CHARS = 40
MAX_SEARCHES = 1            # 一次回答最多查几次（7B 查第二次多半是重复同一个词，白白多等几秒）
SEARCH_REPLY_CHARS = 120    # 用过工具（查资料、问云端）的回答可以比平时长一点（平时 MAX_REPLY_CHARS）

# ---- 情景记忆（recall.py）：按线索想起往事 ----
EMBED_MODEL = os.environ.get("PET_EMBED_MODEL", "bge-m3")  # Ollama 里的向量模型，强制跑 CPU
MEMORY_RECALL_K = 3          # 每次最多想起几件事
# 相关度低于这个的不想起。bge-m3 实测：相关的问法 0.50~0.78，不相关的闲聊最高也到 0.58（“今天天气不错”），
# 所以门槛取 0.6 宁可少想起；再给“出现同一个具体的词”（咪咪、日本）加 0.08，补回短名字被漏掉的
MEMORY_MIN_SIM = 0.6
MEMORY_KEYWORD_BONUS = 0.08
# 白天即时合并只管几乎原话重复的：实测“妈妈住院了”和“妈妈出院了”也有 0.89，0.88 会把住院的事覆盖掉。
# 说法差得多的同一件事（0.82~0.92）留给夜间整理，由模型判断是不是同一件事、合成一句（变化也写进去）
MEMORY_DEDUP_SIM = 0.92
MEMORY_MERGE_SIM = 0.82      # 夜间整理：这么像的列为“可能是同一件事”，再交给模型判断
MEMORY_MERGE_MAX = 8         # 每晚最多处理几组（每组一次 LLM 调用）
# 遗忘曲线：重要度 3 的事半衰期 14 天（每高一级翻倍、低一级减半），被提起/想起越多越牢；5 级永远记得。
# 记得的程度掉到 0.15 以下就“淡忘”：不再主动想起，但不删，以后又提起会重新想起来
MEMORY_HALF_LIFE_DAYS = 14
MEMORY_FADE_BELOW = 0.15
FORGET_MIN_SIM = 0.6         # 主人说“忘掉……”时，相关度到这个的记忆才删（最多 3 条）
SELF_VIEW_MAX_CHARS = 120    # 团子“我是怎样的团子”的自我认识，每晚小小修订（nightly.py）
MEMORY_MAX_CHARS = 80

# ---- 小工具箱（toolbox.py） ----
MAX_TOOL_CALLS = 3          # 一次回答最多调几次工具（各工具自己的上限见 toolbox.limit）
WEATHER = os.environ.get("PET_WEATHER", "1") == "1"   # 天气（Open-Meteo，会发出城市名）
HOME_CITY = os.environ.get("PET_CITY", "")           # 主人所在城市；空就从记忆的画像里找，再没有就问主人

# ---- 问云端大模型（cloud.py，Gemini）：只发一句不含个人信息的问题 ----
CLOUD = os.environ.get("PET_CLOUD", "1") == "1"      # 还要配了 GEMINI_API_KEY 才真的可用
# 实测（2026-10，免费层）：gemini-3.5-flash-lite 约 5 秒、答得也好；gemini-3.8-flash 要 60~75 秒（排队/想太久），
# 宠物说话等不了。更强的 gemini-3.8-flash / gemini-4-argon 可以改这里，付费层可能快些
CLOUD_MODEL = os.environ.get("PET_CLOUD_MODEL", "gemini-3.5-flash-lite")
CLOUD_DAILY_LIMIT = int(os.environ.get("PET_CLOUD_DAILY", "20"))     # 每天最多问几次（防止花费失控）
CLOUD_MAX_TOKENS = 1024
CLOUD_TIMEOUT_S = 30        # 超过就当没问到，团子照实说（总不能让主人干等一分钟）
CLOUD_MAX_QUESTION_CHARS = 300
# 本地大脑（Ollama）没开时，由云端替团子说话：人设、最近几句对话、勾起的记忆会发给 Gemini（主人同意过）
CLOUD_STANDIN = os.environ.get("PET_CLOUD_STANDIN", "1") == "1"
CLOUD_STANDIN_DAILY = int(os.environ.get("PET_CLOUD_STANDIN_DAILY", "300"))  # 代答每天最多几次，和“问云端”分开算
CLOUD_STANDIN_HISTORY = 6   # 代答时带上最近几句对话
MEMORY_BACKLOG_MAX = 200    # 代答期间最多攒多少轮对话等本地大脑回来补记（再多就丢最早的）
LOCAL_PROBE_S = 0.5         # 探本地大脑在不在：Windows 上连关着的端口要等 2 秒才报错，所以只等这么久

# ---- 时间窗口（"HH:MM"，可以跨午夜） ----
QUIET_HOURS = ("23:30", "08:30")   # 免打扰：这段时间绝不主动找主人
SLEEPY_HOURS = ("01:00", "06:00")  # 这段时间里对话会犯困、催主人睡觉
LONELY_HOURS = ("10:00", "22:30")  # 只在这段时间里因为孤单探头

# ---- 内在状态 ----
LONELY_HOURS_TO_FULL = 5.0   # 多少小时不说话孤单值涨满（4 小时 = 0.8）
PAT_LONELINESS = 0.3         # 被摸之后孤单值最多从这里重新开始涨
LONELY_THRESHOLD = 0.8
CURIOUS_THRESHOLD = 0.85
CURIOUS_MIN_ENERGY = 0.5
CURIOUS_GROWTH_PER_HOUR = 0.12

# ---- 主动搭话的闸门 ----
PROACTIVE_COOLDOWN_HOURS = 2.0     # 两次主动之间至少间隔（被无视时翻倍）
MAX_IGNORE_BACKOFF = 3             # 冷却最多翻 2^3 倍
IGNORE_WINDOW_MINUTES = 30         # 主动搭话后这么久没回应，算被无视一次
DAILY_PROACTIVE_LIMIT = 4
FOLLOWUP_MIN_GAP_MINUTES = 30      # 回访可以不等冷却，但离上次主动至少隔这么久
FOLLOWUP_STALE_HOURS = 12          # 过期太久的回访直接作废，免得第二天早上问“眼睛还酸吗”
FOLLOWUP_DEFAULT_HOURS = 6         # 负面情绪但模型没给时间时，默认几小时后回访
MAX_PENDING_FOLLOWUPS = 10
EVENT_TTL_HOURS = 2                # 设备离线太久，没送达的主动搭话就不播了
REMINDER_MAX_DAYS = 7             # 定时提醒最远能定几天后
MAX_PENDING_REMINDERS = 20
REMINDER_STALE_HOURS = 2          # 大脑关着时到点的提醒，开机后超过这么久就不补了
NEGATIVE_MOOD_WORDS = ["累", "疲", "酸", "痛", "疼", "难过", "伤心", "焦虑", "烦", "压力", "不舒服",
                       "病", "生气", "郁闷", "紧张", "崩溃", "失眠", "困"]

# 只属于这台电脑的设置（服务器地址、密钥位置……）写在 local_settings.py 里覆盖上面的值，不进 git。
# 例子见 local_settings.example.py
try:
    from local_settings import *  # noqa: F401,F403
except ImportError:
    pass
