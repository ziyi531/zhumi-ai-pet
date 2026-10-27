"""团子的小工具箱：本地 9B 大脑通过工具调用来用，不好在本地处理的事交给工具，团子只负责用自己的口吻说。

| 工具 | 干什么 | 什么会离开这台电脑 |
|---|---|---|
| date_calc  | 算日子：某天星期几、距今多少天、N 天后是几号 | 无 |
| calculator | 算数（小模型最容易算错）                  | 无 |
| weather    | 天气（Open-Meteo，免费免注册）             | 城市名 / 经纬度 |
| web_search | 上网搜索（websearch.py）                   | 搜索词 |
| ask_cloud  | 问云端大模型（cloud.py，Gemini）           | 一句不含个人信息的问题 |
| set_reminder | 定时提醒：N 秒/分钟/小时后，或某个钟点（到点由 brain.fire_reminders 出声） | 无 |

规矩都在代码里：能不能用（开关、key、额度）、每轮最多用几次、参数里的主人称呼先抹掉。
"""
import ast
import logging
import math
import operator
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

import requests

import clock
import cloud
import config
import db
import websearch

log = logging.getLogger("pet.tools")
WEEKDAYS = "一二三四五六日"


@dataclass
class Result:
    detail: str   # 设备上显示的“查了什么”（搜索词、城市、算式……）
    text: str     # 交回给大脑的工具返回
    ok: bool = True


def _spec(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "required": required, "properties": props}}}


SPECS = {
    "date_calc": _spec(
        "date_calc", "算日子：节日是哪天、离今天还有多少天，某一天是星期几，或者今天往后/往前 N 天是几号。"
        "春节、中秋这些农历节日每年日期不一样，一定用它算，别自己猜。",
        {"holiday": {"type": "string", "description": "节日名，比如 春节、除夕、元宵、端午、七夕、中秋、重阳、元旦、国庆、圣诞；算的是下一次"},
         "date": {"type": "string", "description": "要算的那一天，格式 YYYY-MM-DD"},
         "days": {"type": "integer", "description": "从今天往后数几天（负数是往前）"}}, []),
    "calculator": _spec(
        "calculator", "算数：加减乘除、百分比、乘方、开方等。只要涉及具体数字的计算就用它，别自己心算。",
        {"expression": {"type": "string", "description": "算式，比如 128*3+45、2**10、sqrt(2)、(1500-320)/12"}},
        ["expression"]),
    "weather": _spec(
        "weather", "查天气：现在的天气和今明后三天的预报（气温、下雨概率）。主人问天气、要不要带伞、冷不冷时用。",
        {"city": {"type": "string", "description": "城市名；主人没说就留空，用主人所在的城市"}}, []),
    "web_search": _spec(
        "web_search", "上网搜索，返回几条网页标题和摘要。用来查外面世界里你不知道、拿不准或可能过时的事（新游戏、新闻、人物、价格……）。"
        "主人说自己累了、开心、吃了什么这类自己的事和心情时，不要用它。",
        {"query": {"type": "string", "description": "搜索词，简短，只写要查的东西本身，不含主人的名字和隐私"}},
        ["query"]),
    "ask_cloud": _spec(
        "ask_cloud", "请教一个更聪明的云端大模型。只在你自己答不好的难题上用：需要多步推理的问题、讲清楚一个原理、"
        "写一段稍长的东西、代码问题。闲聊、陪伴、主人自己的事绝对不要用；查新闻价格这类事实用 web_search。",
        {"question": {"type": "string", "description": "要问的问题，改写成能单独看懂的一句话；不要写主人的名字、住址等任何个人信息"}},
        ["question"]),
    "set_reminder": _spec(
        "set_reminder", "定时提醒：主人说“X 秒/分钟/小时后提醒我……”“下午三点叫我……”时用，到点你会自己出声提醒主人。"
        "答应提醒主人就一定要调用它，光嘴上答应到时候是不会提醒的。相对时间填 seconds/minutes/hours，钟点填 at，二选一。",
        {"text": {"type": "string", "description": "到点要提醒主人做的事，写成动作，比如 去上厕所、开会、关火、吃药"},
         "seconds": {"type": "integer", "description": "多少秒后"},
         "minutes": {"type": "integer", "description": "多少分钟后"},
         "hours": {"type": "number", "description": "多少小时后"},
         "at": {"type": "string", "description": "几点提醒，24 小时制 HH:MM，比如 15:00；不是今天就写 YYYY-MM-DD HH:MM"}},
        ["text"]),
}
LABEL = {"date_calc": "算日子", "calculator": "算一算", "weather": "看天气", "web_search": "上网查", "ask_cloud": "问云端",
         "set_reminder": "定提醒"}


def available() -> list[str]:
    """现在能用的工具（顺序就是给大脑看的顺序）。"""
    names = ["date_calc", "calculator", "set_reminder"]
    if config.WEATHER:
        names.append("weather")
    if websearch.available():
        names.append("web_search")
    if cloud.available():
        names.append("ask_cloud")
    return names


def limit(name) -> int:
    """一轮对话里每个工具最多用几次。"""
    return {"web_search": config.MAX_SEARCHES, "ask_cloud": 1}.get(name, 2)


def describe(name: str, args: dict, profile=()) -> str:
    """用工具之前就能告诉设备的“查什么”（和 run 里实际发出去的一致）。"""
    args = args if isinstance(args, dict) else {}
    if name == "web_search":
        return websearch.clean_query(args.get("query", ""), profile)
    if name == "ask_cloud":
        return websearch.clean_query(args.get("question", ""), profile, max_chars=config.CLOUD_MAX_QUESTION_CHARS)
    if name == "weather":
        return str(args.get("city") or "").strip() or home_city(profile)
    if name == "calculator":
        return str(args.get("expression", ""))
    if name == "set_reminder":
        return reminder_text(args.get("text"))
    if name == "date_calc":
        return str(args.get("holiday") or args.get("date")
                   or (f"{args['days']} 天" if args.get("days") is not None else ""))
    return ""


def run(name: str, args: dict, profile=()) -> Result:
    args = args if isinstance(args, dict) else {}
    try:
        if name == "date_calc":
            return date_calc(args.get("date"), args.get("days"), args.get("holiday"))
        if name == "calculator":
            return calculator(str(args.get("expression", "")))
        if name == "weather":
            return weather(str(args.get("city") or "").strip() or home_city(profile))
        if name == "web_search":
            q = websearch.clean_query(args.get("query", ""), profile)
            return Result(q, websearch.results_text(websearch.search(q)) if q else "", bool(q))
        if name == "set_reminder":
            return set_reminder(args)
        if name == "ask_cloud":
            q = websearch.clean_query(args.get("question", ""), profile, max_chars=config.CLOUD_MAX_QUESTION_CHARS)
            if not q:
                return Result("", "（问题是空的）", False)
            answer = cloud.ask(q)
            return Result(q, answer or "（云端这次没答上来，可能网络不通或者今天的次数用完了）", bool(answer))
    except Exception as e:
        log.warning("工具 %s 出错：%s", name, e)
        return Result("", f"（{LABEL.get(name, name)}出错了：{e}）", False)
    return Result("", f"（没有叫 {name} 的工具）", False)


# ---------- 定时提醒 ----------
REMIND_PREFIX = re.compile(r"^(到点|到时候?)?(提醒|叫|喊)(一下)?(主人|我|你)?|^该")
AT_RE = re.compile(r"(?:(\d{4})-(\d{1,2})-(\d{1,2})\s+)?(\d{1,2})[:：](\d{2})(?:[:：](\d{2}))?")


def reminder_text(text) -> str:
    t = REMIND_PREFIX.sub("", str(text or "").strip()).strip(" ，,。！!")
    return t[:40]


def span_text(seconds: float) -> str:
    """多久之后，说成人话：30 秒、5 分钟、1 小时 30 分钟。"""
    s = int(round(seconds))
    if s < 60:
        return f"{s} 秒"
    h, m, sec = s // 3600, s % 3600 // 60, s % 60
    parts = [f"{h} 小时" if h else "", f"{m} 分钟" if m else "", f"{sec} 秒" if sec and not h else ""]
    return " ".join(p for p in parts if p)


def _num(v) -> float:
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def reminder_due(args: dict, now: datetime) -> datetime | None:
    """到点时间：相对时间（秒/分/时加起来）优先；否则钟点，今天的已经过了就是明天的。"""
    delta = _num(args.get("seconds")) + _num(args.get("minutes")) * 60 + _num(args.get("hours")) * 3600
    if delta > 0:
        return now + timedelta(seconds=round(delta))
    m = AT_RE.search(str(args.get("at") or ""))
    if not m:
        return None
    y, mo, d, h, mi, sec = (int(x) if x else None for x in m.groups())
    if h > 23 or mi > 59:
        return None
    due = (datetime(y, mo, d) if y else now).replace(hour=h, minute=mi, second=sec or 0, microsecond=0)
    if not y and due <= now:
        due += timedelta(days=1)
    return due


def set_reminder(args: dict) -> Result:
    now = clock.now()
    text = reminder_text(args.get("text")) or "主人交代的事"
    try:
        due = reminder_due(args, now)
    except ValueError:
        due = None
    if due is None:
        return Result(text, "（没看懂几点提醒：相对时间填 seconds/minutes/hours，钟点填 at）", False)
    if due <= now or due - now > timedelta(days=config.REMINDER_MAX_DAYS):
        return Result(text, f"（只能定 {config.REMINDER_MAX_DAYS} 天以内、还没到的时间）", False)
    if len(db.pending_reminders()) >= config.MAX_PENDING_REMINDERS:
        return Result(text, "（待办的提醒太多了，先让主人等前面的提醒完）", False)
    db.add_reminder(clock.fmt(due), text, clock.fmt(now))
    span = span_text((due - now).total_seconds())
    day = "今天" if due.date() == now.date() else "明天" if due.date() == now.date() + timedelta(days=1)         else f"{due.month}月{due.day}日"
    log.info("定了提醒：%s（%s后）%s", clock.fmt(due), span, text)
    return Result(f"{span}后 {text}",
                  f"定好了：{day} {due:%H:%M:%S}（{span}后）提醒主人「{text}」。到点你会自己出声提醒，现在不用再做别的。")


# ---------- 算日子 ----------
# 农历节日（月, 日）；除夕 = 春节前一天。用 lunardate 换算（GPL-3.0，个人自用没问题，分发前要换掉）
LUNAR = {"春节": (1, 1), "元宵": (1, 15), "端午": (5, 5), "七夕": (7, 7), "中秋": (8, 15), "重阳": (9, 9), "腊八": (12, 8)}
SOLAR = {"元旦": (1, 1), "情人节": (2, 14), "妇女节": (3, 8), "劳动节": (5, 1), "儿童节": (6, 1), "国庆": (10, 1),
         "万圣节": (10, 31), "圣诞": (12, 25)}


ALIAS = {"过年": "春节", "新年": "元旦", "大年三十": "除夕", "年三十": "除夕", "端阳": "端午", "圣诞夜": "圣诞"}


def next_holiday(name: str, today):
    """下一次（含今天）这个节日的公历日期，返回 (标准名, 日期)；不认识返回 (None, None)。"""
    name = ALIAS.get(name, name)
    key = next((k for k in [*LUNAR, *SOLAR, "除夕"] if name in (k, k + "节") or name + "节" == k), None)
    if key is None:
        return None, None
    for year in (today.year - 1, today.year, today.year + 1):
        if key in SOLAR:
            d = datetime(year, *SOLAR[key]).date()
        else:
            from lunardate import LunarDate
            m, dd = LUNAR["春节"] if key == "除夕" else LUNAR[key]
            d = LunarDate(year, m, dd).to_solar_date()
            if key == "除夕":
                d -= timedelta(days=1)
        if d >= today:
            return key, d
    return key, None


def date_calc(day=None, days=None, holiday=None) -> Result:
    today = clock.now().date()
    if holiday:
        key, target = next_holiday(str(holiday).strip(), today)
        if not target:
            return Result(str(holiday), f"（不认识「{holiday}」这个节日，可以上网查一下它是哪天）", False)
        diff = (target - today).days
        text = (f"今天是 {today:%Y-%m-%d} 星期{WEEKDAYS[today.weekday()]}；下一个{key}是 {target:%Y-%m-%d} "
                f"星期{WEEKDAYS[target.weekday()]}，" + ("就是今天。" if diff == 0 else f"还有 {diff} 天。"))
        return Result(key, text)
    if day:
        target = datetime.strptime(str(day).strip()[:10], "%Y-%m-%d").date()
    elif days is not None:
        target = today + timedelta(days=int(days))
    else:
        target = today
    diff = (target - today).days
    rel = "就是今天" if diff == 0 else (f"在 {diff} 天之后" if diff > 0 else f"在 {-diff} 天之前")
    text = (f"今天是 {today:%Y-%m-%d} 星期{WEEKDAYS[today.weekday()]}；"
            f"{target:%Y-%m-%d} 是星期{WEEKDAYS[target.weekday()]}，{rel}。")
    return Result(f"{target:%Y-%m-%d}", text)


# ---------- 计算器（只认数字和算术，不执行任何代码） ----------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round, "log": math.log, "log10": math.log10,
          "sin": math.sin, "cos": math.cos, "tan": math.tan, "floor": math.floor, "ceil": math.ceil}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("指数太大了")
        return _OPS[type(node.op)](left, right)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS
            and not node.keywords and len(node.args) <= 2):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("只能算数字")


def calculator(expr: str) -> Result:
    e = (expr.replace("×", "*").replace("÷", "/").replace("（", "(").replace("）", ")")
         .replace("^", "**").replace("，", "").replace(",", ""))
    e = re.sub(r"(\d+(?:\.\d+)?)%", r"(\1/100)", e)
    value = _eval(ast.parse(e[:200], mode="eval"))
    if isinstance(value, float):
        value = round(value, 6)
        if value == int(value) and abs(value) < 1e15:
            value = int(value)
    return Result(expr, f"{expr} = {value}")


# ---------- 天气（Open-Meteo） ----------
WMO = {0: "晴", 1: "大致晴朗", 2: "多云", 3: "阴", 45: "有雾", 48: "有雾凇", 51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
       56: "冻毛毛雨", 57: "冻毛毛雨", 61: "小雨", 63: "中雨", 65: "大雨", 66: "冻雨", 67: "冻雨", 71: "小雪",
       73: "中雪", 75: "大雪", 77: "雪粒", 80: "阵雨", 81: "阵雨", 82: "强阵雨", 85: "阵雪", 86: "强阵雪",
       95: "雷阵雨", 96: "雷阵雨伴冰雹", 99: "强雷阵雨伴冰雹"}
_geo_cache: dict = {}
CITY_KEYS = ("城市", "所在地", "住在", "居住")


def home_city(profile=()) -> str:
    """主人所在的城市：先看配置，再看记忆里的画像。"""
    if config.HOME_CITY:
        return config.HOME_CITY
    for p in profile:
        if any(k in p.get("key", "") for k in CITY_KEYS):
            return p.get("value", "")
    return ""


def _geocode(city):
    if city not in _geo_cache:
        r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                         params={"name": city, "count": 1, "language": "zh", "format": "json"}, timeout=8)
        r.raise_for_status()
        hits = r.json().get("results") or []
        _geo_cache[city] = hits[0] if hits else None
    return _geo_cache[city]


def weather(city: str) -> Result:
    if not city:
        return Result("", "（不知道主人在哪个城市，问问主人）", False)
    place = _geocode(city)
    if not place:
        return Result(city, f"（没找到叫「{city}」的地方）", False)
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": place["latitude"], "longitude": place["longitude"], "timezone": "auto", "forecast_days": 3,
        "current": "temperature_2m,apparent_temperature,weather_code,relative_humidity_2m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"}, timeout=8)
    r.raise_for_status()
    d = r.json()
    cur, day = d["current"], d["daily"]
    lines = [f"{place.get('name', city)}现在{WMO.get(cur['weather_code'], '天气未知')}，{cur['temperature_2m']:.0f}°C"
             f"（体感 {cur['apparent_temperature']:.0f}°C），湿度 {cur['relative_humidity_2m']}%。"]
    for i, name in enumerate(["今天", "明天", "后天"][: len(day["time"])]):
        rain = day["precipitation_probability_max"][i]
        lines.append(f"{name}：{WMO.get(day['weather_code'][i], '未知')}，{day['temperature_2m_min'][i]:.0f}~"
                     f"{day['temperature_2m_max'][i]:.0f}°C" + (f"，下雨概率 {rain}%" if rain is not None else ""))
    return Result(place.get("name", city), "\n".join(lines))
