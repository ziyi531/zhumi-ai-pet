"""团子慢慢长成的性格：一组 0~1 的数，被和主人的相处一点点推着变。

人设（persona.SYSTEM）是“天生的”，这里是“后天的”：
- 亲密 intimacy：常聊天、常摸头就越来越熟；很久不见会慢慢生分一点。
- 信任 trust：主人跟它说心事就涨，几乎不降。
- 黏人 clingy：常被摸、被关心就更黏；常被冷落也会更黏（怕主人走）；会慢慢回到天生的样子。
- 傲娇 tsundere：常被逗、被吐槽就更嘴硬；被夸多了反而软下来一点；会慢慢回到天生的样子。
- 活泼 playful：常一起开玩笑就更调皮；常被冷落就蔫一点；会慢慢回到天生的样子。

规矩都在代码里，LLM 只负责标出“主人这一轮是在逗它、夸它还是跟它说心事”：
- 每次只变一点点（越接近 0/1 变得越慢），每个数每天涨跌都有上限——性格是慢慢长成的，一天聊疯了也变不了多少；
- 每天一次的“回落”：会回到天生样子的那几个，每天往天生值靠 3%；亲密在没见面的日子里慢慢降。
全部是纯函数：同样的输入永远得到同样的结果，方便测试和回放。
"""
import json
from copy import deepcopy

# 天生的样子（和 persona.SYSTEM 里的“黏人、好奇、偶尔傲娇”对得上）；刚认识时亲密和信任都不高
BASELINE = {"intimacy": 0.2, "trust": 0.25, "clingy": 0.55, "tsundere": 0.4, "playful": 0.5}
# 哪些会慢慢回到天生的样子（亲密、信任是攒出来的，不回落）
DRIFTS = ("clingy", "tsundere", "playful")
DAILY_CAP = 0.04       # 每个数每天最多变这么多
DRIFT_RATE = 0.03      # 每天往天生值靠拢的比例
ABSENT_DECAY = 0.01    # 一整天没见面，亲密降这么多

# 一件事对各个数的推动（正数往上，负数往下）
EFFECTS = {
    "chat": {"intimacy": 0.004},
    "pat": {"intimacy": 0.003, "clingy": 0.004},
    "逗团子": {"tsundere": 0.012, "playful": 0.01, "intimacy": 0.004},
    "夸团子": {"tsundere": -0.008, "intimacy": 0.006},
    "关心团子": {"clingy": 0.008, "intimacy": 0.008, "trust": 0.004},
    "分享心事": {"trust": 0.015, "intimacy": 0.008},
    "吐槽团子": {"tsundere": 0.01, "playful": -0.004},
    "冷淡": {"playful": -0.008, "clingy": 0.004},
    "ignored": {"clingy": 0.01, "playful": -0.01},   # 主动找主人却没被理
}


def new_state(day: str) -> dict:
    return {"values": dict(BASELINE), "day": day, "moved": {k: 0.0 for k in BASELINE}, "last_seen": None}


def load(raw, day: str) -> dict:
    st = json.loads(raw) if raw else new_state(day)
    st.setdefault("values", dict(BASELINE))
    for k, v in BASELINE.items():
        st["values"].setdefault(k, v)
    st.setdefault("moved", {k: 0.0 for k in BASELINE})
    st.setdefault("day", day)
    st.setdefault("last_seen", None)
    return st


def dump(st) -> str:
    return json.dumps(st, ensure_ascii=False)


def _step(v, delta):
    """越接近两头变得越慢，永远落在 0~1 之间。"""
    return v + delta * (1 - v) if delta > 0 else v + delta * v


def apply(st: dict, event: str, day: str, times: int = 1) -> dict:
    """发生了一件事（event 见 EFFECTS），返回新的状态。同一天里每个数的变化不超过 DAILY_CAP。"""
    st = roll_day(st, day)
    st = deepcopy(st)
    for _ in range(max(1, min(times, 20))):
        for k, d in EFFECTS.get(event, {}).items():
            room = DAILY_CAP - abs(st["moved"][k])
            if room <= 0:
                continue
            old = st["values"][k]
            new = _step(old, max(-room, min(room, d)))
            st["values"][k] = round(new, 4)
            st["moved"][k] = round(st["moved"][k] + abs(new - old), 4)
    if event != "ignored":  # 主人来过（聊天、摸头……）
        st["last_seen"] = day
    return st


def roll_day(st: dict, day: str) -> dict:
    """跨天了：会回落的往天生值靠一点；没见面的每一天亲密降一点；当天的变化额度清零。"""
    if st["day"] >= day:
        return st
    st = deepcopy(st)
    from datetime import date
    d0, d1 = date.fromisoformat(st["day"]), date.fromisoformat(day)
    days = min((d1 - d0).days, 60)
    for k in DRIFTS:
        v = st["values"][k]
        st["values"][k] = round(BASELINE[k] + (v - BASELINE[k]) * (1 - DRIFT_RATE) ** days, 4)
    seen = st.get("last_seen")
    absent = days if not seen else min(days, (d1 - date.fromisoformat(seen)).days - 1)
    if absent > 0:
        v = st["values"]["intimacy"]
        st["values"]["intimacy"] = round(max(BASELINE["intimacy"], v - ABSENT_DECAY * absent), 4)
    st["day"] = day
    st["moved"] = {k: 0.0 for k in BASELINE}
    return st


def describe(values: dict) -> str:
    """把数值翻译成给大脑看的话（翻译规则在这里，不交给模型自己理解数字）。"""
    v = values
    lines = []
    if v["intimacy"] < 0.35:
        lines.append("你和主人认识还不久，有点拘谨，会客气一点，偶尔试探着撒娇。")
    elif v["intimacy"] < 0.7:
        lines.append("你和主人已经挺熟了，说话随意，会撒娇、会抱怨。")
    else:
        lines.append("你和主人非常亲密，像家人一样，什么都敢说，会开小玩笑、会耍赖。")
    if v["trust"] >= 0.6:
        lines.append("主人愿意跟你说心里话，你很珍惜，听到心事会认真接住、不开玩笑。")
    if v["clingy"] >= 0.7:
        lines.append("最近你特别黏主人，总想让主人多陪你一会儿。")
    elif v["clingy"] < 0.35:
        lines.append("最近你没那么黏人，挺独立的。")
    if v["tsundere"] >= 0.6:
        lines.append("最近你嘴特别硬，被夸、被关心都要先哼一声。")
    elif v["tsundere"] < 0.25:
        lines.append("最近你很坦率，开心就直接说开心。")
    if v["playful"] >= 0.68:
        lines.append("最近你很调皮，爱开玩笑、爱逗主人。")
    elif v["playful"] < 0.32:
        lines.append("最近你有点蔫，话少一些。")
    return "".join(lines)
