"""内在状态：全部由时间戳推导出来的纯函数，不碰数据库，也不调用 LLM。

精力看生物钟，孤单看多久没说话，好奇看时段和多久没交流。
同样的时间戳永远得到同样的状态，所以快进时钟就能复现任何一天。
"""
from dataclasses import asdict, dataclass
from datetime import datetime

import config

# 精力的生物钟曲线：(小时, 精力)，中间线性插值
ENERGY_CURVE = [(0, 0.3), (1, 0.2), (2, 0.1), (6, 0.1), (8, 0.6), (10, 0.9), (13, 0.75),
                (15, 0.85), (20, 0.8), (22, 0.55), (24, 0.3)]


def parse_hhmm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def in_window(t: datetime, window) -> bool:
    """t 是否落在 ("HH:MM", "HH:MM") 里，支持跨午夜。"""
    start, end = parse_hhmm(window[0]), parse_hhmm(window[1])
    m = t.hour * 60 + t.minute
    return start <= m < end if start <= end else (m >= start or m < end)


def hours_between(a, b):
    if a is None:
        return None
    return max(0.0, (b - a).total_seconds() / 3600)


def circadian_energy(t: datetime) -> float:
    h = t.hour + t.minute / 60
    for (h0, e0), (h1, e1) in zip(ENERGY_CURVE, ENERGY_CURVE[1:]):
        if h0 <= h <= h1:
            return e0 + (e1 - e0) * (h - h0) / (h1 - h0)
    return ENERGY_CURVE[-1][1]


def curiosity_base(t: datetime) -> float:
    h = t.hour
    if 8 <= h < 12:
        return 0.35
    if 12 <= h < 18:
        return 0.4
    if 18 <= h < 22:
        return 0.3
    return 0.1


@dataclass
class PetState:
    energy: float
    loneliness: float
    curiosity: float
    phase: str          # awake / sleepy / asleep
    sleepy_talk: bool   # 说话要不要带困意
    hours_alone: float  # 距离上次和主人说话

    def to_dict(self):
        d = asdict(self)
        for k in ("energy", "loneliness", "curiosity", "hours_alone"):
            d[k] = round(d[k], 3)
        return d


def compute(now: datetime, last_interaction=None, last_pat=None, last_proactive=None) -> PetState:
    alone = hours_between(last_interaction, now) or 0.0
    lonely = min(1.0, alone / config.LONELY_HOURS_TO_FULL)
    # 被摸一下没有说话那么管用：孤单值降到 PAT_LONELINESS，然后重新开始涨
    if last_pat is not None and (last_interaction is None or last_pat > last_interaction):
        lonely = min(lonely, config.PAT_LONELINESS + hours_between(last_pat, now) / config.LONELY_HOURS_TO_FULL)

    exchanges = [t for t in (last_interaction, last_proactive) if t is not None]
    quiet = hours_between(max(exchanges), now) if exchanges else 0.0
    curiosity = min(1.0, curiosity_base(now) + config.CURIOUS_GROWTH_PER_HOUR * quiet)

    energy = circadian_energy(now)
    phase = "asleep" if energy < 0.2 else "sleepy" if energy < 0.45 else "awake"
    sleepy_talk = in_window(now, config.SLEEPY_HOURS) or phase == "asleep"
    return PetState(energy, lonely, curiosity, phase, sleepy_talk, alone)
