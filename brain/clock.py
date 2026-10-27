"""虚拟时钟：真实时间 + 偏移量，用来快进测试。偏移量存在数据库里，重启后保留。

只允许往前拨，否则“上次互动时间”会跑到未来，状态计算就乱了。
"""
from datetime import datetime, timedelta

import db

FMT = "%Y-%m-%d %H:%M:%S"


def fmt(dt: datetime) -> str:
    return dt.strftime(FMT)


def parse(s):
    return datetime.strptime(s, FMT) if s else None


def offset_seconds() -> float:
    return float(db.get_kv("clock_offset", "0"))


def now() -> datetime:
    return (datetime.now() + timedelta(seconds=offset_seconds())).replace(microsecond=0)


def advance(hours: float) -> datetime:
    if hours < 0:
        raise ValueError("时钟只能往前拨")
    db.set_kv("clock_offset", offset_seconds() + hours * 3600)
    return now()


def set_time(hhmm: str) -> datetime:
    """拨到下一个 HH:MM（今天的已经过了就是明天的）。"""
    h, m = (int(x) for x in hhmm.split(":"))
    cur = now()
    target = cur.replace(hour=h, minute=m, second=0)
    if target <= cur:
        target += timedelta(days=1)
    return advance((target - cur).total_seconds() / 3600)
