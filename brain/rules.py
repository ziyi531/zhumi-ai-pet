"""主动搭话的决策：纯函数。

要不要开口、为什么开口，全部由这里的规则决定；LLM 只负责把已经决定好的事说出口。
"""
from dataclasses import dataclass
from datetime import datetime, timedelta

import config
from state import PetState, hours_between, in_window


@dataclass
class Gate:
    last_proactive: datetime | None
    ignored_streak: int
    proactive_today: int


@dataclass
class Decision:
    kind: str            # followup / lonely / curious
    reason: str
    face: str
    vibe: str
    followup: dict | None = None
    topic: dict | None = None


def cooldown_hours(streak: int) -> float:
    """被无视得越多越识趣：冷却时间翻倍。"""
    return config.PROACTIVE_COOLDOWN_HOURS * 2 ** min(streak, config.MAX_IGNORE_BACKOFF)


def update_ignored(now, last_proactive, last_interaction, counted_for, streak):
    """主动搭话后一段时间没有回应，被无视次数 +1（每次主动只算一次）。

    返回新的 (streak, counted_for)。主人说话时由调用方把 streak 清零。
    """
    if last_proactive is None or counted_for == last_proactive:
        return streak, counted_for
    if last_interaction is not None and last_interaction >= last_proactive:
        return streak, counted_for
    if now - last_proactive >= timedelta(minutes=config.IGNORE_WINDOW_MINUTES):
        return streak + 1, last_proactive
    return streak, counted_for


def decide(st: PetState, now: datetime, gate: Gate, due_followups, topics):
    """返回 (Decision 或 None, 原因说明)。"""
    if in_window(now, config.QUIET_HOURS):
        return None, "免打扰时段"
    if gate.proactive_today >= config.DAILY_PROACTIVE_LIMIT:
        return None, f"今天已经主动找了 {gate.proactive_today} 次"

    since = hours_between(gate.last_proactive, now)

    # 回访最重要：不用等完整冷却，只要别和上一次主动挨得太近
    if due_followups:
        if since is None or since * 60 >= config.FOLLOWUP_MIN_GAP_MINUTES:
            f = due_followups[0]
            return Decision("followup", f"回访：{f['fact']}", "peek", "nudge", followup=f), "有到期的回访"
        return None, "有到期的回访，但刚主动找过主人"

    cd = cooldown_hours(gate.ignored_streak)
    if since is not None and since < cd:
        return None, f"冷却中（还要 {cd - since:.1f} 小时）"

    if st.loneliness >= config.LONELY_THRESHOLD and in_window(now, config.LONELY_HOURS):
        return Decision("lonely", f"孤单值 {st.loneliness:.2f}", "peek", "nudge"), "太孤单了"

    if st.curiosity >= config.CURIOUS_THRESHOLD and st.energy >= config.CURIOUS_MIN_ENERGY and topics:
        t = topics[0]
        return Decision("curious", f"好奇：{t['key']}", "curious", "tap", topic=t), "好奇心满了"

    return None, "没有想说的"
