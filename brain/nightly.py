"""夜间整理：团子睡觉时（免打扰时段）把白天的事整理一遍，每晚一次。

1. 合并：相似度 ≥ MEMORY_MERGE_SIM 的同类记忆由代码列为候选，模型判断是不是同一件事；
   是就合成一句（后来的变化也写进去，“住过院、后来出院了”），不是就分开留着。
2. 淡忘：按遗忘曲线（recall.retention，代码算）把记不清的小事标成 faded——不删，以后又提起会重新想起来。
3. 日记：用团子第一人称写今天的小日记；“今天最想记住的事”存成一条往事，以后能被线索勾起来。
4. 自我认知：在昨天的“我是怎样的团子”上小小地修订，放进每次聊天的提示词。

日记不会每次聊天都整篇读一遍（那就成了“扮演昨天的自己”）；靠的是被线索勾起的往事 + 慢慢长成的性格数值，
日记只是素材和自我认知的来源。
"""
import logging
from datetime import datetime, timedelta

import numpy as np

import clock
import config
import db
import memory
import persona
import recall
import traits
from state import in_window

log = logging.getLogger("pet.nightly")


def night_of(now: datetime) -> str:
    """现在这一晚整理的是哪一天：23:45 整理当天，凌晨 2 点整理的是前一天。"""
    return (now - timedelta(hours=9)).strftime("%Y-%m-%d")


def due(now: datetime) -> bool:
    return in_window(now, config.QUIET_HOURS) and (db.get_kv("last_night") or "") < night_of(now)


def maybe_run(client, now: datetime):
    """心跳里调：在免打扰时段、今晚还没整理过才跑。"""
    return run(client, now) if due(now) else None


def run(client, now: datetime, day: str | None = None) -> dict:
    day = day or night_of(now)
    report = {"day": day, "merged": [], "faded": [], "diary": None, "highlight": None, "self_view": None}
    try:
        report["merged"] = _merge(client, now)
        report["faded"] = _fade(now)
        diary, highlight = _diary(client, now, day)
        report["diary"], report["highlight"] = diary, highlight
        if diary:
            report["self_view"] = _self_view(client, now, diary)
    finally:
        db.set_kv("last_night", day)  # 出错也不在同一晚反复重试（明晚再来）
    log.info("夜间整理（%s）：合并 %d 组，淡忘 %d 条，%s", day, len(report["merged"]), len(report["faded"]),
             "写了日记" if report["diary"] else "今天没聊天，没写日记")
    return report


# ---------- 1. 合并 ----------
def _clusters(rows):
    """相似度够高的同类记忆连成一组（并查集），按时间先后排好。"""
    if len(rows) < 2:
        return []
    m = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    sims = m @ m.T
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if rows[i]["kind"] == rows[j]["kind"] and sims[i, j] >= config.MEMORY_MERGE_SIM:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(rows[i])
    return [sorted(g, key=lambda r: r["created_at"]) for g in groups.values() if len(g) > 1]


def _merge(client, now):
    merged = []
    for group in _clusters(db.active_memories())[: config.MEMORY_MERGE_MAX]:
        group = group[:5]
        try:
            res = client.chat_json(persona.merge_messages([r["text"] for r in group]))
        except Exception as e:
            log.warning("合并判断失败：%s", e)
            continue
        if not (isinstance(res, dict) and res.get("same") is True and str(res.get("text") or "").strip()):
            continue
        text = str(res["text"]).strip()[: config.MEMORY_MAX_CHARS]
        vec = recall.embed([text])
        if vec is None:
            continue
        keep = group[0]  # 留最早的那条（记住“从什么时候开始”），内容换成合成后的
        db.merge_memories(keep["id"], text, max(r["importance"] for r in group), sum(r["mentions"] for r in group),
                          vec[0].tobytes(), [r["id"] for r in group[1:]], clock.fmt(now))
        merged.append({"from": [r["text"] for r in group], "to": text})
    return merged


# ---------- 2. 淡忘 ----------
def _fade(now):
    faded = []
    for r in db.active_memories(with_embedding=False):
        if recall.should_fade(r, now):
            db.set_memory_status(r["id"], "faded", r["updated_at"])  # 不改时间：淡忘不算“被想起”
            faded.append(r["text"])
    return faded


# ---------- 3. 日记 ----------
def _diary(client, now, day):
    start = f"{day} 05:00:00"
    end = (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d 05:00:00")
    msgs = db.messages_between(start, end)
    if not msgs:
        return None, None
    lines = [f"{'主人' if m['role'] == 'user' else '我'}：{m['text']}" for m in msgs][-40:]
    tr = memory.traits_now(now)
    try:
        res = client.chat_json(persona.diary_messages(day, "\n".join(lines), traits.describe(tr["values"]),
                                                      db.get_kv("self_view", "")), temperature=0.7)
    except Exception as e:
        log.warning("写日记失败：%s", e)
        return None, None
    if not isinstance(res, dict) or not str(res.get("diary") or "").strip():
        return None, None
    diary = persona.EMOJI_RE.sub("", str(res["diary"])).strip()[:400]
    db.set_diary(day, diary, clock.fmt(now))
    highlight = str(res.get("highlight") or "").strip()
    if highlight:
        importance = max(1, min(4, int(res.get("importance") or 3)))
        # 记成“团子自己的经历”，时间算在那一天（让它说得出“那天”）
        recall.remember("self", highlight, "", importance, datetime.strptime(f"{day} 21:00", "%Y-%m-%d %H:%M"))
    return diary, highlight


# ---------- 4. 自我认知 ----------
def _self_view(client, now, diary):
    old = db.get_kv("self_view", "")
    tr = memory.traits_now(now)
    try:
        res = client.chat_json(persona.self_view_messages(old, diary, traits.describe(tr["values"])), temperature=0.5)
    except Exception as e:
        log.warning("修订自我认识失败：%s", e)
        return None
    view = str((res or {}).get("self_view") or "").strip() if isinstance(res, dict) else ""
    view = persona.EMOJI_RE.sub("", view)[: config.SELF_VIEW_MAX_CHARS]
    if len(view) < 10 or persona.off_character(view):
        return None
    db.set_kv("self_view", view)
    return view
