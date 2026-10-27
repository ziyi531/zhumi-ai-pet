"""记忆管理：管理页（/memory）用的查看、修改、删除、导出，和对团子说“忘掉这件事”。

删除都是真删：整行从数据库抹掉，不留“已删除”的痕迹（这是主人的隐私）。
“忘掉”时找哪几条记忆由代码决定：说清楚了是哪件事就按那句话找，只说“忘掉这件事”就指主人上一句说的事。
"""
import json
import re
from datetime import datetime

import numpy as np

import clock
import config
import db
import memory
import recall
import traits

# 主人在让团子忘掉什么（只认祈使的说法；“你是不是忘了我的生日？”不算）
FORGET_RE = re.compile(r"忘掉|忘了吧|忘记吧|把.{0,20}忘了|别记|不要记|别再记|删掉.{0,6}(记忆|这件|那件|这个|那个)")
FILLER_RE = re.compile(r"忘掉|忘了吧|忘记吧|忘了|忘记|别记住?着?|不要记住?着?|别再记|删掉|记忆|帮我|请你?|你把|把|这件事|那件事|"
                       r"这个|那个|刚才|刚刚|说的|的事|吧|好吗|好不好|拜托|猪咪|团子|小糯米|[，。！？、,.!?～~\s]")


def is_forget_request(text: str) -> bool:
    t = text or ""
    return bool(FORGET_RE.search(t)) and not re.search(r"(是不是|有没有|会不会|会).{0,4}忘|忘.{0,6}[吗呢]", t)


def _hanzi(s):
    return len(re.findall(r"[一-鿿]", s))


def forget(text: str, now: datetime, history) -> dict:
    """按主人的话忘掉一件事：删掉相关的记忆、取消相关回访、删掉那句原话。返回删了什么（给日志和设备看）。"""
    cue = FILLER_RE.sub("", text)
    from_previous = _hanzi(cue) < 2
    prev_user = next((m for m in reversed(history) if m["role"] == "user"), None)
    if from_previous:
        cue = prev_user["text"] if prev_user else ""
    report = {"cue": cue, "memories": [], "followups": [], "messages": 0}
    if not cue:
        return report

    rows = db.memories_with_status(("active", "faded"))
    if rows:
        vec = recall.embed([cue])
        if vec is not None:
            mat = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
            sims = mat @ vec[0] + np.array([recall.keyword_bonus(cue, r["text"]) for r in rows], dtype=np.float32)
        else:  # 本地向量模型没开（云端代答时）：按共有的词找
            sims = np.array([recall.keyword_sim(cue, r["text"]) for r in rows], dtype=np.float32)
        for i in np.argsort(-sims)[:3]:
            if sims[i] >= config.FORGET_MIN_SIM:
                db.delete_memory(rows[i]["id"])
                report["memories"].append(rows[i]["text"])
    for f in db.pending_followups():
        if recall.keyword_bonus(cue, f["fact"]) or any(f["fact"] in m or m in f["fact"] for m in report["memories"]):
            db.close_followup(f["id"], "forgotten", clock.fmt(now))
            report["followups"].append(f["fact"])

    # 原话也删掉：只说“忘掉这件事”就删上一句和团子的回答；说清楚了（“怕老板的事”）就删提到它的每一句——
    # 主人明说要忘的事，宁可多删（“主人”“今天”这类到处都有的词不算重合）
    ids = []
    msgs = db.recent_messages(200)
    for i, m in enumerate(msgs):
        hit = (from_previous and prev_user and m["id"] == prev_user["id"]) or \
              (not from_previous and m["role"] == "user" and recall._grams(cue) & recall._grams(m["text"]))
        if hit:
            ids.append(m["id"])
            if i + 1 < len(msgs) and msgs[i + 1]["role"] == "pet":
                ids.append(msgs[i + 1]["id"])
    db.delete_messages(ids)
    report["messages"] = len(ids)
    return report


# ---------- 管理页 ----------
def overview(now: datetime) -> dict:
    tr = memory.traits_now(now)
    mems = []
    for r in db.memories_with_status(("active", "faded")):
        r = {k: v for k, v in r.items() if k != "embedding"}
        r["retention"] = round(recall.retention(r, now), 2)
        mems.append(r)
    return {
        "time": clock.fmt(now),
        "traits": tr["values"], "traits_baseline": traits.BASELINE, "traits_text": traits.describe(tr["values"]),
        "self_view": db.get_kv("self_view", ""),
        "memories": mems[::-1],
        "profile": db.get_profile(),
        "followups": db.pending_followups(),
        "diary": db.recent_diary(60)[::-1],
        "last_night": db.get_kv("last_night", ""),
    }


def edit(mid: int, text=None, kind=None, importance=None, now=None):
    row = db.get_memory(mid)
    if not row:
        return None
    text = str(text).strip()[: config.MEMORY_MAX_CHARS] if text is not None else row["text"]
    kind = kind if kind in recall.KINDS else row["kind"]
    importance = max(1, min(5, int(importance))) if importance is not None else row["importance"]
    emb = row["embedding"]
    if text != row["text"]:
        vec = recall.embed([text])
        if vec is None:
            raise RuntimeError("向量模型不可用，改不了文字")
        emb = vec[0].tobytes()
    db.edit_memory(mid, text, kind, importance, emb, clock.fmt(now or clock.now()))
    return {k: v for k, v in db.get_memory(mid).items() if k != "embedding"}


def set_traits(values: dict, now: datetime):
    """管理页里手动调性格数值（调试人设用）。"""
    with memory._traits_lock:
        day = now.strftime("%Y-%m-%d")
        st = traits.roll_day(traits.load(db.get_kv("traits"), day), day)
        for k, v in values.items():
            if k in traits.BASELINE:
                st["values"][k] = round(max(0.0, min(1.0, float(v))), 4)
        db.set_kv("traits", traits.dump(st))
        return st["values"]


EXPORT_TABLES = ["memories", "profile", "followups", "diary", "messages", "events", "device_log"]
EXPORT_KV = ["traits", "self_view", "born_at", "last_night", "tts_choice"]


def export(now: datetime) -> dict:
    """完整备份：团子的记忆、日记、画像、性格、聊天记录。向量不导出（换电脑后用 bge-m3 重新算就行）。"""
    data = {"exported_at": clock.fmt(now), "real_time": datetime.now().isoformat(timespec="seconds"),
            "pet": config.PET_NAME, "version": 1}
    for t in EXPORT_TABLES:
        data[t] = [{k: v for k, v in r.items() if k != "embedding"} for r in db.all_rows(t)]
    data["kv"] = {k: db.get_kv(k) for k in EXPORT_KV if db.get_kv(k) is not None}
    if "traits" in data["kv"]:
        data["kv"]["traits"] = json.loads(data["kv"]["traits"])
    return data
