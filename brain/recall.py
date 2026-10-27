"""情景记忆：存下值得记的事，聊天时按线索自动想起来——像人被一句话勾起往事，而不是每天读一遍日记。

每条记忆是一句话（谁、什么事、什么时候），分三类：owner 主人的事、us 我们俩之间的事、self 团子自己的经历。
向量用本地的 bge-m3（Ollama，强制跑在 CPU 上：显卡已经被大脑和声音服务占满了，算一句只要几十毫秒）。

想起哪几条由代码决定：相关度为主，重要度和新鲜度加一点分，再设一道门槛，不相关的不硬塞。
同一件事再提起时不新增，而是更新原来那条（换成新的说法、提起次数加一）。
"""
import logging
import math
from datetime import datetime

import numpy as np
import requests

import clock
import config
import db

log = logging.getLogger("pet.recall")
KINDS = ("owner", "us", "self")


def embed(texts):
    """一批文本 → 单位化的向量（N×1024 float32）；向量模型不在时返回 None，不抛异常。"""
    if not texts:
        return np.zeros((0, 1024), dtype=np.float32)
    try:
        r = requests.post(f"{config.OLLAMA_URL}/api/embed", json={
            "model": config.EMBED_MODEL, "input": list(texts), "keep_alive": "30m",
            "options": {"num_gpu": 0}}, timeout=30)  # num_gpu=0：只用 CPU，不去挤显卡
        r.raise_for_status()
        v = np.asarray(r.json()["embeddings"], dtype=np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
    except Exception as e:
        log.warning("算记忆向量失败（%s 在吗？）：%s", config.EMBED_MODEL, e)
        return None


def _matrix(rows):
    return np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]) if rows else None


def remember(kind, text, mood, importance, now: datetime):
    """记下一件事；和已有的同类记忆几乎一样就更新那条（淡忘了的会被重新想起来）。
    返回 ("added"|"reinforced"|"revived"|"skipped", id)。"""
    kind = kind if kind in KINDS else "owner"
    text = str(text or "").strip()[: config.MEMORY_MAX_CHARS]
    if len(text) < 4:
        return "skipped", None
    importance = max(1, min(5, int(importance or 3)))
    vec = embed([text])
    if vec is None:
        return "skipped", None
    ts = clock.fmt(now)
    same_kind = [r for r in db.memories_with_status(("active", "faded")) if r["kind"] == kind]
    if same_kind:
        sims = _matrix(same_kind) @ vec[0]
        best = int(np.argmax(sims))
        if sims[best] >= config.MEMORY_DEDUP_SIM:
            row = same_kind[best]
            db.reinforce_memory(row["id"], text, mood, importance, vec[0].tobytes(), ts)
            if row["status"] == "faded":
                db.set_memory_status(row["id"], "active", ts)
                return "revived", row["id"]
            return "reinforced", row["id"]
    return "added", db.add_memory(kind, text, mood, importance, vec[0].tobytes(), ts)


# ---------- 遗忘曲线（纯函数，夜间整理用） ----------
def retention(row, now: datetime) -> float:
    """还记得多清楚（0~1）：从上次被提起/想起算起按指数衰减。
    半衰期看重要度（3 级约 14 天，每差一级翻倍/减半），被提起、被想起的次数越多记得越牢；5 级永远记得。"""
    if row["importance"] >= 5:
        return 1.0
    touched = max(row["updated_at"], row.get("recalled_at") or "")
    age_days = max(0.0, (now - clock.parse(touched)).total_seconds() / 86400)
    reinforce = (row["mentions"] - 1) + (row.get("recall_count") or 0)
    half_life = config.MEMORY_HALF_LIFE_DAYS * 2 ** (row["importance"] - 3) * (1 + 0.5 * min(reinforce, 6))
    return 0.5 ** (age_days / half_life)


def should_fade(row, now: datetime) -> bool:
    return retention(row, now) < config.MEMORY_FADE_BELOW


# 到处都有的词不算“关键词重合”
COMMON = {"主人", "猪咪", "团子","我们", "你们", "他们", "今天", "昨天", "明天", "最近", "什么", "一个", "这个", "那个",
          "喜欢", "觉得", "时候", "现在", "还有", "没有", "就是", "可以", "因为", "所以", "已经", "好像", "怎么"}


def _grams(text):
    t = "".join(ch for ch in text if "一" <= ch <= "鿿" or ch.isalnum())
    return {t[i:i + 2] for i in range(len(t) - 1)} - COMMON


def keyword_bonus(query, memory_text) -> float:
    """问句和记忆里出现同一个具体的词（咪咪、日本、考研……）就加分：短名字在向量里信息太少，光看相似度会漏。"""
    return config.MEMORY_KEYWORD_BONUS if _grams(query) & _grams(memory_text) else 0.0


def score(sim, row, now: datetime) -> float:
    """想起来的优先级：相关度为主，重要的、最近的、常被提起的稍微加分。"""
    age_days = max(0.0, (now - clock.parse(row["updated_at"])).total_seconds() / 86400)
    return (sim + 0.03 * (row["importance"] - 3) + 0.06 * math.exp(-age_days / 14)
            + 0.02 * min(row["mentions"] - 1, 3))


def keyword_sim(query, memory_text) -> float:
    """没有向量时的相关度：共有的具体词越多越像（一个词 0.6 刚过门槛，两个 0.7，封顶 0.8）。"""
    n = len(_grams(query) & _grams(memory_text))
    return min(0.5 + 0.1 * n, 0.8) if n else 0.0


def relevant(query: str, now: datetime, k=None, vectors=True):
    """和这句话相关的往事（最多 k 条，相关度不够的不要）。顺带记下“想起过”。
    vectors=False 或向量模型不在（Ollama 没开）时只按共有的词找，粗一些但不至于什么都想不起来。"""
    k = k or config.MEMORY_RECALL_K
    rows = db.active_memories()
    if not rows or not query.strip():
        return []
    vec = embed([query]) if vectors else None
    if vec is None:
        sims = [keyword_sim(query, r["text"]) for r in rows]
    else:
        sims = _matrix(rows) @ vec[0] + np.array([keyword_bonus(query, r["text"]) for r in rows], dtype=np.float32)
    picked = sorted(((score(float(s), r, now), float(s), r) for s, r in zip(sims, rows)
                     if s >= config.MEMORY_MIN_SIM), key=lambda x: -x[0])[:k]
    if picked:
        db.mark_recalled([r["id"] for _, _, r in picked], clock.fmt(now))
        log.info("想起：%s", [(round(s, 2), r["text"]) for _, s, r in picked])
    return [{**{c: r[c] for c in ("id", "kind", "text", "mood", "importance", "created_at", "updated_at")},
             "sim": round(s, 3)} for _, s, r in picked]
