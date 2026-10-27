"""记忆提炼：每轮对话后让模型把值得记的事整理成结构化数据。

模型只“建议”回访时间，最终的到期时间由 resolve_due 用代码校验，保证可复现。
"""
import hashlib
import json
import logging
import re
from datetime import datetime, timedelta

import threading

import clock
import config
import db
import persona
import recall
import traits
from state import in_window, parse_hhmm

_traits_lock = threading.Lock()


def traits_now(now: datetime) -> dict:
    """当前的性格数值（顺带把跨过的天数结算掉：回落、生分）。"""
    with _traits_lock:
        st = traits.roll_day(traits.load(db.get_kv("traits"), now.strftime("%Y-%m-%d")), now.strftime("%Y-%m-%d"))
        db.set_kv("traits", traits.dump(st))
        return st


def brain_traits_event(event: str, now: datetime, times: int = 1) -> dict:
    """发生了一件影响性格的事（聊天、摸头、被逗、被冷落……），记进数据库。"""
    with _traits_lock:
        day = now.strftime("%Y-%m-%d")
        st = traits.apply(traits.load(db.get_kv("traits"), day), event, day, times)
        db.set_kv("traits", traits.dump(st))
        return st

log = logging.getLogger("pet.memory")

HHMM_RE = re.compile(r"(\d{1,2})[:：](\d{2})")


def is_negative(mood: str) -> bool:
    return any(w in (mood or "") for w in config.NEGATIVE_MOOD_WORDS)


def resolve_due(now: datetime, hhmm, mood) -> datetime | None:
    """算出回访时间：已经过了的时间点顺延到明天；落在免打扰时段的推迟到醒来。"""
    due = None
    m = HHMM_RE.search(str(hhmm or ""))
    if m and int(m.group(1)) < 24 and int(m.group(2)) < 60:
        due = now.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
        if due <= now + timedelta(minutes=30):
            due += timedelta(days=1)
    if due is None and is_negative(mood):
        due = now + timedelta(hours=config.FOLLOWUP_DEFAULT_HOURS)
    if due is None:
        return None
    if in_window(due, config.QUIET_HOURS):
        wake = parse_hhmm(config.QUIET_HOURS[1])
        moved = due.replace(hour=wake // 60, minute=wake % 60)
        if moved <= due:
            moved += timedelta(days=1)
        due = moved
    return due.replace(second=0, microsecond=0)


def apply(result, now: datetime, traits_at: datetime | None = None, followups=True):
    """把模型提炼的结果写进数据库（情景记忆、回访、画像、相处方式对性格的推动），返回新增的回访列表。
    traits_at：性格按哪个时刻结算（补记旧对话时用现在，性格的日子不能倒着走）。
    followups=False：这一轮主人定了提醒，回访一条都不加（提醒有闹钟管）。"""
    if not isinstance(result, dict):
        return []
    ts = clock.fmt(now)

    for m in result.get("memories") or []:
        if isinstance(m, dict) and m.get("text"):
            how, _ = recall.remember(m.get("kind"), m.get("text"), str(m.get("mood") or "")[:10],
                                     m.get("importance"), now)
            if how != "skipped":
                log.info("记住（%s，%s）：%s", m.get("kind"), how, m.get("text"))

    interaction = str(result.get("interaction") or "").strip()
    if interaction in traits.EFFECTS:
        brain_traits_event(interaction, traits_at or now)

    pending = db.pending_followups()
    known = {f["fact"] for f in pending}
    added = []
    for f in (result.get("facts") or []) if followups else []:
        if not isinstance(f, dict):
            continue
        fact = str(f.get("fact") or "").strip()[:60]
        if not fact or fact in known or len(pending) + len(added) >= config.MAX_PENDING_FOLLOWUPS:
            continue
        mood = str(f.get("mood") or "").strip()[:10]
        hint = str(f.get("follow_up") or "").strip()[:60]
        # 要回访的事本身也是值得想起的往事（“在准备考研”过两天说“复习好累”时要能勾起来）
        recall.remember("owner", fact, mood, 3, now)
        due = resolve_due(now, f.get("follow_up_at"), mood)
        if due is None:
            continue
        db.add_followup(fact, mood, clock.fmt(due), hint, ts)
        known.add(fact)
        added.append({"fact": fact, "mood": mood, "due_at": clock.fmt(due), "hint": hint})

    profile = result.get("profile")
    if isinstance(profile, dict):
        for k, v in profile.items():
            if k and isinstance(v, (str, int, float)) and str(v).strip():
                db.set_profile(str(k).strip()[:20], str(v).strip()[:60], ts)
    return added


def extract(client, user_text: str, pet_reply: str, now: datetime, known=(), traits_at=None, followups=True):
    """known：这轮聊天时勾起的往事文本，帮整理员分清说的是谁。返回新增的回访；大脑没答上来返回 None。"""
    msgs = persona.memory_messages(user_text, pet_reply, now, known)
    try:
        result = client.chat_json(msgs)
    except Exception as e:
        log.warning("记忆提炼失败：%s", e)
        return None
    added = apply(result, now, traits_at, followups)
    if added:
        log.info("新增回访：%s", added)
    return added


# ---------- 补记：云端代答时没法提炼（存记忆要本地向量模型），本地大脑回来后补上 ----------
_backlog_lock = threading.Lock()   # 改积压列表时短暂持有
_catch_up_lock = threading.Lock()  # 同一时间只有一个线程在补


def _backlog():
    raw = db.get_kv("memory_backlog")
    return json.loads(raw) if raw else []


def _fingerprint(user_text, pet_text):
    return hashlib.sha1(f"{user_text}\n{pet_text}".encode("utf-8")).hexdigest()[:16]


def defer(user_id, pet_id, now: datetime, known=()):
    """记下一轮还没提炼的对话。只记消息编号和指纹，不存原话：主人后来让忘掉、原话被删了，就不补了
    （SQLite 会把删掉的编号再分给新消息，所以要靠指纹认出是不是原来那两句）。"""
    user, pet = db.get_message(user_id), db.get_message(pet_id)
    with _backlog_lock:
        items = _backlog()
        items.append({"user": user_id, "pet": pet_id, "at": clock.fmt(now), "known": list(known),
                      "fp": _fingerprint(user["text"], pet["text"])})
        db.set_kv("memory_backlog", json.dumps(items[-config.MEMORY_BACKLOG_MAX:], ensure_ascii=False))


def _drop(item):
    with _backlog_lock:
        db.set_kv("memory_backlog", json.dumps([i for i in _backlog() if i != item], ensure_ascii=False))


def backlog_size() -> int:
    return len(_backlog())


def catch_up(client, now: datetime) -> int:
    """按时间顺序补提炼积压的对话；当时的时间照旧（记忆、回访按当时算），性格按现在结算。
    大脑中途又不行了就停下，剩下的下次再补。返回补了几轮。"""
    if not _catch_up_lock.acquire(blocking=False):
        return 0  # 别的线程正在补
    done = 0
    try:
        for item in _backlog():
            user, pet = db.get_message(item["user"]), db.get_message(item["pet"])
            if user and pet and _fingerprint(user["text"], pet["text"]) == item.get("fp"):
                if extract(client, user["text"], pet["text"], clock.parse(item["at"]), item.get("known") or (),
                           traits_at=now) is None:
                    break
                done += 1
            _drop(item)
        if done:
            log.info("补记了云端代答时的 %d 轮对话", done)
        return done
    finally:
        _catch_up_lock.release()
