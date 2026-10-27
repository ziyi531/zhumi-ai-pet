"""SQLite 存储：状态键值、对话、待回访事项、主人画像、待推送事件、设备日志。

时间统一存成 "YYYY-MM-DD HH:MM:SS" 字符串（虚拟时钟时间），字符串比较即时间比较。
"""
import sqlite3
import threading
from contextlib import contextmanager

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY, ts TEXT, role TEXT, text TEXT, proactive INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS followups (
    id INTEGER PRIMARY KEY, created_at TEXT, fact TEXT, mood TEXT, due_at TEXT, hint TEXT,
    status TEXT DEFAULT 'pending', done_at TEXT);
CREATE TABLE IF NOT EXISTS profile (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT, asked_at TEXT);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY, created_at TEXT, kind TEXT, text TEXT, face TEXT, vibe TEXT, voice TEXT,
    delivered_at TEXT);
CREATE TABLE IF NOT EXISTS device_log (id INTEGER PRIMARY KEY, ts TEXT, kind TEXT, count INTEGER);
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY, created_at TEXT, updated_at TEXT, kind TEXT, text TEXT, mood TEXT,
    importance INTEGER DEFAULT 3, mentions INTEGER DEFAULT 1, recalled_at TEXT, recall_count INTEGER DEFAULT 0,
    embedding BLOB, status TEXT DEFAULT 'active');
CREATE TABLE IF NOT EXISTS diary (day TEXT PRIMARY KEY, text TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY, created_at TEXT, due_at TEXT, text TEXT, status TEXT DEFAULT 'pending', done_at TEXT);
"""

# 调度线程和请求线程共用数据库，写入量很小，一把锁最省心
_lock = threading.Lock()


@contextmanager
def tx():
    with _lock:
        config.DB_FILE.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(config.DB_FILE, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def init():
    with tx() as c:
        c.executescript(SCHEMA)


def _rows(cur):
    return [dict(r) for r in cur.fetchall()]


# ---------- kv ----------
def get_kv(key, default=None):
    with tx() as c:
        r = c.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
    return r["value"] if r else default


def set_kv(key, value):
    with tx() as c:
        c.execute("INSERT OR REPLACE INTO kv(key, value) VALUES(?, ?)", (key, str(value)))


# ---------- 对话 ----------
def add_message(role, text, ts, proactive=False):
    with tx() as c:
        return c.execute("INSERT INTO messages(ts, role, text, proactive) VALUES(?,?,?,?)",
                         (ts, role, text, int(proactive))).lastrowid


def get_message(mid):
    with tx() as c:
        rows = _rows(c.execute("SELECT * FROM messages WHERE id=?", (mid,)))
    return rows[0] if rows else None


def recent_messages(n):
    with tx() as c:
        rows = _rows(c.execute("SELECT * FROM messages ORDER BY id DESC LIMIT ?", (n,)))
    return rows[::-1]


# ---------- 回访 ----------
def add_followup(fact, mood, due_at, hint, ts):
    with tx() as c:
        c.execute("INSERT INTO followups(created_at, fact, mood, due_at, hint) VALUES(?,?,?,?,?)",
                  (ts, fact, mood, due_at, hint))


def pending_followups():
    with tx() as c:
        return _rows(c.execute("SELECT * FROM followups WHERE status='pending' ORDER BY due_at"))


def close_followup(fid, status, ts):
    with tx() as c:
        c.execute("UPDATE followups SET status=?, done_at=? WHERE id=?", (status, ts, fid))


# ---------- 画像 ----------
def set_profile(key, value, ts):
    with tx() as c:
        c.execute("""INSERT INTO profile(key, value, updated_at) VALUES(?,?,?)
                     ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
                  (key, value, ts))


def get_profile():
    with tx() as c:
        return _rows(c.execute("SELECT * FROM profile ORDER BY updated_at DESC"))


def mark_profile_asked(key, ts):
    with tx() as c:
        c.execute("UPDATE profile SET asked_at=? WHERE key=?", (ts, key))


# ---------- 推送事件 ----------
def add_event(kind, text, face, vibe, voice, ts):
    with tx() as c:
        c.execute("INSERT INTO events(created_at, kind, text, face, vibe, voice) VALUES(?,?,?,?,?,?)",
                  (ts, kind, text, face, vibe, voice))


def take_pending_events(ts, not_before):
    """设备拉取：取出未送达的事件并标记为已送达。

    早于 not_before 的事件已经过时（比如设备离线了半天），标记为 expired，不再播放。
    """
    with tx() as c:
        rows = _rows(c.execute("SELECT * FROM events WHERE delivered_at IS NULL ORDER BY id"))
        fresh = [r for r in rows if r["created_at"] >= not_before]
        c.executemany("UPDATE events SET delivered_at=? WHERE id=?",
                      [(ts if r in fresh else "expired", r["id"]) for r in rows])
    return fresh


def count_events_on(day):
    """某天（"YYYY-MM-DD"）一共主动找了主人几次。主人定的提醒不算（那是主人让它说的）。"""
    with tx() as c:
        return c.execute("SELECT COUNT(*) FROM events WHERE substr(created_at, 1, 10)=? AND kind!='reminder'",
                         (day,)).fetchone()[0]


# ---------- 定时提醒 ----------
def add_reminder(due_at, text, ts):
    with tx() as c:
        return c.execute("INSERT INTO reminders(created_at, due_at, text) VALUES(?,?,?)", (ts, due_at, text)).lastrowid


def pending_reminders():
    with tx() as c:
        return _rows(c.execute("SELECT * FROM reminders WHERE status='pending' ORDER BY due_at"))


def due_reminders(ts):
    with tx() as c:
        return _rows(c.execute("SELECT * FROM reminders WHERE status='pending' AND due_at<=? ORDER BY due_at", (ts,)))


def close_reminder(rid, status, ts):
    with tx() as c:
        c.execute("UPDATE reminders SET status=?, done_at=? WHERE id=?", (status, ts, rid))


# ---------- 情景记忆（recall.py） ----------
def add_memory(kind, text, mood, importance, embedding: bytes, ts):
    with tx() as c:
        cur = c.execute("""INSERT INTO memories(created_at, updated_at, kind, text, mood, importance, embedding)
                           VALUES(?,?,?,?,?,?,?)""", (ts, ts, kind, text, mood, importance, embedding))
        return cur.lastrowid


def active_memories(with_embedding=True):
    cols = "*" if with_embedding else ("id, created_at, updated_at, kind, text, mood, importance, mentions, "
                                       "recalled_at, recall_count, status")
    with tx() as c:
        return _rows(c.execute(f"SELECT {cols} FROM memories WHERE status='active' ORDER BY id"))


def memories_with_status(statuses):
    q = ",".join("?" * len(statuses))
    with tx() as c:
        return _rows(c.execute(f"SELECT * FROM memories WHERE status IN ({q}) ORDER BY id", tuple(statuses)))


def set_memory_status(mid, status, ts):
    with tx() as c:
        c.execute("UPDATE memories SET status=?, updated_at=? WHERE id=?", (status, ts, mid))


def merge_memories(keep_id, text, importance, mentions, embedding: bytes, drop_ids, ts):
    """夜间整理：几条说的是同一件事，合成 keep_id 这一条，其余标成 merged。"""
    with tx() as c:
        c.execute("""UPDATE memories SET text=?, importance=?, mentions=?, embedding=?, updated_at=?, status='active'
                     WHERE id=?""", (text, importance, mentions, embedding, ts, keep_id))
        c.executemany("UPDATE memories SET status='merged', updated_at=? WHERE id=?", [(ts, i) for i in drop_ids])


def reinforce_memory(mid, text, mood, importance, embedding: bytes, ts):
    """同一件事又提起了：换成新的说法，重要度取大，提起次数加一。"""
    with tx() as c:
        c.execute("""UPDATE memories SET text=?, mood=?, importance=MAX(importance, ?), embedding=?,
                     updated_at=?, mentions=mentions+1 WHERE id=?""", (text, mood, importance, embedding, ts, mid))


def mark_recalled(ids, ts):
    with tx() as c:
        c.executemany("UPDATE memories SET recalled_at=?, recall_count=recall_count+1 WHERE id=?",
                      [(ts, i) for i in ids])


def delete_memory(mid):
    """真删（主人要求忘掉、或在管理页删除）：整行抹掉，不留痕迹。"""
    with tx() as c:
        return c.execute("DELETE FROM memories WHERE id=?", (mid,)).rowcount


def get_memory(mid):
    with tx() as c:
        r = c.execute("SELECT * FROM memories WHERE id=?", (mid,)).fetchone()
    return dict(r) if r else None


def edit_memory(mid, text, kind, importance, embedding: bytes, ts):
    """管理页里改一条记忆。"""
    with tx() as c:
        c.execute("UPDATE memories SET text=?, kind=?, importance=?, embedding=?, updated_at=? WHERE id=?",
                  (text, kind, importance, embedding, ts, mid))


def delete_messages(ids):
    with tx() as c:
        c.executemany("DELETE FROM messages WHERE id=?", [(i,) for i in ids])


def delete_profile(key):
    with tx() as c:
        return c.execute("DELETE FROM profile WHERE key=?", (key,)).rowcount


def delete_diary(day):
    with tx() as c:
        return c.execute("DELETE FROM diary WHERE day=?", (day,)).rowcount


def all_rows(table):
    """导出备份用（表名只来自代码里的固定列表）。"""
    with tx() as c:
        return _rows(c.execute(f"SELECT * FROM {table}"))


# ---------- 日记（nightly.py） ----------
def messages_between(start, end):
    with tx() as c:
        return _rows(c.execute("SELECT * FROM messages WHERE ts>=? AND ts<? ORDER BY id", (start, end)))


def set_diary(day, text, ts):
    with tx() as c:
        c.execute("INSERT OR REPLACE INTO diary(day, text, created_at) VALUES(?,?,?)", (day, text, ts))


def recent_diary(n):
    with tx() as c:
        rows = _rows(c.execute("SELECT * FROM diary ORDER BY day DESC LIMIT ?", (n,)))
    return rows[::-1]


# ---------- 设备日志 ----------
def add_device_log(kind, count, ts):
    with tx() as c:
        c.execute("INSERT INTO device_log(ts, kind, count) VALUES(?,?,?)", (ts, kind, count))
