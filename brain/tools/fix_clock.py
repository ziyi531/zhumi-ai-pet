"""把大脑的虚拟时钟拨回真实时间。

调试页（/debug）的“快进”会给时钟加偏移量，存在数据库里、重启也不丢。忘了拨回来，大脑就一直活在“将来”：
主人凌晨三点问几点，它以为是第二天傍晚（2026-10-28 真机踩过，快了约 40 小时）。
时钟只能往前拨，所以不能简单清零：数据库里的时间都是按快进后的时钟记的，清零后它们全在“将来”，
孤单值、回访、遗忘曲线都会算乱。这里把所有记下的时间一起往回挪同样的偏移量，再把偏移量清零。

先关掉大脑再跑（从 ai_pet/ 目录）：python tools/fix_clock.py
会先备份到 data/pet.db.bak-clock-<时间>。
"""
import json
import re
import shutil
import socket
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

TS_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
FMT = "%Y-%m-%d %H:%M:%S"


def shift_ts(s: str, off: timedelta) -> str:
    return TS_RE.sub(lambda m: (datetime.strptime(m.group(0), FMT) - off).strftime(FMT), s)


def shift_day(s: str, off: timedelta) -> str:
    """只有日期的（性格结算到哪天）：按当天中午挪，挪完不超过今天。"""
    d = (datetime.strptime(s, "%Y-%m-%d") + timedelta(hours=12) - off).date()
    return min(d, datetime.now().date()).isoformat()


def shift_json(v, off):
    if isinstance(v, dict):
        return {k: shift_json(x, off) for k, x in v.items()}
    if isinstance(v, list):
        return [shift_json(x, off) for x in v]
    if isinstance(v, str):
        return shift_day(v, off) if DAY_RE.match(v) else shift_ts(v, off)
    return v


def brain_running() -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", config.PORT)) == 0


def main():
    if brain_running():
        sys.exit(f"大脑还开着（端口 {config.PORT}），先关掉再跑。")
    path = Path(config.DB_FILE)
    c = sqlite3.connect(path)
    row = c.execute("SELECT value FROM kv WHERE key='clock_offset'").fetchone()
    seconds = float(row[0]) if row else 0.0
    if not seconds:
        print("时钟没有偏移，不用改。")
        return
    off = timedelta(seconds=seconds)
    backup = path.with_name(f"{path.name}.bak-clock-{datetime.now():%Y%m%d-%H%M%S}")
    c.close()
    shutil.copy2(path, backup)
    print(f"时钟快了 {seconds / 3600:.1f} 小时，已备份到 {backup.name}")

    c = sqlite3.connect(path)
    changed = 0
    tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    for t in tables:
        if t == "kv":
            continue
        cols = [r[1] for r in c.execute(f"PRAGMA table_info({t})") if r[2].upper() == "TEXT"]
        key = "rowid" if t != "diary" else "day"
        for col in cols:
            for k, v in c.execute(f"SELECT {key}, {col} FROM {t}").fetchall():
                if not isinstance(v, str):
                    continue
                if t == "diary" and col == "day" and DAY_RE.match(v):
                    new = shift_day(v, off)
                elif TS_RE.fullmatch(v):
                    new = shift_ts(v, off)
                else:
                    continue
                if new != v:
                    c.execute(f"UPDATE {t} SET {col}=? WHERE {key}=?", (new, k))
                    changed += 1
    for k, v in c.execute("SELECT key, value FROM kv").fetchall():
        if k == "clock_offset" or not isinstance(v, str):
            continue
        if v.startswith(("{", "[")):
            try:
                data = json.loads(v)
            except ValueError:
                continue
            if k in ("cloud_usage", "cloud_standin_usage"):  # 这两个本来就按真实日期记
                continue
            new = json.dumps(shift_json(data, off), ensure_ascii=False)
        elif TS_RE.fullmatch(v):
            new = shift_ts(v, off)
        elif DAY_RE.match(v):
            new = shift_day(v, off)
        else:
            continue
        if new != v:
            c.execute("UPDATE kv SET value=? WHERE key=?", (new, k))
            changed += 1
    c.execute("UPDATE kv SET value='0' WHERE key='clock_offset'")
    c.commit()
    last = c.execute("SELECT value FROM kv WHERE key='last_interaction'").fetchone()
    print(f"改了 {changed} 处时间，偏移量清零。现在真实时间 {datetime.now():%Y-%m-%d %H:%M}，"
          f"最后一次聊天记为 {last[0] if last else '无'}")


if __name__ == "__main__":
    main()
