"""问云端大模型（Gemini API）：本地 9B 答不好的难题，交给云端想，再由团子用自己的口吻说。

隐私：问难题时只发一句问题（toolbox 里已抹掉主人的称呼），不带对话历史、记忆和画像。
例外是“代答”（standin）：本地大脑（Ollama）没开时，人设、最近几句对话、画像和被勾起的记忆会发给云端，
让它替团子说话（主人同意过，PET_CLOUD_STANDIN=0 关掉）。
开关：没配 key 或 PET_CLOUD=0 就不可用；每天最多 CLOUD_DAILY_LIMIT 次（真实日期，花的是真钱）。

key 在 Google AI Studio 生成，放环境变量 GEMINI_API_KEY，或者写进 data/gemini_api_key.txt。注意：API 免费层的提问会被 Google 用来改进产品，
要开通 Cloud 结算用付费层（Google AI Pro 每月送的 Cloud 额度可以抵扣），付费层的数据不拿去训练。
"""
import json
import logging
import os
import threading
from datetime import date

import requests

import config
import db

log = logging.getLogger("pet.cloud")
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
SYSTEM = ("你是一个知识渊博、讲解清楚的助手。用简体中文回答，准确、简洁，300 字以内；"
          "不确定的地方直接说不确定，不要编造。你的回答会再由一个小宠物用口语转述给它的主人。")
_lock = threading.Lock()


KEY_FILE = config.DATA_DIR / "gemini_api_key.txt"


def _key() -> str:
    """环境变量 GEMINI_API_KEY 优先；没有就读 data/gemini_api_key.txt（只放 key 本身一行）。
    文件的好处是改了马上生效，不用重启终端和程序去刷新环境变量。key 只用在请求头里，不写日志、不发给别处。"""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key and KEY_FILE.exists():
        # utf-8-sig：记事本 / PowerShell 存 UTF-8 时会在开头加个看不见的 BOM，不去掉请求头就发不出去
        key = KEY_FILE.read_text(encoding="utf-8-sig").strip()
    return key


def available() -> bool:
    return config.CLOUD and bool(_key())


def used_today(key="cloud_usage") -> int:
    raw = db.get_kv(key)
    usage = json.loads(raw) if raw else {}
    return usage.get("count", 0) if usage.get("date") == date.today().isoformat() else 0


def _take_quota(key="cloud_usage", limit=None) -> bool:
    """占用一次当天额度；用完了返回 False。"""
    limit = config.CLOUD_DAILY_LIMIT if limit is None else limit
    with _lock:
        n = used_today(key)
        if n >= limit:
            return False
        db.set_kv(key, json.dumps({"date": date.today().isoformat(), "count": n + 1}))
        return True


def _post(payload) -> str:
    r = requests.post(API.format(model=config.CLOUD_MODEL), headers={"x-goog-api-key": _key()}, json=payload,
                      timeout=config.CLOUD_TIMEOUT_S)
    r.raise_for_status()
    cand = (r.json().get("candidates") or [{}])[0]
    return "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", [])
                   if not p.get("thought")).strip()


def ask(question: str) -> str:
    """返回云端的回答；不可用、额度用完或出错时返回空字符串（不抛异常）。"""
    if not available():
        return ""
    if not _take_quota():
        log.warning("今天问云端的次数用完了（%d 次）", config.CLOUD_DAILY_LIMIT)
        return ""
    try:
        text = _post({
            "systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": question}]}],
            "generationConfig": {"maxOutputTokens": config.CLOUD_MAX_TOKENS},
        })
        log.info("问云端（%s）：%s → %d 字", config.CLOUD_MODEL, question, len(text))
        return text[:1500]
    except Exception as e:
        # 不把 key 写进日志：requests 的异常信息里只有地址，key 在请求头里
        log.warning("问云端失败：%s", str(e)[:200])
        return ""


# ---------- 代答：本地大脑没开时，云端照着同一套人设和上下文替团子说话 ----------
def standin_available() -> bool:
    return config.CLOUD_STANDIN and available()


def to_gemini(messages):
    """Ollama 格式的对话 → Gemini 的 systemInstruction + contents。
    system 消息都并进 systemInstruction；连续同一方的话合并；开头是团子的话（主动搭话）就当背景放进 system。"""
    system, contents = [], []
    for m in messages:
        if m["role"] == "system":
            system.append(m["content"])
            continue
        role = "user" if m["role"] == "user" else "model"
        if not contents and role == "model":
            system.append(f"（你刚才对主人说过：{m['content']}）")
        elif contents and contents[-1]["role"] == role:
            contents[-1]["parts"][0]["text"] += "\n" + m["content"]
        else:
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
    return "\n\n".join(system), contents


def standin(messages, json_mode=False, temperature=0.7):
    """返回云端替团子说的话（原文，要 JSON 就是 JSON 文本）；不可用、额度用完或出错时返回 None。"""
    if not standin_available():
        return None
    if not _take_quota("cloud_standin_usage", config.CLOUD_STANDIN_DAILY):
        log.warning("今天云端代答的次数用完了（%d 次）", config.CLOUD_STANDIN_DAILY)
        return None
    system, contents = to_gemini(messages)
    gen = {"maxOutputTokens": config.CLOUD_MAX_TOKENS, "temperature": temperature}
    if json_mode:
        gen["responseMimeType"] = "application/json"
    try:
        text = _post({"systemInstruction": {"parts": [{"text": system}]}, "contents": contents,
                      "generationConfig": gen})
        log.info("云端代答（%s）：%d 字", config.CLOUD_MODEL, len(text))
        return text
    except Exception as e:
        log.warning("云端代答失败：%s", str(e)[:200])
        return None
