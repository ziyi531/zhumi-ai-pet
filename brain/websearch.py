"""联网搜索：团子唯一会“出门”的地方。

只把搜索词发出去（经 ddgs 轮流问几个公开搜索引擎，免注册、免 API key），
不带对话、记忆和主人的任何信息；搜索词里出现主人的称呼也会先抹掉。
单个引擎经常限流或没结果，所以用 backend="auto" 让它自动换下一个，失败再重试一次。
"""
import logging
import re
import time

import config

log = logging.getLogger("pet.search")

NAME_KEYS = ("称呼", "名字", "姓名", "昵称")


def clean_query(query: str, profile=(), max_chars=None) -> str:
    """要发出去的词（搜索词、问云端的问题）清洗：去掉主人的称呼（隐私），压缩空白，限长。"""
    q = str(query or "")
    for p in profile:
        if any(k in p.get("key", "") for k in NAME_KEYS) and len(p.get("value", "")) >= 2:
            q = q.replace(p["value"], " ")
    q = " ".join(q.replace("「", " ").replace("」", " ").split())
    return q[: max_chars or config.SEARCH_MAX_QUERY_CHARS]


def search(query: str):
    """返回 [{title, body, href}]；没结果或网络不通返回 []，不抛异常。

    auto 每次挑的引擎不一样，偶尔只回一两条、关键信息不在里面：少于 3 条就再搜一次，合并去重。
    """
    from ddgs import DDGS  # 延迟导入：没装时其它功能照常

    found, seen = [], set()
    t = time.perf_counter()
    for attempt in range(2):
        try:
            results = DDGS(timeout=config.SEARCH_TIMEOUT_S).text(
                query, region="cn-zh", safesearch="moderate", max_results=config.SEARCH_RESULTS, backend="auto")
        except Exception as e:
            log.warning("搜索「%s」第 %d 次失败：%s", query, attempt + 1, e)
            results = []
        for r in results:
            key = r.get("href") or r.get("title")
            if key and key not in seen:
                seen.add(key)
                found.append({"title": r.get("title", ""), "body": r.get("body", ""), "href": r.get("href", "")})
        if len(found) >= 3:
            break
    log.info("搜索「%s」：%d 条（%.1f 秒）", query, len(found), time.perf_counter() - t)
    return found[: config.SEARCH_RESULTS + 2]


def available() -> bool:
    if not config.WEB_SEARCH:
        return False
    try:
        import ddgs  # noqa: F401  第一次导入要 1~2 秒，所以 warmup() 在启动时先导入
        return True
    except ImportError:
        return False


def warmup():
    if available():
        log.info("可以上网查资料（只发搜索词）")
    elif config.WEB_SEARCH:
        log.warning("没装 ddgs，团子暂时不能上网查资料（pip install ddgs）")


def results_text(results) -> str:
    """给大脑看的资料：标题 + 摘要，不给网址（团子不念网址）。"""
    lines = []
    for i, r in enumerate(results, 1):
        body = re.sub(r"\s+", " ", r["body"]).strip()[:200]
        lines.append(f"{i}. {r['title'].strip()}：{body}")
    return "\n".join(lines)
