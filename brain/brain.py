"""大脑：对话和心跳。数值和决定来自 state / rules，LLM 只负责说话。"""
import json
import logging
import re
import threading
import time
from datetime import timedelta

import clock
import cloud
import config
import db
import memory
import memory_admin
import nightly
import persona
import rules
import state
import recall
import toolbox
import traits
from llm import OllamaClient, parse_json_loose

log = logging.getLogger("pet.brain")

client = OllamaClient()
BACKGROUND_MEMORY = True  # 测试时设成 False，让记忆提炼同步完成
TOPIC_REASK_DAYS = 3

_tick_lock = threading.Lock()


def _dt(key):
    return clock.parse(db.get_kv(key))


def current_state(now=None):
    now = now or clock.now()
    return state.compute(now, _dt("last_interaction"), _dt("last_pat"), _dt("last_proactive"))


def gate(now):
    return rules.Gate(_dt("last_proactive"), int(db.get_kv("ignored_streak", "0")),
                      db.count_events_on(now.strftime("%Y-%m-%d")))


def event_cutoff(now):
    return clock.fmt(now - timedelta(hours=config.EVENT_TTL_HOURS))


def voice_for(st):
    return "sleepy" if st.sleepy_talk else "normal"


# ---------- 对话 ----------
def chat(text: str, tone: str = "", on_tool=None):
    """on_tool(工具名, 细节)：要用工具时先回调一下（设备显示“在网上查…”“在问云端…”）。"""
    now = clock.now()
    ts = clock.fmt(now)
    history = db.recent_messages(config.HISTORY_TURNS)
    st = current_state(now)  # 说话前的状态：孤单值反映它有多想你
    if memory_admin.is_forget_request(text):
        return _forget(text, now, history, st)
    user_id = db.add_message("user", text, ts)
    db.set_kv("last_interaction", ts)
    db.set_kv("ignored_streak", 0)

    profile = db.get_profile()
    standin = use_standin()
    native = config.TOOL_MODE == "native"
    tools = toolbox.available() if native else [t for t in toolbox.available() if t == "web_search"]
    quiet = standin or (bool(tools) and not wants_tools(text))
    if quiet:  # 主人只是在说自己的事 / 闲聊：工具连给都不给（9B 偶尔手痒，光靠提示词压不住）；云端代答也不给
        tools = []
    # 这句话勾起的往事（太短的话带上团子上一句，不然“是啊”“然后呢”什么也勾不起来）
    cue = text if len(text) >= 6 or not history else f"{history[-1]['text']} {text}"
    memories = recall.relevant(cue, now, vectors=not standin)  # 向量模型也在 Ollama 里，代答时只按词找
    tr = memory.brain_traits_event("chat", now)  # 聊天本身就让关系近一点
    if standin:
        history = history[-config.CLOUD_STANDIN_HISTORY:]
    msgs = persona.chat_messages(st, now, profile, history, text, tone, tools, native, quiet,
                                 memories, traits.describe(tr["values"]), db.get_kv("self_view", ""))
    farewell = ends_conversation(text)
    if farewell:
        msgs = msgs + [{"role": "system", "content": FAREWELL_NOTE}]
    if ASKS_TIME.search(text):
        msgs = msgs + [{"role": "system", "content": persona.time_note(now)}]
    if standin:
        reply, face, ok, used = _generate_standin(msgs, st)
    else:
        reply, face, ok, used = _generate_chat(msgs, st, profile, tools, on_tool, native)
        if not ok and cloud.standin_available():
            # Ollama 开着、本地模型却出错了（实测过：显卡出状况时加载模型报 CUDA 错误，返回 500）：这一句也让云端代答
            log.warning("本地大脑出错，这一句改由云端代答")
            standin = True
            msgs = persona.chat_messages(st, now, profile, history[-config.CLOUD_STANDIN_HISTORY:], text, tone, (),
                                         native, True, memories, traits.describe(tr["values"]),
                                         db.get_kv("self_view", ""))
            if farewell:
                msgs = msgs + [{"role": "system", "content": FAREWELL_NOTE}]
            reply, face, ok, used = _generate_standin(msgs, st)
    if not ASKS_OLD_NAME.search(text) and not TALKS_OLD_NAME.search(reply):
        # 改名后它还会自称旧名字；主人问起、或者它自己在讲以前的名字时不换
        # （“以前那个名字团子就不提啦”换成“猪咪”就说反了，真机踩过）
        reply = persona.new_name(reply)
    pet_id = db.add_message("pet", reply, ts)

    known = [m["text"] for m in memories]
    if ok and standin:  # 代答时提炼不了（存记忆要算向量，本地向量模型也没开）：记下来，本地大脑回来再补
        memory.defer(user_id, pet_id, now, known)
    elif ok:
        reminded = any(u["tool"] == "set_reminder" for u in used)

        def remember():  # 定了提醒的这句不再记成回访（不然到点提醒完，过一天又“回访”一遍）
            memory.extract(client, text, reply, now, known, followups=not reminded)
            memory.catch_up(client, now)  # 本地大脑回来了：顺带补记云端代答时攒下的对话
        _background(remember)

    result = {"text": reply, "face": face, "vibe": "purr" if face == "happy" else "tap",
              "voice": voice_for(st), "state": current_state(now).to_dict()}
    if memories:
        result["recalled"] = [m["text"] for m in memories]  # 调试面板看它想起了什么
    if used:
        result["tools"] = used  # [{"tool": 名字, "detail": 查了什么}]
        searched = [u["detail"] for u in used if u["tool"] == "web_search"]
        if searched:
            result["searched"] = searched
    if standin and ok:
        result["brain"] = "cloud"  # 设备上注明“云端代答”：这次的对话离开过这台电脑
    if farewell:
        result["end"] = True  # 主人道别了：设备说完这句就不再接着听
    return result


def use_standin() -> bool:
    """本地大脑（Ollama）没开、又能用云端代答，就让云端替团子说话。"""
    return cloud.standin_available() and not client.is_up(config.LOCAL_PROBE_S)


def _generate_standin(msgs, st):
    """云端照同一套人设、上下文替团子说话；返回值同 _generate_chat。云端也不行，就老实说脑袋没连上。"""
    reply, face = None, None
    for _ in range(2):
        raw = cloud.standin(msgs, json_mode=True)
        if raw is None:
            return persona.fallback_line("offline"), "sleep", False, []
        data = parse_json_loose(raw)
        data = data if isinstance(data, dict) else _salvage(raw)
        reply, face = persona.clean_reply(str(data.get("reply") or "")), data.get("face")
        if reply and not persona.off_character(reply):
            break
        reply = None
        msgs = msgs + [{"role": "system", "content": persona.STAY_IN_CHARACTER}]
    if st.sleepy_talk:
        face = "sleepy"
    elif face not in persona.CHAT_FACES:
        face = "happy"
    return reply or persona.fallback_line("in_character"), face, True, []


# 主人是在提问或请团子办事，才把工具交给大脑。“我中午吃了火锅”“我升职了！”这种陈述，连工具都看不到。
ASKING = re.compile(r"[?？]|吗|呢|么|啥|什么|几|多少|多久|多大|多远|怎么|怎样|如何|为什么|为啥|哪|谁|是不是|有没有"
                    r"|帮我|帮忙|给我|告诉我|查|搜|算|讲讲|说说|介绍|推荐|解释|翻译|天气|带伞|(.)不\1"
                    r"|提醒|叫我|喊我|闹钟|定时|计时")


def wants_tools(text: str) -> bool:
    return bool(ASKING.search(text or ""))


# 主人在道别：这句回完，设备就不再接着听（连续对话结束，回去等唤醒词）。只认明确的说法，
# “好了”“没事”这种太宽的不算（“我病好了”“没事就好”）
FAREWELL = re.compile(r"拜拜|再见|晚安|bye|不聊了|不说了|下次(再)?聊|回头(再)?聊|改天(再)?聊|先这样|就聊到这|聊到这|"
                      r"我(先|要)?去(忙|睡|洗澡|吃饭|上班|上课|工作|学习|写作业)|我(先|要)?睡(了|觉)|你(先)?(去玩|休息)吧", re.I)
FAREWELL_NOTE = "主人要去忙别的了（或者要睡了），这段聊天到这里结束：温柔地道个别，一两句就好，不要再问问题。"


# 主人在问现在几点、今天几号、星期几
ASKS_TIME = re.compile(r"几点|什么时间|啥时间|多少点|几号|星期几|礼拜几|周几|什么时候了|今天.{0,2}日期")


# 主人在问它以前的名字：这时回复里的旧名字是在回答问题，不替换
ASKS_OLD_NAME = re.compile("以前|原来|之前|从前|旧名|改名|改过名|" + "|".join(map(re.escape, config.PET_OLD_NAMES or ("$^",))))
TALKS_OLD_NAME = re.compile("以前|原来|之前|从前|旧名|改名|改过名|名字")


# 识别出来的不是话：只有标点，或者只是零碎的一个英文词（“The.”“Yeah.”——没人说话时环境声常被认成这样）。
# 这种当“没听清”，设备就结束这段连续对话，不去回应一个杂音（真机踩过：对着“The.”聊了好几轮）
def meaningful(heard: str) -> str:
    text = (heard or "").strip()
    core = re.sub(r"[\s\W_]+", "", text)
    if not core:
        return ""
    if not re.search(r"[一-鿿]", core) and len(text.split()) <= 1:
        return ""
    return text


def ends_conversation(text: str) -> bool:
    return bool(FAREWELL.search(text or ""))


def _background(fn):
    """记忆相关的慢活放后台做（要再问一次大脑），自测时同步做。"""
    if BACKGROUND_MEMORY:
        threading.Thread(target=fn, daemon=True).start()
    else:
        fn()


FORMAT_AGAIN ='现在不能再查了，只能输出 {"reply": "你要对主人说的话", "face": "表情"}。'
FORGOT_NOTE = ("主人刚让你忘掉一件事，你已经把它从记忆里删掉了。乖乖答应主人，一两句话就好；"
               "不要提那件事的内容，不要追问，也别说“我记住了要忘掉”这种绕口令。")


def _forget(text, now, history, st):
    """主人说“忘掉这件事”：真删相关记忆和原话，团子乖乖答应。这一轮不提炼新记忆，原话也不存。"""
    rep = memory_admin.forget(text, now, history)
    ts = clock.fmt(now)
    log.info("按主人要求忘掉：%d 条记忆、%d 个回访、%d 句原话", len(rep["memories"]), len(rep["followups"]),
             rep["messages"])
    db.add_message("user", f"（主人让{config.PET_NAME}忘掉了一件事）", ts)
    db.set_kv("last_interaction", ts)
    history = db.recent_messages(config.HISTORY_TURNS)[:-1]  # 删过的原话已经不在了
    tr = memory.traits_now(now)
    msgs = persona.chat_messages(st, now, db.get_profile(), history, "把那件事忘掉吧。",
                                 traits_text=traits.describe(tr["values"]), self_view=db.get_kv("self_view", ""))
    msgs = msgs + [{"role": "system", "content": FORGOT_NOTE}]
    reply, face, _, _ = _generate_standin(msgs, st) if use_standin() else _generate_chat(msgs, st)
    db.add_message("pet", reply, ts)
    return {"text": reply, "face": face, "vibe": "tap", "voice": voice_for(st), "state": current_state(now).to_dict(),
            "forgot": {"memories": len(rep["memories"]), "followups": len(rep["followups"]),
                       "messages": rep["messages"]}}


# 嘴上说要查/要去问、手上没调工具（9B 偶尔这样）：提醒一次，让它真的去做
SAYS_WILL = {
    "web_search": (re.compile(r"查一?查|去查|查一下|帮你查|搜一?搜|搜一下|上网看看"),
                   "你刚才说要去查，那就现在调用 web_search 查一下，查到再回答主人。"),
    "set_reminder": (re.compile(r"(提醒|叫|喊)(一下)?(主人|你)|定好|闹钟"),
                     "你刚才答应到点提醒主人，那就现在调用 set_reminder 定好；光嘴上答应，到时候你是不会自己想起来的。"),
    "ask_cloud": (re.compile(r"云端|去问问|问一问|问一下别|请教|聪明的大家伙"),
                  "你刚才说要去请教，那就现在调用 ask_cloud 问一下，拿到答案再用你的口吻讲给主人听。"),
}


def _ask(msgs, open_tools, native, temperature):
    """问一次大脑。返回 (回复 dict, 想调的工具 (名字, 参数) 或 None, 要记进对话的 assistant 消息)。

    native：带上 open_tools 里的工具，模型自己发调用；没有可用工具时就不给工具、要求 JSON。
    json：模型在 JSON 里写 {"search": "词"}（只有上网搜索）。
    """
    if native and open_tools:
        msg = client.chat_tools(msgs, [toolbox.SPECS[n] for n in open_tools], temperature)
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            if fn.get("name") in open_tools:
                call_msg = {"role": "assistant", "content": msg.get("content", ""), "tool_calls": [call]}
                return {}, (fn["name"], fn.get("arguments") or {}), call_msg
        text = msg.get("content", "")
        data = parse_json_loose(text)
        return (data if isinstance(data, dict) else _salvage(text) if text else {}), None, None
    data = client.chat_json(msgs, temperature=temperature)
    data = data if isinstance(data, dict) else {}
    if data.get("search") and not data.get("reply"):
        query = data["search"]
        call = ("web_search", {"query": query}) if "web_search" in open_tools else ("web_search", {})
        return {}, call, {"role": "assistant", "content": json.dumps({"search": query}, ensure_ascii=False)}
    return data, None, None


REPLY_RE = re.compile(r'"reply"\s*:\s*["“]?(.*?)["”]?\s*(?:,\s*"face"|\}\s*$|$)', re.S)
FACE_RE = re.compile(r'"face"\s*:\s*["“]?(\w+)')


def _salvage(text: str) -> dict:
    """没约束解码时，模型偶尔把 JSON 写坏（比如用了中文引号 “）：把 reply 和 face 抠出来，别把半截 JSON 念给主人。"""
    if '"reply"' not in text:
        return {"reply": text}
    m, f = REPLY_RE.search(text), FACE_RE.search(text)
    return {"reply": m.group(1).strip() if m else text, "face": f.group(1) if f else None}


# 工具参数里会发出去的那一项：记进对话时换成清洗过的（和工具返回对得上）
SENT_ARG = {"web_search": "query", "ask_cloud": "question"}


def _generate_chat(msgs, st, profile=(), tools=(), on_tool=None, native=False):
    """返回 (回复, 表情, LLM 是否可用, 用过的工具 [{"tool", "detail"}])。

    工具调用：大脑想用哪个工具就调（toolbox.py），把结果作为工具返回交回去再问；每轮最多 MAX_TOOL_CALLS 次，
    每个工具还有自己的上限（toolbox.limit）。出戏或格式不对会重试一次。
    """
    reply, face, used, retries, nudged = None, None, [], 0, False
    while True:
        t = time.perf_counter()
        count = lambda n: sum(u["tool"] == n for u in used)  # noqa: E731
        open_tools = ([n for n in tools if count(n) < toolbox.limit(n)]
                      if len(used) < config.MAX_TOOL_CALLS else [])
        try:
            data, call, call_msg = _ask(msgs, open_tools, native, 0.5 if used else 0.7)
        except Exception as e:
            log.warning("对话失败：%s", e)
            kind = "stuck" if "timed out" in str(e).lower() or "timeout" in type(e).__name__.lower() else "offline"
            return persona.fallback_line(kind), "sleep", False, _public(used)
        log.debug("大脑想了 %.1f 秒：%s", time.perf_counter() - t, call and f"用 {call}" or data)
        if call is not None:
            name, args = call
            if name in open_tools and not any(u["tool"] == name and u.get("args") == args for u in used):
                if on_tool:
                    on_tool(name, toolbox.describe(name, args, profile))  # 设备先显示“在查…”，再去跑工具
                res = toolbox.run(name, args, profile)
                used.append({"tool": name, "detail": res.detail, "args": args})
                if name in SENT_ARG:  # 对话里记下实际发出去的（清洗过的），和工具返回对得上
                    if "tool_calls" in call_msg:
                        c = call_msg["tool_calls"][0]
                        call_msg["tool_calls"] = [{**c, "function": {**c["function"],
                                                                     "arguments": {SENT_ARG[name]: res.detail}}}]
                    else:
                        call_msg = {"role": "assistant", "content": json.dumps({"search": res.detail}, ensure_ascii=False)}
                more = count(name) < toolbox.limit(name)
                msgs = msgs + [call_msg, *persona.tool_result_messages(name, res.detail, res.text, res.ok, more, native)]
                continue
            hint = FORMAT_AGAIN  # 用够了、不能用，或者重复调同一个
        elif data.get("reply"):
            reply = persona.clean_reply(str(data["reply"]), config.SEARCH_REPLY_CHARS if used else None)
            face = data.get("face")
            promised = next((n for n, (pat, _) in SAYS_WILL.items()
                             if n in open_tools and pat.search(reply or "")), None)
            if reply and promised and not used and not nudged:
                nudged = True
                log.info("嘴上说要用 %s 却没用，提醒一次：%s", promised, reply)
                msgs = msgs + [{"role": "assistant", "content": json.dumps(data, ensure_ascii=False)},
                               {"role": "system", "content": SAYS_WILL[promised][1]}]
                continue
            if reply and not persona.off_character(reply, topic_ok=bool(used)):
                break
            hint = persona.STAY_IN_CHARACTER
        else:
            hint = FORMAT_AGAIN
        reply = None
        retries += 1
        if retries >= 2:
            break
        msgs = msgs + [{"role": "system", "content": hint}]
    if not reply:
        reply = persona.fallback_line("in_character")
    if st.sleepy_talk:
        face = "sleepy"
    elif face not in persona.CHAT_FACES:
        face = "happy"
    return reply, face, True, _public(used)


def _public(used):
    """给设备看的工具记录：只有工具名和清洗过的细节，不带模型原始参数（里面可能有主人的称呼）。"""
    return [{"tool": u["tool"], "detail": u["detail"]} for u in used]


def record_device_event(kind: str, count: int = 1):
    """设备上报本地发生的事（比如被摸了几次）。摸头算一种回应：会降低孤单值、不算“被无视”。"""
    ts = clock.fmt(clock.now())
    db.add_device_log(kind, count, ts)
    if kind == "pat" and count > 0:
        db.set_kv("last_pat", ts)
        memory.brain_traits_event("pat", clock.now(), count)  # 常被摸头，会越来越亲、越来越黏


def take_events():
    """取出待推送的主动事件（过时太久的自动作废）。"""
    now = clock.now()
    return db.take_pending_events(clock.fmt(now), event_cutoff(now))


# ---------- 定时提醒 ----------
def fire_reminders():
    """到点的提醒变成主动事件（pusher 每秒调一次，马上推给设备出声）。
    不受免打扰、冷却和每日次数限制：是主人自己让它到点说的。大脑关着时错过太久的就不补了。"""
    now = clock.now()
    ts = clock.fmt(now)
    fired = []
    for r in db.due_reminders(ts):
        if now - clock.parse(r["due_at"]) > timedelta(hours=config.REMINDER_STALE_HOURS):
            db.close_reminder(r["id"], "expired", ts)
            continue
        text = persona.reminder_line(r["text"])
        db.add_event("reminder", text, "peek", "nudge", "normal", ts)
        db.add_message("pet", text, ts, proactive=True)
        db.close_reminder(r["id"], "done", ts)
        log.info("到点提醒：%s", text)
        fired.append(text)
    return fired


# ---------- 心跳 ----------
def tick():
    with _tick_lock:
        return _tick()


def _tick():
    now = clock.now()
    ts = clock.fmt(now)

    # 说话或摸头都算回应过
    responses = [t for t in (_dt("last_interaction"), _dt("last_pat")) if t is not None]
    streak, counted = rules.update_ignored(
        now, _dt("last_proactive"), max(responses, default=None), _dt("ignore_counted_for"),
        int(db.get_kv("ignored_streak", "0")))
    db.set_kv("ignored_streak", streak)
    if counted:
        db.set_kv("ignore_counted_for", clock.fmt(counted))
        memory.brain_traits_event("ignored", now)  # 主动找主人没被理：更黏、更蔫一点
    tr = memory.traits_now(now)  # 顺带结算跨天：会回落的回落、没见面的生分一点

    due = []
    for f in db.pending_followups():
        due_at = clock.parse(f["due_at"])
        if due_at > now:
            continue
        if now - due_at > timedelta(hours=config.FOLLOWUP_STALE_HOURS):
            db.close_followup(f["id"], "expired", ts)
        else:
            due.append(f)

    st = current_state(now)
    stale = clock.fmt(now - timedelta(days=TOPIC_REASK_DAYS))
    topics = [p for p in db.get_profile() if not p["asked_at"] or p["asked_at"] < stale]
    decision, reason = rules.decide(st, now, gate(now), due, topics)

    result = {"time": ts, "state": st.to_dict(), "fired": decision is not None, "reason": reason}
    if decision:
        text = _compose(decision, st, now, tr)
        db.add_event(decision.kind, text, decision.face, decision.vibe, voice_for(st), ts)
        db.add_message("pet", text, ts, proactive=True)
        db.set_kv("last_proactive", ts)
        if decision.followup:
            db.close_followup(decision.followup["id"], "done", ts)
        if decision.topic:
            db.mark_profile_asked(decision.topic["key"], ts)
        result.update(kind=decision.kind, text=text, detail=decision.reason)
        log.info("主动搭话（%s）：%s", decision.reason, text)
    if memory.backlog_size() and client.is_up(config.LOCAL_PROBE_S):  # 本地大脑回来了，主人还没开口也先补记
        _background(lambda: memory.catch_up(client, now))
    night = nightly.maybe_run(client, now)  # 睡觉时段、今晚还没整理过：合并、淡忘、写日记、修订自我认识
    if night:
        result["nightly"] = {k: night[k] for k in ("day", "diary", "highlight", "self_view")}
        result["nightly"].update(merged=len(night["merged"]), faded=len(night["faded"]))
    db.set_kv("last_tick", json.dumps(result, ensure_ascii=False))
    return result


def run_nightly():
    """调试用：不管现在几点，马上把今天整理一遍。"""
    with _tick_lock:
        now = clock.now()
        return nightly.run(client, now, now.strftime("%Y-%m-%d"))


def _compose(decision, st, now, tr=None):
    # 回访时顺带想起和这件事相关的往事（比如“考研”勾起“十二月考试”）
    memories = recall.relevant(decision.followup["fact"], now) if decision.followup else []
    traits_text = traits.describe(tr["values"]) if tr else ""
    msgs = persona.proactive_messages(decision, st, now, db.get_profile(), db.recent_messages(4),
                                      memories, traits_text, db.get_kv("self_view", ""))
    try:
        text = persona.clean_reply(client.chat(msgs, temperature=0.9))
    except Exception as e:
        log.warning("主动措辞失败：%s", e)
        text = persona.clean_reply(cloud.standin(msgs, temperature=0.9) or "")  # 本地大脑没开：云端代为措辞
    if not text or persona.off_character(text):
        text = persona.fallback_line(decision.kind)
    return persona.new_name(text)


# ---------- 调试面板 ----------
def status():
    now = clock.now()
    st = current_state(now)
    g = gate(now)
    cd = rules.cooldown_hours(g.ignored_streak)
    since = state.hours_between(g.last_proactive, now)
    last_tick = db.get_kv("last_tick")
    return {
        "time": clock.fmt(now),
        "state": st.to_dict(),
        "gate": {
            "proactive_today": g.proactive_today,
            "daily_limit": config.DAILY_PROACTIVE_LIMIT,
            "ignored_streak": g.ignored_streak,
            "cooldown_hours": cd,
            "cooldown_left": round(max(0.0, cd - since), 2) if since is not None else 0.0,
            "quiet": state.in_window(now, config.QUIET_HOURS),
            "last_proactive": db.get_kv("last_proactive"),
        },
        "followups": db.pending_followups(),
        "reminders": db.pending_reminders(),
        "profile": db.get_profile(),
        "traits": memory.traits_now(now)["values"],
        "self_view": db.get_kv("self_view", ""),
        "memories": db.active_memories(with_embedding=False)[-30:],
        "diary": db.recent_diary(3),
        "messages": db.recent_messages(12),
        "last_tick": json.loads(last_tick) if last_tick else None,
    }
