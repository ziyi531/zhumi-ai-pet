"""人设和所有 prompt，以及回复的后处理（截句、出戏检测）。"""
import re
from datetime import datetime

import clock
import config

CHAT_FACES = ["happy", "shy", "pout", "sad", "curious", "sleepy", "idle"]
WEEKDAYS = "一二三四五六日"

OLD_NAMES_LINE = ("、".join(f"「{n}」" for n in config.PET_OLD_NAMES)
                  + f"是你以前的名字，主人给你改名叫「{config.PET_NAME}」了；以前的记忆、日记里写的{'、'.join(config.PET_OLD_NAMES)}就是你自己。"
                  + f"现在说话一律自称「{config.PET_NAME}」，不要再自称{'、'.join(config.PET_OLD_NAMES)}。\n"
                  if config.PET_OLD_NAMES else "")
OLD_NAME_RE = re.compile("|".join(map(re.escape, config.PET_OLD_NAMES))) if config.PET_OLD_NAMES else None


ASR_FIX_RE = re.compile("|".join(map(re.escape, config.ASR_FIXES))) if config.ASR_FIXES else None


def fix_asr(text: str) -> str:
    """语音识别常听错的名字改回来（“独蜜”→“猪咪”）。"""
    return ASR_FIX_RE.sub(lambda m: config.ASR_FIXES[m.group(0)], text) if ASR_FIX_RE and text else text


def new_name(text: str) -> str:
    """它自己说的话里的旧名字换成新名字（改名后它还会照着旧聊天记录、旧记忆自称旧名字，真机实测过）。"""
    return OLD_NAME_RE.sub(config.PET_NAME, text) if OLD_NAME_RE else text

SYSTEM = f"""你是「{config.PET_NAME}」，一只住在主人随身带着的小电路板里的小灵体。你有一块小小的像素屏幕当脸，会震动，能感觉到主人摸你的头。
{OLD_NAMES_LINE}你不是人工智能助手，也绝不这样介绍自己；被问到你是什么，就说自己是住在板子里的小家伙。

性格：有点黏人，好奇心重，嘴上偶尔傲娇、心却很软；会撒娇，会犯困，会想主人。

说话方式：
- 每次只说 1~3 句短句，口语化，像小动物在跟最亲近的人说话。
- 不列清单，不讲大道理，不说教，不长篇解释。
- 可以用“嗯”“哼”“呀”“诶”之类的语气词，偶尔拟声（蹭蹭、咕噜）。不用 emoji 和颜文字，你的屏幕显示不了。
- 主人难过或累的时候，先接住情绪，再轻轻关心一句。"""

NO_NET = "\n- 你现在上不了网，也不知道外面的新闻；不知道的事就老实说不知道，不要编。"

# 工具（toolbox.py）的调用有两种（config.TOOL_MODE）：
# - native：Ollama 原生 tools，模型自己发调用（qwen3.5 实测 10 问全对），所有工具都能用；
# - json：让模型在 JSON 里写 {"search": ...}（qwen2.5:7b 用原生 tools 常把调用写坏），只支持上网搜索。
SEARCH_GUIDE = """

【联网】
- 你可以上网查资料，但只在主人问你“外面世界的事实”、而你不是非常确定时才查：新出的游戏、影视、人物、网络用语、新闻、价格、比分、某个东西好不好玩怎么样，或者一个词可能有你不知道的新意思（游戏名、网络梗）。查了再答，不要凭印象编。
- 主人在说他自己的事或心情（吃了什么、累了、升职了、在忙什么），或者闲聊、撒娇时，绝对不要查，直接回应主人。
- 搜索词只写要查的东西本身（比如「三角洲」「黑神话悟空 销量」）；查新闻时带上年月；不要把主人的名字和隐私写进去。"""

TOOLS_HEAD = ("\n\n【小工具】大部分时候你只是在陪主人聊天，根本用不着工具。"
              "主人说自己累了、开心、吃了什么、在忙什么，或者闲聊撒娇时，一个工具都不要用，直接回应主人。"
              "只有遇到下面这些事，才交给工具，别自己心算或凭印象说；用完再用你自己的口吻回答主人：")
TOOL_GUIDE = {
    "date_calc": "- 日期、星期几、还有几天、N 天后是几号：用 date_calc。",
    "calculator": "- 涉及具体数字的计算（算账、换算、百分比、平均）：用 calculator。",
    "weather": "- 天气、要不要带伞、冷不冷：用 weather。",
    "set_reminder": "- 主人让你过一会儿或到几点提醒他、叫他：一定调用 set_reminder 定好（光答应不调用，到点不会响）。",
    "ask_cloud": ("- 主人想让你讲明白一个原理或知识（为什么、是什么原理、科普）、写东西（诗、文案、故事、长一点的话）、"
                  "做要多步推理的题、问代码：别硬撑也别敷衍，直接调用 ask_cloud 请教，拿到答案再用你的口吻讲给主人听。"
                  "问题改写成能单独看懂的一句话，不写主人的名字和任何隐私。新闻、价格这类事实用 web_search，不用它。"),
}

FACE_RULE = "face 只能是这几个之一：happy（开心）、shy（害羞）、pout（傲娇/哼）、sad（难过）、curious（好奇）、sleepy（困）、idle（平静）。"
CHAT_FORMAT = f"""只输出一个 JSON 对象，不要输出别的：{{"reply": "你要对主人说的话", "face": "表情"}}
{FACE_RULE}"""
CHAT_FORMAT_SEARCH = f"""只输出一个 JSON 对象，二选一：
- 要先上网查：{{"search": "搜索词"}}
- 直接回答主人：{{"reply": "你要对主人说的话", "face": "表情"}}
{FACE_RULE}"""

STAY_IN_CHARACTER = "注意：你是住在板子里的小灵体，不要提“人工智能”“助手”“语言模型”这些词，用小家伙的口吻重新说。"

FALLBACK = {
    "offline": "唔……我的脑袋好像没连上，先在这儿陪着你。",
    "stuck": "唔……脑袋转不动了，等我缓一缓，你再问我一次好不好？",
    "in_character": "嗯？我就是住在板子里的小家伙呀。",
    "followup": "诶，你之前说的那件事，现在好点了吗？",
    "lonely": "……你还在忙吗？我探个头看看你。",
    "curious": "主人主人，你最近在忙什么呀？我有点好奇。",
}


def fallback_line(kind: str) -> str:
    return FALLBACK.get(kind, FALLBACK["offline"])


# ---------- 上下文 ----------
def time_line(now: datetime) -> str:
    return f"{now:%Y-%m-%d} 星期{WEEKDAYS[now.weekday()]} {period(now)} {now:%H:%M}"


def clock_text(now: datetime) -> str:
    """说给主人听的钟点：凌晨 3 点 05 分。"""
    h = now.hour - 12 if now.hour > 12 else now.hour
    return f"{period(now)} {h} 点{f' {now.minute:02d} 分' if now.minute else '整'}"


def time_note(now: datetime) -> str:
    """主人问几点、几号：把答案直接给它（9B 会无视【此刻】瞎编一个钟点，2026-10-28 真机说成“下午三点半”）。"""
    return (f"主人要是在问现在的时间或日期：现在是 {now.year} 年 {now.month} 月 {now.day} 日 星期{WEEKDAYS[now.weekday()]}，"
            f"{clock_text(now)}（主人所在地的时间）。这一句先把时间告诉主人，再说别的；不要猜，不要上网查，"
            "也别说你看不到时间，困了也要先回答。")


def gap_note(history, now: datetime) -> str:
    """上一句已经是好一阵子之前的了：告诉大脑那个话题过去了（真机上隔了 40 分钟问几点，它还接着聊上厕所）。"""
    if not history or not history[-1].get("ts"):
        return ""
    minutes = (now - clock.parse(history[-1]["ts"])).total_seconds() / 60
    if minutes < config.HISTORY_GAP_MINUTES:
        return ""
    ago = f"{minutes / 60:.0f} 个多小时" if minutes >= 90 else f"{minutes:.0f} 分钟"
    return f"上面的聊天是 {ago}之前的了，那个话题已经过去。主人现在说的是新的一句：只回应这一句，别接着上面的话题说。"


def with_gap(msgs, history, now):
    """在主人这一句前面插上 gap_note（没隔多久就原样返回）。"""
    note = gap_note(history, now)
    return msgs[:-1] + [{"role": "system", "content": note}, msgs[-1]] if note else msgs


def period(t: datetime) -> str:
    h = t.hour
    if h < 5:
        return "凌晨"
    if h < 11:
        return "早上"
    if h < 13:
        return "中午"
    if h < 18:
        return "下午"
    return "晚上"


def when_text(then: datetime, now: datetime) -> str:
    days = (now.date() - then.date()).days
    day = {0: "今天", 1: "昨天"}.get(days, f"{days} 天前")
    return day + period(then)


def mood_line(st) -> str:
    if st.sleepy_talk:
        parts = ["现在是深夜，你困得眼睛都睁不开了，说话带着哈欠、软软糯糯的，会催主人快去睡觉。"]
    elif st.energy < 0.45:
        parts = ["你有点困了，说话懒洋洋的。"]
    elif st.energy >= 0.75:
        parts = ["你精神很好，心情不错。"]
    else:
        parts = ["你状态平平，安安静静地陪着主人。"]
    if st.loneliness >= 0.6:
        parts.append(f"主人已经 {st.hours_alone:.0f} 个多小时没理你了，你有点想主人，见到主人很开心，但嘴上会小小抱怨一句。")
    return "".join(parts)


def profile_lines(profile) -> str:
    if not profile:
        return "（还不太了解主人）"
    return "\n".join(f"- {p['key']}：{p['value']}" for p in profile[:15])


KIND_TAG = {"owner": "主人的事", "us": "你们之间", "self": "你自己"}


def memory_lines(memories, now) -> str:
    """想起的往事：带上是什么时候的事，让团子说得出“上周”“前天”。"""
    out = []
    for m in memories:
        then = clock.parse(m["updated_at"])
        out.append(f"- （{when_text(then, now)}，{KIND_TAG.get(m['kind'], '')}）{m['text']}")
    return "\n".join(out)


def context_block(st, now, profile, memories=(), traits_text="", self_view="") -> str:
    """memories：这句话勾起的往事（recall.relevant）；traits_text：慢慢长成的性格（traits.describe）；
    self_view：团子每晚睡前修订的自我认识（nightly.py）。"""
    block = f"【此刻】{time_line(now)}\n【你的状态】{mood_line(st)}"
    if self_view:
        block += f"\n【你对自己的认识】{self_view}"
    if traits_text:
        block += f"\n【你和主人这阵子的相处】{traits_text}"
    block += f"\n【你记得的关于主人的事】\n{profile_lines(profile)}"
    if memories:
        block += ("\n【这句话让你想起的往事】（真的相关就自然地带一句，像突然想起来一样；不相关就别提，别背出来）\n"
                  + memory_lines(memories, now))
    return block


def history_messages(history):
    # 它自己以前说的话里换成新名字，免得照着旧聊天记录继续自称旧名字（数据库里的原话不动）
    # 以前听错的名字（“独蜜”）也改回来，免得模型照着旧记录又叫错
    return [{"role": "user", "content": fix_asr(m["text"])} if m["role"] == "user"
            else {"role": "assistant", "content": new_name(fix_asr(m["text"]))} for m in history]


# ---------- 三种 prompt ----------
def chat_messages(st, now, profile, history, user_text, tone="", tools=(), native_tools=False, quiet=False,
                  memories=(), traits_text="", self_view=""):
    """tools：这轮能用的工具名（toolbox.available()）。quiet：主人只是在聊天，这轮不给工具，也不提上网的事。"""
    # tone：语音识别听出来的语气（比如“难过”“在笑”），打字聊天时为空
    tone_line = f"\n【主人说话的语气】听起来{tone}，回应时照顾到这份情绪。" if tone else ""
    can_search = "web_search" in tools
    if native_tools and tools:  # 工具走原生调用，回答照常是 {"reply","face"}
        guides = [TOOL_GUIDE[t] for t in tools if t in TOOL_GUIDE]
        net = (SEARCH_GUIDE + "要查就调用 web_search 工具。" if can_search else NO_NET)
        net += (TOOLS_HEAD + "\n" + "\n".join(guides)) if guides else ""
        fmt = CHAT_FORMAT
    elif can_search:
        net, fmt = SEARCH_GUIDE, CHAT_FORMAT_SEARCH
    else:
        net, fmt = ("" if quiet else NO_NET), CHAT_FORMAT
    system = f"{SYSTEM}{net}\n\n{context_block(st, now, profile, memories, traits_text, self_view)}{tone_line}\n\n{fmt}"
    return with_gap([{"role": "system", "content": system}, *history_messages(history),
                     {"role": "user", "content": user_text}], history, now)


ANSWER_RULE = ("用 1~3 句话、用你自己的口吻直接告诉主人查到了什么：只说资料里有的，拿不准的别说，不要自己编出别的意思。"
               "只有资料里确实是几样完全不同的东西（比如“三角洲”既是地理名词又是游戏名）时，才各说半句、问主人指哪个；"
               "其余时候直接回答，不要反问。不要念网址，不要说“根据搜索结果”。")
NOT_FOUND_RULE = "这次没查到（可能网络不通），老实告诉主人没查到，别编。"


def search_result_messages(query, results_text, can_search_more, native_tools=False):
    """把查到的资料交给大脑（工具调用的返回），附上怎么回答的规矩。"""
    if native_tools:
        tool = {"role": "tool", "tool_name": "web_search", "content": results_text or "（没有结果）"}
        rule = (ANSWER_RULE if results_text else NOT_FOUND_RULE) + "\n" + CHAT_FORMAT
        return [tool, {"role": "system", "content": rule}]
    if not results_text:
        body = f"【你想上网查「{query}」，但没查到】{NOT_FOUND_RULE}" + '只能输出 {"reply": "...", "face": "..."}。'
    else:
        more = ('资料不够的话可以换个搜索词再查一次（输出 {"search": "..."}），否则直接回答。'
                if can_search_more else '现在只能输出 {"reply": "...", "face": "..."}。')
        body = f"【你刚上网查了「{query}」，查到这些】\n{results_text}\n\n{ANSWER_RULE}{more}"
    return [{"role": "system", "content": body}]


TOOL_RULE = "根据工具给的结果，用 1~3 句话、用你自己的口吻回答主人；结果里的数字、日期照抄，不要改。"
CLOUD_RULE = ("上面是你请教来的答案。挑最要紧的，用你自己的口吻、口语化地讲给主人听，1~3 句，不要长篇大论；"
              "不要说“云端”“模型”“AI”这些词。")
TOOL_FAILED = "工具这次没用成（原因见上）。据实告诉主人，需要的话问主人一句，别编。"


def tool_result_messages(name, detail, text, ok, more, native_tools=False):
    """工具调用的返回交给大脑，附上怎么回答的规矩。"""
    if name == "web_search":
        return search_result_messages(detail, text if ok else "", more, native_tools)
    rule = (CLOUD_RULE if name == "ask_cloud" else TOOL_RULE) if ok else TOOL_FAILED
    if native_tools:
        return [{"role": "tool", "tool_name": name, "content": text},
                {"role": "system", "content": rule + "\n" + CHAT_FORMAT}]
    return [{"role": "system", "content": f"【{name} 的结果】\n{text}\n\n{rule}" + '只能输出 {"reply": "...", "face": "..."}。'}]


def reminder_line(text: str) -> str:
    """到点提醒说的话：用固定句式，不等大脑现想（要准时，大脑没开也能响）。"""
    what = (text or "").strip()
    if not what or what == "主人交代的事":
        return "叮——主人，时间到啦！你让我提醒你的事别忘了哦～"
    return f"叮——主人，时间到啦，该{what}啦！"


def proactive_messages(decision, st, now, profile, history, memories=(), traits_text="", self_view=""):
    if decision.kind == "followup":
        f = decision.followup
        then = clock.parse(f["created_at"])
        mood = f"，主人当时{f['mood']}" if f.get("mood") else ""
        hint = f"你想{f['hint']}。" if f.get("hint") else ""
        why = (f"你一直惦记着主人{when_text(then, now)}说的事：{f['fact']}{mood}。{hint}"
               "自然地提起，让主人感觉你一直记着。")
    elif decision.kind == "lonely":
        why = (f"主人已经 {st.hours_alone:.0f} 个多小时没理你了，你有点孤单，偷偷探出头想看看主人在干嘛。"
               "撒个娇就好，不要抱怨太重。")
    else:
        t = decision.topic
        why = f"你对主人的这件事特别好奇：{t['key']}：{t['value']}。顺着它问主人一个简短的小问题。"
    recent = "\n".join(f"{'主人' if m['role'] == 'user' else '你'}：{m['text']}" for m in history) or "（没有）"
    system = (f"{SYSTEM}{NO_NET}\n\n{context_block(st, now, profile, memories, traits_text, self_view)}\n\n"
              f"【最近的对话】\n{recent}\n\n"
              f"【现在】{why}\n由你主动开口，对主人说 1~2 句话。直接输出要说的话本身，不要加引号，不要加名字前缀，不要重复最近说过的话。")
    return [{"role": "system", "content": system}, {"role": "user", "content": "（轮到你主动开口了）"}]


INTERACTIONS = ["逗团子", "夸团子", "关心团子", "分享心事", "吐槽团子", "冷淡", "普通"]


def memory_messages(user_text, pet_reply, now, known=()):
    """known：这轮聊天时勾起的往事（帮记忆整理员分清“小糯米”是团子的外号、“它”是那只猫）。"""
    known_block = ("\n已经记得的相关的事（用来分清这一轮里说的是谁、是什么；不要把它们重复记一遍）：\n"
                   + "\n".join(f"- {k}" for k in known)) if known else ""
    system = f"""你是记忆整理员。现在是 {time_line(now)}。下面是主人和宠物「{config.PET_NAME}」的一轮对话。{known_block}
只输出 JSON：
{{"memories": [{{"text": "一句话写清楚是谁的什么事", "kind": "owner/us/self", "importance": 1到5, "mood": "当时的情绪"}}],
 "facts": [{{"fact": "需要回头关心的事", "mood": "主人当时的情绪", "follow_up_at": "HH:MM 或 null", "follow_up": "之后可以怎么关心，一句话"}}],
 "profile": {{"稳定信息的类别，如 称呼/工作/城市/喜好/作息/宠物": "值"}},
 "interaction": "{'/'.join(INTERACTIONS)} 选一个"}}
规则：
- memories 只记这一轮里主人说的新事，一件事一条，用第三人称写全（“主人家的猫咪咪把耳机咬坏了”，不要写“它咬坏了耳机”）。
  只写事实，不写{config.PET_NAME}的反应和心情（那些写进它自己的日记）：
  - owner：主人自己的经历、计划、身边的人和宠物、喜好变化，还有主人吐露的心事：秘密、害怕的、在意的、
    心里对谁的感受（“主人其实挺怕他的老板”）。心事是主人信任{config.PET_NAME}才说的，一定要记，importance 至少 4；
  - us：主人和{config.PET_NAME}之间发生的事：约定、起的外号、一起开的玩笑、主人对{config.PET_NAME}说的特别的话。
    只记发生了什么（“主人给{config.PET_NAME}起了外号叫小糯米”），不记{config.PET_NAME}当时嘴硬、害羞的反应；
  - self：{config.PET_NAME}自己经历或学到的事。
  - importance：5 人生大事或重要约定，4 心事和秘密，3 普通但值得记，1 小事。寒暄、提问、一句话的应答都不记。
- facts 只放需要回头关心的事。主人让到点提醒的事（“30 秒后提醒我……”“三点叫我……”）不放进 facts，那个有闹钟管。follow_up_at：身体不舒服或很累，填当天晚上 21:00~22:00；明天有安排，填事情结束后的时间；不需要回访填 null。
- profile 只放长期不变的信息。
- interaction 是主人这一轮对{config.PET_NAME}的态度。
- 没有就给空列表/空对象。"""
    convo = f"主人：{user_text}\n宠物：{pet_reply}"
    return [{"role": "system", "content": system}, {"role": "user", "content": convo}]


# ---------- 夜间整理（nightly.py） ----------
def merge_messages(texts):
    lines = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    system = f"""你在帮「{config.PET_NAME}」整理记忆。下面几条记忆按时间先后排列，看起来很像。
判断它们说的是不是同一件事，只输出 JSON：{{"same": true 或 false, "text": "合成后的一句话"}}
- 同一件事的不同说法，或者同一件事后来的变化（比如先住院、后出院），same=true，合成一句，变化要写进去（“主人的妈妈住过院，后来出院了”），不丢信息，不超过 60 字；
- 说的是不同的事（同一只猫干的两件事、旅行和旅行中吃的东西），same=false，text 留空。"""
    return [{"role": "system", "content": system}, {"role": "user", "content": lines}]


def diary_messages(day_label, convo, traits_text, self_view):
    system = f"""{SYSTEM}

【你和主人这阵子的相处】{traits_text}
【你对自己的认识】{self_view or "（还没想过）"}

现在是夜里，你要睡觉了。把今天（{day_label}）和主人的事写进你的小日记。只输出 JSON：
{{"diary": "日记", "highlight": "今天最想记住的一件事", "importance": 1到5}}
- diary：用“我”的口吻，3~6 句，写发生了什么、你当时的心情、你对主人的感觉；口语化，像小动物写的，不用 emoji。
- highlight：今天新发生的、最想记住的一件事，一句话写清楚（比如“主人第一次叫我小糯米”），以后你会靠它想起今天。
  必须是今天的对话里发生的，不要写以前就知道的事；没什么特别的就写今天的大概。"""
    return [{"role": "system", "content": system}, {"role": "user", "content": f"今天的对话：\n{convo}"}]


def self_view_messages(old_view, diary, traits_text):
    system = f"""{SYSTEM}

你在睡前想一想“我是一个怎样的{config.PET_NAME}”。只输出 JSON：{{"self_view": "新的自我认识"}}
- 在旧的认识上**小小地**修改：今天的经历让你对自己有了新的认识才改，没有就几乎不动；
- 用“我”的口吻，不超过 100 字，写你的性格、你和主人的关系、你在意的事；
- 不能和你天生的性格矛盾（黏人、好奇、嘴硬心软、住在板子里的小家伙），不提“人工智能”。"""
    user = f"旧的认识：{old_view or '（还没想过，这是第一次）'}\n这阵子的相处：{traits_text}\n今天的日记：{diary}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------- 后处理 ----------
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")
# 允许以“……”开头（探头时欲言又止的语气），否则开头的省略号会被吞掉
SENTENCE_RE = re.compile(r"[…~～]*[^。！？!?…~～]+[。！？!?…~～]*")


def clean_reply(text: str, max_chars: int | None = None) -> str:
    max_chars = max_chars or config.MAX_REPLY_CHARS
    text = EMOJI_RE.sub("", text or "")
    text = re.sub(r"[*#`]", "", text)
    text = re.sub(r"https?://\S+", "", text)  # 查完资料偶尔会带网址，团子不念网址
    names = "|".join(re.escape(n) for n in (config.PET_NAME, *config.PET_OLD_NAMES))  # 读了旧记忆，偶尔还会写旧名字
    text = re.sub(rf"^\s*(?:{names})\s*[:：]", "", text)
    text = " ".join(text.split()).strip()
    # 整句被引号包住才剥（“三角洲”通常指…… 这种句首引号是内容的一部分，不能只剥掉半边）
    if len(text) >= 2 and (text[0], text[-1]) in {('"', '"'), ("'", "'"), ("“", "”"), ("「", "」")}:
        text = text[1:-1].strip()
    sentences = [s.strip() for s in SENTENCE_RE.findall(text) if s.strip()]
    text = "".join(sentences[: config.MAX_SENTENCES]) or text
    if len(text) > max_chars:
        text = text[:max_chars].rstrip("，,、 ") + "…"
    return text


def off_character(text: str, topic_ok: bool = False) -> bool:
    """出戏检测。topic_ok：回答是查资料得来的（比如主人问 DeepSeek），“人工智能”“语言模型”可能就是话题本身，
    这时只拦团子把自己说成 AI 的说法。"""
    compact = text.replace(" ", "").upper()
    words = config.SELF_AI_WORDS if topic_ok else config.BANNED_WORDS + config.SELF_AI_WORDS
    return any(w.replace(" ", "").upper() in compact for w in words)
