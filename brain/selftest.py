"""不依赖 Ollama 的自测：纯函数断言 + 用假 LLM 跑一遍“下午说眼睛酸 → 晚上主动关心”的完整闭环。

运行：python selftest.py（在 ai_pet/ 目录）。用临时数据库，不碰 data/pet.db。
"""
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import config

TMP = Path(tempfile.mkdtemp(prefix="pet_selftest_"))
config.DB_FILE = TMP / "a.db"
config.TTS_AUTOSTART = False  # 自测不去拉起真的声音服务（不占显卡）
config.CLOUD = False          # 自测不问真的云端（花钱），要测时单独打开并换成假的
config.RELAY_HOST = ""        # 自测不开真的 SSH 隧道

import numpy as np  # noqa: E402

import recall  # noqa: E402


def fake_embed(texts):
    """自测用的假向量模型：按相邻两个字哈希成 1024 维，同样的话得到同样的向量。不依赖 Ollama。"""
    out = np.zeros((len(texts), 1024), dtype=np.float32)
    for i, t in enumerate(texts):
        for a, b in zip(t, t[1:]):
            out[i, hash(a + b) % 1024] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)


recall.embed = fake_embed

import brain  # noqa: E402

brain.BACKGROUND_MEMORY = False  # 记忆提炼同步做：不然后台线程会写到下一个测试的临时库里
import clock  # noqa: E402
import db  # noqa: E402
import memory  # noqa: E402
import persona  # noqa: E402
import rules  # noqa: E402
import state  # noqa: E402

D = datetime(2026, 10, 18)


def at(hhmm, days=0):
    h, m = map(int, hhmm.split(":"))
    return D + timedelta(days=days, hours=h, minutes=m)


def test_state():
    s = state.compute(at("03:00"), at("02:00"))
    assert s.phase == "asleep" and s.sleepy_talk, s
    s = state.compute(at("10:00"), at("06:00"))
    assert s.phase == "awake" and not s.sleepy_talk and abs(s.loneliness - 0.8) < 1e-9, s
    s = state.compute(at("14:00"), at("08:00"), last_pat=at("13:30"))
    assert abs(s.loneliness - (config.PAT_LONELINESS + 0.1)) < 1e-9, s  # 被摸过，孤单值回落
    s = state.compute(at("01:30"), at("01:00"))
    assert s.sleepy_talk, s
    assert state.in_window(at("23:45"), ("23:30", "08:30")) and state.in_window(at("02:00"), ("23:30", "08:30"))
    assert not state.in_window(at("12:00"), ("23:30", "08:30"))


def test_rules():
    lonely = state.compute(at("15:00"), at("10:30"))
    g = rules.Gate(None, 0, 0)
    d, _ = rules.decide(lonely, at("15:00"), g, [], [])
    assert d and d.kind == "lonely"
    # 免打扰
    d, why = rules.decide(state.compute(at("23:45"), at("10:00")), at("23:45"), g, [], [])
    assert d is None and "免打扰" in why
    # 冷却
    d, why = rules.decide(lonely, at("15:00"), rules.Gate(at("14:00"), 0, 1), [], [])
    assert d is None and "冷却" in why
    # 被无视一次，冷却翻倍：3 小时前主动过，正常冷却 2h 已过，但翻倍后 4h 没过
    d, why = rules.decide(lonely, at("15:00"), rules.Gate(at("12:00"), 1, 1), [], [])
    assert d is None and "冷却" in why
    # 每日上限
    d, why = rules.decide(lonely, at("15:00"), rules.Gate(None, 0, config.DAILY_PROACTIVE_LIMIT), [], [])
    assert d is None
    # 回访不等冷却，但离上次主动至少 30 分钟
    fu = [{"id": 1, "fact": "眼睛酸"}]
    d, _ = rules.decide(lonely, at("15:00"), rules.Gate(at("14:20"), 0, 1), fu, [])
    assert d and d.kind == "followup"
    d, _ = rules.decide(lonely, at("15:00"), rules.Gate(at("14:45"), 0, 1), fu, [])
    assert d is None
    # 被无视计数：主动后 30 分钟没回应才算，且每次主动只算一次
    s, c = rules.update_ignored(at("14:20"), at("14:00"), at("13:00"), None, 0)
    assert s == 0
    s, c = rules.update_ignored(at("14:31"), at("14:00"), at("13:00"), None, 0)
    assert s == 1 and c == at("14:00")
    s, c = rules.update_ignored(at("15:00"), at("14:00"), at("13:00"), c, s)
    assert s == 1
    s, _ = rules.update_ignored(at("15:00"), at("14:00"), at("14:10"), None, 0)
    assert s == 0  # 主人回过话


def test_resolve_due():
    assert memory.resolve_due(at("14:00"), "22:00", "疲惫") == at("22:00")
    assert memory.resolve_due(at("22:50"), "22:00", "疲惫") == at("22:00", days=1)
    assert memory.resolve_due(at("14:00"), "23:50", "平静") == at("08:30", days=1)  # 免打扰 → 推到醒来
    assert memory.resolve_due(at("14:00"), None, "有点累") == at("20:00")          # 负面情绪默认 6 小时
    assert memory.resolve_due(at("14:00"), None, "开心") is None
    assert memory.resolve_due(at("14:00"), "null", "开心") is None


def test_clean_reply():
    r = persona.clean_reply("团子：呜，辛苦啦。快闭眼歇两分钟嘛！我陪着你～第四句不该出现。😊")
    assert r == "呜，辛苦啦。快闭眼歇两分钟嘛！我陪着你～", r
    assert persona.clean_reply("……你还在忙吗？") == "……你还在忙吗？"
    assert persona.clean_reply("“三角洲”通常指河口。") == "“三角洲”通常指河口。"  # 句首引号是内容，不能只剥半边
    assert persona.clean_reply("“嗯，我在呢。”") == "嗯，我在呢。"  # 整句被引号包住才剥
    assert persona.clean_reply(f"{config.PET_NAME}：嗯嗯") == "嗯嗯"  # 现在的名字和以前的名字开头都剥掉
    assert "团子" in persona.SYSTEM and config.PET_NAME in persona.SYSTEM  # 它知道自己改过名
    assert persona.new_name("团子乖乖睡啦") == f"{config.PET_NAME}乖乖睡啦"
    hist = persona.history_messages([{"role": "pet", "text": "团子在呢"}, {"role": "user", "text": "团子你好"}])
    assert hist == [{"role": "assistant", "content": f"{config.PET_NAME}在呢"}, {"role": "user", "content": "团子你好"}]
    assert brain.ASKS_OLD_NAME.search("那你以前叫什么") and not brain.ASKS_OLD_NAME.search("你在干嘛")
    assert brain.TALKS_OLD_NAME.search("以前那个笨笨的名字团子就不提啦")
    assert persona.fix_asr("他叫独蜜。") == "他叫猪咪。" and persona.fix_asr("杜米这个词") == "猪咪这个词"
    assert persona.history_messages([{"role": "user", "text": "他叫独蜜"}])[0]["content"] == "他叫猪咪"
    for noise in [".", "The.", "Yeah.", "  ", "。", "嗯…"[1:]]:
        assert brain.meaningful(noise) == "", noise
    for real in ["你在干嘛呀？", "嗯", "OK没问题，那我先去忙啦。", "I love you"]:
        assert brain.meaningful(real) == real, real
    assert persona.off_character("我是一个AI 助手哦")
    assert not persona.off_character("我是住在板子里的小家伙")


class FakeLLM:
    model = "fake"

    def chat_json(self, msgs, temperature=0.0):
        if "记忆整理员" in msgs[0]["content"]:
            return {"facts": [{"fact": "下午改Bug眼睛很酸", "mood": "疲惫", "follow_up_at": "22:00",
                               "follow_up": "问问眼睛好点没"}], "profile": {"工作": "程序员"}}
        return {"reply": "呜，辛苦啦。快闭眼歇两分钟嘛。", "face": "sad"}

    def chat_tools(self, msgs, tools, temperature=0.7):  # 原生工具调用模式：不调工具，照常回 JSON 文本
        return {"role": "assistant", "content": json.dumps(self.chat_json(msgs, temperature), ensure_ascii=False)}

    def chat(self, msgs, temperature=0.2, json_mode=False):
        assert "下午改Bug眼睛很酸" in msgs[0]["content"] or "孤单" in msgs[0]["content"]
        return "诶，你下午说眼睛酸，现在好点了吗？"

    def is_up(self, connect_timeout=None):
        return True


def fresh_db(name):
    config.DB_FILE = TMP / name
    db.init()
    ts = clock.fmt(clock.now())
    db.set_kv("last_interaction", ts)


def take():
    now = clock.now()
    return db.take_pending_events(clock.fmt(now), brain.event_cutoff(now))


def test_followup_loop():
    fresh_db("followup.db")
    brain.client, brain.BACKGROUND_MEMORY = FakeLLM(), False
    clock.set_time("14:00")
    r = brain.chat("改了一下午 Bug，眼睛好酸")
    assert r["face"] == "sad" and r["text"]
    fus = db.pending_followups()
    assert len(fus) == 1 and fus[0]["due_at"].endswith("22:00:00"), fus

    clock.set_time("21:30")
    r = brain.tick()
    assert not r["fired"] or r["kind"] != "followup", r  # 还没到期
    take()

    clock.set_time("22:00")
    db.set_kv("last_proactive", "")  # 21:30 可能因为孤单开过口，这里只测回访
    r = brain.tick()
    assert r["fired"] and r["kind"] == "followup", r
    events = take()
    assert len(events) == 1 and events[0]["face"] == "peek"
    assert not db.pending_followups()
    r = brain.tick()
    assert not r["fired"], r  # 同一件事不会再问
    assert take() == []


def test_lonely_and_backoff():
    fresh_db("lonely.db")
    brain.client = FakeLLM()
    clock.set_time("10:00")
    db.set_kv("last_interaction", clock.fmt(clock.now()))
    clock.advance(4)
    r = brain.tick()
    assert r["fired"] and r["kind"] == "lonely", r
    r = brain.tick()
    assert not r["fired"] and "冷却" in r["reason"], r
    clock.advance(0.6)
    brain.tick()
    assert db.get_kv("ignored_streak") == "1"
    brain.chat("在呢在呢")
    assert db.get_kv("ignored_streak") == "0"


def test_pat_counts_as_response():
    fresh_db("pat.db")
    brain.client = FakeLLM()
    clock.set_time("10:00")
    db.set_kv("last_interaction", clock.fmt(clock.now()))
    clock.advance(4)
    assert brain.tick()["kind"] == "lonely"
    clock.advance(0.1)
    db.set_kv("last_pat", clock.fmt(clock.now()))  # 它探头后被摸了一下
    clock.advance(0.6)
    brain.tick()
    assert db.get_kv("ignored_streak") == "0"


def test_stale_events_expire():
    fresh_db("stale.db")
    brain.client = FakeLLM()
    clock.set_time("10:00")
    db.set_kv("last_interaction", clock.fmt(clock.now()))
    clock.advance(4)
    assert brain.tick()["fired"]
    clock.advance(config.EVENT_TTL_HOURS + 0.5)  # 设备离线太久才来拉取
    assert take() == []
    assert take() == []  # 过期的也不会留到下次


def test_websocket_link():
    """设备协议：hello → welcome；chat → thinking/say/state；pat 记为回应；主动搭话实时推送。"""
    from fastapi.testclient import TestClient
    import server

    fresh_db("ws.db")
    brain.client = FakeLLM()
    config.PUSH_INTERVAL_S = 0.1
    clock.set_time("10:00")
    db.set_kv("last_interaction", clock.fmt(clock.now()))
    with TestClient(server.app) as c, c.websocket_connect("/ws") as ws:
        ws.send_json({"type": "chat", "text": "hi"})
        assert ws.receive_json()["type"] == "error"  # 没 hello 不理
        ws.send_json({"type": "hello", "device": "test", "proto": 1})
        welcome = ws.receive_json()
        assert welcome["type"] == "welcome" and "loneliness" in welcome["state"], welcome

        ws.send_json({"type": "chat", "text": "改了一下午 Bug，眼睛好酸"})
        assert ws.receive_json()["type"] == "thinking"
        say = ws.receive_json()
        assert say["type"] == "say" and not say["proactive"] and say["text"] and say["face"] == "sad", say
        assert ws.receive_json()["type"] == "state"

        ws.send_json({"type": "pat", "count": 2})
        assert ws.receive_json()["type"] == "state"
        assert db.get_kv("last_pat")

        # 4 小时没理它 → 心跳决定主动搭话 → 推送给在线设备
        clock.advance(4.2)
        assert brain.tick()["fired"]
        pushed = ws.receive_json()
        while pushed["type"] == "state":
            pushed = ws.receive_json()
        assert pushed["type"] == "say" and pushed["proactive"] and pushed["face"] == "peek", pushed

        ws.send_json({"type": "ping"})
        msg = ws.receive_json()
        while msg["type"] == "state":
            msg = ws.receive_json()
        assert msg["type"] == "pong"


def test_chat_does_not_block_socket():
    """大脑想得很慢（冷启动加载模型）时，连接照常收发：ping 马上有 pong，不会因为心跳没人回而被断开。"""
    import threading
    from fastapi.testclient import TestClient
    import server

    fresh_db("slow.db")
    gate = threading.Event()

    class SlowLLM(FakeLLM):
        def chat_tools(self, msgs, tools, temperature=0.7):
            gate.wait(10)
            return super().chat_tools(msgs, tools, temperature)

        def chat_json(self, msgs, temperature=0.0):
            if "记忆整理员" not in msgs[0]["content"]:
                gate.wait(10)
            return super().chat_json(msgs, temperature)

    brain.client = SlowLLM()
    with TestClient(server.app) as c, c.websocket_connect("/ws") as ws:
        ws.send_json({"type": "hello", "device": "test"})
        assert ws.receive_json()["type"] == "welcome"
        ws.send_json({"type": "chat", "text": "在吗"})
        assert ws.receive_json()["type"] == "thinking"
        ws.send_json({"type": "ping"})
        msg = ws.receive_json()
        while msg["type"] == "state":
            msg = ws.receive_json()
        assert msg == {"type": "pong"}, msg  # 大脑还卡在 gate 上，接收循环照样回了
        gate.set()
        msg = ws.receive_json()
        while msg["type"] == "state":
            msg = ws.receive_json()
        assert msg["type"] == "say", msg


def test_voice_listen():
    """按住说话：listen start → 二进制录音 → stop → heard → 对话；语气会传给大脑；太短的不识别。"""
    from fastapi.testclient import TestClient
    import asr
    import server

    fresh_db("voice.db")
    seen = {}

    class ToneLLM(FakeLLM):
        def chat_json(self, msgs, temperature=0.0):
            if "记忆整理员" not in msgs[0]["content"]:
                seen["system"] = msgs[0]["content"]
            return super().chat_json(msgs, temperature)

    brain.client = ToneLLM()
    real = (asr.available, asr.transcribe_pcm16)
    asr.available = lambda: True
    asr.transcribe_pcm16 = lambda pcm, rate=16000: asr.Heard("今天好累呀", "难过", "")
    try:
        with TestClient(server.app) as c, c.websocket_connect("/ws") as ws:
            ws.send_json({"type": "hello", "device": "test"})
            assert ws.receive_json()["type"] == "welcome"

            ws.send_json({"type": "listen", "state": "start", "format": "pcm16", "rate": 16000})
            ws.send_bytes(b"\x00\x00" * 16000)  # 1 秒
            ws.send_json({"type": "listen", "state": "stop"})
            heard = ws.receive_json()
            assert heard == {"type": "heard", "text": "今天好累呀", "emotion": "难过", "event": ""}, heard
            assert ws.receive_json()["type"] == "thinking"
            assert ws.receive_json()["type"] == "say"
            assert "听起来难过" in seen["system"]

            ws.send_json({"type": "listen", "state": "start"})
            ws.send_bytes(b"\x00\x00" * 1600)  # 0.1 秒：当作手滑
            ws.send_json({"type": "listen", "state": "stop"})
            msg = ws.receive_json()
            while msg["type"] == "state":
                msg = ws.receive_json()
            assert msg == {"type": "heard", "text": ""}, msg
    finally:
        asr.available, asr.transcribe_pcm16 = real


def test_real_asr_sample():
    """装了 SenseVoice 模型时，用模型自带的样例音频真跑一次识别。"""
    import wave
    import asr

    sample = config.ASR_MODEL_DIR / "test_wavs" / "zh.wav"
    if not (asr.available() and sample.exists()):
        print("  （跳过：没装语音识别模型）")
        return
    with wave.open(str(sample)) as w:
        pcm = w.readframes(w.getnframes())
    heard = asr.transcribe_pcm16(pcm)
    assert "开饭时间" in heard.text, heard


def test_tts_text_and_mood():
    import tts

    assert tts.speech_text("（蹭蹭）嘿嘿，好舒服～😊") == "嘿嘿，好舒服～"
    assert tts.speech_text("（摸摸头）") == "" and tts.speech_text("……") == ""
    assert tts.mood_for({"face": "shy"}) == "happy" and tts.mood_for({"face": "lonely"}) == "sad"
    assert tts.mood_for({"face": "happy", "voice": "sleepy"}) == "sleepy"  # 深夜一律犯困
    assert tts.mood_for({"face": "peek"}) == "normal"


def test_tts_voice_stream():
    """团子的声音：声明 audio_out 的设备在 say 之后收到 tts start → PCM 帧 → stop，情绪跟表情走；
    没声明的设备（固件）只收文字；主人开口会打断；合成失败 stop 带 error。用假的声音服务，不碰显卡。"""
    import json
    import threading
    import time
    from fastapi.testclient import TestClient
    import server
    import tts

    fresh_db("tts.db")
    brain.client = FakeLLM()
    config.PUSH_INTERVAL_S = 0.1
    clock.set_time("10:00")
    db.set_kv("last_interaction", clock.fmt(clock.now()))
    calls, mode = [], {"frames": 3, "delay": 0.0, "fail": False}
    started = threading.Event()

    def fake_stream(text, mood, cancelled=lambda: False):
        calls.append((text, mood))
        if mode["fail"]:
            raise ConnectionError("声音服务挂了")
        for _ in range(mode["frames"]):
            if cancelled():
                return
            started.set()
            time.sleep(mode["delay"])
            yield b"\x01\x00" * 2400  # 0.1 秒

    def recv(ws):  # 跳过定期的状态同步
        m = ws.receive_json()
        while m["type"] == "state":
            m = ws.receive_json()
        return m

    real = (tts.available, tts.stream)
    tts.available, tts.stream = (lambda: True), fake_stream
    try:
        with TestClient(server.app) as c, c.websocket_connect("/ws") as ws, c.websocket_connect("/ws") as fw:
            ws.send_json({"type": "hello", "device": "web", "audio_out": "pcm16"})
            assert ws.receive_json()["type"] == "welcome"
            fw.send_json({"type": "hello", "device": "esp32"})
            assert fw.receive_json()["type"] == "welcome"

            ws.send_json({"type": "chat", "text": "改了一下午 Bug，眼睛好酸"})
            assert recv(ws)["type"] == "thinking"
            say = recv(ws)
            assert say["type"] == "say" and say["audio"] and say["id"], say
            start = recv(ws)
            assert start == {"type": "tts", "state": "start", "id": say["id"], "format": "pcm16", "rate": 24000}, start
            assert sum(len(ws.receive_bytes()) for _ in range(3)) == 3 * 4800
            assert recv(ws) == {"type": "tts", "state": "stop", "id": say["id"]}
            assert calls[-1] == ("呜，辛苦啦。快闭眼歇两分钟嘛。", "sad"), calls  # 难过的表情 → 委屈的语气

            # 主动搭话：能出声的设备收到声音，固件只收文字
            clock.advance(4.2)
            assert brain.tick()["fired"]
            fsay = recv(fw)
            assert fsay["type"] == "say" and fsay["proactive"] and not fsay["audio"] and "id" not in fsay, fsay
            psay = recv(ws)
            assert psay["proactive"] and psay["audio"] and psay["id"] > say["id"], psay
            assert recv(ws)["state"] == "start"
            for _ in range(3):
                ws.receive_bytes()
            assert recv(ws) == {"type": "tts", "state": "stop", "id": psay["id"]}

            # 说到一半主人开口：剩下的不推了
            mode.update(frames=30, delay=0.05)
            started.clear()
            ws.send_json({"type": "chat", "text": "再陪我聊聊"})
            assert recv(ws)["type"] == "thinking"
            say2 = recv(ws)
            assert recv(ws)["state"] == "start"
            started.wait(5)
            ws.send_json({"type": "listen", "state": "start", "format": "pcm16", "rate": 16000})
            frames = 0
            while True:
                m = ws.receive()
                if m.get("bytes") is not None:
                    frames += 1
                    continue
                msg = json.loads(m["text"])
                if msg["type"] == "tts":
                    break
            assert msg == {"type": "tts", "state": "stop", "id": say2["id"], "cancelled": True}, msg
            assert frames < 30, frames
            ws.send_json({"type": "listen", "state": "cancel"})

            # 声音服务挂了：say 照常，stop 带 error，设备自己兜底
            mode.update(fail=True)
            ws.send_json({"type": "chat", "text": "你在吗"})
            assert recv(ws)["type"] == "thinking"
            say3 = recv(ws)
            assert recv(ws)["state"] == "start"
            assert recv(ws) == {"type": "tts", "state": "stop", "id": say3["id"], "error": "声音没合成出来"}
    finally:
        tts.available, tts.stream = real


def test_opus_device():
    """ESP32 走 Opus：hello 声明 audio_out=opus，录音一帧一个 Opus 包，大脑解码后识别；
    团子的声音编成 60ms 一包的 Opus 推过去，解出来长度对得上；摸头（hush）让它别说了。没装 av 就跳过。"""
    import time
    import numpy as np
    from fastapi.testclient import TestClient
    import asr
    import opus
    import server
    import tts

    if not opus.available():
        print("  （跳过：没装 av）")
        return
    fresh_db("opus.db")
    brain.client = FakeLLM()
    got = {}

    def fake_asr(pcm, rate=16000):
        got["pcm"], got["rate"] = pcm, rate
        return asr.Heard("今天好累呀", "", "")

    def fake_stream(text, mood, cancelled=lambda: False):
        for _ in range(5):
            yield b"\x10\x00" * 2400  # 5 × 0.1 秒

    real = (asr.available, asr.transcribe_pcm16, tts.available, tts.stream)
    asr.available, asr.transcribe_pcm16 = (lambda: True), fake_asr
    tts.available, tts.stream = (lambda: True), fake_stream
    try:
        with TestClient(server.app) as c, c.websocket_connect("/ws") as ws:
            ws.send_json({"type": "hello", "device": "esp32", "audio_out": "opus"})
            assert ws.receive_json()["type"] == "welcome"

            t = np.arange(16000) / 16000  # 1 秒 440Hz
            enc = opus.Encoder(16000)
            packets = enc.encode((5000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16).tobytes()) + enc.flush()
            ws.send_json({"type": "listen", "state": "start", "format": "opus", "rate": 16000})
            for p in packets:
                ws.send_bytes(p)
            ws.send_json({"type": "listen", "state": "stop"})
            assert ws.receive_json()["text"] == "今天好累呀"
            assert got["rate"] == 16000 and 0.9 < len(got["pcm"]) / 2 / 16000 < 1.3, len(got["pcm"])
            a = np.frombuffer(got["pcm"], dtype=np.int16).astype(float)
            assert np.sqrt((a ** 2).mean()) > 2000  # 解出来的是那段声音，不是静音

            assert ws.receive_json()["type"] == "thinking"
            say = ws.receive_json()
            assert say["type"] == "say" and say["audio"], say
            m = ws.receive_json()
            while m["type"] == "state":
                m = ws.receive_json()
            assert m == {"type": "tts", "state": "start", "id": say["id"], "format": "opus", "rate": 24000,
                         "frame_ms": 60}, m
            dec, pcm = opus.Decoder(24000), b""
            while True:
                f = ws.receive()
                if f.get("bytes") is None:
                    assert json.loads(f["text"]) == {"type": "tts", "state": "stop", "id": say["id"]}
                    break
                assert len(f["bytes"]) < 1000  # 一帧一个 Opus 包，不是整段 PCM
                pcm += dec.decode(f["bytes"])
            assert 0.5 <= len(pcm) / 2 / 24000 <= 0.7, len(pcm)  # 0.5 秒的声音，最后一包补了静音

            ws.send_json({"type": "listen", "state": "start", "format": "mp3"})
            assert "不支持" in ws.receive_json()["message"]

            # 摸头让团子别说了（hush）：正在推的这段声音停下，stop 带 cancelled
            def slow_stream(text, mood, cancelled=lambda: False):
                for _ in range(40):
                    if cancelled():
                        return
                    time.sleep(0.05)
                    yield b"\x10\x00" * 2400

            tts.stream = slow_stream
            ws.send_json({"type": "chat", "text": "讲个长故事"})
            m = ws.receive_json()
            while not (m["type"] == "tts" and m["state"] == "start"):
                m = ws.receive_json()
            ws.receive_bytes()
            ws.send_json({"type": "hush"})
            frames = 0
            while True:
                f = ws.receive()
                if f.get("bytes") is not None:
                    frames += 1
                    continue
                m = json.loads(f["text"])
                if m["type"] == "tts":
                    break
            assert m["state"] == "stop" and m.get("cancelled") and frames < 50, (m, frames)  # 4 秒的声音约 67 包
    finally:
        asr.available, asr.transcribe_pcm16, tts.available, tts.stream = real


class SearchLLM(FakeLLM):
    """会用搜索工具的假大脑：问“三角洲”先要求查（原生 tools 发调用 / JSON 里写 search），拿到资料后再回答。"""

    def __init__(self, query="三角洲"):
        self.query, self.seen = query, []

    def _answer(self, msgs):
        self.seen.append(msgs)
        found = [m["content"] for m in msgs
                 if m["role"] == "tool" or (m["role"] == "system" and m["content"].startswith("【你"))]
        if not found and "三角洲" in msgs[-1]["content"]:
            return None  # 要查
        if found and "三角洲行动" in found[-1]:
            return {"reply": "三角洲是河口冲出来的平原，也是腾讯的射击游戏《三角洲行动》，你说哪个呀？", "face": "curious"}
        if found:
            return {"reply": "唔，我刚刚没查到，网好像不太通。", "face": "sad"}
        return {"reply": "嗯嗯，我在呢。", "face": "happy"}

    def chat_json(self, msgs, temperature=0.0):
        if "记忆整理员" in msgs[0]["content"]:
            return {"facts": [], "profile": {}}
        return self._answer(msgs) or {"search": self.query}

    def chat_tools(self, msgs, tools, temperature=0.7):
        data = self._answer(msgs)
        if data is None:
            return {"role": "assistant", "content": "",
                    "tool_calls": [{"function": {"name": "web_search", "arguments": {"query": self.query}}}]}
        return {"role": "assistant", "content": json.dumps(data, ensure_ascii=False)}


def test_web_search_tool():
    """联网搜索（工具调用，原生 tools 和 JSON 两种）：该查就查、结果交回大脑再回答；只有清洗过的搜索词交给搜索；
    查不到老实说；闲聊不查；关掉开关就不查。"""
    import websearch

    from fastapi.testclient import TestClient
    import server

    calls, notified = [], []
    real = (websearch.search, websearch.available, config.WEB_SEARCH, config.TOOL_MODE)
    found = [{"title": "三角洲行动", "body": "腾讯天美 J3 工作室开发的第一人称射击游戏，2024 年 9 月上线。", "href": "https://x"}]
    websearch.available = lambda: config.WEB_SEARCH
    try:
        for mode in ("native", "json"):
            config.TOOL_MODE, config.WEB_SEARCH = mode, True
            fresh_db(f"search_{mode}.db")
            clock.set_time("10:00")  # 白天：深夜跑自测时表情会被换成犯困
            db.set_profile("称呼", "小蔡", clock.fmt(clock.now()))
            calls.clear(), notified.clear()
            websearch.search = lambda q: calls.append(q) or found
            brain.client = SearchLLM(query="小蔡 想知道 「三角洲」")
            r = brain.chat("什么是三角洲", on_tool=lambda name, detail: notified.append(detail))
            assert calls == ["想知道 三角洲"] == notified, (mode, calls, notified)  # 主人的称呼被抹掉了
            assert r["searched"] == ["想知道 三角洲"] and "三角洲行动" in r["text"] and r["face"] == "curious", r
            last = brain.client.seen[-1]
            if mode == "native":  # 调用 → 工具返回（资料）→ 怎么回答的规矩
                assert last[-3]["tool_calls"][0]["function"]["name"] == "web_search", last[-3]
                assert last[-2]["role"] == "tool" and "腾讯天美" in last[-2]["content"]
                assert "https://x" not in last[-2]["content"]  # 不给网址
                assert '{"search"' not in last[0]["content"] and "web_search" in last[0]["content"]
            else:
                assert last[-2] == {"role": "assistant", "content": '{"search": "想知道 三角洲"}'}
                assert "腾讯天美" in last[-1]["content"] and "https://x" not in last[-1]["content"]

            calls.clear()
            websearch.search = lambda q: calls.append(q) or []  # 网络不通
            brain.client = SearchLLM()
            r = brain.chat("三角洲到底是啥")
            assert calls == ["三角洲"] and "没查到" in r["text"], (mode, r)

            brain.client = SearchLLM()
            assert "searched" not in brain.chat("今天好累")  # 闲聊不查
            config.WEB_SEARCH = False
            calls.clear()
            brain.client = SearchLLM()
            r = brain.chat("什么是三角洲")  # 关掉联网：想查也不让，提示里也没有搜索这个选项
            assert calls == [] and "searched" not in r, (mode, r)
            assert '{"search"' not in brain.client.seen[0][0]["content"]
            assert "web_search" not in brain.client.seen[0][0]["content"]

        # 设备协议：先 thinking，查资料时再来一条带 search 的 thinking，say 里带 searched
        config.TOOL_MODE, config.WEB_SEARCH = "native", True
        websearch.search = lambda q: found
        brain.client = SearchLLM()
        with TestClient(server.app) as c, c.websocket_connect("/ws") as ws:
            ws.send_json({"type": "hello", "device": "test"})
            assert ws.receive_json()["type"] == "welcome"
            ws.send_json({"type": "chat", "text": "什么是三角洲"})
            assert ws.receive_json() == {"type": "thinking"}
            assert ws.receive_json() == {"type": "thinking", "tool": "web_search", "detail": "三角洲", "search": "三角洲"}
            say = ws.receive_json()
            assert say["type"] == "say" and say["searched"] == ["三角洲"], say
            assert say["tools"] == [{"tool": "web_search", "detail": "三角洲"}], say
    finally:
        websearch.search, websearch.available, config.WEB_SEARCH, config.TOOL_MODE = real

    assert websearch.clean_query("「黑神话悟空」   销量", [{"key": "称呼", "value": "小蔡"}]) == "黑神话悟空 销量"
    assert not persona.off_character("DeepSeek 做的大语言模型挺厉害", topic_ok=True)  # 查资料聊 AI 话题不算出戏
    assert persona.off_character("作为AI，我不能……", topic_ok=True)
    assert persona.off_character("我是一个语言模型")


def test_toolbox():
    """小工具：计算器只认算式；日期按大脑的时钟算；问云端只发清洗过的问题、受每日额度和开关约束；
    大脑调工具（计算器、问云端）→ 拿结果 → 用自己的口吻回答，设备收到用了什么工具。"""
    import cloud
    import toolbox

    assert toolbox.calculator("1500*12%").text == "1500*12% = 180"
    assert toolbox.calculator("(1500-320)/12").text.endswith("98.333333")
    for bad in ['__import__("os")', "9**99999", "open('x')"]:
        try:
            toolbox.calculator(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    fresh_db("tools.db")
    clock.set_time("10:00")
    today = clock.now().date()
    r = toolbox.date_calc(days=7)
    assert f"{today + timedelta(days=7):%Y-%m-%d}" in r.text and "7 天之后" in r.text, r.text
    # 农历节日由代码算（模型不知道每年春节是哪天）
    from datetime import date as _date
    assert toolbox.next_holiday("春节", _date(2026, 10, 25)) == ("春节", _date(2027, 2, 6))
    assert toolbox.next_holiday("大年三十", _date(2026, 10, 25)) == ("除夕", _date(2027, 2, 5))
    assert toolbox.next_holiday("中秋节", _date(2026, 9, 25)) == ("中秋", _date(2026, 9, 25))  # 当天就是
    assert toolbox.next_holiday("国庆节", _date(2026, 10, 25)) == ("国庆", _date(2027, 10, 1))
    assert not toolbox.date_calc(holiday="植树节").ok  # 不认识的老实说
    # 放几层进显卡按空余现算：桌面多占了显存就少放几层，放不下就全放 CPU，不会超出模型层数
    import llm
    real_fit = dict(config.GPU_FIT)
    config.GPU_FIT.update(total_mb=8188, reserved_mb=232, margin_mb=400, base_mb=1050, per_layer_mb=125, layers=33)
    try:
        assert llm.fit_layers(3200) == 26           # 声音服务 + 桌面 744MB 时
        assert llm.fit_layers(2500) == 32           # 桌面占得少时多放几层
        assert llm.fit_layers(500) == 33            # 显卡空着：全放，但不超过 33 层
        assert llm.fit_layers(7900) == 0            # 显卡满了：全放 CPU，不硬塞
        assert llm.fit_layers(7600, ours_mb=4400) == 26  # 模型已加载时，它自己占的算作可用
    finally:
        config.GPU_FIT.update(real_fit)
    assert persona.fallback_line("stuck") != persona.fallback_line("offline")
    # 只有提问 / 请它办事时才给工具；说自己的事、闲聊不给
    for q in ["什么是三角洲", "三角洲行动好玩吗", "离春节还有几天", "北京明天冷不冷", "帮我算算房租", "明天要带伞吗",
              "你知道 DeepSeek 吗", "最近有什么大新闻"]:
        assert brain.wants_tools(q), q
    for s in ["我中午吃了火锅", "我升职了！", "今天好累啊", "晚安", "嘿嘿", "我在玩三角洲行动，好难啊"]:
        assert not brain.wants_tools(s), s
    # 模型把 JSON 写坏（中文引号）时，把回复抠出来，不念半截 JSON
    assert brain._salvage('{"reply": “诶？主人住哪儿呀？”, "face": "curious"}') == {"reply": "诶？主人住哪儿呀？", "face": "curious"}
    assert brain._salvage("就是普通的一句话") == {"reply": "就是普通的一句话"}

    # 云端：没开 / 没 key 时不在工具列表里
    real = (config.CLOUD, cloud.ask, cloud._key, config.CLOUD_DAILY_LIMIT, config.WEB_SEARCH, config.TOOL_MODE)
    config.TOOL_MODE = "native"
    assert "ask_cloud" not in toolbox.available()
    asked = []
    try:
        config.CLOUD, config.CLOUD_DAILY_LIMIT = True, 2
        cloud._key = lambda: "test-key"
        assert "ask_cloud" in toolbox.available()
        # 额度：真实的 ask 会先扣次数；这里把网络请求换掉，只测扣次数
        orig_post = cloud.requests.post
        cloud.requests.post = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("no net"))
        try:
            assert cloud.ask("一") == "" and cloud.ask("二") == "" and cloud.used_today() == 2
            assert cloud.ask("三") == "" and cloud.used_today() == 2  # 额度用完，不再发
        finally:
            cloud.requests.post = orig_post

        cloud.ask = lambda q: asked.append(q) or "量子纠缠是两个粒子的状态互相关联，测一个就知道另一个。"
        db.set_profile("称呼", "小蔡", clock.fmt(clock.now()))

        class ToolLLM(FakeLLM):
            def __init__(self):
                self.seen = []

            def chat_tools(self, msgs, tools, temperature=0.7):
                self.seen.append((msgs, [t["function"]["name"] for t in tools]))
                tool_msgs = [m for m in msgs if m["role"] == "tool"]
                user = [m for m in msgs if m["role"] == "user"][-1]["content"]
                if not tool_msgs and "算" in user:
                    return {"role": "assistant", "content": "", "tool_calls": [
                        {"function": {"name": "calculator", "arguments": {"expression": "128*3+45"}}}]}
                if not tool_msgs and "量子" in user:
                    return {"role": "assistant", "content": "", "tool_calls": [
                        {"function": {"name": "ask_cloud", "arguments": {"question": "小蔡想知道量子纠缠是什么"}}}]}
                if tool_msgs:
                    return {"role": "assistant", "content": json.dumps(
                        {"reply": "嗯！" + tool_msgs[-1]["content"][:30], "face": "happy"}, ensure_ascii=False)}
                return {"role": "assistant", "content": '{"reply": "嗯嗯", "face": "happy"}'}

        brain.client = ToolLLM()
        seen = []
        r = brain.chat("帮我算一下 128*3+45", on_tool=lambda n, d: seen.append((n, d)))
        assert r["tools"] == [{"tool": "calculator", "detail": "128*3+45"}] and "429" in r["text"], r
        assert seen == [("calculator", "128*3+45")] and "searched" not in r

        brain.client = ToolLLM()
        r = brain.chat("量子纠缠是什么呀")
        assert asked == ["想知道量子纠缠是什么"], asked  # 主人的称呼抹掉了，只发这一句
        assert r["tools"] == [{"tool": "ask_cloud", "detail": "想知道量子纠缠是什么"}] and "量子纠缠" in r["text"], r
        sent = brain.client.seen[0][0]
        assert "ask_cloud" in sent[0]["content"] and "ask_cloud" in brain.client.seen[0][1]
        # 第二轮（拿到云端答案后）不能再问云端：每轮只问一次
        assert "ask_cloud" not in brain.client.seen[1][1], brain.client.seen[1][1]
    finally:
        (config.CLOUD, cloud.ask, cloud._key, config.CLOUD_DAILY_LIMIT, config.WEB_SEARCH, config.TOOL_MODE) = real


def test_nudge_when_says_will_search():
    """嘴上说“我去查查看”却没调工具：提醒一次后真的去查，再用查到的回答。"""
    import websearch

    class LazyLLM(SearchLLM):
        def chat_tools(self, msgs, tools, temperature=0.7):
            if not any(m["role"] == "system" and "调用 web_search" in m["content"] for m in msgs[1:]):
                return {"role": "assistant", "content": '{"reply": "这个团子不清楚，我得去查查看。", "face": "curious"}'}
            if any(m["role"] == "tool" for m in msgs):  # 查到了，正常回答
                return super().chat_tools(msgs, tools, temperature)
            return {"role": "assistant", "content": "",
                    "tool_calls": [{"function": {"name": "web_search", "arguments": {"query": "三角洲"}}}]}

    real = (websearch.search, websearch.available, config.TOOL_MODE, config.WEB_SEARCH)
    calls = []
    websearch.search = lambda q: calls.append(q) or [{"title": "三角洲行动", "body": "腾讯的射击游戏", "href": ""}]
    websearch.available = lambda: True
    config.TOOL_MODE, config.WEB_SEARCH = "native", True
    try:
        fresh_db("nudge.db")
        brain.client = LazyLLM()
        r = brain.chat("你知道三角洲吗")
        assert calls == ["三角洲"] and "三角洲行动" in r["text"], (calls, r)
    finally:
        websearch.search, websearch.available, config.TOOL_MODE, config.WEB_SEARCH = real


def test_reminder():
    """定时提醒：大脑调 set_reminder 定好（嘴上答应没调会被提醒一次）；到点精确到秒变成主动事件，
    不受免打扰和每日次数限制，也不算进主动次数；这句不再记成回访；大脑关太久错过的不补。"""
    import toolbox

    assert toolbox.span_text(30) == "30 秒" and toolbox.span_text(90) == "1 分钟 30 秒"
    assert toolbox.span_text(5400) == "1 小时 30 分钟"
    assert toolbox.reminder_text("提醒我去上厕所") == "去上厕所" and toolbox.reminder_text("该开会了") == "开会了"
    assert persona.reminder_line("去上厕所") == "叮——主人，时间到啦，该去上厕所啦！"
    # 问几点：把真实钟点直接交给大脑（真机上它无视【此刻】编了个“下午三点半”）
    from datetime import datetime as _dt
    assert persona.clock_text(_dt(2026, 10, 27, 3, 5)) == "凌晨 3 点 05 分"
    assert persona.clock_text(_dt(2026, 10, 27, 15, 0)) == "下午 3 点整"
    assert persona.clock_text(_dt(2026, 10, 27, 12, 30)) == "中午 12 点 30 分"
    assert "凌晨 03:05" in persona.time_line(_dt(2026, 10, 27, 3, 5))
    for q in ["现在几点啦？", "今天几号", "今天星期几呀"]:
        assert brain.ASKS_TIME.search(q), q
    assert not brain.ASKS_TIME.search("我今天好累")
    # 上一句隔了 40 分钟：告诉大脑那是旧话题（真机上问几点，它还接着聊 40 分钟前的上厕所）
    old = [{"role": "user", "ts": "2026-10-27 02:58:47", "text": "那你想办法上个厕所给我看"}]
    msgs = persona.chat_messages(state.compute(_dt(2026, 10, 27, 3, 38), None), _dt(2026, 10, 27, 3, 38, 48), [],
                                 old, "现在几点啦？")
    assert msgs[-1]["content"] == "现在几点啦？" and "40 分钟之前" in msgs[-2]["content"], msgs[-2:]
    msgs = persona.chat_messages(state.compute(_dt(2026, 10, 27, 3, 0), None), _dt(2026, 10, 27, 3, 0), [],
                                 old, "然后呢")
    assert msgs[-2]["role"] == "assistant" or msgs[-2]["content"] == old[0]["text"]  # 刚聊过的不插
    assert brain.wants_tools("30秒后提醒我上厕所") and brain.wants_tools("三点叫我开会")

    fresh_db("remind.db")
    clock.set_time("10:00")
    now = clock.now()
    assert toolbox.reminder_due({"seconds": 30}, now) == now + timedelta(seconds=30)
    assert toolbox.reminder_due({"minutes": 1, "seconds": 5}, now) == now + timedelta(seconds=65)
    assert toolbox.reminder_due({"at": "10:30"}, now) == now.replace(minute=30)
    assert toolbox.reminder_due({"at": "09:00"}, now) == now.replace(hour=9) + timedelta(days=1)  # 过了就是明天
    assert toolbox.reminder_due({"at": "下午"}, now) is None
    assert not toolbox.set_reminder({"text": "x", "hours": 24 * 30}).ok  # 太远
    assert not toolbox.set_reminder({"text": "x", "at": "2020-01-01 10:00"}).ok  # 已经过了
    assert db.pending_reminders() == []

    class RemindLLM(FakeLLM):
        def __init__(self, lazy=False):
            self.lazy = lazy

        def chat_tools(self, msgs, tools, temperature=0.7):
            tool_msgs = [m for m in msgs if m["role"] == "tool"]
            if tool_msgs:
                return {"role": "assistant", "content": '{"reply": "定好啦，到点我叫你～", "face": "happy"}'}
            nudged = any(m["role"] == "system" and "调用 set_reminder" in m["content"] for m in msgs[1:])
            if self.lazy and not nudged:  # 光嘴上答应（真机上就是这样没响的）
                return {"role": "assistant", "content": '{"reply": "好哒，30 秒后我提醒你～", "face": "happy"}'}
            assert "set_reminder" in [t["function"]["name"] for t in tools]
            return {"role": "assistant", "content": "", "tool_calls": [{"function": {
                "name": "set_reminder", "arguments": {"text": "提醒我去上厕所", "seconds": 30}}}]}

    real = (config.TOOL_MODE, brain.BACKGROUND_MEMORY)
    config.TOOL_MODE, brain.BACKGROUND_MEMORY = "native", False
    try:
        for lazy in (False, True):
            fresh_db(f"remind_{lazy}.db")
            clock.set_time("10:00")
            brain.client = RemindLLM(lazy)
            r = brain.chat("30秒后提醒我去上厕所")
            assert r["tools"] == [{"tool": "set_reminder", "detail": "30 秒后 去上厕所"}], r
            [rem] = db.pending_reminders()
            assert rem["text"] == "去上厕所" and rem["due_at"] == clock.fmt(clock.now() + timedelta(seconds=30)), rem
            assert db.pending_followups() == []  # 记忆整理员给了回访，这句也不加
    finally:
        config.TOOL_MODE, brain.BACKGROUND_MEMORY = real

    assert brain.fire_reminders() == []
    clock.advance(29 / 3600)
    assert brain.fire_reminders() == [] and take() == []
    clock.advance(1 / 3600)
    assert brain.fire_reminders() == ["叮——主人，时间到啦，该去上厕所啦！"]
    [ev] = take()
    assert ev["kind"] == "reminder" and ev["face"] == "peek" and ev["voice"] == "normal"
    assert brain.fire_reminders() == [] and db.pending_reminders() == []  # 只响一次
    assert db.count_events_on(clock.now().strftime("%Y-%m-%d")) == 0  # 不占每天主动找主人的次数

    # 免打扰时段照样响（主人自己定的）
    clock.set_time("23:59")
    db.add_reminder(clock.fmt(clock.now()), "关灯睡觉", clock.fmt(clock.now()))
    assert brain.fire_reminders() == ["叮——主人，时间到啦，该关灯睡觉啦！"]
    # 大脑关了三小时、错过的不补
    db.add_reminder(clock.fmt(clock.now() - timedelta(hours=3)), "关火", clock.fmt(clock.now()))
    assert brain.fire_reminders() == [] and db.pending_reminders() == []


def test_traits():
    """慢慢长成的性格：每天变化有上限；会回落的慢慢回到天生样子；很久不见会生分；数值翻译成话由代码做。"""
    import traits as T

    st = T.new_state("2026-10-01")
    for _ in range(200):  # 一天里被逗了 200 次
        st = T.apply(st, "逗团子", "2026-10-01")
    v = st["values"]
    assert v["tsundere"] - T.BASELINE["tsundere"] <= T.DAILY_CAP + 1e-6, v  # 一天最多变 DAILY_CAP
    assert v["tsundere"] > T.BASELINE["tsundere"] and v["playful"] > T.BASELINE["playful"]
    up = v["tsundere"]
    st = T.apply(st, "逗团子", "2026-10-02")  # 第二天额度重置，还能再涨
    assert st["values"]["tsundere"] > up
    for d in range(3, 31):  # 一个月天天被逗：慢慢变得很傲娇，但不会超过 1
        for _ in range(10):
            st = T.apply(st, "逗团子", f"2026-10-{d:02d}")
    assert 0.6 < st["values"]["tsundere"] < 1.0, st["values"]
    assert "嘴特别硬" in T.describe(st["values"])
    # 之后再也不逗：慢慢回到天生的样子（但不会一下子回去）
    later = T.roll_day(st, "2026-11-30")
    assert T.BASELINE["tsundere"] < later["values"]["tsundere"] < st["values"]["tsundere"]
    # 亲密：攒出来的，常来就涨；很久没见面会降，但不低于天生值
    st = T.new_state("2026-10-01")
    for d in range(1, 31):
        st = T.apply(st, "chat", f"2026-10-{d:02d}", times=20)
        st = T.apply(st, "pat", f"2026-10-{d:02d}", times=5)
    close = st["values"]["intimacy"]
    assert close > 0.5, close
    gone = T.roll_day(st, "2026-11-20")  # 20 天没见
    assert T.BASELINE["intimacy"] <= gone["values"]["intimacy"] < close
    assert T.roll_day(st, "2026-10-31")["values"]["intimacy"] == close  # 只隔一天、昨天见过：不算没见面
    assert "认识还不久" in T.describe(T.BASELINE) and "非常亲密" in T.describe({**T.BASELINE, "intimacy": 0.8})


def test_episodic_memory():
    """情景记忆：记住、同一件事合并、按线索想起（关键词加分）、聊天时放进提示词；相处方式推动性格。"""
    fresh_db("episodic.db")
    clock.set_time("10:00")
    now = clock.now()
    real_min = config.MEMORY_MIN_SIM
    config.MEMORY_MIN_SIM = 0.15  # 假向量模型的相似度普遍偏低（只有一个词重合时约 0.12 + 关键词加分 0.08）
    try:
        assert recall.remember("owner", "主人家的猫咪咪把耳机咬坏了", "无奈", 3, now)[0] == "added"
        assert recall.remember("owner", "主人在准备十二月的考研考试", "焦虑", 4, now)[0] == "added"
        assert recall.remember("us", "主人给团子起了外号叫小糯米", "开心", 4, now)[0] == "added"
        how, _ = recall.remember("owner", "主人家的猫咪咪把耳机咬坏了", "生气", 5, now)
        assert how == "reinforced"  # 同一件事再提起：更新原来那条
        rows = db.active_memories(with_embedding=False)
        assert len(rows) == 3 and rows[0]["mentions"] == 2 and rows[0]["importance"] == 5, rows
        assert recall.remember("owner", "嗯", "", 3, now)[0] == "skipped"  # 太短的不记

        got = recall.relevant("咪咪今天又闹了", now)
        assert got and "咪咪" in got[0]["text"], got  # 出现同一个名字：想起来
        assert recall.relevant("今天天气怎么样", now) == []  # 不相关的不硬塞
        assert db.active_memories(with_embedding=False)[0]["recall_count"] == 1

        # 聊天时：勾起的往事放进提示词；提炼出的新记忆和相处方式写进去
        seen = {}

        class MemLLM(FakeLLM):
            def chat_tools(self, msgs, tools, temperature=0.7):
                seen["system"] = msgs[0]["content"]
                return super().chat_tools(msgs, tools, temperature)

            def chat_json(self, msgs, temperature=0.0):
                if "记忆整理员" in msgs[0]["content"]:
                    return {"memories": [{"text": "主人周末要去海边看日出", "kind": "owner", "importance": 3}],
                            "facts": [], "profile": {}, "interaction": "逗团子"}
                seen["system"] = msgs[0]["content"]
                return {"reply": "嘿嘿", "face": "happy"}

        brain.client, brain.BACKGROUND_MEMORY = MemLLM(), False
        before = memory.traits_now(now)["values"]["tsundere"]
        r = brain.chat("咪咪今天又把我的耳机咬了")
        assert "这句话让你想起的往事" in seen["system"] and "咪咪把耳机咬坏了" in seen["system"], seen["system"]
        assert r["recalled"] and "咪咪" in r["recalled"][0]
        assert "你和主人这阵子的相处" in seen["system"]
        assert any("海边看日出" in m["text"] for m in db.active_memories(with_embedding=False))
        assert memory.traits_now(now)["values"]["tsundere"] > before  # 被逗了：更傲娇一点
        brain.record_device_event("pat", 3)
        assert memory.traits_now(now)["values"]["clingy"] > traits_baseline("clingy")
    finally:
        config.MEMORY_MIN_SIM = real_min


def test_nightly():
    """夜间整理：遗忘曲线；同一件事合成一句（变化写进去）、不同的事分开留；记不清的淡忘、再提起会想起来；
    写日记、今天最想记住的事、自我认识；只在睡觉时段、每晚一次；自我认识进聊天提示词。"""
    import nightly

    fresh_db("nightly.db")
    clock.set_time("10:00")
    now = clock.now()
    # 遗忘曲线（纯函数）
    row = {"importance": 3, "mentions": 1, "recall_count": 0, "recalled_at": None,
           "updated_at": clock.fmt(now - timedelta(days=14))}
    assert abs(recall.retention(row, now) - 0.5) < 1e-6  # 3 级：14 天记得一半
    assert recall.retention({**row, "importance": 5}, now) == 1.0  # 5 级永远记得
    assert recall.retention({**row, "mentions": 3}, now) > 0.5  # 常被提起记得更牢
    assert not recall.should_fade({**row, "updated_at": clock.fmt(now - timedelta(days=30))}, now)
    assert recall.should_fade({**row, "updated_at": clock.fmt(now - timedelta(days=45))}, now)
    assert recall.should_fade({**row, "importance": 1, "updated_at": clock.fmt(now - timedelta(days=10))}, now)

    real = (config.MEMORY_MIN_SIM, config.MEMORY_MERGE_SIM)
    config.MEMORY_MIN_SIM, config.MEMORY_MERGE_SIM = 0.15, 0.5  # 假向量模型的相似度偏低
    try:
        recall.remember("owner", "主人的妈妈住院了", "担心", 4, now - timedelta(days=2))
        recall.remember("owner", "主人的妈妈出院了", "开心", 4, now)
        recall.remember("owner", "主人家的猫咪咪把耳机咬坏了", "", 3, now)
        recall.remember("owner", "主人家的猫咪咪把拖鞋咬坏了", "", 3, now)
        recall.remember("owner", "主人中午吃了一碗牛肉面", "", 1, now - timedelta(days=12))  # 小事，很久没提
        db.add_message("user", "以后叫你小糯米吧", clock.fmt(now))
        db.add_message("pet", "哼，才不要呢", clock.fmt(now))

        class NightLLM(FakeLLM):
            def chat_json(self, msgs, temperature=0.0):
                sys_ = msgs[0]["content"]
                if "整理记忆" in sys_:
                    if "妈妈" in msgs[1]["content"]:
                        return {"same": True, "text": "主人的妈妈住过院，后来出院了"}
                    return {"same": False, "text": ""}  # 咪咪咬耳机、咬拖鞋是两件事
                if "小日记" in sys_:
                    assert "以后叫你小糯米吧" in msgs[1]["content"]
                    return {"diary": "今天主人说要叫我小糯米，我嘴上说不要，其实偷偷开心了一下午。😊",
                            "highlight": "主人第一次叫我小糯米", "importance": 4}
                if "我是一个怎样的" in sys_:
                    return {"self_view": "我是住在板子里的团子，嘴硬心软，主人叫我小糯米。"}
                return super().chat_json(msgs, temperature)

        brain.client = NightLLM()
        assert nightly.maybe_run(brain.client, now) is None  # 白天不整理
        clock.set_time("23:45")
        night = clock.now()
        rep = nightly.maybe_run(brain.client, night)
        assert rep and rep["day"] == night.strftime("%Y-%m-%d"), rep
        texts = [m["text"] for m in db.active_memories(with_embedding=False)]
        assert "主人的妈妈住过院，后来出院了" in texts and "主人的妈妈住院了" not in texts, texts
        assert "主人家的猫咪咪把耳机咬坏了" in texts and "主人家的猫咪咪把拖鞋咬坏了" in texts  # 不同的事都留着
        assert "主人中午吃了一碗牛肉面" in rep["faded"] and "主人中午吃了一碗牛肉面" not in texts
        assert "主人第一次叫我小糯米" in texts  # 今天最想记住的事成了一条往事
        diary = db.recent_diary(1)[0]
        assert "小糯米" in diary["text"] and "😊" not in diary["text"]
        assert db.get_kv("self_view") == "我是住在板子里的团子，嘴硬心软，主人叫我小糯米。"
        assert nightly.maybe_run(brain.client, night + timedelta(hours=2)) is None  # 一晚只整理一次
        # 淡忘的事再提起：重新想起来，而不是当成新的事
        assert recall.remember("owner", "主人中午吃了一碗牛肉面", "", 1, night)[0] == "revived"

        # 自我认识进聊天提示词
        seen = {}

        class SeeLLM(FakeLLM):
            def chat_json(self, msgs, temperature=0.0):
                if "记忆整理员" not in msgs[0]["content"]:
                    seen["system"] = msgs[0]["content"]
                return super().chat_json(msgs, temperature)

        clock.set_time("10:00")
        brain.client = SeeLLM()
        brain.chat("早呀")
        assert "【你对自己的认识】我是住在板子里的团子" in seen["system"], seen["system"][:300]
    finally:
        config.MEMORY_MIN_SIM, config.MEMORY_MERGE_SIM = real


def test_memory_admin():
    """记忆管理：说“忘掉……”真删记忆、原话和回访（不误伤别的）；管理页的查看、修改、删除、改性格和自我认识、导出。"""
    from fastapi.testclient import TestClient
    import memory_admin as MA
    import server

    for s in ["把我妈住院的事忘掉", "忘掉这件事", "这个别记了", "不要记这个", "帮我把刚才那句忘了吧"]:
        assert MA.is_forget_request(s), s
    for s in ["你是不是忘了我的生日", "你会忘记我吗", "我忘带钥匙了", "今天好累"]:
        assert not MA.is_forget_request(s), s

    fresh_db("admin.db")
    clock.set_time("10:00")
    now = clock.now()
    real_min = (config.MEMORY_MIN_SIM, config.FORGET_MIN_SIM)
    config.MEMORY_MIN_SIM, config.FORGET_MIN_SIM = 0.15, 0.15
    try:
        recall.remember("owner", "主人的妈妈住院了", "担心", 4, now)
        recall.remember("owner", "主人家的猫咪咪把耳机咬坏了", "", 3, now)
        db.add_followup("主人的妈妈住院了", "担心", clock.fmt(now + timedelta(hours=8)), "问问妈妈怎么样", clock.fmt(now))
        db.add_message("user", "我妈妈今天住院了", clock.fmt(now))
        db.add_message("pet", "别担心，我陪着你", clock.fmt(now))

        class ForgetLLM(FakeLLM):
            def chat_json(self, msgs, temperature=0.0):
                assert "记忆整理员" not in msgs[0]["content"], "忘掉的这一轮不该提炼记忆"
                return {"reply": "嗯，忘掉啦。", "face": "shy"}

            def chat_tools(self, msgs, tools, temperature=0.7):
                raise AssertionError("忘掉的这一轮不给工具")

        brain.client = ForgetLLM()
        r = brain.chat("把我妈住院的事忘掉吧")
        assert r["text"] == "嗯，忘掉啦。" and r["forgot"]["memories"] == 1, r
        left = [m["text"] for m in db.active_memories(with_embedding=False)]
        assert left == ["主人家的猫咪咪把耳机咬坏了"], left  # 只删相关的
        assert db.pending_followups() == []
        texts = [m["text"] for m in db.recent_messages(20)]
        assert "我妈妈今天住院了" not in texts and "别担心，我陪着你" not in texts, texts  # 原话和团子的回答都删了
        assert f"（主人让{config.PET_NAME}忘掉了一件事）" in texts and "把我妈住院的事忘掉吧" not in texts

        # 只说“忘掉这件事”：指上一句
        brain.client = FakeLLM()
        db.add_message("user", "咪咪又把耳机咬坏了", clock.fmt(now))
        brain.client = ForgetLLM()
        r = brain.chat("忘掉这件事")
        assert r["forgot"]["memories"] == 1 and db.active_memories(with_embedding=False) == [], r
        assert "咪咪又把耳机咬坏了" not in [m["text"] for m in db.recent_messages(20)]
    finally:
        config.MEMORY_MIN_SIM, config.FORGET_MIN_SIM = real_min

    # 管理页接口
    recall.remember("us", "主人给团子起了外号叫小糯米", "", 4, now)
    db.set_profile("城市", "北京", clock.fmt(now))
    db.set_diary("2026-10-01", "今天主人叫我小糯米", clock.fmt(now))
    with TestClient(server.app) as c:
        ov = c.get("/api/memories").json()
        assert ov["memories"][0]["text"] == "主人给团子起了外号叫小糯米" and "embedding" not in ov["memories"][0]
        assert 0 < ov["memories"][0]["retention"] <= 1 and set(ov["traits"]) == set(ov["traits_baseline"])
        mid = ov["memories"][0]["id"]
        old_emb = db.get_memory(mid)["embedding"]
        row = c.patch(f"/api/memories/{mid}", json={"text": "主人叫团子小糯米", "importance": 5}).json()
        assert row["text"] == "主人叫团子小糯米" and row["importance"] == 5
        assert db.get_memory(mid)["embedding"] != old_emb  # 改了文字就重新算向量
        assert c.patch("/api/memories/99999", json={"text": "x"}).status_code == 404
        assert c.put("/api/traits", json={"values": {"clingy": 1.7, "bogus": 0.3}}).json()["traits"]["clingy"] == 1.0
        assert c.put("/api/self_view", json={"text": "我是小糯米"}).json()["self_view"] == "我是小糯米"
        exp = c.get("/api/export")
        assert exp.status_code == 200 and "attachment" in exp.headers["content-disposition"]
        backup = exp.json()
        assert backup["memories"][0]["text"] == "主人叫团子小糯米" and "embedding" not in backup["memories"][0]
        assert backup["kv"]["self_view"] == "我是小糯米" and backup["kv"]["traits"]["values"]["clingy"] == 1.0
        assert any(d["day"] == "2026-10-01" for d in backup["diary"])
        assert c.delete(f"/api/memories/{mid}").json() == {"ok": True} and db.get_memory(mid) is None
        assert c.delete("/api/profile/城市").json() == {"ok": True} and db.get_profile() == []
        assert c.delete("/api/diary/2026-10-01").json() == {"ok": True} and db.recent_diary(5) == []
        assert c.get("/memory", follow_redirects=False).headers["location"] == "/static/memory.html"


def test_relay_security():
    """从公网（隧道）进来的：HTTP 接口一律 403（记忆、导出不外泄），/ws 的 hello 必须带对设备密钥；局域网照常。"""
    from fastapi.testclient import TestClient
    import relay
    import server

    fresh_db("relay.db")
    real = (config.RELAY_LOCAL_HOST, relay.TOKEN_FILE)
    relay.TOKEN_FILE = TMP / "device_token.txt"
    token = relay.device_token()
    assert len(token) >= 24 and relay.device_token() == token  # 第一次生成，之后不变
    try:
        with TestClient(server.app) as c:
            assert c.get("/api/memories").status_code == 200  # 局域网
            with c.websocket_connect("/ws") as ws:
                ws.send_json({"type": "hello", "device": "web"})
                assert ws.receive_json()["type"] == "welcome"

            config.RELAY_LOCAL_HOST = "testserver"  # TestClient 的连接都当成从隧道进来的
            for path in ("/api/memories", "/api/export", "/memory", "/static/memory.html"):
                assert c.get(path).status_code == 403, path
            with c.websocket_connect("/ws") as ws:
                ws.send_json({"type": "hello", "device": "esp32", "token": "猜的"})
                assert ws.receive_json() == {"type": "error", "message": "密钥不对"}
            with c.websocket_connect("/ws") as ws:
                ws.send_json({"type": "hello", "device": "esp32", "token": token})
                assert ws.receive_json()["type"] == "welcome"
    finally:
        config.RELAY_LOCAL_HOST, relay.TOKEN_FILE = real


def test_llm_load_retry():
    """模型现加载时显卡偶发 CUDA 错误，Ollama 返回 500：等一下再试一次；请求都带 keep_alive，别 5 分钟就卸掉模型。"""
    import llm

    class Resp:
        def __init__(self, code):
            self.status_code = code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"{self.status_code} Server Error")

        def json(self):
            return {"message": {"content": "好"}}

    codes, seen = [500, 200], []
    c = llm.OllamaClient()
    c.num_gpu, c.capabilities = (lambda: None), (lambda: set())
    real = (llm.requests.post, config.LOAD_RETRY_S)
    llm.requests.post = lambda url, json, timeout: seen.append(json) or Resp(codes.pop(0))
    config.LOAD_RETRY_S = 0
    try:
        assert c.chat([{"role": "user", "content": "嗯"}]) == "好" and len(seen) == 2
        assert seen[0]["keep_alive"] == config.KEEP_ALIVE
        codes[:] = [500, 500]
        try:
            c.chat([{"role": "user", "content": "嗯"}])
            raise AssertionError("两次都失败应该抛出去，让大脑改用云端代答")
        except RuntimeError:
            pass
    finally:
        llm.requests.post, config.LOAD_RETRY_S = real


def test_cloud_standin():
    """本地大脑（Ollama）没开：云端照同一套人设、最近几句和被勾起的记忆代答；额度用完就老实说没连上。"""
    import cloud
    fresh_db("standin.db")
    clock.set_time("10:00")
    now = clock.now()
    recall.remember("owner", "主人家的猫咪咪把耳机咬坏了", "", 3, now)
    db.add_message("pet", "主人早呀", clock.fmt(now))  # 开头是团子的主动搭话

    class DownLLM(FakeLLM):
        def is_up(self, connect_timeout=None):
            return False

        def chat_json(self, msgs, temperature=0.0):
            raise ConnectionError("ollama 没开")

        chat_tools = chat = lambda self, *a, **k: (_ for _ in ()).throw(ConnectionError("ollama 没开"))

    replies, sent = [], []

    class Resp:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            pass

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": self.text}]}}]}

    real = (config.CLOUD, cloud._key, cloud.requests.post, config.CLOUD_STANDIN_DAILY, brain.client, recall.embed)
    config.CLOUD, cloud._key = True, lambda: "test-key"
    cloud.requests.post = lambda url, headers, json, timeout: sent.append(json) or Resp(replies.pop(0))
    recall.embed = lambda texts: (_ for _ in ()).throw(AssertionError("代答时不该去算向量（Ollama 没开）"))
    brain.client = DownLLM()
    try:
        replies[:] = ['{"reply": "我是一个AI助手。", "face": "idle"}', '{"reply": "咪咪又闹啦？耳机还好吗。", "face": "curious"}']
        r = brain.chat("咪咪今天又闹了")
        assert r["brain"] == "cloud" and r["text"] == "咪咪又闹啦？耳机还好吗。" and r["face"] == "curious", r
        assert len(sent) == 2  # 出戏重试了一次
        body = sent[0]
        system = body["systemInstruction"]["parts"][0]["text"]
        assert "住在主人随身带着的小电路板里" in system and "咪咪把耳机咬坏了" in system, system  # 人设 + 按词想起的记忆
        assert "你刚才对主人说过：主人早呀" in system  # 开头团子的话并进背景
        assert body["contents"][0]["role"] == "user" and body["contents"][-1]["parts"][0]["text"] == "咪咪今天又闹了"
        assert body["generationConfig"]["responseMimeType"] == "application/json" and "tools" not in body
        assert "x-goog-api-key" not in json.dumps(body)
        assert len(db.active_memories(with_embedding=False)) == 1  # 代答时不提炼新记忆，先攒着
        assert [m["text"] for m in db.recent_messages(3)][-1] == "咪咪又闹啦？耳机还好吗。"
        assert memory.backlog_size() == 1
        said_at = db.recent_messages(2)[0]["ts"]
        replies[:] = ['{"reply": "嗯嗯，我在听。", "face": "shy"}']
        brain.chat("这句后来被删掉了")
        db.delete_messages([m["id"] for m in db.recent_messages(2)])  # 像主人让它忘掉了一样
        assert memory.backlog_size() == 2

        # 主动搭话也能代为措辞
        replies[:] = ["你还在忙吗？我探个头。"]
        text = brain._compose(rules.Decision("lonely", "孤单", "peek", "nudge"), brain.current_state(now), now)
        assert text == "你还在忙吗？我探个头。", text

        # 代答额度用完：不再发，老实说没连上
        config.CLOUD_STANDIN_DAILY = cloud.used_today("cloud_standin_usage")
        n = len(sent)
        r = brain.chat("在吗")
        assert len(sent) == n and "brain" not in r and r["text"] == persona.fallback_line("offline"), r
        assert memory.backlog_size() == 2  # 没答上来的那轮不用补
        assert memory.catch_up(DownLLM(), now) == 0 and memory.backlog_size() == 2  # 本地还没回来：留着下次补

        # 本地大脑回来：照常走本地、不发云端，顺带按当时的时间补记攒下的对话（被删的那轮不补）
        heard = []

        class LocalLLM(FakeLLM):
            def chat_json(self, msgs, temperature=0.0):
                if "记忆整理员" not in msgs[0]["content"]:
                    return super().chat_json(msgs, temperature)
                heard.append(msgs[1]["content"])
                text = "咪咪今天又闹腾了一回" if "咪咪" in msgs[1]["content"] else "主人今天上班很累"
                return {"memories": [{"text": text, "kind": "owner", "importance": 3}], "interaction": "普通"}

        config.CLOUD_STANDIN_DAILY = 300
        brain.client, recall.embed = LocalLLM(), fake_embed
        r = brain.chat("今天好累")
        assert len(sent) == n and "brain" not in r, r
        assert memory.backlog_size() == 0 and len(heard) == 2, heard
        assert "今天好累" in heard[0] and "咪咪今天又闹了" in heard[1] and "被删掉" not in "".join(heard), heard
        row = next(m for m in db.active_memories(with_embedding=False) if m["text"] == "咪咪今天又闹腾了一回")
        assert row["created_at"] == said_at, (row, said_at)

        # Ollama 开着、本地模型却出错（加载失败返回 500）：这一句也让云端代答
        class BrokenLLM(FakeLLM):
            def chat_json(self, msgs, temperature=0.0):
                raise RuntimeError("500 Server Error: Internal Server Error")

            chat_tools = chat = lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("500 Server Error"))

        brain.client = BrokenLLM()
        replies[:] = ['{"reply": "嗯嗯，团子在呢。", "face": "happy"}']
        r = brain.chat("在吗")
        assert r["brain"] == "cloud" and r["text"] == f"嗯嗯，{config.PET_NAME}在呢。", r  # 旧名字也被换掉了
        assert "tools" not in sent[-1] and memory.backlog_size() == 1  # 这一轮也攒着等本地好了再补记
    finally:
        (config.CLOUD, cloud._key, cloud.requests.post, config.CLOUD_STANDIN_DAILY, brain.client, recall.embed) = real


def traits_baseline(k):
    import traits as T
    return T.BASELINE[k]


def test_conversation_end():
    """连续对话：主人道别时这句回复带 end（设备说完不再接着听），团子被提醒好好道别；平常的话不带。"""
    for s in ["拜拜啦", "晚安团子", "好啦不聊了，我去洗澡", "我先去忙了", "下次再聊", "我要睡觉了", "先这样吧"]:
        assert brain.ends_conversation(s), s
    for s in ["你在干嘛呀", "我病好了", "没事就好", "在想我什么呀", "我刚睡醒"]:
        assert not brain.ends_conversation(s), s
    # 已知局限：“你会说再见吗”这种问句也会被当成道别（只是这段对话提前结束，再喊一声就行）

    fresh_db("bye.db")
    seen = {}

    class ByeLLM(FakeLLM):
        def chat_json(self, msgs, temperature=0.0):
            if "记忆整理员" not in msgs[0]["content"]:
                seen["msgs"] = msgs
            return super().chat_json(msgs, temperature)

    brain.client = ByeLLM()
    clock.set_time("10:00")
    assert "end" not in brain.chat("你在干嘛呀")
    r = brain.chat("好啦我去忙了，拜拜")
    assert r["end"] is True and any(brain.FAREWELL_NOTE in m["content"] for m in seen["msgs"]), r
    from link import say_msg
    assert say_msg(r, proactive=False)["end"] is True and "end" not in say_msg({"text": "嗯"}, proactive=False)


def test_sleepy_chat():
    fresh_db("sleepy.db")
    brain.client = FakeLLM()
    clock.set_time("01:30")
    r = brain.chat("还在写代码")
    assert r["face"] == "sleepy" and r["voice"] == "sleepy", r


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("✓", t.__name__)
    print(f"全部通过（{len(tests)} 项），临时数据在 {TMP}")
