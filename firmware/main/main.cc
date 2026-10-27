// 团子固件入口（第二阶段骨架）。
//
// 现在能做的：
//   1. 屏幕上显示八条风表情，待机会眨眼、东张西望；摸屏幕会开心冒爱心。
//   2. 连上 WiFi 和 PC 大脑（WebSocket，协议见 ai_pet/PROTOCOL.md）：
//      基础表情跟着大脑的状态走（犯困、睡着、孤单），摸头会上报，大脑主动来找你时会探头。
//   3. 对着它聊天：说“Hi，喵喵”（或者按住屏幕）→ 竖耳朵倾听，把你说的话（Opus）传给大脑，说完自动停 →
//      思考 → 用猪咪的声音（大脑合成、Opus 推过来）回答你 → 接着听（连续对话），没人接话或道别就结束。
//   4. 没连上大脑时喊唤醒词只会迷糊地歪歪头；“鹦鹉学舌”自检（录 3 秒再放出来）留着，kParrotSelfTest 打开。
// 还没做的：配网页面、震动、省电。见 README。
#include <algorithm>
#include <atomic>
#include <deque>
#include <mutex>

#include <esp_log.h>
#include <esp_random.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>
#include <nvs_flash.h>

#include "audio_service.h"
#include "face_display.h"
#include "lichuang_board.h"
#include "pet_link.h"

#define TAG "Pet"

namespace {

using face::FaceId;

enum class Event : uint8_t {
    kWakeWord, kPlaybackDrained, kPat, kSay, kThinking, kRefresh,
    kMicPacket, kHeard, kTtsStart, kTtsStop, kHoldStart, kHoldEnd,
};

QueueHandle_t g_events;
AudioService g_audio;
face::FaceDisplay g_face;

// 网络任务塞进来的数据，主循环取走
std::mutex g_mu;
net::BrainState g_brain;
bool g_has_brain = false;
std::deque<net::Say> g_says;
std::string g_heard;
bool g_tts_ok = true;

// 团子的声音：网络任务收到 Opus 包直接塞进解码队列（不经过主循环，免得卡顿）
std::atomic<bool> g_accept_audio{false};  // 主人开口后就不再收上一段没推完的声音
std::atomic<bool> g_listening{false};     // 正在听主人说话：这时推来的（上一轮迟到的）声音不播
std::atomic<int> g_tts_rate{24000}, g_tts_frame_ms{60};

// 倾听：说完停顿这么久就算说完了；一直不开口、或者说太久，也收工
constexpr int64_t kEndSilenceUs = 1200 * 1000;  // 说过话之后停顿这么久算说完
// 刚唤醒的这一小段不算（“小智”的尾音、喇叭回声会被人声检测当成开口，接着主人一停顿就被当成说完了）；
// 太短的一下也不算说话（咔哒声、咳一下）
constexpr int64_t kWakeTailUs = 500 * 1000;
// 连续对话：团子说完一句后接着听。喇叭刚停，开头 0.3 秒不算（防余音）；6 秒没接话，这段对话就结束
// 接着听时判断“开口”更严：头 0.6 秒不算（喇叭余音），连续 0.5 秒以上的人声才算开口——
// 0.3 秒 / 0.2 秒时，环境杂音一直被当成说话，“没人接话就结束”永远等不到，对着杂音聊个没完（真机踩过）
constexpr int64_t kFollowTailUs = 600 * 1000;
constexpr int64_t kFollowNoSpeechUs = 6 * 1000 * 1000;
constexpr int64_t kFollowMinSpeechUs = 500 * 1000;
constexpr int64_t kMinSpeechUs = 200 * 1000;
constexpr int64_t kNoSpeechUs = 5 * 1000 * 1000;
constexpr int64_t kMaxListenUs = 12 * 1000 * 1000;
constexpr int64_t kReplyTimeoutUs = 90 * 1000 * 1000;  // 大脑冷启动要加载模型，等久一点
// 没连上大脑时喊唤醒词要不要“鹦鹉学舌”（录 3 秒再放出来）。语音链路已经验证过，平时关着
constexpr bool kParrotSelfTest = false;
constexpr int64_t kAckTimeoutUs = 2 * 1000 * 1000;  // 应答最多等这么久就开始听（万一“播完了”的通知没来）

void Post(Event e) { xQueueSend(g_events, &e, 0); }

void InitNvs() {
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);
}

// 基础表情：和网页团子 pet.html 的 baseFace() 同一套规则
FaceId BaseFace() {
    auto& link = net::PetLink::Get();
    if (!link.Online()) {
        int h = link.BrainHour();
        if (h < 0) return FaceId::kIdle;  // 还没连上过大脑
        return (h >= 23 || h < 7) ? FaceId::kSleep : FaceId::kLonely;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_has_brain) return FaceId::kIdle;
    if (g_brain.phase == "asleep") return FaceId::kSleep;
    if (g_brain.phase == "sleepy") return FaceId::kSleepy;
    if (g_brain.loneliness >= 0.8f) return FaceId::kLonely;
    return FaceId::kIdle;
}

void StartLink() {
    auto& link = net::PetLink::Get();
    link.on_state = [](const net::BrainState& s) {
        {
            std::lock_guard<std::mutex> lock(g_mu);
            g_brain = s;
            g_has_brain = true;
        }
        Post(Event::kRefresh);
    };
    link.on_say = [](const net::Say& s) {
        {
            std::lock_guard<std::mutex> lock(g_mu);
            g_says.push_back(s);
        }
        Post(Event::kSay);
    };
    link.on_thinking = [] { Post(Event::kThinking); };
    link.on_online = [](bool) { Post(Event::kRefresh); };
    link.on_heard = [](const std::string& text) {
        {
            std::lock_guard<std::mutex> lock(g_mu);
            g_heard = text;
        }
        Post(Event::kHeard);
    };
    link.on_tts_start = [](int rate, int frame_ms) {
        if (g_listening) return;
        g_tts_rate = rate;
        g_tts_frame_ms = frame_ms;
        g_accept_audio = true;
        Post(Event::kTtsStart);
    };
    link.on_audio = [](const uint8_t* data, size_t len) {
        if (!g_accept_audio) return;
        auto packet = std::make_unique<AudioStreamPacket>();
        packet->sample_rate = g_tts_rate;
        packet->frame_duration = g_tts_frame_ms;
        packet->payload.assign(data, data + len);
        // 大脑合成得比播放快：解码队列满了就在这里等一会儿（每次最多一包的时间），相当于按播放速度收
        g_audio.PushPacketToDecodeQueue(std::move(packet), true);
    };
    link.on_tts_stop = [](bool ok) {
        {
            std::lock_guard<std::mutex> lock(g_mu);
            g_tts_ok = ok;
        }
        Post(Event::kTtsStop);
    };
    link.Start();
}

// 被唤醒时的应答：编译时嵌进固件的 Ogg Opus（猪咪的声音，ai_pet/tools/make_wake_acks.py 生成）
#define WAKE_ACK(n)                                                     \
    extern const char wake_ack_##n##_start[] asm("_binary_wake_ack_" #n "_ogg_start"); \
    extern const char wake_ack_##n##_end[] asm("_binary_wake_ack_" #n "_ogg_end");
WAKE_ACK(0)
WAKE_ACK(1)
WAKE_ACK(2)
WAKE_ACK(3)

std::string_view RandomWakeAck() {  // “在呢～”“怎么啦？”“嗯？”“我在呢。”随机一句
    static const std::string_view acks[] = {
        {wake_ack_0_start, size_t(wake_ack_0_end - wake_ack_0_start)},
        {wake_ack_1_start, size_t(wake_ack_1_end - wake_ack_1_start)},
        {wake_ack_2_start, size_t(wake_ack_2_end - wake_ack_2_start)},
        {wake_ack_3_start, size_t(wake_ack_3_end - wake_ack_3_start)},
    };
    return acks[esp_random() % 4];
}

enum class Mode {
    kIdle,
    kAcking,                               // 被叫醒，先应一声（应完再听，免得把自己的声音录进去）
    kRecording, kReplaying,               // 鹦鹉学舌（没连大脑时）
    kListening, kWaitingReply, kSpeaking,  // 和大脑聊天
};

}  // namespace

extern "C" void app_main() {
    InitNvs();
    // AFE 缓冲区一满，esp-sr 每 50ms 打一条警告，115200 波特率下每条要发近 10ms，
    // 刷屏本身就占 CPU，让唤醒词更追不上（真机踩过）：只留错误
    esp_log_level_set("AFE", ESP_LOG_ERROR);
    g_events = xQueueCreate(16, sizeof(Event));

    auto& board = board::LichuangBoard::Get();
    board.Init();
    g_face.on_pat = [] { Post(Event::kPat); };
    g_face.on_hold_start = [] { Post(Event::kHoldStart); };
    g_face.on_hold_end = [] { Post(Event::kHoldEnd); };
    g_face.Start(&board);

    // 语音：回调都在语音任务里触发，只投递事件，不在回调里干活
    AudioServiceCallbacks cb;
    cb.on_wake_word_detected = [](const std::string& word) {
        ESP_LOGI(TAG, "wake word: %s", word.c_str());
        Post(Event::kWakeWord);
    };
    cb.on_playback_drained = [] { Post(Event::kPlaybackDrained); };
    cb.on_send_queue_available = [] { Post(Event::kMicPacket); };
    g_audio.Initialize(board.Audio());
    g_audio.SetCallbacks(cb);
    g_audio.Start();
    g_audio.EnableWakeWordDetection(true);

    StartLink();
    ESP_LOGI(TAG, "ready");

    Mode mode = Mode::kIdle;
    int64_t record_until = 0, listen_start = 0, silence_since = 0, speech_since = 0, wait_start = 0, ack_start = 0;
    bool heard_voice = false, tts_done = false;
    bool follow_up = false;  // 这次倾听是连续对话里的“接着听”（不是喊唤醒词开始的）
    bool held = false;       // 按住屏幕说话：松手才算说完，不靠停顿判断
    bool convo_end = false;  // 刚才那句 say 带了 end：主人道别了，说完不再接着听
    int pending_pats = 0;
    auto& link = net::PetLink::Get();

    auto back_to_idle = [&] {
        mode = Mode::kIdle;
        g_accept_audio = false;
        g_listening = false;
        g_audio.EnableWakeWordDetection(true);
    };
    // 开始听主人说话：喊唤醒词开始的，或者团子说完一句后接着听（连续对话，不用再喊“你好小智”）
    auto start_listening = [&](bool follow) {
        g_listening = true;
        g_accept_audio = false;
        g_audio.EnableWakeWordDetection(false);
        g_audio.EnableVoiceProcessing(true);  // 开始录音（带回声消除），顺带清掉还没播完的声音
        link.ListenStart();                   // 大脑收到就不再往这台设备推上一段声音（tts stop 带 cancelled）
        g_face.SetBaseFace(FaceId::kListen);
        listen_start = esp_timer_get_time();
        silence_since = speech_since = 0;
        heard_voice = false;
        follow_up = follow;
        held = false;
        mode = Mode::kListening;
    };
    // 团子说完一句：主人没道别就竖起耳朵接着听，道别了（或者断线了）就结束这段对话
    auto after_speaking = [&] {
        if (!convo_end && link.Online()) {
            ESP_LOGI(TAG, "listening for a follow-up");
            start_listening(true);
        } else {
            if (convo_end) ESP_LOGI(TAG, "conversation ended (goodbye)");
            back_to_idle();
        }
    };
    auto send_mic = [&] {  // 把攒下的 Opus 包发给大脑
        while (auto packet = g_audio.PopPacketFromSendQueue()) {
            if (mode == Mode::kListening) link.SendAudio(packet->payload.data(), packet->payload.size());
        }
    };
    // 说完了（或者不说了）：停录音；有话就交给大脑等回复，没话就回去等唤醒
    auto finish_listening = [&](bool keep) {
        g_audio.EnableVoiceProcessing(false);
        send_mic();
        g_listening = false;
        g_audio.EnableWakeWordDetection(true);  // 等回复、说话的时候也能用唤醒词打断
        if (keep && link.ListenStop()) {
            ESP_LOGI(TAG, "listen done (%.1fs), waiting for reply", (esp_timer_get_time() - listen_start) / 1e6);
            mode = Mode::kWaitingReply;
            wait_start = esp_timer_get_time();
            g_face.SetBaseFace(FaceId::kThink);
        } else {
            link.ListenCancel();
            if (follow_up) {
                ESP_LOGI(TAG, "conversation ended (no reply)");  // 主人没接话：这段对话就到这儿，安静地回去等唤醒
            } else {
                ESP_LOGI(TAG, "listen cancelled (no speech)");
                g_face.FlashFace(FaceId::kCurious, 1500);
            }
            back_to_idle();
        }
    };

    while (true) {
        Event e;
        // 倾听时要及时判断“说完了没有”，等事件的间隔短一些
        TickType_t wait = (mode == Mode::kListening) ? pdMS_TO_TICKS(100) : pdMS_TO_TICKS(500);
        bool got = xQueueReceive(g_events, &e, wait) == pdTRUE;
        int64_t now = esp_timer_get_time();

        // ---- 唤醒：连着大脑就聊天（团子正在说、或者在想的时候也行：打断它），没连上就鹦鹉学舌自检 ----
        bool interruptible = mode == Mode::kIdle || mode == Mode::kSpeaking || mode == Mode::kWaitingReply;
        if (got && e == Event::kWakeWord && interruptible && link.Online()) {
            if (mode == Mode::kIdle) {
                // 先应一声“在呢～”，应完再听（不然自己的声音会被录进去）
                g_face.SetBaseFace(FaceId::kListen);
                g_audio.PlaySound(RandomWakeAck());
                ack_start = now;
                mode = Mode::kAcking;
            } else {  // 正在说、在想的时候被叫：直接听，不应
                ESP_LOGI(TAG, "interrupted while %s", mode == Mode::kSpeaking ? "speaking" : "thinking");
                start_listening(false);
            }
            continue;
        }
        if (mode == Mode::kAcking && ((got && e == Event::kPlaybackDrained) || now - ack_start > kAckTimeoutUs)) {
            start_listening(false);
            continue;
        }
        // ---- 按住屏幕说话：在说、在想、在接着听的时候都能按（在说就先闭嘴）；松手就交给大脑 ----
        if (got && e == Event::kHoldStart && link.Online()) {
            if (mode == Mode::kSpeaking) {
                g_accept_audio = false;
                g_audio.ResetDecoder();
                link.SendHush();
            }
            if (mode != Mode::kListening) start_listening(false);
            held = true;
            follow_up = false;
            ESP_LOGI(TAG, "hold to talk");
            continue;
        }
        if (got && e == Event::kHoldEnd && mode == Mode::kListening && held) {
            held = false;
            finish_listening(heard_voice || now - listen_start > 1000 * 1000);  // 按了一秒多就当说了话（人声检测可能漏）
            continue;
        }
        if (got && e == Event::kWakeWord && mode == Mode::kIdle) {
            if (kParrotSelfTest) {  // 鹦鹉学舌自检：录 3 秒再放出来（早期验证麦克风、喇叭用）
                g_audio.EnableWakeWordDetection(false);
                g_audio.EnableAudioTesting(true);  // 开始录音
                g_face.SetBaseFace(FaceId::kListen);
                record_until = now + 3000 * 1000;
                mode = Mode::kRecording;
            } else {
                // 没连上大脑：应一声、迷糊地歪歪头，不学舌（放出主人自己的声音只会让人困惑）
                ESP_LOGI(TAG, "woken but the brain is not connected");
                g_audio.PlaySound(RandomWakeAck());
                g_face.FlashFace(FaceId::kCurious, 2000);
                g_audio.EnableWakeWordDetection(true);  // 唤醒后引擎会关掉唤醒词，重新打开
            }
            continue;
        }

        // ---- 倾听：边录边发，靠人声检测判断说完了没有 ----
        if (got && e == Event::kMicPacket) send_mic();
        if (mode == Mode::kListening) {
            // 每 0.1 秒看一眼“现在有没有人在说话”，不用开口/闭口的瞬间事件：唤醒词尾音会触发一次假的开口
            if (now - listen_start > (follow_up ? kFollowTailUs : kWakeTailUs)) {
                if (g_audio.IsVoiceDetected()) {
                    if (!speech_since) speech_since = now;
                    // 短于 0.2 秒（接着听时 0.5 秒）的一下不算开口，也不打断停顿计时
                    if (now - speech_since >= (follow_up ? kFollowMinSpeechUs : kMinSpeechUs)) {
                        heard_voice = true;                     // （环境里时不时的小声响会让“说完了”一直等不到）
                        silence_since = 0;
                    }
                } else {
                    speech_since = 0;
                    if (heard_voice && !silence_since) silence_since = now;
                }
            }
            if (!link.Online()) {
                g_audio.EnableVoiceProcessing(false);
                back_to_idle();
            } else if (held) {
                if (now - listen_start > kMaxListenUs) finish_listening(true);  // 按住说话：只等松手（最长 12 秒）
            } else if (heard_voice && silence_since && now - silence_since > kEndSilenceUs) {
                finish_listening(true);
            } else if (now - listen_start > kMaxListenUs) {
                finish_listening(heard_voice);
            } else if (!heard_voice && now - listen_start > (follow_up ? kFollowNoSpeechUs : kNoSpeechUs)) {
                finish_listening(false);
            }
            continue;
        }

        // ---- 等回复、说话 ----
        if (got && e == Event::kHeard && mode == Mode::kWaitingReply) {
            std::string text;
            {
                std::lock_guard<std::mutex> lock(g_mu);
                text = g_heard;
            }
            if (text.empty()) {  // 没听清
                g_face.FlashFace(FaceId::kCurious, 1500);
                back_to_idle();
            }
        }
        if (got && e == Event::kTtsStart && g_accept_audio) {
            mode = Mode::kSpeaking;  // 唤醒词开着：说话时喊“你好小智”就打断（回声消除会减掉团子自己的声音）
            tts_done = false;
        }
        if (got && e == Event::kTtsStop && mode == Mode::kSpeaking) {
            tts_done = true;
            bool ok;
            {
                std::lock_guard<std::mutex> lock(g_mu);
                ok = g_tts_ok;
            }
            if (!ok) {
                back_to_idle();  // 被打断、没合成出来：这段对话就算了
            } else if (g_audio.IsPlaybackIdle()) {
                after_speaking();
            }
        }
        if (got && e == Event::kPlaybackDrained && mode == Mode::kSpeaking && tts_done) after_speaking();
        if (mode == Mode::kWaitingReply && (!link.Online() || now - wait_start > kReplyTimeoutUs)) back_to_idle();
        if (mode == Mode::kRecording && now >= record_until) {
            g_face.SetBaseFace(FaceId::kThink);
            g_audio.EnableAudioTesting(false);  // 停止录音并开始回放
            mode = Mode::kReplaying;
            continue;
        }
        if (got && e == Event::kPlaybackDrained && mode == Mode::kReplaying) {
            g_face.FlashFace(FaceId::kHappy, 1500);
            g_audio.EnableWakeWordDetection(true);
            mode = Mode::kIdle;
        }

        // ---- 和大脑的互动 ----
        if (got && e == Event::kPat && mode == Mode::kSpeaking) {
            // 团子说话时摸摸头：马上闭嘴。比喊“你好小智”打断快得多也准得多——
            // 团子自己在出声时，唤醒词要等回声消除减掉它的声音才认得出，实测晚了好几秒
            ESP_LOGI(TAG, "hushed by a pat");
            g_accept_audio = false;
            g_audio.ResetDecoder();
            link.SendHush();
            back_to_idle();
        }
        if (got && e == Event::kPat) pending_pats++;
        if (pending_pats > 0 && link.SendPat(pending_pats)) pending_pats = 0;  // 离线时攒着，连上后一起报

        if (got && e == Event::kThinking) g_face.FlashFace(FaceId::kThink, 15000);
        if (got && e == Event::kSay) {
            net::Say s;
            {
                std::lock_guard<std::mutex> lock(g_mu);
                if (g_says.empty()) continue;
                s = g_says.front();
                g_says.pop_front();
            }
            // 做表情（声音随后由 tts 推过来）；主动来找你时先探头
            FaceId f = FaceId::kIdle;
            face::FaceFromName(s.face.c_str(), &f);
            if (s.proactive) f = FaceId::kPeek;
            uint32_t ms = std::max<uint32_t>(3000, s.text.size() / 3 * 250);  // UTF-8 中文一个字 3 字节
            g_face.FlashFace(f, ms);
            ESP_LOGI(TAG, "%s猪咪：%s", s.proactive ? "【主动】" : "", s.text.c_str());
            convo_end = s.end;  // 主动来找你的话也没带 end：说完会接着听，可以直接回它
            if (mode == Mode::kWaitingReply && !s.audio) back_to_idle();  // 没有声音（声音服务没开）：直接回去等唤醒
        }

        if (mode == Mode::kIdle || mode == Mode::kWaitingReply || mode == Mode::kSpeaking) {
            g_face.SetBaseFace(mode == Mode::kWaitingReply ? FaceId::kThink : BaseFace());
        }
    }
}
