// 和 PC 大脑的连接：WiFi + WebSocket，协议见 ai_pet/PROTOCOL.md（和网页团子说同一套协议）。
// 回调都在网络任务里触发，里面只做轻量的事（改表情、投递事件）。
#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace net {

struct BrainState {
    std::string phase;       // awake / sleepy / asleep
    float loneliness = 0;
    float energy = 1;
    bool sleepy_talk = false;
};

struct Say {
    std::string text, face, vibe, voice;
    bool proactive = false;
    bool audio = false;  // 这段话的声音随后推过来（tts start → Opus 包 → tts stop）
    bool end = false;    // 主人道别了：说完这句不再接着听（连续对话结束）
};

class PetLink {
public:
    static PetLink& Get();

    // 连 WiFi（menuconfig 里的 SSID/密码），连上后自动连大脑，断了自动重连
    void Start();
    bool Online() const { return online_; }

    // 扫描后决定连哪个 WiFi（0 家里、1 手机热点）
    void UseWifi(int index, const char* ssid, const char* password);
    int WifiIndex() const { return wifi_index_; }

    bool SendChat(const std::string& text);
    bool SendPat(int count);
    // 语音：ListenStart 之后用 SendAudio 发 Opus 包（16kHz、60ms 一包），说完 ListenStop，不要了 ListenCancel
    bool ListenStart();
    bool ListenStop();
    bool ListenCancel();
    bool SendAudio(const uint8_t* data, size_t len);
    bool SendHush();  // 让大脑别再推正在说的这段声音（摸头打断）

    // 大脑那边的“现在几点”（用服务端时间，所以调试面板快进时设备也跟着走）；没连过时返回 -1
    int BrainHour() const;

    std::function<void(const BrainState&)> on_state;
    std::function<void(const Say&)> on_say;
    std::function<void()> on_thinking;
    std::function<void(bool online)> on_online;
    std::function<void(const std::string& text)> on_heard;  // 大脑听到的话（空 = 没听清）
    // 团子的声音：start 带采样率和每包毫秒数，之后每个 Opus 包回调一次 on_audio，最后 stop（被打断/出错时 ok=false）
    std::function<void(int rate, int frame_ms)> on_tts_start;
    std::function<void(const uint8_t* data, size_t len)> on_audio;
    std::function<void(bool ok)> on_tts_stop;

private:
    PetLink() = default;
    void* client_ = nullptr;  // esp_websocket_client_handle_t
    // 两个地址：家里局域网、出门经服务器中转。一个连续连不上就换另一个（换地址要在单独的任务里做）
    int url_index_ = 0;
    int failures_ = 0;
    int wifi_index_ = 0;     // 连的是哪个 WiFi：0 家里、1 手机热点
    int wifi_failures_ = 0;  // 同一个 WiFi 连续连不上的次数，多了就重新扫
    void* switch_task_ = nullptr;  // TaskHandle_t
    const char* Url() const;
    void OnConnectFailed();
    static constexpr uint32_t kSwitchBit = 1;   // 换另一个地址
    static constexpr uint32_t kRestartBit = 2;  // 同一个地址重新连（大脑正常关闭后客户端不会自己重连）
    void Notify(uint32_t bit);
    static void SwitchTask(void* arg);
    volatile bool online_ = false;
    int64_t time_base_us_ = 0;      // 收到服务端时间时的本地时间
    int brain_seconds_of_day_ = -1;  // 那一刻服务端是一天中的第几秒

    void StartWifi();
    void StartSocket();
    bool SendJson(const std::string& json);
    bool SendListen(const char* state);
    void HandleText(const char* data, int len);
    void SetOnline(bool on);
    friend struct LinkCallbacks;
};

}  // namespace net
