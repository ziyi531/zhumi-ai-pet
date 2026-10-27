// ai_pet: 自定义唤醒词（“猪咪猪咪”）——用 ESP-SR 的 MultiNet 命令词识别当唤醒词，拼音写在 sdkconfig 里
// （CONFIG_CUSTOM_WAKE_WORD），不用训练模型。
// 改自小智 main/audio/wake_words/custom_wake_word.*（上游提交见 ../UPSTREAM_COMMIT.txt）：
// 去掉了它的资源系统（assets / index.json 里配命令词）和“把唤醒词录音编码上传”的功能，只留识别本身。
// 模型分区里有 MultiNet 模型时，AfeAudioEngine 就用它代替 WakeNet（“你好小智”）。
//
// 现在没在用（2026-10-26 真机实测）：“猪咪猪咪”五次只认出一次左右——MultiNet 是认命令词的，
// 按 3~6 秒一个窗口清空重来，唤醒词跨窗口就丢；调过送入音量（放大 4 倍，大声喊会截平）、窗口、门槛都不行。
// 唤醒词改用乐鑫训练好的 WakeNet“Hi,喵喵”（menuconfig 里选模型）。想再试自定义唤醒词，
// 在 ESP Speech Recognition 里选上 mn7_cn 就会自动用回这里。
#ifndef CUSTOM_WAKE_WORD_H
#define CUSTOM_WAKE_WORD_H

#include <esp_mn_iface.h>
#include <model_path.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include "audio_codec.h"
#include "wake_word.h"

class CustomWakeWord : public WakeWord {
public:
    ~CustomWakeWord();

    bool Initialize(AudioCodec* codec, srmodel_list_t* models_list);
    void Feed(const std::vector<int16_t>& data);
    void FeedMono(const int16_t* data, size_t samples);
    void OnWakeWordDetected(std::function<void(const std::string& wake_word)> callback);
    void Start();
    void Stop();
    size_t GetFeedSize();
    void EncodeWakeWordData() {}                                   // 不上传唤醒词录音
    bool GetWakeWordOpus(std::vector<uint8_t>&) { return false; }
    const std::string& GetLastDetectedWakeWord() const { return last_detected_wake_word_; }
    std::vector<std::string> GetWakeWords() const override;

private:
    static constexpr int kInputGain = 2;  // 送给 MultiNet 前放大几倍（4 倍时大声喊会截平失真）
    esp_mn_iface_t* multinet_ = nullptr;
    model_iface_data_t* multinet_model_data_ = nullptr;
    std::string display_;  // 唤醒词的中文（日志、回调里用）
    std::function<void(const std::string& wake_word)> wake_word_detected_callback_;
    AudioCodec* codec_ = nullptr;
    std::string last_detected_wake_word_;
    std::atomic<bool> running_ = false;
    std::vector<int16_t> input_buffer_;
    std::mutex input_buffer_mutex_;

    void FeedSamples(const int16_t* data, size_t samples, bool mono);
};

#endif
