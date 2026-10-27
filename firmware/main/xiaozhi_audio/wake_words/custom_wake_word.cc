#include "custom_wake_word.h"

#include <algorithm>

#include <esp_log.h>
#include <esp_mn_models.h>
#include <esp_mn_speech_commands.h>

#define TAG "CustomWakeWord"

CustomWakeWord::~CustomWakeWord() {
    if (multinet_model_data_ != nullptr && multinet_ != nullptr) {
        multinet_->destroy(multinet_model_data_);
        multinet_model_data_ = nullptr;
    }
}

bool CustomWakeWord::Initialize(AudioCodec* codec, srmodel_list_t* models_list) {
    codec_ = codec;
    display_ = CONFIG_CUSTOM_WAKE_WORD_DISPLAY;
    if (models_list == nullptr || models_list->num <= 0) {
        ESP_LOGE(TAG, "No speech models");
        return false;
    }
    char* mn_name = esp_srmodel_filter(models_list, ESP_MN_PREFIX, "cn");
    if (mn_name == nullptr) {
        ESP_LOGE(TAG, "No Chinese MultiNet model (menuconfig -> ESP Speech Recognition -> mn7_cn)");
        return false;
    }

    multinet_ = esp_mn_handle_from_name(mn_name);
    // 一个识别窗口 6 秒：窗口到点就清空重来，唤醒词正好跨在两个窗口之间就认不出（3 秒时实测五次才认出一次）
    multinet_model_data_ = multinet_->create(mn_name, 6000);
    multinet_->set_det_threshold(multinet_model_data_, CONFIG_CUSTOM_WAKE_WORD_THRESHOLD / 100.0f);
    input_buffer_.reserve(multinet_->get_samp_chunksize(multinet_model_data_));
    esp_mn_commands_clear();
    esp_mn_commands_add(1, CONFIG_CUSTOM_WAKE_WORD);  // 拼音，例如 "zhu mi zhu mi"
    esp_mn_commands_update();
    multinet_->print_active_speech_commands(multinet_model_data_);
    ESP_LOGI(TAG, "Wake word: %s (%s, threshold %d%%) on %s", display_.c_str(), CONFIG_CUSTOM_WAKE_WORD,
             CONFIG_CUSTOM_WAKE_WORD_THRESHOLD, mn_name);
    return true;
}

void CustomWakeWord::OnWakeWordDetected(std::function<void(const std::string& wake_word)> callback) {
    wake_word_detected_callback_ = callback;
}

std::vector<std::string> CustomWakeWord::GetWakeWords() const {
    if (multinet_model_data_ == nullptr) return {};
    return {display_};
}

void CustomWakeWord::Start() { running_ = true; }

void CustomWakeWord::Stop() {
    running_ = false;
    std::lock_guard<std::mutex> lock(input_buffer_mutex_);
    input_buffer_.clear();
}

void CustomWakeWord::Feed(const std::vector<int16_t>& data) { FeedSamples(data.data(), data.size(), false); }

void CustomWakeWord::FeedMono(const int16_t* data, size_t samples) { FeedSamples(data, samples, true); }

void CustomWakeWord::FeedSamples(const int16_t* data, size_t samples, bool mono) {
    if (multinet_model_data_ == nullptr || data == nullptr || samples == 0) return;

    std::lock_guard<std::mutex> lock(input_buffer_mutex_);
    if (!running_) return;  // 在锁里看，免得和 Stop() 抢

    // 只给唤醒词识别这一路放大：AFE 出来的声音偏小（30cm 外喊，峰值才满刻度的 5%~8%），MultiNet 对小声认不出
    auto push = [this](int16_t v) { input_buffer_.push_back(int16_t(std::clamp<int>(v * kInputGain, -32768, 32767))); };
    if (!mono && codec_->input_channels() > 1) {  // 多声道只取第一路（麦克风）
        for (size_t i = 0; i < samples; i += codec_->input_channels()) push(data[i]);
    } else {
        for (size_t i = 0; i < samples; ++i) push(data[i]);
    }

    int chunksize = multinet_->get_samp_chunksize(multinet_model_data_);
    while (input_buffer_.size() >= static_cast<size_t>(chunksize)) {
        esp_mn_state_t state = multinet_->detect(multinet_model_data_, input_buffer_.data());
        if (state == ESP_MN_STATE_DETECTED) {
            esp_mn_results_t* r = multinet_->get_results(multinet_model_data_);
            if (r->num > 0 && r->command_id[0] == 1) {
                ESP_LOGI(TAG, "Wake word detected: %s (prob %.2f)", display_.c_str(), r->prob[0]);
                last_detected_wake_word_ = display_;
                running_ = false;
                input_buffer_.clear();
                multinet_->clean(multinet_model_data_);
                if (wake_word_detected_callback_) wake_word_detected_callback_(last_detected_wake_word_);
                return;
            }
            multinet_->clean(multinet_model_data_);
        } else if (state == ESP_MN_STATE_TIMEOUT) {
            multinet_->clean(multinet_model_data_);  // 3 秒没听到唤醒词，清掉重新听
        }
        input_buffer_.erase(input_buffer_.begin(), input_buffer_.begin() + chunksize);
    }
}

size_t CustomWakeWord::GetFeedSize() {
    return multinet_model_data_ ? multinet_->get_samp_chunksize(multinet_model_data_) : 0;
}
