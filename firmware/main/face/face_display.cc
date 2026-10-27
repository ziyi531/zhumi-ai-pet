#include "face_display.h"

#include <algorithm>

#include <esp_heap_caps.h>
#include <esp_log.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#define TAG "FaceDisplay"

namespace face {
namespace {

constexpr int kStripRows = 24;         // 每次 DMA 发 24 行
constexpr uint32_t kFrameMs = 50;      // 约 20 帧/秒
constexpr uint32_t kHeartLifeMs = 1300;
constexpr uint32_t kHoldMs = 700;  // 按住超过这么久算“按住说话”，短于它只是摸头

uint32_t NowMs() { return uint32_t(esp_timer_get_time() / 1000); }

}  // namespace

void FaceDisplay::Start(board::LichuangBoard* board) {
    board_ = board;
    canvas_ = new Canvas(board::kScreenW, board::kScreenH);
    for (auto& buf : tx_buf_) {
        buf = static_cast<uint16_t*>(
            heap_caps_malloc(board::kScreenW * kStripRows * sizeof(uint16_t), MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL));
    }
    anim_.SetFace(FaceId::kIdle, NowMs());
    // 优先级 2：比声音、唤醒词、联网都低——CPU 紧的时候宁可表情掉帧，也不能让它们饿着
    xTaskCreatePinnedToCore([](void* arg) { static_cast<FaceDisplay*>(arg)->Loop(); }, "face", 6144, this, 2,
                            nullptr, 1);
}

void FaceDisplay::SetBaseFace(FaceId id) {
    std::lock_guard<std::mutex> lock(mu_);
    base_ = id;
}

void FaceDisplay::FlashFace(FaceId id, uint32_t ms) {
    std::lock_guard<std::mutex> lock(mu_);
    flash_ = id;
    flash_until_ = NowMs() + ms;
}

FaceId FaceDisplay::BaseFace() {
    std::lock_guard<std::mutex> lock(mu_);
    return base_;
}

void FaceDisplay::HandleTouch(uint32_t now) {
    auto tp = board_->ReadTouch();
    if (tp.pressed && !was_pressed_) {
        // 摸头：冒个小爱心；睡着时被摸只会迷迷糊糊地犯困，醒着就开心
        hearts_.push_back({float(tp.x), float(tp.y), now});
        FaceId base = BaseFace();
        FlashFace(base == FaceId::kSleep ? FaceId::kSleepy : FaceId::kHappy, 1500);
        if (on_pat) on_pat();
        press_start_ = now;
    }
    // 按住不放 = 按住说话：唤醒词靠不住时（团子自己在说话、周围吵）这个最准
    if (tp.pressed && !holding_ && now - press_start_ >= kHoldMs) {
        holding_ = true;
        if (on_hold_start) on_hold_start();
    }
    if (!tp.pressed && holding_) {
        holding_ = false;
        if (on_hold_end) on_hold_end();
    }
    was_pressed_ = tp.pressed;
    hearts_.erase(std::remove_if(hearts_.begin(), hearts_.end(),
                                 [now](const Heart& h) { return now - h.t_ms > kHeartLifeMs; }),
                  hearts_.end());
}

void FaceDisplay::Present() {
    // 整帧在 PSRAM 里画好，分条拷到内部 DMA 缓冲（顺便把字节序换成屏幕要的大端）再发出去
    const uint16_t* src = canvas_->pixels();
    int slot = 0;
    for (int y = 0; y < board::kScreenH; y += kStripRows) {
        int rows = std::min(kStripRows, board::kScreenH - y);
        board_->AcquireTxSlot();
        uint16_t* dst = tx_buf_[slot];
        const uint16_t* s = src + y * board::kScreenW;
        for (int i = 0, n = rows * board::kScreenW; i < n; ++i) dst[i] = __builtin_bswap16(s[i]);
        board_->Blit(0, y, board::kScreenW, rows, dst);
        slot = (slot + 1) % board::LichuangBoard::kTxSlots;
    }
}

void FaceDisplay::Loop() {
    TickType_t last = xTaskGetTickCount();
    uint32_t frames = 0, render_us = 0;
    ESP_LOGI(TAG, "face loop started");
    while (true) {
        uint32_t now = NowMs();
        HandleTouch(now);

        FaceId want;
        {
            std::lock_guard<std::mutex> lock(mu_);
            want = now < flash_until_ ? flash_ : base_;
        }
        if (want != shown_) {
            anim_.SetFace(want, now);
            shown_ = want;
        }

        int64_t t0 = esp_timer_get_time();
        DrawBajo(*canvas_, anim_.Step(now), hearts_);
        render_us += uint32_t(esp_timer_get_time() - t0);
        Present();

        if (++frames == 100) {
            ESP_LOGI(TAG, "avg render %lu ms/frame", (unsigned long)(render_us / 100 / 1000));
            frames = 0;
            render_us = 0;
        }
        // 一帧超时（表情切换时会重画头）也至少让出 1 个 tick，不然 CPU1 的空闲任务饿死，看门狗会报警
        TickType_t next = last + pdMS_TO_TICKS(kFrameMs), now_tick = xTaskGetTickCount();
        vTaskDelay(next > now_tick ? next - now_tick : 1);
        last = std::max(next, now_tick);
    }
}

}  // namespace face
