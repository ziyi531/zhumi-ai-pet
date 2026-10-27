// 表情显示任务：按固定帧率推进动画、画八条、推到屏幕，顺便读触摸做“摸头”反应。
// 这一层就是网页假设备里“设备本地本能”的固件版：摸头的反应不等 PC，当场就有。
#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <vector>

#include "bajo_face.h"
#include "face_anim.h"
#include "lichuang_board.h"

namespace face {

class FaceDisplay {
public:
    void Start(board::LichuangBoard* board);

    // 基础表情：一直保持，直到下次设置
    void SetBaseFace(FaceId id);
    // 临时表情：显示 ms 毫秒后回到基础表情（比如被摸时开心一下）
    void FlashFace(FaceId id, uint32_t ms);
    FaceId BaseFace();

    // 被摸头时回调（在显示任务里调用，别在里面做耗时的事）
    std::function<void()> on_pat;
    // 按住屏幕超过 0.7 秒 / 松手：按住说话
    std::function<void()> on_hold_start;
    std::function<void()> on_hold_end;

private:
    board::LichuangBoard* board_ = nullptr;
    Canvas* canvas_ = nullptr;
    FaceAnimator anim_;
    std::mutex mu_;
    FaceId base_ = FaceId::kIdle;
    FaceId shown_ = FaceId::kIdle;
    FaceId flash_ = FaceId::kIdle;
    uint32_t flash_until_ = 0;
    std::vector<Heart> hearts_;
    bool was_pressed_ = false;
    bool holding_ = false;
    uint32_t press_start_ = 0;
    uint16_t* tx_buf_[board::LichuangBoard::kTxSlots] = {};

    void Loop();
    void HandleTouch(uint32_t now);
    void Present();
};

}  // namespace face
