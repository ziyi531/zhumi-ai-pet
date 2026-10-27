// 表情参数和动画：每个表情就是一组数字，切换时逐帧插值，所以会“变形”过去。
// 数值和网页预览 ai_pet/static/faces.html 的 BASE / FACE_DEFS 完全一致，改表情时两边一起改。
#pragma once

#include <cstdint>

namespace face {

enum class FaceId : uint8_t {
    kIdle, kListen, kThink, kHappy, kShy, kPout, kSad, kLonely, kCurious, kSleepy, kSleep, kPeek, kCount
};

enum class EyeMode : uint8_t { kOpen, kHappy, kClosed, kPout };
enum class Mouth : uint8_t { kW, kO, kFlat, kFrown, kSmile, kNone };
enum class Extra : uint8_t { kNone, kTear, kDots, kSparkle, kAnger, kWaves, kOther };

struct FaceParams {
    // 会插值的数值
    float eye_w = 1, eye_h = 1, lid_t = 0, lid_b = 0, brow = 0;
    float look_x = 0, look_y = 0, eye_y = 0, scale_l = 1, scale_r = 1;
    float blush = 0.15f, ear = 0, pupil = 0.35f;
    // 直接切换的
    EyeMode mode = EyeMode::kOpen;
    Mouth mouth = Mouth::kW;
    Extra extra = Extra::kNone;
    bool peek = false;
    bool look_anim = false;
};

// 一帧要画的全部信息
struct FrameState {
    FaceParams p;
    FaceId face = FaceId::kIdle;
    float blink = 0;       // 0..1，眨眼进度
    uint32_t t_ms = 0;     // 动画时间
    uint32_t age_ms = 0;   // 进入当前表情多久了（探头动画用）
};

FaceParams GoalOf(FaceId id);
const char* FaceName(FaceId id);
bool FaceFromName(const char* name, FaceId* out);

class FaceAnimator {
public:
    FaceAnimator();
    void SetFace(FaceId id, uint32_t now_ms);
    FaceId face() const { return face_; }
    FrameState Step(uint32_t now_ms);

private:
    FaceId face_ = FaceId::kIdle;
    FaceParams goal_, cur_;
    uint32_t start_ms_ = 0, blink_at_ = 1500, blink_start_ = 0;
    uint32_t glance_until_ = 0;
    float glance_ = 0;
    bool blinked_ = false;
};

}  // namespace face
