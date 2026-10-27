#include "face_anim.h"

#include <cmath>
#include <cstring>

#include <esp_random.h>

namespace face {
namespace {

float Rand01() { return (esp_random() & 0xFFFF) / 65535.0f; }
float EaseOut(float x) { return 1 - (1 - x) * (1 - x) * (1 - x); }

bool Blinkable(FaceId f) {
    switch (f) {
        case FaceId::kIdle: case FaceId::kListen: case FaceId::kShy: case FaceId::kLonely:
        case FaceId::kCurious: case FaceId::kPeek: case FaceId::kSad: case FaceId::kThink:
            return true;
        default:
            return false;
    }
}

constexpr const char* kNames[] = {"idle", "listen", "think", "happy", "shy", "pout",
                                  "sad", "lonely", "curious", "sleepy", "sleep", "peek"};

}  // namespace

FaceParams GoalOf(FaceId id) {
    FaceParams p;  // 默认值就是网页里的 BASE
    switch (id) {
        case FaceId::kIdle: break;
        case FaceId::kListen:
            p.eye_w = 1.06f; p.eye_h = 1.12f; p.look_y = -4; p.mouth = Mouth::kO; p.extra = Extra::kWaves;
            p.ear = 1; p.pupil = 0.8f; break;
        case FaceId::kThink:
            p.look_y = -10; p.lid_b = 0.15f; p.mouth = Mouth::kFlat; p.extra = Extra::kDots; p.look_anim = true;
            p.pupil = 0.3f; break;
        case FaceId::kHappy:
            p.mode = EyeMode::kHappy; p.mouth = Mouth::kSmile; p.blush = 0.9f; p.extra = Extra::kSparkle;
            p.ear = 0.4f; break;
        case FaceId::kShy:
            p.look_x = -12; p.look_y = 10; p.lid_t = 0.18f; p.eye_h = 0.92f; p.mouth = Mouth::kFlat; p.blush = 1;
            p.ear = -0.3f; p.pupil = 0.6f; break;
        case FaceId::kPout:
            p.mode = EyeMode::kPout; p.mouth = Mouth::kFrown; p.blush = 0.4f; p.extra = Extra::kAnger;
            p.ear = -0.7f; break;
        case FaceId::kSad:
            p.brow = 0.4f; p.look_y = 8; p.lid_t = 0.08f; p.mouth = Mouth::kFrown; p.extra = Extra::kTear;
            p.blush = 0; p.ear = -1; p.pupil = 0.8f; break;
        case FaceId::kLonely:
            p.brow = 0.3f; p.look_x = -10; p.look_y = 12; p.lid_t = 0.28f; p.mouth = Mouth::kFlat; p.blush = 0;
            p.ear = -0.8f; p.pupil = 0.7f; break;
        case FaceId::kCurious:
            p.scale_l = 1.15f; p.scale_r = 0.82f; p.look_x = 6; p.mouth = Mouth::kO; p.extra = Extra::kOther;
            p.ear = 0.6f; p.pupil = 0.9f; break;
        case FaceId::kSleepy:
            p.lid_t = 0.55f; p.eye_y = 6; p.mouth = Mouth::kFlat; p.extra = Extra::kOther; p.blush = 0.1f;
            p.ear = -0.5f; p.pupil = 0.2f; break;
        case FaceId::kSleep:
            p.mode = EyeMode::kClosed; p.eye_y = 8; p.mouth = Mouth::kFlat; p.extra = Extra::kOther;
            p.blush = 0.1f; p.ear = -0.6f; break;
        case FaceId::kPeek:
            p.look_y = -10; p.extra = Extra::kOther; p.peek = true; p.mouth = Mouth::kNone; p.ear = 0.8f;
            p.pupil = 0.9f; break;
        default: break;
    }
    return p;
}

const char* FaceName(FaceId id) {
    int i = int(id);
    return i < int(FaceId::kCount) ? kNames[i] : "idle";
}

bool FaceFromName(const char* name, FaceId* out) {
    for (int i = 0; i < int(FaceId::kCount); ++i) {
        if (strcmp(name, kNames[i]) == 0) {
            *out = FaceId(i);
            return true;
        }
    }
    return false;
}

FaceAnimator::FaceAnimator() : goal_(GoalOf(FaceId::kIdle)), cur_(goal_) {}

void FaceAnimator::SetFace(FaceId id, uint32_t now_ms) {
    face_ = id;
    goal_ = GoalOf(id);
    start_ms_ = now_ms;
    cur_.mode = goal_.mode;
    cur_.mouth = goal_.mouth;
    cur_.extra = goal_.extra;
    cur_.peek = goal_.peek;
    cur_.look_anim = goal_.look_anim;
}

FrameState FaceAnimator::Step(uint32_t t) {
    uint32_t age = t - start_ms_;
    FaceParams g = goal_;
    if (g.look_anim) g.look_x = std::sin(t / 380.0f) * 14;
    if (face_ == FaceId::kIdle) {  // 待机时偶尔东张西望
        if (t > glance_until_) {
            glance_ = Rand01() < 0.5f ? 0 : (Rand01() < 0.5f ? -12 : 12);
            glance_until_ = t + 1500 + uint32_t(Rand01() * 2500);
        }
        g.look_x += glance_;
    }
    if (g.peek) g.eye_y = 140 * (1 - EaseOut(std::fmin(1.0f, age / 1100.0f))) + 50;

    // 和网页一样每帧逼近目标 18%；设备上帧率稍低，所以逼近得稍快一点
    constexpr float k = 0.25f;
    auto lerp = [k](float& v, float goal) { v += (goal - v) * k; };
    lerp(cur_.eye_w, g.eye_w); lerp(cur_.eye_h, g.eye_h); lerp(cur_.lid_t, g.lid_t); lerp(cur_.lid_b, g.lid_b);
    lerp(cur_.brow, g.brow); lerp(cur_.look_x, g.look_x); lerp(cur_.look_y, g.look_y);
    lerp(cur_.scale_l, g.scale_l); lerp(cur_.scale_r, g.scale_r); lerp(cur_.blush, g.blush);
    lerp(cur_.ear, g.ear); lerp(cur_.pupil, g.pupil);
    if (g.peek) cur_.eye_y = g.eye_y; else lerp(cur_.eye_y, g.eye_y);

    FrameState fs;
    fs.p = cur_;
    fs.face = face_;
    fs.t_ms = t;
    fs.age_ms = age;
    if (cur_.mode == EyeMode::kOpen && Blinkable(face_)) {
        if (t > blink_at_) {
            blink_start_ = t;
            blinked_ = true;
            blink_at_ = t + 2200 + uint32_t(Rand01() * 4000);
        }
        uint32_t d = t - blink_start_;
        if (blinked_ && d < 170) fs.blink = 1 - std::fabs(d / 85.0f - 1);
    }
    return fs;
}

}  // namespace face
