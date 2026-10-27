// 八条风表情的画法，逐段对应网页预览 faces.html 里的 STYLES.bajo。
#pragma once

#include <vector>

#include "canvas.h"
#include "face_anim.h"

namespace face {

struct Heart {
    float x, y;
    uint32_t t_ms;
};

void DrawBajo(Canvas& cv, const FrameState& fs, const std::vector<Heart>& hearts);

}  // namespace face
