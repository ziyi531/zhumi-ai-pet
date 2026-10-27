#include "bajo_face.h"

#include <algorithm>
#include <cmath>

#include "captions.h"

namespace face {
namespace {

constexpr float kPi = 3.14159265f;

// 和网页里 bajo.G 一致
constexpr float kGap = 64, kCy = 140, kEyeW = 84, kEyeH = 84;

constexpr Rgb kInk = Hex(0x141414);
constexpr Rgb kBg = Hex(0xf7f6f2);
constexpr Rgb kWhite = Hex(0xffffff);
constexpr Rgb kIris = Hex(0xe4e98f);
constexpr Rgb kEarGray = Hex(0xb4b6bf);
constexpr Rgb kEarPink = Hex(0xf6b8c8);
constexpr Rgb kBlush = Hex(0xffb6c8);
constexpr Rgb kTongue = Hex(0xff8fa3);
constexpr Rgb kExtra = Hex(0x5a5f6b);
constexpr Rgb kTear = Hex(0x7fb8ff);
constexpr Rgb kSparkle = Hex(0xf5c518);
constexpr Rgb kAnger = Hex(0xff5d73);
constexpr Rgb kLedge = Hex(0xece8df);
constexpr Rgb kHeart = Hex(0xff6f91);

struct EyeBox {
    float cx, cy, w, h;
};

EyeBox Eye(const FaceParams& p, int side) {
    float s = side < 0 ? p.scale_l : p.scale_r;
    return {160 + side * kGap + p.look_x, kCy + p.eye_y + p.look_y, kEyeW * p.eye_w * s, kEyeH * p.eye_h * s};
}

// 眼皮线的高度：brow>0 外眼角下垂（难过），brow<0 内眼角下压（生气）
float LidY(const EyeBox& e, const FaceParams& p, int side, float lt, float x) {
    float u = std::clamp((x - (e.cx - e.w / 2)) / e.w, 0.0f, 1.0f);
    float outer = side < 0 ? 1 - u : u;
    float slope = p.brow > 0 ? p.brow * outer : -p.brow * (1 - outer);
    return e.cy - e.h / 2 + lt * e.h + slope * e.h * 0.7f;
}

Rgb Lerp(Rgb a, Rgb b, float t) {
    return Rgb{uint8_t(a.r + (b.r - a.r) * t), uint8_t(a.g + (b.g - a.g) * t), uint8_t(a.b + (b.b - a.b) * t)};
}

// 圆角三角形（只圆耳朵尖），采样成多边形
std::vector<Vec2> RoundTri(Vec2 a, Vec2 tip, Vec2 b, float rad) {
    auto near = [&](Vec2 q) {
        float len = Len(q.x - tip.x, q.y - tip.y);
        return Vec2{tip.x + (q.x - tip.x) * rad / len, tip.y + (q.y - tip.y) * rad / len};
    };
    Vec2 p1 = near(a), p2 = near(b);
    std::vector<Vec2> pts{a, p1};
    for (int i = 1; i < 6; ++i) {
        float s = i / 6.0f, u = 1 - s;
        pts.push_back({u * u * p1.x + 2 * u * s * tip.x + s * s * p2.x, u * u * p1.y + 2 * u * s * tip.y + s * s * p2.y});
    }
    pts.push_back(p2);
    pts.push_back(b);
    return pts;
}

float Twitch(uint32_t t) {  // 左耳偶尔抖一下
    uint32_t ph = t % 6100;
    return ph < 240 ? std::sin(ph / 240.0f * kPi) * 8 : 0;
}

void DrawHead(Canvas& cv, float ear, float twitch) {
    float droop = std::max(0.0f, -ear), perk = std::max(0.0f, ear);
    for (int side : {-1, 1}) {
        Vec2 tip{160 + side * (100 + droop * 30), 16 + droop * 26 - perk * 8 - (side < 0 ? twitch : 0)};
        Vec2 b1{160.0f + side * 40, 60}, b2{160.0f + side * 118, 100};
        auto outer = RoundTri(b1, tip, b2, 14);
        cv.Polygon(outer, kEarGray);
        cv.PolygonStroke(outer, 5, kInk);
        Vec2 c{(b1.x + b2.x + tip.x) / 3, (b1.y + b2.y + tip.y) / 3};
        auto sh = [&](Vec2 q) { return Vec2{c.x + (q.x - c.x) * 0.5f, c.y + (q.y - c.y) * 0.5f + 4}; };
        cv.Polygon(RoundTri(sh(b1), sh(tip), sh(b2), 8), kEarPink);
    }
    // 头：大圆团，灰色头顶渐变到白色下巴（银渐层）
    auto head = [](float x, float y) { return Canvas::SdCircle(x, y, 160, 196, 150); };
    cv.FillVGradient(head, 10, 46, 310, 346, [](float y) {
        float u = std::clamp((y - 46) / 194.0f, 0.0f, 1.0f);
        if (u < 0.45f) return Lerp(Hex(0xaeb0b9), Hex(0xd5d6db), u / 0.45f);
        if (u < 0.75f) return Lerp(Hex(0xd5d6db), Hex(0xf4f4f6), (u - 0.45f) / 0.3f);
        return Lerp(Hex(0xf4f4f6), Hex(0xffffff), (u - 0.75f) / 0.25f);
    });
    // 白肚皮整个在头的圆里面（不用再按头裁剪，裁剪会让里面没法整段写）
    cv.Ellipse(160, 216, 92, 58, kWhite);
    cv.Fill([](float x, float y) { return std::fabs(Canvas::SdCircle(x, y, 160, 196, 150)) - 2.5f; }, 8, 44, 312, 348,
            kInk);
}

// ---------- 头部缓存 ----------
// 头（耳朵、渐变大圆、白肚皮、描边）每帧几乎不变，却占了一帧大半的时间：画进缓存，之后每帧整块拷贝。
// 头只会上下平移（呼吸、表情高低），按 1/4 像素分档缓存；耳朵姿势按 0.1、抖耳朵按 1 像素分档，
// 表情切换时耳朵慢慢动，那几帧要重画。探头时头移得太远、超出缓存上下的余量，就直接画（那时头大半在屏幕外）。
constexpr int kHeadMargin = 40;
constexpr int kHeadSlots = 6;

struct HeadSlot {
    Canvas* cv = nullptr;
    int key = -1;
    uint32_t used = 0;
};
HeadSlot g_head[kHeadSlots];
uint32_t g_head_clock = 0;

// 画背景和头，代替 Clear + DrawHead；offset 是头整体的上下平移
void DrawBackgroundAndHead(Canvas& cv, float ear, uint32_t t, float offset) {
    int q = int(std::lround(offset * 4));
    int phase = ((q % 4) + 4) % 4, ip = (q - phase) / 4;
    int ear_q = int(std::lround(ear * 10)), tw_q = int(std::lround(Twitch(t)));
    if (std::abs(ip) >= kHeadMargin) {
        cv.SetOffset(0, 0);
        cv.Clear(kBg);
        cv.SetOffset(0, offset);
        DrawHead(cv, ear, Twitch(t));
        return;
    }
    int key = (ear_q + 32) * 1000 + tw_q * 10 + phase;
    HeadSlot* slot = nullptr;
    for (auto& s : g_head) {
        if (s.key == key) slot = &s;
    }
    if (!slot) {  // 没缓存过：挑最久没用的那格重画
        slot = &g_head[0];
        for (auto& s : g_head) {
            if (s.used < slot->used) slot = &s;
        }
        if (!slot->cv) slot->cv = new Canvas(cv.width(), cv.height() + 2 * kHeadMargin);
        slot->key = key;
        slot->cv->SetOffset(0, 0);
        slot->cv->Clear(kBg);
        slot->cv->SetOffset(0, kHeadMargin + phase / 4.0f);
        DrawHead(*slot->cv, ear_q / 10.0f, float(tw_q));
    }
    slot->used = ++g_head_clock;
    cv.CopyRows(*slot->cv, kHeadMargin - ip);
}

void DrawOpenEye(Canvas& cv, const EyeBox& e, const FaceParams& p, int side, float blink) {
    float lt = std::max(p.lid_t, blink), lb = p.lid_b, bot = e.cy + e.h / 2;
    bool has_top = !(lt < 0.01f && std::fabs(p.brow) < 0.01f);
    // 眼皮：用裁剪把眼睛上下盖住（睁大眼睛没有眼皮时不裁剪：裁剪会让圆里面没法整段写）
    if (has_top || lb > 0.01f) {
        cv.SetClip([=, &e, &p](float x, float y) {
            float d = -1e9f;
            if (has_top) d = std::max(d, LidY(e, p, side, lt, x) - y);
            if (lb > 0.01f) d = std::max(d, y - (bot - lb * e.h));
            return d;
        });
    }
    float r = std::min(e.w, e.h) / 2;
    cv.Circle(e.cx, e.cy, r, kInk);
    cv.Circle(e.cx, e.cy, r - 4, kIris);  // 一圈细细的黄绿色虹膜
    cv.Circle(e.cx + p.look_x * 0.12f, e.cy + p.look_y * 0.12f, (r - 4) * (0.86f + p.pupil * 0.08f), kInk);
    cv.Circle(e.cx - r * 0.32f, e.cy - r * 0.32f, r * 0.26f, kWhite);
    cv.Circle(e.cx + r * 0.28f, e.cy + r * 0.27f, r * 0.11f, kWhite);
    cv.SetClip(nullptr);
    // 半闭眼时补一道粗眼皮线，“呆”和“无语”的神态全靠它
    if (has_top) {
        float x0 = e.cx - e.w / 2, x1 = e.cx + e.w / 2;
        cv.SetClip([&e, r](float x, float y) { return Canvas::SdCircle(x, y, e.cx, e.cy, r + 1); });
        cv.Line({x0, LidY(e, p, side, lt, x0)}, {x1, LidY(e, p, side, lt, x1)}, 6, kInk);
        cv.SetClip(nullptr);
    }
}

void DrawStrokeEye(Canvas& cv, const EyeBox& e, EyeMode mode, int side) {
    constexpr float lw = 7;
    if (mode == EyeMode::kHappy) {
        cv.Arc(e.cx, e.cy + e.h * 0.2f, e.w * 0.5f, kPi * 1.12f, kPi * 1.88f, lw, kInk);
    } else if (mode == EyeMode::kPout) {
        float d = -side;  // 左眼“>”、右眼“<”，尖角朝内
        Vec2 a{e.cx - d * e.w * 0.3f, e.cy - e.h * 0.26f}, m{e.cx + d * e.w * 0.3f, e.cy},
            b{e.cx - d * e.w * 0.3f, e.cy + e.h * 0.26f};
        cv.Line(a, m, lw, kInk);
        cv.Line(m, b, lw, kInk);
    } else {  // 闭眼
        cv.Arc(e.cx, e.cy - e.w * 0.12f, e.w * 0.5f, kPi * 0.15f, kPi * 0.85f, lw, kInk);
    }
}

void DrawMouth(Canvas& cv, const FaceParams& p) {
    float my = kCy + p.eye_y + 56;
    constexpr float lw = 3.5f;
    switch (p.mouth) {
        case Mouth::kO:
            cv.Ellipse(160, my + 2, 3.8f, 4.7f, kTongue);
            cv.Fill([=](float x, float y) { return std::fabs(Canvas::SdEllipse(x, y, 160, my + 2, 3.8f, 4.7f)) - lw / 2; },
                    150, my - 8, 170, my + 12, kInk);
            break;
        case Mouth::kFrown: cv.Arc(160, my + 8.8f, 7.6f, kPi * 1.25f, kPi * 1.75f, lw, kInk); break;
        case Mouth::kFlat: cv.Line({155.6f, my + 2}, {164.4f, my + 2}, lw, kInk); break;
        case Mouth::kNone: break;
        default:  // ω，开心时吐一点小舌头
            if (p.mouth == Mouth::kSmile) {
                cv.Fill([=](float x, float y) {
                    return std::max(Canvas::SdCircle(x, y, 160, my + 9, 5), my + 9 - y);
                }, 154, my + 8, 166, my + 15, kTongue);
            }
            cv.Arc(157, my, 3, 0, kPi, lw, kInk);
            cv.Arc(163, my, 3, 0, kPi, lw, kInk);
            break;
    }
}

void Star(Canvas& cv, float x, float y, float r, Rgb c) {
    std::vector<Vec2> pts;
    for (int i = 0; i < 8; ++i) {
        float a = i * kPi / 4, rr = i % 2 ? r * 0.35f : r;
        pts.push_back({x + std::cos(a) * rr, y + std::sin(a) * rr});
    }
    cv.Polygon(pts, c);
}

void DrawExtras(Canvas& cv, const FrameState& fs) {
    const FaceParams& p = fs.p;
    float t = float(fs.t_ms);
    switch (p.extra) {
        case Extra::kTear: {
            float ph = std::fmod(t / 1500, 1.0f);
            float x = 160 - kGap - kEyeW * 0.32f + p.look_x, y = kCy + p.eye_y + kEyeH * 0.35f + ph * 46;
            cv.Fill([=](float px, float py) {
                float drop = Canvas::SdCircle(px, py, x, y, 6);
                float top = Canvas::SdPolygon(px, py, {{x - 6, y}, {x, y - 13}, {x + 6, y}});
                return std::min(drop, top);
            }, x - 7, y - 14, x + 7, y + 7, kTear, 1 - ph);
            break;
        }
        case Extra::kDots: {
            int n = int(t / 350) % 4;
            for (int i = 0; i < n; ++i) cv.Circle(214 + i * 15, 40, 5, kExtra);
            break;
        }
        case Extra::kSparkle: {
            const float pos[4][2] = {{62, 54}, {258, 44}, {272, 150}, {48, 150}};
            for (int i = 0; i < 4; ++i)
                Star(cv, pos[i][0], pos[i][1], 6 + 5 * std::max(0.0f, std::sin(t / 260 + i * 1.7f)), kSparkle);
            break;
        }
        case Extra::kAnger: {
            float k = 1 + 0.12f * std::sin(t / 150);
            for (int i = 0; i < 4; ++i) {
                float a = i * kPi / 2 + kPi / 4;
                cv.Arc(262 + std::cos(a) * 13 * k, 74 + std::sin(a) * 13 * k, 8 * k, a + kPi * 0.75f, a + kPi * 1.25f, 5,
                       kAnger);
            }
            break;
        }
        case Extra::kWaves:
            for (int side : {-1, 1})
                for (int i = 0; i < 2; ++i) {
                    float ph = std::fmod(t / 900 + i / 2.0f, 1.0f), r = 10 + ph * 22;
                    float cx = 160 + side * 122, cy = kCy + p.eye_y;
                    if (side > 0) cv.Arc(cx, cy, r, -kPi * 0.3f, kPi * 0.3f, 4, kExtra, 1 - ph);
                    else cv.Arc(cx, cy, r, kPi * 0.7f, kPi * 1.3f, 4, kExtra, 1 - ph);
                }
            break;
        default: break;
    }
}

// 探头：眼睛从“桌沿”后面冒出来，两只小爪子扒着边
void DrawLedge(Canvas& cv, const FrameState& fs) {
    if (!fs.p.peek) return;
    cv.Rect(0, 182, 320, 58, kLedge);
    cv.Rect(0, 182, 320, 5, kInk);
    if (fs.age_ms > 500) {
        for (float x : {62.0f, 258.0f}) {
            cv.HalfEllipseTop(x, 186, 18, 12, kWhite);
            cv.Fill([=](float px, float py) {
                return std::max(std::fabs(Canvas::SdEllipse(px, py, x, 186, 18, 12)) - 1.25f, py - 187);
            }, x - 20, 172, x + 20, 188, kInk);
            for (float dx : {-6.0f, 0.0f, 6.0f}) cv.Line({x + dx, 186}, {x + dx, 180}, 2.5f, kInk);
        }
    }
}

void DrawCaption(Canvas& cv, const FrameState& fs) {
    const Caption& cap = kCaptions[int(fs.face)];
    if (!cap.fill || (fs.face == FaceId::kPeek && fs.age_ms < 1000)) return;
    int base_y = int(std::lround(40 + std::sin(fs.t_ms / 400.0f) * 2));
    cv.BlitMask(cap.outline, cap.w, cap.h, 160 + cap.dx, base_y + cap.dy, kBg);
    cv.BlitMask(cap.fill, cap.w, cap.h, 160 + cap.dx, base_y + cap.dy, kInk);
}

void DrawHearts(Canvas& cv, const std::vector<Heart>& hearts, uint32_t now) {
    for (const auto& h : hearts) {
        float ph = (now - h.t_ms) / 1200.0f;
        if (ph < 0 || ph > 1) continue;
        float x = h.x, y = h.y - ph * 40, s = 9;
        // 两个圆加一个倒三角拼成的爱心
        cv.Fill([=](float px, float py) {
            float l = Canvas::SdCircle(px, py, x - s * 0.5f, y - s * 0.3f, s * 0.58f);
            float r = Canvas::SdCircle(px, py, x + s * 0.5f, y - s * 0.3f, s * 0.58f);
            float tri = Canvas::SdPolygon(px, py, {{x - s * 1.05f, y - s * 0.1f}, {x + s * 1.05f, y - s * 0.1f}, {x, y + s}});
            return std::min(std::min(l, r), tri);
        }, x - s * 1.2f, y - s * 1.0f, x + s * 1.2f, y + s * 1.1f, kHeart, 1 - ph);
    }
}

}  // namespace

void DrawBajo(Canvas& cv, const FrameState& fs, const std::vector<Heart>& hearts) {
    const FaceParams& p = fs.p;
    float breathe = std::sin(fs.t_ms / 1300.0f) * 2;

    DrawBackgroundAndHead(cv, p.ear, fs.t_ms, breathe + (p.peek ? p.eye_y : p.eye_y * 0.4f));
    cv.SetOffset(0, breathe);

    // 腮红：位置对应网页里 gap+24、h*0.7 的写法
    float blush = std::min(1.0f, std::max(p.blush, 0.3f));
    for (int side : {-1, 1}) cv.Ellipse(160 + side * 102, kCy + p.eye_y + kEyeH * 0.7f * 0.62f, 20, 10, kBlush, blush);

    for (int side : {-1, 1}) {
        EyeBox e = Eye(p, side);
        if (p.mode == EyeMode::kOpen && fs.blink < 0.8f) DrawOpenEye(cv, e, p, side, fs.blink);
        else DrawStrokeEye(cv, e, p.mode == EyeMode::kOpen ? EyeMode::kClosed : p.mode, side);
    }
    if (!p.peek) DrawMouth(cv, p);
    DrawExtras(cv, fs);
    DrawLedge(cv, fs);
    DrawCaption(cv, fs);
    cv.SetOffset(0, 0);
    DrawHearts(cv, hearts, fs.t_ms);
}

}  // namespace face
