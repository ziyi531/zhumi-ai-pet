#include "canvas.h"

#include <algorithm>
#include <cstring>

#include <esp_heap_caps.h>

namespace face {
namespace {

constexpr float kPi = 3.14159265f;
constexpr float kSkip = 0.7f;  // 按距离跳格子时只跳七成，见 Raster

inline uint16_t Pack(Rgb c) { return uint16_t(((c.r & 0xF8) << 8) | ((c.g & 0xFC) << 3) | (c.b >> 3)); }
inline Rgb Unpack(uint16_t v) {
    uint8_t r = (v >> 8) & 0xF8, g = (v >> 3) & 0xFC, b = (v << 3) & 0xF8;
    return Rgb{uint8_t(r | r >> 5), uint8_t(g | g >> 6), uint8_t(b | b >> 5)};
}
inline float Clamp01(float v) { return v < 0 ? 0 : (v > 1 ? 1 : v); }

// 多边形的边先算好（每条边的方向和 1/长度²），逐像素求距离时就不用每条边都做一次除法
struct Edge {
    float x, y, ex, ey, inv, yj;  // 起点 v[i]、方向 v[j]-v[i]、1/|e|²、终点的 y
};

std::vector<Edge> Edges(const std::vector<Vec2>& v) {
    std::vector<Edge> es;
    for (size_t i = 0, j = v.size() - 1; i < v.size(); j = i, ++i) {
        float ex = v[j].x - v[i].x, ey = v[j].y - v[i].y, l2 = ex * ex + ey * ey;
        es.push_back({v[i].x, v[i].y, ex, ey, l2 > 0 ? 1 / l2 : 0, v[j].y});
    }
    return es;
}

// 和 Canvas::SdPolygon 同一个算法，只是用了预先算好的边
float SdEdges(float x, float y, const std::vector<Edge>& es) {
    float d = 1e30f, s = 1.0f;
    for (const Edge& e : es) {
        float wx = x - e.x, wy = y - e.y;
        float t = Clamp01((wx * e.ex + wy * e.ey) * e.inv);
        float bx = wx - e.ex * t, by = wy - e.ey * t;
        d = std::min(d, bx * bx + by * by);
        bool c1 = y >= e.y, c2 = y < e.yj, c3 = e.ex * wy > e.ey * wx;
        if ((c1 && c2 && c3) || (!c1 && !c2 && !c3)) s = -s;
    }
    return s * sqrtf(d);
}

}  // namespace

Canvas::Canvas(int w, int h) : w_(w), h_(h) {
    px_ = static_cast<uint16_t*>(heap_caps_malloc(w * h * sizeof(uint16_t), MALLOC_CAP_SPIRAM));
}

Canvas::~Canvas() { heap_caps_free(px_); }

void Canvas::Clear(Rgb c) {
    uint16_t v = Pack(c);
    std::fill(px_, px_ + w_ * h_, v);
}

void Canvas::BlendPixel(int x, int y, Rgb c, float a) {
    uint16_t& p = px_[y * w_ + x];
    if (a >= 0.996f) {
        p = Pack(c);
        return;
    }
    Rgb d = Unpack(p);
    d.r = uint8_t(d.r + (c.r - d.r) * a);
    d.g = uint8_t(d.g + (c.g - d.g) * a);
    d.b = uint8_t(d.b + (c.b - d.b) * a);
    p = Pack(d);
}

template <bool kRowConst, typename ColorFn>
void Canvas::Raster(const Sdf& sdf, float x0, float y0, float x1, float y1, ColorFn color, float alpha) {
    if (alpha <= 0.004f) return;
    // bbox 是未平移坐标，转成屏幕像素范围（多留 1 像素给抗锯齿）
    int ix0 = std::max(0, int(std::floor(x0 + ox_)) - 1), ix1 = std::min(w_ - 1, int(std::ceil(x1 + ox_)) + 1);
    int iy0 = std::max(0, int(std::floor(y0 + oy_)) - 1), iy1 = std::min(h_ - 1, int(std::ceil(y1 + oy_)) + 1);
    // 跳格子：距离场每走 1 像素最多变 1，所以离边界 d 像素远时，接下来约 d 个像素都在同一侧——
    // 外面的直接跳过，里面的不用再算距离（整格覆盖）。乘 kSkip 留余量：椭圆的距离是近似值，不严格满足“每像素最多变 1”。
    // 不跳的话大圆头、头的描边圈要把包围盒里几万个像素挨个算一遍，一帧要 400ms。
    for (int y = iy0; y <= iy1; ++y) {
        float fy = y + 0.5f - oy_;
        for (int x = ix0; x <= ix1; ++x) {
            float fx = x + 0.5f - ox_;
            float d = sdf(fx, fy);
            if (d >= 0.5f) {
                x += int((d - 0.5f) * kSkip);
                continue;
            }
            if (clip_) {
                d = std::max(d, clip_(fx, fy));  // 裁剪函数（眼皮）不是距离场，带裁剪时里面不能跳
            } else if (d <= -1.5f) {
                int run = std::min(int((-d - 0.5f) * kSkip), ix1 - x);  // 这些像素一定整格在形状里
                if (kRowConst && alpha >= 0.996f) {
                    // 颜色只随行变（纯色、上下渐变）又不透明：整段直接写，不用每个像素再算颜色、再混合
                    uint16_t* row = px_ + y * w_;
                    std::fill(row + x, row + x + run + 1, Pack(color(fx, fy)));
                } else {
                    for (int k = 0; k <= run; ++k) BlendPixel(x + k, y, color(fx + k, fy), alpha);
                }
                x += run;
                continue;
            }
            float cov = Clamp01(0.5f - d);  // 边缘 1 像素宽的过渡就是抗锯齿
            if (cov > 0) BlendPixel(x, y, color(fx, fy), cov * alpha);
        }
    }
}

void Canvas::Fill(const Sdf& sdf, float x0, float y0, float x1, float y1, Rgb c, float alpha) {
    Raster<true>(sdf, x0, y0, x1, y1, [c](float, float) { return c; }, alpha);
}

void Canvas::FillShader(const Sdf& sdf, float x0, float y0, float x1, float y1,
                        const std::function<Rgb(float, float)>& shader, float alpha) {
    Raster<false>(sdf, x0, y0, x1, y1, shader, alpha);
}

void Canvas::FillVGradient(const Sdf& sdf, float x0, float y0, float x1, float y1,
                           const std::function<Rgb(float y)>& shader, float alpha) {
    Raster<true>(sdf, x0, y0, x1, y1, [&shader](float, float y) { return shader(y); }, alpha);
}

void Canvas::BlitMask(const uint8_t* mask, int mw, int mh, int x, int y, Rgb c) {
    x += int(std::lround(ox_));
    y += int(std::lround(oy_));
    for (int j = 0; j < mh; ++j) {
        int py = y + j;
        if (py < 0 || py >= h_) continue;
        for (int i = 0; i < mw; ++i) {
            int px = x + i;
            uint8_t a = mask[j * mw + i];
            if (a == 0 || px < 0 || px >= w_) continue;
            BlendPixel(px, py, c, a / 255.0f);
        }
    }
}

// ---------- 距离函数 ----------
float Canvas::SdCircle(float x, float y, float cx, float cy, float r) {
    return Len(x - cx, y - cy) - r;
}

float Canvas::SdEllipse(float x, float y, float cx, float cy, float rx, float ry) {
    // Inigo Quilez 的椭圆近似距离，对画图足够准
    float px = (x - cx) / rx, py = (y - cy) / ry;
    float k0 = Len(px, py);
    float k1 = Len(px / rx, py / ry);
    if (k1 < 1e-6f) return -std::min(rx, ry);
    return k0 * (k0 - 1.0f) / k1;
}

float Canvas::SdSegment(float x, float y, Vec2 a, Vec2 b) {
    float pax = x - a.x, pay = y - a.y, bax = b.x - a.x, bay = b.y - a.y;
    float len2 = bax * bax + bay * bay;
    float h = len2 > 0 ? Clamp01((pax * bax + pay * bay) / len2) : 0;
    return Len(pax - bax * h, pay - bay * h);
}

float Canvas::SdArc(float x, float y, float cx, float cy, float r, float a0, float a1) {
    float dx = x - cx, dy = y - cy;
    float th = std::atan2(dy, dx);
    while (th < a0) th += 2 * kPi;
    while (th >= a0 + 2 * kPi) th -= 2 * kPi;
    if (th <= a1) return std::fabs(Len(dx, dy) - r);
    float d0 = Len(x - (cx + std::cos(a0) * r), y - (cy + std::sin(a0) * r));
    float d1 = Len(x - (cx + std::cos(a1) * r), y - (cy + std::sin(a1) * r));
    return std::min(d0, d1);
}

float Canvas::SdPolygon(float x, float y, const std::vector<Vec2>& v) {
    // Inigo Quilez 的多边形距离：最近边的距离 + 射线法判断内外
    size_t n = v.size();
    float d = (x - v[0].x) * (x - v[0].x) + (y - v[0].y) * (y - v[0].y);
    float s = 1.0f;
    for (size_t i = 0, j = n - 1; i < n; j = i, ++i) {
        float ex = v[j].x - v[i].x, ey = v[j].y - v[i].y;
        float wx = x - v[i].x, wy = y - v[i].y;
        float t = Clamp01((wx * ex + wy * ey) / (ex * ex + ey * ey));
        float bx = wx - ex * t, by = wy - ey * t;
        d = std::min(d, bx * bx + by * by);
        bool c1 = y >= v[i].y, c2 = y < v[j].y, c3 = ex * wy > ey * wx;
        if ((c1 && c2 && c3) || (!c1 && !c2 && !c3)) s = -s;
    }
    return s * std::sqrt(d);
}

// ---------- 常用形状 ----------
void Canvas::Circle(float cx, float cy, float r, Rgb c, float alpha) {
    Fill([=](float x, float y) { return SdCircle(x, y, cx, cy, r); }, cx - r, cy - r, cx + r, cy + r, c, alpha);
}

void Canvas::Ellipse(float cx, float cy, float rx, float ry, Rgb c, float alpha) {
    Fill([=](float x, float y) { return SdEllipse(x, y, cx, cy, rx, ry); }, cx - rx, cy - ry, cx + rx, cy + ry, c,
         alpha);
}

void Canvas::HalfEllipseTop(float cx, float cy, float rx, float ry, Rgb c, float alpha) {
    Fill([=](float x, float y) { return std::max(SdEllipse(x, y, cx, cy, rx, ry), y - cy); }, cx - rx, cy - ry,
         cx + rx, cy, c, alpha);
}

void Canvas::Rect(float x, float y, float w, float h, Rgb c, float alpha) {
    Fill([=](float px, float py) { return std::max(std::max(x - px, px - (x + w)), std::max(y - py, py - (y + h))); },
         x, y, x + w, y + h, c, alpha);
}

void Canvas::Line(Vec2 a, Vec2 b, float width, Rgb c, float alpha) {
    float hw = width / 2;
    Fill([=](float x, float y) { return SdSegment(x, y, a, b) - hw; }, std::min(a.x, b.x) - hw,
         std::min(a.y, b.y) - hw, std::max(a.x, b.x) + hw, std::max(a.y, b.y) + hw, c, alpha);
}

void Canvas::Arc(float cx, float cy, float r, float a0, float a1, float width, Rgb c, float alpha) {
    float hw = width / 2, e = r + hw;
    Fill([=](float x, float y) { return SdArc(x, y, cx, cy, r, a0, a1) - hw; }, cx - e, cy - e, cx + e, cy + e, c,
         alpha);
}

void Canvas::Polygon(const std::vector<Vec2>& pts, Rgb c, float alpha) {
    float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
    for (auto& p : pts) {
        x0 = std::min(x0, p.x); y0 = std::min(y0, p.y);
        x1 = std::max(x1, p.x); y1 = std::max(y1, p.y);
    }
    auto es = Edges(pts);
    Fill([&es](float x, float y) { return SdEdges(x, y, es); }, x0, y0, x1, y1, c, alpha);
}

void Canvas::PolygonStroke(const std::vector<Vec2>& pts, float width, Rgb c, float alpha) {
    float hw = width / 2, x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
    for (auto& p : pts) {
        x0 = std::min(x0, p.x); y0 = std::min(y0, p.y);
        x1 = std::max(x1, p.x); y1 = std::max(y1, p.y);
    }
    auto es = Edges(pts);
    Fill([&es, hw](float x, float y) { return std::fabs(SdEdges(x, y, es)) - hw; }, x0 - hw, y0 - hw, x1 + hw,
         y1 + hw, c, alpha);
}

void Canvas::CopyRows(const Canvas& src, int src_y) {
    src_y = std::clamp(src_y, 0, src.h_ - h_);
    memcpy(px_, src.px_ + src_y * w_, size_t(w_) * h_ * sizeof(uint16_t));
}

}  // namespace face
