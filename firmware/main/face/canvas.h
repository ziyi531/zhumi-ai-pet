// 极简软件光栅器：用“有向距离场”画抗锯齿的圆、椭圆、弧线、线段和多边形。
// 网页预览（ai_pet/static/faces.html）里用 Canvas 2D 画的东西，都在这里有对应的函数，
// 这样八条风的画法可以几乎逐行移植。像素格式 RGB565，存在 PSRAM 的整帧缓冲里。
#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

namespace face {

struct Vec2 {
    float x = 0, y = 0;
};

struct Rgb {
    uint8_t r = 0, g = 0, b = 0;
};

constexpr Rgb Hex(uint32_t v) { return Rgb{uint8_t(v >> 16), uint8_t(v >> 8), uint8_t(v)}; }

// 向量长度。不用 std::hypot：它为防溢出做了很多额外运算，在 ESP32 上慢几十倍，
// 而像素坐标远远到不了会溢出的大小。每帧要算几十万次，差别就是 2 帧/秒和 20 帧/秒。
inline float Len(float x, float y) { return sqrtf(x * x + y * y); }

// 距离函数：返回点到形状边界的距离，形状内部为负
using Sdf = std::function<float(float x, float y)>;

class Canvas {
public:
    Canvas(int w, int h);
    ~Canvas();

    int width() const { return w_; }
    int height() const { return h_; }
    const uint16_t* pixels() const { return px_; }

    void Clear(Rgb c);
    // 从同样宽、更高的画布 src 的第 src_y 行起，整块拷 height() 行过来（头部缓存用）
    void CopyRows(const Canvas& src, int src_y);
    // 整体平移（呼吸、探头时用），后面所有绘制都会加上这个偏移
    void SetOffset(float dx, float dy) { ox_ = dx; oy_ = dy; }
    // 裁剪：只在 clip(x,y) <= 0 的区域内画，传 nullptr 取消
    void SetClip(Sdf clip) { clip_ = std::move(clip); }

    // 按距离场填充；bbox 是形状在“未平移”坐标下的包围盒，用来少算像素
    void Fill(const Sdf& sdf, float x0, float y0, float x1, float y1, Rgb c, float alpha = 1.0f);
    // 同上，但颜色随位置变化（渐变）
    void FillShader(const Sdf& sdf, float x0, float y0, float x1, float y1,
                    const std::function<Rgb(float x, float y)>& shader, float alpha = 1.0f);
    // 上下渐变（颜色只随 y 变）：形状里面整段直接写，比 FillShader 快得多
    void FillVGradient(const Sdf& sdf, float x0, float y0, float x1, float y1,
                       const std::function<Rgb(float y)>& shader, float alpha = 1.0f);
    // 贴一张 8 位透明度蒙版（用来画预先渲染好的中文字）
    void BlitMask(const uint8_t* mask, int mw, int mh, int x, int y, Rgb c);

    // ---- 常用形状 ----
    void Circle(float cx, float cy, float r, Rgb c, float alpha = 1.0f);
    void Ellipse(float cx, float cy, float rx, float ry, Rgb c, float alpha = 1.0f);
    // 上半个椭圆（探头时的小爪子）
    void HalfEllipseTop(float cx, float cy, float rx, float ry, Rgb c, float alpha = 1.0f);
    void Rect(float x, float y, float w, float h, Rgb c, float alpha = 1.0f);
    void Line(Vec2 a, Vec2 b, float width, Rgb c, float alpha = 1.0f);
    // 圆弧描边，角度和 Canvas 2D 一样：弧度、y 轴向下、从 a0 顺时针到 a1（a1 > a0）
    void Arc(float cx, float cy, float r, float a0, float a1, float width, Rgb c, float alpha = 1.0f);
    void Polygon(const std::vector<Vec2>& pts, Rgb c, float alpha = 1.0f);
    void PolygonStroke(const std::vector<Vec2>& pts, float width, Rgb c, float alpha = 1.0f);

    // ---- 距离函数（也给外面拼形状用）----
    static float SdCircle(float x, float y, float cx, float cy, float r);
    static float SdEllipse(float x, float y, float cx, float cy, float rx, float ry);
    static float SdSegment(float x, float y, Vec2 a, Vec2 b);
    static float SdArc(float x, float y, float cx, float cy, float r, float a0, float a1);
    static float SdPolygon(float x, float y, const std::vector<Vec2>& pts);

private:
    int w_, h_;
    uint16_t* px_ = nullptr;
    float ox_ = 0, oy_ = 0;
    Sdf clip_;

    void BlendPixel(int x, int y, Rgb c, float a);
    // kRowConst：颜色只随行变，形状里面可以整段写
    template <bool kRowConst, typename ColorFn>
    void Raster(const Sdf& sdf, float x0, float y0, float x1, float y1, ColorFn color, float alpha);
};

}  // namespace face
