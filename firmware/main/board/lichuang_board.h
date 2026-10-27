// 立创实战派 ESP32-S3 成品机的板级初始化。
// 引脚和初始化顺序参考小智固件 main/boards/lckfb/szpi-esp32s3（MIT）。
#pragma once

#include <driver/i2c_master.h>
#include <esp_lcd_panel_io.h>
#include <esp_lcd_panel_ops.h>
#include <esp_lcd_touch.h>

#include "audio_codec.h"

namespace board {

constexpr int kScreenW = 320;
constexpr int kScreenH = 240;

struct TouchPoint {
    bool pressed = false;
    int x = 0;  // 屏幕坐标，0..319
    int y = 0;  // 0..239
};

class LichuangBoard {
public:
    static LichuangBoard& Get();

    void Init();
    AudioCodec* Audio();
    esp_lcd_panel_handle_t Panel() const { return panel_; }

    // 把一块 RGB565 像素（大端，屏幕要的格式）交给 DMA 发到屏幕。
    // 发送是异步的：调用前先 AcquireTxSlot()，DMA 发完后槽位自动归还，
    // 所以最多同时有 kTxSlots 块缓冲在路上，复用缓冲前一定要先拿到槽位。
    static constexpr int kTxSlots = 2;
    void AcquireTxSlot();
    void Blit(int x, int y, int w, int h, const uint16_t* pixels);
    TouchPoint ReadTouch();
    void SetBacklight(int percent);  // 0..100
    void SetCameraPower(bool on);

private:
    LichuangBoard() = default;
    void InitI2c();
    void InitLcd();
    void InitTouch();
    void InitBacklight();

    i2c_master_bus_handle_t i2c_bus_ = nullptr;
    class Pca9557* pca9557_ = nullptr;
    esp_lcd_panel_io_handle_t panel_io_ = nullptr;
    esp_lcd_panel_handle_t panel_ = nullptr;
    esp_lcd_touch_handle_t touch_ = nullptr;
    void* tx_slots_ = nullptr;  // FreeRTOS 计数信号量
};

}  // namespace board
