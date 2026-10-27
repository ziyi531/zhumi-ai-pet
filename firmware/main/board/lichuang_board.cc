#include "lichuang_board.h"

#include <driver/gpio.h>
#include <driver/ledc.h>
#include <driver/spi_common.h>
#include <es8311_codec.h>
#include <esp_lcd_panel_vendor.h>
#include <esp_lcd_touch_ft5x06.h>
#include <esp_log.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>

#include "box_audio_codec.h"

#define TAG "Board"

namespace board {
namespace {

// ---- 引脚（来自小智 szpi-esp32s3/config.h）----
constexpr gpio_num_t kI2cSda = GPIO_NUM_1;
constexpr gpio_num_t kI2cScl = GPIO_NUM_2;
constexpr gpio_num_t kI2sMclk = GPIO_NUM_38;
constexpr gpio_num_t kI2sWs = GPIO_NUM_13;
constexpr gpio_num_t kI2sBclk = GPIO_NUM_14;
constexpr gpio_num_t kI2sDin = GPIO_NUM_12;
constexpr gpio_num_t kI2sDout = GPIO_NUM_45;
constexpr gpio_num_t kLcdMosi = GPIO_NUM_40;
constexpr gpio_num_t kLcdSclk = GPIO_NUM_41;
constexpr gpio_num_t kLcdDc = GPIO_NUM_39;
constexpr gpio_num_t kBacklight = GPIO_NUM_42;  // 低电平点亮（反相）
constexpr int kAudioSampleRate = 24000;
constexpr uint8_t kEs7210Addr = 0x82;

// PCA9557 是 I2C 扩展 IO：bit0 屏幕片选，bit1 功放使能，bit2 摄像头电源（低有效）
constexpr uint8_t kPcaLcdCs = 0;
constexpr uint8_t kPcaSpeakerAmp = 1;
constexpr uint8_t kPcaCameraPower = 2;

}  // namespace

class Pca9557 {
public:
    explicit Pca9557(i2c_master_bus_handle_t bus) {
        i2c_device_config_t cfg = {};
        cfg.dev_addr_length = I2C_ADDR_BIT_LEN_7;
        cfg.device_address = 0x19;
        cfg.scl_speed_hz = 400000;
        ESP_ERROR_CHECK(i2c_master_bus_add_device(bus, &cfg, &dev_));
        WriteReg(0x01, 0x03);  // 输出初值
        WriteReg(0x03, 0xf8);  // 低 3 位设为输出
    }
    void Set(uint8_t bit, bool level) {
        uint8_t data = ReadReg(0x01);
        data = (data & ~(1 << bit)) | (static_cast<uint8_t>(level) << bit);
        WriteReg(0x01, data);
    }

private:
    i2c_master_dev_handle_t dev_ = nullptr;
    void WriteReg(uint8_t reg, uint8_t value) {
        uint8_t buf[2] = {reg, value};
        ESP_ERROR_CHECK(i2c_master_transmit(dev_, buf, 2, 100));
    }
    uint8_t ReadReg(uint8_t reg) {
        uint8_t value = 0;
        ESP_ERROR_CHECK(i2c_master_transmit_receive(dev_, &reg, 1, &value, 1, 100));
        return value;
    }
};

// 小智的 BoxAudioCodec（ES8311 放音 + ES7210 收音），开关喇叭时顺带控制功放
class PetAudioCodec : public BoxAudioCodec {
public:
    PetAudioCodec(i2c_master_bus_handle_t bus, Pca9557* pca)
        : BoxAudioCodec(bus, kAudioSampleRate, kAudioSampleRate, kI2sMclk, kI2sBclk, kI2sWs, kI2sDout,
                        kI2sDin, GPIO_NUM_NC, ES8311_CODEC_DEFAULT_ADDR, kEs7210Addr,
                        true,   // 第三路麦克风输入接的是喇叭回采，用来做回声消除
                        37.5f,  // 麦克风增益（ES7210 最大值；28dB 时 30cm 外说话录到的 RMS 才 60~300，太小）
                        2, 0.0f),
          pca_(pca) {
        // 只插 USB 时，喇叭一响的大电流会把板子拉到掉电重启（真机实测 70 且功放和解码芯片同时开会），
        // 功放先开之后 50 稳，但听团子说话偏小
        output_volume_ = kDefaultVolume;
    }

    void EnableOutput(bool enable) override {
        if (enable == output_enabled()) return;
        // 开：先开功放、等它稳住，再让解码芯片出声，两股电流错开；关：反过来，先关功放免得“啪”一声
        if (enable) {
            pca_->Set(kPcaSpeakerAmp, true);
            vTaskDelay(pdMS_TO_TICKS(30));
            BoxAudioCodec::EnableOutput(true);
        } else {
            pca_->Set(kPcaSpeakerAmp, false);
            BoxAudioCodec::EnableOutput(false);
        }
    }

private:
    static constexpr int kDefaultVolume = 65;

private:
    Pca9557* pca_;
};

LichuangBoard& LichuangBoard::Get() {
    static LichuangBoard board;
    return board;
}

void LichuangBoard::Init() {
    InitI2c();
    InitLcd();
    InitTouch();
    InitBacklight();
    SetCameraPower(false);  // 摄像头先关着，用到时再开，省电
}

void LichuangBoard::InitI2c() {
    i2c_master_bus_config_t cfg = {};
    cfg.i2c_port = I2C_NUM_1;
    cfg.sda_io_num = kI2cSda;
    cfg.scl_io_num = kI2cScl;
    cfg.clk_source = I2C_CLK_SRC_DEFAULT;
    cfg.glitch_ignore_cnt = 7;
    cfg.flags.enable_internal_pullup = 1;
    ESP_ERROR_CHECK(i2c_new_master_bus(&cfg, &i2c_bus_));
    pca9557_ = new Pca9557(i2c_bus_);
}

void LichuangBoard::InitLcd() {
    spi_bus_config_t bus = {};
    bus.mosi_io_num = kLcdMosi;
    bus.miso_io_num = GPIO_NUM_NC;
    bus.sclk_io_num = kLcdSclk;
    bus.quadwp_io_num = GPIO_NUM_NC;
    bus.quadhd_io_num = GPIO_NUM_NC;
    bus.max_transfer_sz = kScreenW * 40 * sizeof(uint16_t);
    ESP_ERROR_CHECK(spi_bus_initialize(SPI3_HOST, &bus, SPI_DMA_CH_AUTO));

    tx_slots_ = xSemaphoreCreateCounting(kTxSlots, kTxSlots);
    esp_lcd_panel_io_spi_config_t io = {};
    // 每块像素 DMA 发完就归还一个槽位
    io.on_color_trans_done = [](esp_lcd_panel_io_handle_t, esp_lcd_panel_io_event_data_t*, void* ctx) {
        BaseType_t woken = pdFALSE;
        xSemaphoreGiveFromISR(static_cast<SemaphoreHandle_t>(ctx), &woken);
        return woken == pdTRUE;
    };
    io.user_ctx = tx_slots_;
    io.cs_gpio_num = GPIO_NUM_NC;  // 片选走 PCA9557
    io.dc_gpio_num = kLcdDc;
    io.spi_mode = 2;
    io.pclk_hz = 80 * 1000 * 1000;
    io.trans_queue_depth = 10;
    io.lcd_cmd_bits = 8;
    io.lcd_param_bits = 8;
    ESP_ERROR_CHECK(esp_lcd_new_panel_io_spi(SPI3_HOST, &io, &panel_io_));

    esp_lcd_panel_dev_config_t dev = {};
    dev.reset_gpio_num = GPIO_NUM_NC;
    dev.rgb_ele_order = LCD_RGB_ELEMENT_ORDER_RGB;
    dev.bits_per_pixel = 16;
    ESP_ERROR_CHECK(esp_lcd_new_panel_st7789(panel_io_, &dev, &panel_));

    esp_lcd_panel_reset(panel_);
    pca9557_->Set(kPcaLcdCs, false);
    esp_lcd_panel_init(panel_);
    esp_lcd_panel_invert_color(panel_, true);
    esp_lcd_panel_swap_xy(panel_, true);
    esp_lcd_panel_mirror(panel_, true, false);
    esp_lcd_panel_disp_on_off(panel_, true);
    ESP_LOGI(TAG, "LCD ready %dx%d", kScreenW, kScreenH);
}

void LichuangBoard::InitTouch() {
    esp_lcd_touch_config_t cfg = {};
    cfg.x_max = kScreenH;
    cfg.y_max = kScreenW;
    cfg.rst_gpio_num = GPIO_NUM_NC;
    cfg.int_gpio_num = GPIO_NUM_NC;
    cfg.flags.swap_xy = 1;
    cfg.flags.mirror_x = 1;
    cfg.flags.mirror_y = 0;

    esp_lcd_panel_io_handle_t io = nullptr;
    esp_lcd_panel_io_i2c_config_t io_cfg = {};
    io_cfg.dev_addr = ESP_LCD_TOUCH_IO_I2C_FT5x06_ADDRESS;
    io_cfg.control_phase_bytes = 1;
    io_cfg.dc_bit_offset = 0;
    io_cfg.lcd_cmd_bits = 8;
    io_cfg.flags.disable_control_phase = 1;
    io_cfg.scl_speed_hz = 400000;
    if (esp_lcd_new_panel_io_i2c(i2c_bus_, &io_cfg, &io) != ESP_OK) {
        ESP_LOGW(TAG, "Touch IO init failed, continuing without touch");
        return;
    }
    if (esp_lcd_touch_new_i2c_ft5x06(io, &cfg, &touch_) != ESP_OK) {
        ESP_LOGW(TAG, "FT5x06 not found, continuing without touch");
        touch_ = nullptr;
    }
}

void LichuangBoard::InitBacklight() {
    ledc_timer_config_t timer = {};
    timer.speed_mode = LEDC_LOW_SPEED_MODE;
    timer.duty_resolution = LEDC_TIMER_10_BIT;
    timer.timer_num = LEDC_TIMER_0;
    timer.freq_hz = 25000;
    timer.clk_cfg = LEDC_AUTO_CLK;
    ESP_ERROR_CHECK(ledc_timer_config(&timer));

    ledc_channel_config_t ch = {};
    ch.gpio_num = kBacklight;
    ch.speed_mode = LEDC_LOW_SPEED_MODE;
    ch.channel = LEDC_CHANNEL_0;
    ch.timer_sel = LEDC_TIMER_0;
    ch.duty = 0;
    ch.flags.output_invert = 1;
    ESP_ERROR_CHECK(ledc_channel_config(&ch));
    SetBacklight(80);
}

void LichuangBoard::SetBacklight(int percent) {
    if (percent < 0) percent = 0;
    if (percent > 100) percent = 100;
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0, 1023 * percent / 100);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0);
}

void LichuangBoard::SetCameraPower(bool on) {
    pca9557_->Set(kPcaCameraPower, !on);
}

AudioCodec* LichuangBoard::Audio() {
    static PetAudioCodec codec(i2c_bus_, pca9557_);
    return &codec;
}

void LichuangBoard::AcquireTxSlot() {
    // 正常几毫秒就归还；等 500ms 还没还说明 DMA 完成回调没来，报一声别死等（屏幕卡住时好查）
    if (xSemaphoreTake(static_cast<SemaphoreHandle_t>(tx_slots_), pdMS_TO_TICKS(500)) != pdTRUE) {
        ESP_LOGW(TAG, "LCD DMA slot not returned in 500ms");
    }
}

void LichuangBoard::Blit(int x, int y, int w, int h, const uint16_t* pixels) {
    esp_lcd_panel_draw_bitmap(panel_, x, y, x + w, y + h, pixels);
}

TouchPoint LichuangBoard::ReadTouch() {
    TouchPoint tp;
    if (!touch_) return tp;
    esp_lcd_touch_read_data(touch_);
    esp_lcd_touch_point_data_t point = {};
    uint8_t count = 0;
    if (esp_lcd_touch_get_data(touch_, &point, &count, 1) == ESP_OK && count > 0) {
        tp.pressed = true;
        tp.x = point.x;
        tp.y = point.y;
    }
    return tp;
}

}  // namespace board
