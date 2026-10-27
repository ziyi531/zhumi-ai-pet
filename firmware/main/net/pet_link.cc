#include "pet_link.h"

#include <cJSON.h>
#include <atomic>
#include <cstring>
#include <esp_crt_bundle.h>
#include <esp_event.h>
#include <esp_log.h>
#include <esp_netif.h>
#include <esp_timer.h>
#include <esp_websocket_client.h>
#include <esp_wifi.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#define TAG "PetLink"

namespace net {
namespace {

constexpr int kProto = 1;

std::string Str(const cJSON* obj, const char* key, const char* def = "") {
    const cJSON* v = cJSON_GetObjectItemCaseSensitive(obj, key);
    return cJSON_IsString(v) ? v->valuestring : def;
}

// 认识的 WiFi：家里（连局域网的大脑）、手机热点（出门，走服务器中转）。两个都在时连家里的
struct KnownWifi {
    const char* ssid;
    const char* password;
};
const KnownWifi kWifis[] = {{CONFIG_PET_WIFI_SSID, CONFIG_PET_WIFI_PASSWORD},
                            {CONFIG_PET_WIFI2_SSID, CONFIG_PET_WIFI2_PASSWORD}};
std::atomic<bool> g_scanning{false};

// 扫一遍周围的 2.4GHz WiFi，挑认识的里面排在前面、又看得见的那个连。
// 连不上（原因 201）时日志里能看出是名字对不上、还是根本没有 2.4GHz（热点常默认 5GHz）
void ScanThenConnect(void* arg) {
    auto* self = static_cast<PetLink*>(arg);
    int pick = -1;
    wifi_scan_config_t scan = {};
    if (esp_wifi_scan_start(&scan, true) == ESP_OK) {
        uint16_t n = 12;
        wifi_ap_record_t aps[12];
        esp_wifi_scan_get_ap_records(&n, aps);
        for (int i = 0; i < n; ++i) {
            const char* ssid = reinterpret_cast<char*>(aps[i].ssid);
            int known = -1;
            for (int k = 0; k < 2; ++k) {
                if (kWifis[k].ssid[0] && strcmp(ssid, kWifis[k].ssid) == 0) known = k;
            }
            if (known >= 0 && (pick < 0 || known < pick)) pick = known;
            ESP_LOGI(TAG, "scan: \"%s\" ch%d rssi %d auth %d%s", ssid, aps[i].primary, aps[i].rssi, aps[i].authmode,
                     known == 0 ? "  <- home" : known == 1 ? "  <- hotspot" : "");
        }
    }
    if (pick < 0) {
        ESP_LOGW(TAG, "no known WiFi in sight (2.4GHz only), retrying");
        pick = self->WifiIndex() == 0 && kWifis[1].ssid[0] ? 1 : 0;  // 都没看见：轮流试
    }
    self->UseWifi(pick, kWifis[pick].ssid, kWifis[pick].password);
    g_scanning = false;
    esp_wifi_connect();
    vTaskDelete(nullptr);
}

void Rescan(PetLink* self) {
    if (!g_scanning.exchange(true)) xTaskCreate(ScanThenConnect, "wifi_scan", 4096, self, 3, nullptr);
}

float Num(const cJSON* obj, const char* key, float def = 0) {
    const cJSON* v = cJSON_GetObjectItemCaseSensitive(obj, key);
    return cJSON_IsNumber(v) ? float(v->valuedouble) : def;
}

}  // namespace

struct LinkCallbacks {
    static void OnWifi(void* arg, esp_event_base_t base, int32_t id, void* data) {
        auto* self = static_cast<PetLink*>(arg);
        if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
            Rescan(self);  // 事件回调里不能阻塞扫描，放到单独的任务里
        } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
            // 原因码：201 找不到这个网络（走远了、或者热点只开了 5GHz），15/204 密码不对，210/211 加密方式不支持
            auto* ev = static_cast<wifi_event_sta_disconnected_t*>(data);
            ESP_LOGW(TAG, "WiFi disconnected (reason %d, rssi %d), retrying", ev->reason, ev->rssi);
            self->SetOnline(false);
            if (ev->reason == WIFI_REASON_NO_AP_FOUND || ++self->wifi_failures_ >= 3) {
                self->wifi_failures_ = 0;
                Rescan(self);  // 这个 WiFi 不在了（带出门、回到家）：重新扫，换认识的另一个
            } else {
                esp_wifi_connect();
            }
        } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
            ESP_LOGI(TAG, "WiFi connected (%s)", self->WifiIndex() == 0 ? "home" : "hotspot");
            self->wifi_failures_ = 0;
            // 家里 WiFi 连局域网的大脑；手机热点连不到家里的局域网，直接走服务器中转
            int want = (self->WifiIndex() == 0 || !CONFIG_PET_SERVER_URL_REMOTE[0]) ? 0 : 1;
            if (!self->client_) {
                self->url_index_ = want;
                self->StartSocket();
            } else if (self->url_index_ != want) {
                self->url_index_ = 1 - want;  // 让切换任务换到 want
                self->Notify(PetLink::kSwitchBit);
            }
        }
    }

    static void OnSocket(void* arg, esp_event_base_t, int32_t id, void* data) {
        auto* self = static_cast<PetLink*>(arg);
        auto* ev = static_cast<esp_websocket_event_data_t*>(data);
        switch (id) {
            case WEBSOCKET_EVENT_CONNECTED: {
                ESP_LOGI(TAG, "brain connected (%s)", self->url_index_ == 0 ? "LAN" : "remote");
                self->failures_ = 0;
                char hello[160];
                snprintf(hello, sizeof(hello),
                         R"({"type":"hello","device":"esp32","proto":%d,"audio_out":"opus","token":"%s"})", kProto,
                         CONFIG_PET_DEVICE_TOKEN);
                self->SendJson(hello);
                break;
            }
            case WEBSOCKET_EVENT_DISCONNECTED:
                if (!self->online_) self->OnConnectFailed();  // 还没进 welcome 就断了，算这个地址连不上
                self->SetOnline(false);
                break;
            case WEBSOCKET_EVENT_CLOSED:
                // 大脑正常关闭（比如重启大脑）会先发“关闭连接”，客户端收到后就不再自动重连了（真机踩过：
                // 重启大脑后板子再也连不回来，喊唤醒词只能鹦鹉学舌）。我们自己过一会儿重新连
                ESP_LOGI(TAG, "brain closed the connection, reconnecting soon");
                self->SetOnline(false);
                self->Notify(PetLink::kRestartBit);
                break;
            case WEBSOCKET_EVENT_DATA:
                // op_code 1 文本帧、2 二进制帧（团子的声音，一帧一个 Opus 包）。
                // 大消息可能分片，我们的消息和 Opus 包都比缓冲区小，只处理完整的一帧
                if (ev->payload_offset != 0 || ev->data_len != ev->payload_len) break;
                if (ev->op_code == 0x1) {
                    self->HandleText(ev->data_ptr, ev->data_len);
                } else if (ev->op_code == 0x2 && self->on_audio) {
                    self->on_audio(reinterpret_cast<const uint8_t*>(ev->data_ptr), ev->data_len);
                }
                break;
            default:
                break;
        }
    }
};

PetLink& PetLink::Get() {
    static PetLink link;
    return link;
}

void PetLink::Start() {
    if (strlen(CONFIG_PET_WIFI_SSID) == 0) {
        ESP_LOGW(TAG, "WiFi SSID not set (idf.py menuconfig -> AI Pet), staying offline");
        return;
    }
    StartWifi();
}

void PetLink::StartWifi() {
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();
    wifi_init_config_t init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &LinkCallbacks::OnWifi, this));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &LinkCallbacks::OnWifi, this));

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));  // 连哪个 WiFi 由扫描后决定（UseWifi）
    // 省电：空闲时 WiFi 进入调制解调器睡眠，收发时自动醒来
    esp_wifi_set_ps(WIFI_PS_MIN_MODEM);
    ESP_ERROR_CHECK(esp_wifi_start());
}

void PetLink::UseWifi(int index, const char* ssid, const char* password) {
    wifi_index_ = index;
    wifi_config_t cfg = {};
    strncpy(reinterpret_cast<char*>(cfg.sta.ssid), ssid, sizeof(cfg.sta.ssid) - 1);
    strncpy(reinterpret_cast<char*>(cfg.sta.password), password, sizeof(cfg.sta.password) - 1);
    esp_wifi_set_config(WIFI_IF_STA, &cfg);
    ESP_LOGI(TAG, "connecting to %s WiFi \"%s\"", index == 0 ? "home" : "hotspot", ssid);
}

const char* PetLink::Url() const {
    return url_index_ == 0 ? CONFIG_PET_SERVER_URL : CONFIG_PET_SERVER_URL_REMOTE;
}

void PetLink::OnConnectFailed() {
    // 连着手机热点时局域网地址必然连不上；连续失败 2 次就换另一个地址（中转服务器也连不上就换回来）
    if (strlen(CONFIG_PET_SERVER_URL_REMOTE) == 0 || ++failures_ < 2) return;
    failures_ = 0;
    Notify(kSwitchBit);
}

void PetLink::Notify(uint32_t bit) {
    if (switch_task_) xTaskNotify(static_cast<TaskHandle_t>(switch_task_), bit, eSetBits);
}

// 换地址、重新连都要 stop 客户端，不能在 websocket 自己的事件回调里做，所以放在这个任务里
void PetLink::SwitchTask(void* arg) {
    auto* self = static_cast<PetLink*>(arg);
    while (true) {
        uint32_t bits = 0;
        xTaskNotifyWait(0, UINT32_MAX, &bits, portMAX_DELAY);
        auto client = static_cast<esp_websocket_client_handle_t>(self->client_);
        if (bits & kRestartBit) vTaskDelay(pdMS_TO_TICKS(2000));  // 大脑在重启：等它起来
        if (bits & kSwitchBit) {
            self->url_index_ = 1 - self->url_index_;
            ESP_LOGI(TAG, "switching to %s: %s", self->url_index_ == 0 ? "LAN" : "remote", self->Url());
        } else {
            ESP_LOGI(TAG, "reconnecting: %s", self->Url());
        }
        esp_websocket_client_stop(client);
        esp_websocket_client_set_uri(client, self->Url());
        esp_websocket_client_start(client);
    }
}

void PetLink::StartSocket() {
    if (client_) return;  // 已经建过了，断线后客户端会自己重连
    if (!switch_task_) {
        TaskHandle_t t = nullptr;
        xTaskCreate(SwitchTask, "link_switch", 4096, this, 4, &t);
        switch_task_ = t;
    }
    esp_websocket_client_config_t cfg = {};
    cfg.uri = Url();
    cfg.crt_bundle_attach = esp_crt_bundle_attach;  // wss：用内置的根证书校验服务器（Let's Encrypt）
    cfg.reconnect_timeout_ms = 3000;
    cfg.network_timeout_ms = 10000;
    cfg.ping_interval_sec = 20;  // 协议层保活，网页团子用的是应用层 ping
    cfg.buffer_size = 2048;
    auto client = esp_websocket_client_init(&cfg);
    esp_websocket_register_events(client, WEBSOCKET_EVENT_ANY, &LinkCallbacks::OnSocket, this);
    esp_websocket_client_start(client);
    client_ = client;
}

bool PetLink::SendJson(const std::string& json) {
    auto client = static_cast<esp_websocket_client_handle_t>(client_);
    if (!client || !esp_websocket_client_is_connected(client)) return false;
    return esp_websocket_client_send_text(client, json.data(), json.size(), pdMS_TO_TICKS(1000)) >= 0;
}

bool PetLink::SendChat(const std::string& text) {
    if (!online_) return false;
    cJSON* msg = cJSON_CreateObject();
    cJSON_AddStringToObject(msg, "type", "chat");
    cJSON_AddStringToObject(msg, "text", text.c_str());
    char* s = cJSON_PrintUnformatted(msg);
    bool ok = SendJson(s);
    cJSON_free(s);
    cJSON_Delete(msg);
    return ok;
}

bool PetLink::SendPat(int count) {
    if (!online_ || count <= 0) return false;
    char buf[48];
    snprintf(buf, sizeof(buf), R"({"type":"pat","count":%d})", count);
    return SendJson(buf);
}

bool PetLink::SendListen(const char* state) {
    if (!online_) return false;
    char buf[96];
    if (strcmp(state, "start") == 0) {
        snprintf(buf, sizeof(buf), R"({"type":"listen","state":"start","format":"opus","rate":16000})");
    } else {
        snprintf(buf, sizeof(buf), R"({"type":"listen","state":"%s"})", state);
    }
    return SendJson(buf);
}

bool PetLink::SendHush() { return online_ && SendJson(R"({"type":"hush"})"); }

bool PetLink::ListenStart() { return SendListen("start"); }
bool PetLink::ListenStop() { return SendListen("stop"); }
bool PetLink::ListenCancel() { return SendListen("cancel"); }

bool PetLink::SendAudio(const uint8_t* data, size_t len) {
    auto client = static_cast<esp_websocket_client_handle_t>(client_);
    if (!online_ || !client || !esp_websocket_client_is_connected(client)) return false;
    return esp_websocket_client_send_bin(client, reinterpret_cast<const char*>(data), len, pdMS_TO_TICKS(500)) >= 0;
}

int PetLink::BrainHour() const {
    if (brain_seconds_of_day_ < 0) return -1;
    int64_t elapsed = (esp_timer_get_time() - time_base_us_) / 1000000;
    return int(((brain_seconds_of_day_ + elapsed) / 3600) % 24);
}

void PetLink::SetOnline(bool on) {
    if (online_ == on) return;
    online_ = on;
    ESP_LOGI(TAG, "%s", on ? "online" : "offline");
    if (on_online) on_online(on);
}

void PetLink::HandleText(const char* data, int len) {
    cJSON* msg = cJSON_ParseWithLength(data, len);
    if (!msg) return;
    std::string type = Str(msg, "type");

    if (type == "welcome" || type == "state") {
        if (type == "welcome") SetOnline(true);
        std::string t = Str(msg, "time");  // "YYYY-MM-DD HH:MM:SS"
        int hh, mm, ss;
        if (t.size() >= 19 && sscanf(t.c_str() + 11, "%d:%d:%d", &hh, &mm, &ss) == 3) {
            brain_seconds_of_day_ = hh * 3600 + mm * 60 + ss;
            time_base_us_ = esp_timer_get_time();
        }
        const cJSON* st = cJSON_GetObjectItemCaseSensitive(msg, "state");
        if (st && on_state) {
            BrainState s;
            s.phase = Str(st, "phase", "awake");
            s.loneliness = Num(st, "loneliness");
            s.energy = Num(st, "energy", 1);
            s.sleepy_talk = cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(st, "sleepy_talk"));
            on_state(s);
        }
    } else if (type == "say") {
        if (on_say) {
            Say s;
            s.text = Str(msg, "text");
            s.face = Str(msg, "face", "idle");
            s.vibe = Str(msg, "vibe", "tap");
            s.voice = Str(msg, "voice", "normal");
            s.proactive = cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(msg, "proactive"));
            s.audio = cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(msg, "audio"));
            s.end = cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(msg, "end"));
            on_say(s);
        }
    } else if (type == "thinking") {
        if (on_thinking) on_thinking();
    } else if (type == "heard") {
        ESP_LOGI(TAG, "heard: %s", Str(msg, "text").c_str());
        if (on_heard) on_heard(Str(msg, "text"));
    } else if (type == "tts") {
        std::string state = Str(msg, "state");
        if (state == "start") {
            if (Str(msg, "format") != "opus") {
                ESP_LOGW(TAG, "tts format %s not supported", Str(msg, "format").c_str());
            } else if (on_tts_start) {
                on_tts_start(int(Num(msg, "rate", 24000)), int(Num(msg, "frame_ms", 60)));
            }
        } else if (state == "stop" && on_tts_stop) {
            bool bad = cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(msg, "cancelled")) ||
                       cJSON_GetObjectItemCaseSensitive(msg, "error") != nullptr;
            on_tts_stop(!bad);
        }
    } else if (type == "error") {
        ESP_LOGW(TAG, "brain error: %s", Str(msg, "message").c_str());
    }
    cJSON_Delete(msg);
}

}  // namespace net
