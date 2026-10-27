# ESP-IDF 的 SRC_DIRS 不收 .cc 文件（小智的代码都是 .cc），所以自己列源文件。
# 新增 .cc 文件后要重新运行一次 idf.py reconfigure（IDF 的脚本模式不支持 CONFIGURE_DEPENDS）。
file(GLOB_RECURSE PET_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c)

idf_component_register(
    SRCS ${PET_SOURCES}
    INCLUDE_DIRS
        "."
        "board"
        "face"
        "net"
        "xiaozhi_audio"
        "xiaozhi_audio/codecs"
        "xiaozhi_audio/engines"
        "xiaozhi_audio/wake_words"
        "xiaozhi_audio/demuxer"
    PRIV_REQUIRES
        esp_driver_i2c esp_driver_i2s esp_driver_spi esp_driver_gpio esp_driver_ledc
        esp_lcd esp_timer nvs_flash esp_wifi esp_netif esp_event
        espressif__esp_websocket_client espressif__cjson mbedtls
    # 被唤醒时的应答（“在呢～”“怎么啦？”……），猪咪的声音，ai_pet/tools/make_wake_acks.py 生成
    EMBED_FILES
        "assets/wake_ack_0.ogg" "assets/wake_ack_1.ogg" "assets/wake_ack_2.ogg" "assets/wake_ack_3.ogg"
)
# 小智语音模块按 C++23 写的
target_compile_options(${COMPONENT_LIB} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++23>)
