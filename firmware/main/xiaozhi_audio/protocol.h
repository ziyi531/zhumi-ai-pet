// ai_pet: 小智的 protocol.h 里只有这个结构被语音模块用到，单独拿出来。
// 我们自己的通信协议在 main/net/ 里，不依赖小智的协议实现。
#ifndef AI_PET_AUDIO_PROTOCOL_H
#define AI_PET_AUDIO_PROTOCOL_H

#include <cstdint>
#include <vector>

struct AudioStreamPacket {
    int sample_rate = 0;
    int frame_duration = 0;
    uint32_t timestamp = 0;
    uint32_t playback_id = 0;
    uint32_t media_position_ms = 0;
    std::vector<uint8_t> payload;
};

#endif
