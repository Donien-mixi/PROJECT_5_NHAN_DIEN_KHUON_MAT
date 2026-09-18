#ifndef AI_FACE_DETECTOR_H
#define AI_FACE_DETECTOR_H

#include <stdint.h>
#include <stdbool.h>
#include "ai_config.h"

// Đồng bộ Laptop ↔ ESP32: RAW 128x128 RGB
#ifndef FRAME_WIDTH
#define FRAME_WIDTH  RAW_FRAME_SIZE
#endif
#ifndef FRAME_HEIGHT
#define FRAME_HEIGHT RAW_FRAME_SIZE
#endif

struct FaceBox {
    int x;
    int y;
    int width;
    int height;
    // Giữ tọa độ float đã decode để Bilinear trùng khớp host Python,
    // không làm tròn rồi mới crop.
    float center_x;
    float center_y;
    float crop_size;
    float score;
    bool is_valid;
};

extern uint16_t* g_frame_buffer;

// [PERF] Đo thời gian từng chặng preprocess (micros)
extern unsigned long g_us_bilinear; // Bilinear 128→64
extern unsigned long g_us_he;       // Histogram Equalization LUT

void setup_face_detector();
FaceBox detect_face();
bool preprocess_face(const FaceBox& box, float* out_tensor); // 64x64 float [-1,1]

#endif // AI_FACE_DETECTOR_H
