#ifndef AI_FACE_DETECTOR_H
#define AI_FACE_DETECTOR_H

#include <stdint.h>
#include <vector>

struct FaceBox {
    int x;
    int y;
    int width;
    int height;
    bool is_valid;
};

// Global frame buffer to avoid double JPEG decoding
extern uint16_t* g_frame_buffer;

// Hàm khởi tạo bộ dò khuôn mặt (MTMN / BlazeFace)
void setup_face_detector();

// Hàm chạy dò khuôn mặt trên ảnh JPEG
// Trả về bounding box của khuôn mặt đầu tiên tìm thấy
FaceBox detect_face();

// Hàm trích xuất, cắt (crop) và scale khuôn mặt về 64x64 Grayscale
// Đầu vào là JPEG gốc và tọa độ FaceBox.
// Đầu ra là mảng float 64x64 (đã chuẩn hóa (x - 127.5)/128.0) dùng cho TFLite
bool preprocess_face(const FaceBox& box, float* out_tensor);

#endif
