#include "image_decoder.h"
#include "ai_face_detector.h"

// Callback function to decode JPEG pixels to g_frame_buffer
bool jpeg_output(int16_t x, int16_t y, uint16_t w, uint16_t h, uint16_t* bitmap) {
    if (y >= 240) return 0; // Kích thước khung hình ảnh là 240x240
    
    // Lưu vào g_frame_buffer cho AI xử lý (Core 0)
    if (g_frame_buffer) {
        for (int j = 0; j < h; j++) {
            for (int i = 0; i < w; i++) {
                int px = x + i;
                int py = y + j;
                if (px < 240 && py < 240) {
                    g_frame_buffer[py * 240 + px] = bitmap[j * w + i];
                }
            }
        }
    }
    
    return 1; // 1 = continue decoding
}

void setup_image_decoder() {
    // Cấu hình JPEG Decoder
    TJpgDec.setJpgScale(1); 
    TJpgDec.setSwapBytes(false); // Sửa thành false để giải mã RGB565 chuẩn trên Little-Endian (ESP32)
    TJpgDec.setCallback(jpeg_output);
}

void decode_jpeg_frame(const uint8_t* jpeg_data, uint32_t length) {
    // Chỉ giải mã ảnh và lưu vào g_frame_buffer thông qua callback
    TJpgDec.drawJpg(0, 0, jpeg_data, length);
}
