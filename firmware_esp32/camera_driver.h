#ifndef CAMERA_DRIVER_H
#define CAMERA_DRIVER_H

#include <Arduino.h>
#include "esp_camera.h"
#include "camera_pins.h"
#include "ai_config.h"

// Biến cờ và mutex đồng bộ khung hình giữa Camera Task (Core 1) và AI Task (Core 0)
extern volatile bool is_new_frame_available;
extern SemaphoreHandle_t image_mutex;

// Khởi tạo camera OV5640 và cấp phát bộ đệm giải mã
bool setup_camera();

// Thu nhận 1 khung hình từ camera OV5640, giải mã và cắt vuông 128x128 RGB565 nạp vào g_frame_buffer
bool capture_frame_to_buffer();

// Hàm lấy khung hình JPEG mới nhất phục vụ Web Server xem trực tiếp qua trình duyệt
bool get_latest_jpeg_frame(uint8_t* dest, size_t max_len, size_t* out_len);

// Điều chỉnh xoay lật ảnh camera
void set_camera_orientation(int vflip, int hmirror);

#endif // CAMERA_DRIVER_H
