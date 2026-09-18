#ifndef CAMERA_PINS_H
#define CAMERA_PINS_H

// =========================================================================
// Cấu hình chân cứng cho module Camera OV5640 trên mạch ESP32-S3 WROOM-1 N16R8
// (Khớp 100% với bo mạch Freenove / ESP32-S3-EYE từ test_camera_ov5620_on_ESP32-S3)
// =========================================================================

#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     15
#define SIOD_GPIO_NUM     4
#define SIOC_GPIO_NUM     5

#define Y9_GPIO_NUM       16
#define Y8_GPIO_NUM       17
#define Y7_GPIO_NUM       18
#define Y6_GPIO_NUM       12
#define Y5_GPIO_NUM       10
#define Y4_GPIO_NUM       8
#define Y3_GPIO_NUM       9
#define Y2_GPIO_NUM       11

#define VSYNC_GPIO_NUM    6
#define HREF_GPIO_NUM     7
#define PCLK_GPIO_NUM     13

// =========================================================================
// Chân ngoại vi (Buzzer + 2 LED báo trạng thái)
// Lưu ý: Các chân cũ (15, 4, 5) đã bị camera OV5640 chiếm dụng (XCLK, SDA, SCL).
// Chúng ta dời sang các chân Header tự do trên bo mạch ESP32-S3 CAM:
// =========================================================================
#ifndef BUZZER_PIN
#define BUZZER_PIN        1   // Chân còi báo Buzzer (1 bíp thành công, 2 bíp từ chối)
#endif

#ifndef LED_SUCCESS_PIN
#define LED_SUCCESS_PIN   2   // Chân LED Xanh Lá: Điểm danh thành công
#endif

#ifndef LED_FAIL_PIN
#define LED_FAIL_PIN      3   // Chân LED Đỏ: Người lạ / Từ chối
#endif

#endif // CAMERA_PINS_H
