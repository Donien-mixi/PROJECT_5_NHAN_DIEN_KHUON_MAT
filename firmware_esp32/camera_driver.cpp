#include "camera_driver.h"
#include "ai_face_detector.h"
#include <TJpg_Decoder.h>

volatile bool is_new_frame_available = false;
SemaphoreHandle_t image_mutex = nullptr;

static uint16_t* s_temp_decoded_buf = nullptr;
static int s_temp_w = 240;
static int s_temp_h = 240;

static const int MAX_DECODE_W = 640;
static const int MAX_DECODE_H = 480;

// Bộ đệm lưu frame JPEG mới nhất phục vụ Web Server xem trực tiếp
static uint8_t* s_latest_jpeg_buf = nullptr;
static size_t s_latest_jpeg_len = 0;
static SemaphoreHandle_t s_jpeg_mutex = nullptr;
static const size_t MAX_JPEG_STREAM_SIZE = 65536; // 64KB PSRAM

// Callback của TJpg_Decoder: ghi các khối MCU vào bộ đệm tạm
static bool camera_jpeg_output_callback(int16_t x, int16_t y, uint16_t w, uint16_t h, uint16_t* bitmap) {
    if (!s_temp_decoded_buf) return 0;
    for (int j = 0; j < h; j++) {
        int py = y + j;
        if (py >= s_temp_h || py >= MAX_DECODE_H) break;
        uint16_t* row = s_temp_decoded_buf + py * s_temp_w;
        for (int i = 0; i < w; i++) {
            int px = x + i;
            if (px < s_temp_w && px < MAX_DECODE_W) {
                row[px] = bitmap[j * w + i];
            }
        }
    }
    return 1;
}

// Cắt vùng vuông chính giữa (center-crop) và thu phóng về 128x128 RGB565 cực nhanh
static void crop_and_downscale_to_128(const uint16_t* src, int w, int h, uint16_t* dst) {
    if (!src || !dst || w <= 0 || h <= 0) return;
    int side = (w < h) ? w : h;
    int offset_x = (w - side) / 2;
    int offset_y = (h - side) / 2;

    for (int y = 0; y < 128; y++) {
        int sy = offset_y + (y * side) / 128;
        if (sy >= h) sy = h - 1;
        const uint16_t* src_row = src + sy * w;
        uint16_t* dst_row = dst + y * 128;
        for (int x = 0; x < 128; x++) {
            int sx = offset_x + (x * side) / 128;
            if (sx >= w) sx = w - 1;
            dst_row[x] = src_row[sx];
        }
    }
}

bool setup_camera() {
    Serial.println("\n[*] Đang khởi tạo Camera OV5640...");

    if (!image_mutex) image_mutex = xSemaphoreCreateMutex();
    if (!s_jpeg_mutex) s_jpeg_mutex = xSemaphoreCreateMutex();

    // Cấp phát bộ đệm giải mã tạm thời trên PSRAM (cho ảnh tới 640x480)
    s_temp_decoded_buf = (uint16_t*)heap_caps_malloc(MAX_DECODE_W * MAX_DECODE_H * sizeof(uint16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_temp_decoded_buf) {
        Serial.println("❌ LỖI: Không thể cấp phát s_temp_decoded_buf trên PSRAM!");
        return false;
    }

    s_latest_jpeg_buf = (uint8_t*)heap_caps_malloc(MAX_JPEG_STREAM_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_latest_jpeg_buf) {
        Serial.println("❌ LỖI: Không thể cấp phát s_latest_jpeg_buf trên PSRAM!");
        return false;
    }

    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    
    // Sử dụng JPEG để vừa phục vụ AI vừa phục vụ stream Web mượt mà
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size = FRAMESIZE_240X240; // Tỷ lệ vuông 1:1 lý tưởng cho khuôn mặt
    config.jpeg_quality = 12;
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;
    config.fb_location = CAMERA_FB_IN_PSRAM;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("❌ esp_camera_init thất bại! Mã lỗi: 0x%x\n", err);
        // Thử dự phòng với FRAMESIZE_QVGA nếu cảm biến không nhận 240x240
        Serial.println("[*] Đang thử lại với FRAMESIZE_QVGA (320x240)...");
        config.frame_size = FRAMESIZE_QVGA;
        err = esp_camera_init(&config);
        if (err != ESP_OK) {
            Serial.printf("❌ Không thể khởi tạo camera (0x%x)!\n", err);
            return false;
        }
    }

    sensor_t *s = esp_camera_sensor_get();
    if (s != nullptr) {
        // Cân bằng cảm biến cho nhận diện khuôn mặt
        s->set_brightness(s, 1);     // Tăng nhẹ sáng
        s->set_contrast(s, 1);       // Tăng nhẹ tương phản
        s->set_saturation(s, 0);
        s->set_whitebal(s, 1);       // Bật Auto White Balance
        s->set_exposure_ctrl(s, 1);  // Bật Auto Exposure
        s->set_vflip(s, 0);          // Mặc định không lật
        s->set_hmirror(s, 0);        // Mặc định không lật gương

        Serial.printf("✅ Camera OV5640 init thành công! (PID: 0x%x)\n", s->id.PID);
    }

    // Cấu hình bộ giải mã JPEG
    TJpgDec.setJpgScale(1);
    TJpgDec.setSwapBytes(false); // Đảm bảo Little-Endian RGB565 chuẩn của ESP32
    TJpgDec.setCallback(camera_jpeg_output_callback);

    return true;
}

bool capture_frame_to_buffer() {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("[Camera] fb_get fail");
        return false;
    }

    // 1. Sao chép nhanh ảnh JPEG phục vụ Web Server
    if (s_jpeg_mutex && xSemaphoreTake(s_jpeg_mutex, 0) == pdTRUE) {
        if (fb->format == PIXFORMAT_JPEG && fb->len <= MAX_JPEG_STREAM_SIZE) {
            memcpy(s_latest_jpeg_buf, fb->buf, fb->len);
            s_latest_jpeg_len = fb->len;
        }
        xSemaphoreGive(s_jpeg_mutex);
    }

    // 2. CHỈ GIẢI MÃ và nạp vào g_frame_buffer khi AI đã xử lý xong frame trước (!is_new_frame_available)
    // Giúp tiết kiệm 95% CPU trên Core 1, tránh nghẽn mạng WiFi và chống tràn stack tuyệt đối!
    if (!is_new_frame_available && image_mutex && xSemaphoreTake(image_mutex, 0) == pdTRUE) {
        if (g_frame_buffer != nullptr) {
            if (fb->format == PIXFORMAT_JPEG) {
                s_temp_w = fb->width;
                s_temp_h = fb->height;
                if (s_temp_w <= MAX_DECODE_W && s_temp_h <= MAX_DECODE_H) {
                    // Giải mã JPEG vào s_temp_decoded_buf
                    TJpgDec.drawJpg(0, 0, fb->buf, fb->len);
                    // Center-crop và thu về 128x128
                    crop_and_downscale_to_128(s_temp_decoded_buf, fb->width, fb->height, g_frame_buffer);
                    is_new_frame_available = true;
                }
            } else if (fb->format == PIXFORMAT_RGB565) {
                crop_and_downscale_to_128((const uint16_t*)fb->buf, fb->width, fb->height, g_frame_buffer);
                is_new_frame_available = true;
            }
        }
        xSemaphoreGive(image_mutex);
    }

    esp_camera_fb_return(fb);
    return true;
}

bool get_latest_jpeg_frame(uint8_t* dest, size_t max_len, size_t* out_len) {
    if (!dest || !out_len || !s_latest_jpeg_buf || s_latest_jpeg_len == 0) {
        return false;
    }
    bool success = false;
    if (s_jpeg_mutex && xSemaphoreTake(s_jpeg_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
        if (s_latest_jpeg_len <= max_len) {
            memcpy(dest, s_latest_jpeg_buf, s_latest_jpeg_len);
            *out_len = s_latest_jpeg_len;
            success = true;
        }
        xSemaphoreGive(s_jpeg_mutex);
    }
    return success;
}

void set_camera_orientation(int vflip, int hmirror) {
    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        s->set_vflip(s, vflip);
        s->set_hmirror(s, hmirror);
        Serial.printf("[Camera] Orientation updated: vflip=%d, hmirror=%d\n", vflip, hmirror);
    }
}
