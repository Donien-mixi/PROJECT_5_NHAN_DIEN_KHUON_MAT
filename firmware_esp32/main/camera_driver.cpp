#include "camera_driver.h"
#include "camera_pins.h"
#include "ai_face_detector.h"
#include "img_converters.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include <string.h>

static const char* TAG = "CAMERA_DRIVER";

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
    ESP_LOGI(TAG, "Đang khởi tạo Camera OV5640 (ESP-IDF Native)...");

    if (!image_mutex) image_mutex = xSemaphoreCreateMutex();
    if (!s_jpeg_mutex) s_jpeg_mutex = xSemaphoreCreateMutex();

    // Cấp phát bộ đệm giải mã tạm thời trên PSRAM (cho ảnh tới 640x480)
    s_temp_decoded_buf = (uint16_t*)heap_caps_malloc(MAX_DECODE_W * MAX_DECODE_H * sizeof(uint16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_temp_decoded_buf) {
        ESP_LOGE(TAG, "❌ LỖI: Không thể cấp phát s_temp_decoded_buf trên PSRAM!");
        return false;
    }

    s_latest_jpeg_buf = (uint8_t*)heap_caps_malloc(MAX_JPEG_STREAM_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_latest_jpeg_buf) {
        ESP_LOGE(TAG, "❌ LỖI: Không thể cấp phát s_latest_jpeg_buf trên PSRAM!");
        return false;
    }

    camera_config_t config;
    memset(&config, 0, sizeof(config));
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
        ESP_LOGW(TAG, "esp_camera_init thất bại trên FRAMESIZE_240X240 (0x%x), thử FRAMESIZE_QVGA...", err);
        config.frame_size = FRAMESIZE_QVGA;
        err = esp_camera_init(&config);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "❌ Không thể khởi tạo camera OV5640: 0x%x", err);
            return false;
        }
    }

    sensor_t *s = esp_camera_sensor_get();
    if (s != nullptr) {
        s->set_brightness(s, 1);
        s->set_contrast(s, 1);
        s->set_saturation(s, 0);
        s->set_whitebal(s, 1);
        s->set_exposure_ctrl(s, 1);
        s->set_vflip(s, 0);
        s->set_hmirror(s, 0);
        ESP_LOGI(TAG, "✅ Camera OV5640 init thành công! (PID: 0x%x)", s->id.PID);
    }

    return true;
}

bool capture_frame_to_buffer() {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
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
    if (!is_new_frame_available && image_mutex && xSemaphoreTake(image_mutex, 0) == pdTRUE) {
        if (g_frame_buffer != nullptr) {
            if (fb->format == PIXFORMAT_JPEG) {
                s_temp_w = fb->width;
                s_temp_h = fb->height;
                if (s_temp_w <= MAX_DECODE_W && s_temp_h <= MAX_DECODE_H) {
                    // Giải mã JPEG thành RGB565 buffer
                    if (jpg2rgb565(fb->buf, fb->len, (uint8_t*)s_temp_decoded_buf, JPG_SCALE_NONE)) {
                        crop_and_downscale_to_128(s_temp_decoded_buf, fb->width, fb->height, g_frame_buffer);
                        is_new_frame_available = true;
                    }
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
        ESP_LOGI(TAG, "Orientation updated: vflip=%d, hmirror=%d", vflip, hmirror);
    }
}
