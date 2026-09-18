#include <stdio.h>
#include <string.h>
#include <vector>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_psram.h"
#include "esp_heap_caps.h"
#include "esp_spiffs.h"
#include "nvs_flash.h"
#include "driver/gpio.h"

#include "camera_pins.h"
#include "camera_driver.h"
#include "web_server.h"
#include "ai_face_detector.h"
#include "ai_face_recognizer.h"
#include "ai_config.h"

static const char* TAG = "MAIN_APP";

static inline int64_t millis(void) { return esp_timer_get_time() / 1000; }
static inline int64_t micros(void) { return esp_timer_get_time(); }

// Task handles
TaskHandle_t CameraTask;
TaskHandle_t AITask;

// Ghi nhật ký điểm danh vào bộ nhớ Flash SPIFFS
void log_attendance(const char* name) {
    FILE* f = fopen("/spiffs/attendance.csv", "a");
    if (!f) {
        ESP_LOGE(TAG, "❌ SPIFFS open fail");
        return;
    }
    fprintf(f, "%s,%llu\n", name, (unsigned long long)millis());
    fclose(f);
    ESP_LOGI(TAG, "💾 [SPIFFS] Saved attendance: %s", name);
}

// Task Core 1: Thu nhận ảnh liên tục từ camera OV5640 vào PSRAM
void camera_capture_task(void *pvParameters) {
    (void)pvParameters;
    ESP_LOGI(TAG, "📷 [CameraTask] Bắt đầu thu thập ảnh từ OV5640 trên Core 1...");
    for (;;) {
        capture_frame_to_buffer();
        vTaskDelay(pdMS_TO_TICKS(30)); // ~30 FPS
    }
}

// Task Core 0: Độc quyền xử lý toàn bộ thuật toán AI (Detector + Recognizer)
void ai_processing_task(void *pvParameters) {
    (void)pvParameters;
    ESP_LOGI(TAG, "🧠 [AITask] Đang khởi tạo mô hình AI trên Core 0 (SIMD esp-nn)...");
    setup_face_detector();
    setup_face_recognizer();
    ESP_LOGI(TAG, "✅ [AITask] Mô hình AI đã sẵn sàng!");

    std::vector<float> face_tensor(FACE_TARGET_SIZE * FACE_TARGET_SIZE);
    float embedding[128];
    char last_name[64] = "";
    int match_count = 0;
    int64_t last_record = 0;

    // ---- TÁI SỬ DỤNG BOX (Box Reuse): Tăng tốc chu kỳ nhận diện ----
    FaceBox cached_box{};
    bool have_cached_box = false;
    int recon_cycles = 0;
    int unknown_streak = 0;
    bool last_match_ok = false;
    const int MAX_UNKNOWN_PAUSE = 2;

    for (;;) {
        bool has_frame = false;
        xSemaphoreTake(image_mutex, portMAX_DELAY);
        if (is_new_frame_available) {
            is_new_frame_available = false;
            has_frame = true;
        }
        xSemaphoreGive(image_mutex);

        if (has_frame) {
            bool prepared = false;
            bool can_reuse_box = have_cached_box &&
                                 (last_match_ok || (match_count > 0 && unknown_streak <= 1)) &&
                                 (recon_cycles < 10);
            bool need_detect = !can_reuse_box;
            int64_t us_det = 0, us_prep = 0, us_rec = 0;

            xSemaphoreTake(image_mutex, portMAX_DELAY);
            if (need_detect) {
                int64_t t_perf = micros();
                FaceBox box = detect_face();
                us_det = micros() - t_perf;
                if (box.is_valid) {
                    cached_box = box;
                    have_cached_box = true;
                    recon_cycles = 0;
                    t_perf = micros();
                    prepared = preprocess_face(box, face_tensor.data());
                    us_prep = micros() - t_perf;
                } else {
                    have_cached_box = false;
                }
            } else {
                int64_t t_perf = micros();
                prepared = preprocess_face(cached_box, face_tensor.data());
                us_prep = micros() - t_perf;
                recon_cycles++;
            }
            xSemaphoreGive(image_mutex);

            vTaskDelay(1);

            if (prepared) {
                int64_t t_perf = micros();
                extract_face_embedding(face_tensor.data(), embedding);
                us_rec = micros() - t_perf;

                ESP_LOGI(TAG, "[PERF] det:%lld.%02lldms bil:%luus he:%luus rec:%lld.%02lldms cy:%s",
                         (long long)(us_det / 1000), (long long)((us_det % 1000) / 10),
                         g_us_bilinear, g_us_he,
                         (long long)(us_rec / 1000), (long long)((us_rec % 1000) / 10),
                         need_detect ? "DETECT" : "REUSE");

                const float THRESHOLD = FACE_THRESHOLD;
                const char* name = identify_face(embedding, THRESHOLD);
                bool unknown = (strcmp(name, "Unknown") == 0);
                last_match_ok = !unknown;

                // Cập nhật trạng thái AI sang Web Dashboard
                update_ai_status(name, g_last_recognized_score, !unknown);

                if (unknown) {
                    unknown_streak++;
                    if (match_count > 0 && unknown_streak > MAX_UNKNOWN_PAUSE) {
                        match_count = 0;
                        last_name[0] = '\0';
                        have_cached_box = false;
                    }
                    if (match_count == 0) {
                        have_cached_box = false;
                    }
                    if (unknown_streak >= TEMPORAL_VOTES && (millis() - last_record > 5000)) {
                        ESP_LOGW(TAG, "⚠️ [AI] REJECT UNKNOWN (Người lạ)");
                        last_record = millis();
                        unknown_streak = 0;
                        have_cached_box = false;

                        // 2 tiếng bíp dài + LED Đỏ bật
                        for (int i = 0; i < 2; i++) {
                            gpio_set_level((gpio_num_t)BUZZER_PIN, 1);
                            gpio_set_level((gpio_num_t)LED_FAIL_PIN, 1);
                            vTaskDelay(pdMS_TO_TICKS(500));
                            gpio_set_level((gpio_num_t)BUZZER_PIN, 0);
                            gpio_set_level((gpio_num_t)LED_FAIL_PIN, 0);
                            vTaskDelay(pdMS_TO_TICKS(150));
                        }
                    }
                } else {
                    unknown_streak = 0;
                    if (strcmp(name, last_name) == 0) {
                        match_count++;
                    } else {
                        match_count = 1;
                        strncpy(last_name, name, sizeof(last_name) - 1);
                    }

                    // Đủ 3 frame liên tiếp -> Chốt điểm danh thành công
                    if (match_count >= TEMPORAL_VOTES) {
                        if (millis() - last_record > 10000) { // Cooldown 10s
                            ESP_LOGI(TAG, "🎉 [AI] SUCCESS: %s (Độ tin cậy: %.2f)", last_name, g_last_recognized_score);
                            log_attendance(last_name);
                            last_record = millis();

                            // 1 tiếng bíp ngắn + LED Xanh bật
                            gpio_set_level((gpio_num_t)BUZZER_PIN, 1);
                            gpio_set_level((gpio_num_t)LED_SUCCESS_PIN, 1);
                            vTaskDelay(pdMS_TO_TICKS(100));
                            gpio_set_level((gpio_num_t)BUZZER_PIN, 0);
                            gpio_set_level((gpio_num_t)LED_SUCCESS_PIN, 0);
                        }
                    }
                }
            } else {
                match_count = 0;
                last_name[0] = '\0';
                unknown_streak = 0;
                last_match_ok = false;
                have_cached_box = false;
            }
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

extern "C" void app_main(void) {
    ESP_LOGI(TAG, "=================================================================");
    ESP_LOGI(TAG, "🚀 PROJECT 5: STANDALONE FACE ATTENDANCE SYSTEM (OV5640 CAM)");
    ESP_LOGI(TAG, "⚡ 100%% EDGE AI ON ESP32-S3 N16R8 (ESP-IDF 5.3 + SIMD esp-nn)");
    ESP_LOGI(TAG, "=================================================================");

    // 1. Khởi tạo NVS Flash
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // 2. Kiểm tra tài nguyên phần cứng & PSRAM
    size_t psram_size = esp_psram_get_size();
    size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    size_t free_sram  = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    ESP_LOGI(TAG, "CPU Freq: 240 MHz");
    ESP_LOGI(TAG, "Bộ nhớ Octal PSRAM : %d MB (Còn trống: %d KB)", (int)(psram_size / (1024 * 1024)), (int)(free_psram / 1024));
    ESP_LOGI(TAG, "Bộ nhớ Internal SRAM: %d KB", (int)(free_sram / 1024));

    if (psram_size == 0) {
        ESP_LOGE(TAG, "❌ CẢNH BÁO NGUY HIỂM: Không phát hiện PSRAM! Yêu cầu ESP32-S3 N16R8!");
    }

    // 3. Cấu hình GPIO (Buzzer & 2 LED)
    gpio_config_t io_conf;
    memset(&io_conf, 0, sizeof(io_conf));
    io_conf.intr_type = GPIO_INTR_DISABLE;
    io_conf.mode = GPIO_MODE_OUTPUT;
    io_conf.pin_bit_mask = (1ULL << BUZZER_PIN) | (1ULL << LED_SUCCESS_PIN) | (1ULL << LED_FAIL_PIN);
    io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    io_conf.pull_up_en = GPIO_PULLUP_DISABLE;
    gpio_config(&io_conf);
    gpio_set_level((gpio_num_t)BUZZER_PIN, 0);
    gpio_set_level((gpio_num_t)LED_SUCCESS_PIN, 0);
    gpio_set_level((gpio_num_t)LED_FAIL_PIN, 0);

    // 4. Khởi tạo SPIFFS để lưu file điểm danh
    esp_vfs_spiffs_conf_t spiffs_cfg;
    memset(&spiffs_cfg, 0, sizeof(spiffs_cfg));
    spiffs_cfg.base_path = "/spiffs";
    spiffs_cfg.partition_label = "spiffs";
    spiffs_cfg.max_files = 5;
    spiffs_cfg.format_if_mount_failed = true;
    esp_err_t spiffs_ret = esp_vfs_spiffs_register(&spiffs_cfg);
    if (spiffs_ret != ESP_OK) {
        ESP_LOGW(TAG, "⚠️ SPIFFS mount fail: 0x%x (đang format hoặc chưa có phân vùng)", spiffs_ret);
    } else {
        ESP_LOGI(TAG, "✅ SPIFFS đã mount thành công tại /spiffs");
    }

    // 5. Khởi tạo Camera trực tiếp OV5640
    if (!setup_camera()) {
        ESP_LOGE(TAG, "❌ FATAL: Camera OV5640 không thể khởi tạo!");
    }

    // 6. Khởi tạo Web Server HTTP nền
    setup_web_server();

    // 7. Khởi tạo 2 Tasks bất đối xứng trên 2 Core
    // Core 1: Thu nhận camera liên tục (Stack 8KB)
    xTaskCreatePinnedToCore(camera_capture_task, "CamTask", 8192, NULL, 3, &CameraTask, 1);
    // Core 0: Độc quyền tính toán AI (Stack 32KB)
    xTaskCreatePinnedToCore(ai_processing_task, "AITask", 32768, NULL, 5, &AITask, 0);

    ESP_LOGI(TAG, "🚀 [System] Hệ thống đa nhân FreeRTOS đã khởi chạy thành công!");

    // Vòng lặp heartbeat giám sát
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(5000));
        ESP_LOGI(TAG, "[HB] Uptime: %llds | Free PSRAM: %u KB | Free Heap: %u KB",
                 (long long)(esp_timer_get_time() / 1000000),
                 (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024),
                 (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024));
    }
}
