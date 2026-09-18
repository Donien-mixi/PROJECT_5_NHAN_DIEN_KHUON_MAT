#include <Arduino.h>       
#include <WiFi.h>
#include <TensorFlowLite_ESP32.h>   
#include <SPIFFS.h>  
#include <TJpg_Decoder.h>      
#include <esp_task_wdt.h>      

#include "camera_pins.h"  
#include "camera_driver.h"
#include "web_server.h"
#include "ai_face_detector.h"    
#include "ai_face_recognizer.h"          
#include "ai_config.h"        

// Task handles  
TaskHandle_t CameraTask;
TaskHandle_t AITask;

// Ghi nhật ký điểm danh vào bộ nhớ Flash SPIFFS
void log_attendance(const char* name) {
    File f = SPIFFS.open("/attendance.csv", FILE_APPEND);
    if (!f) { 
        Serial.println("❌ SPIFFS open fail"); 
        return; 
    }
    f.printf("%s,%lu\n", name, millis());
    f.close();
    Serial.printf("💾 [SPIFFS] Saved attendance: %s\n", name);
}

// Task Core 1: Thu nhận ảnh liên tục từ camera OV5640 vào PSRAM
void camera_capture_task(void *pvParameters) {
    (void)pvParameters;
    Serial.println("📷 [CameraTask] Bat dau thu thap anh tu OV5640 tren Core 1...");
    for (;;) {
        capture_frame_to_buffer();
        vTaskDelay(pdMS_TO_TICKS(30)); // ~30 FPS, nhường CPU cho WiFi và Web Server
    }
}

// Task Core 0: Độc quyền xử lý toàn bộ thuật toán AI (Detector + Recognizer)
void ai_processing_task(void *pvParameters) {
    (void)pvParameters;
    Serial.println("🧠 [AITask] Dang khoi tao mo hinh AI tren Core 0...");
    setup_face_detector();
    setup_face_recognizer();
    Serial.println("✅ [AITask] Mo hinh AI da san sang!");

    // Đăng ký AITask với Watchdog
    esp_task_wdt_add(NULL);

    std::vector<float> face_tensor(FACE_TARGET_SIZE * FACE_TARGET_SIZE);
    float embedding[128];
    String last_name = ""; 
    int match_count = 0; 
    unsigned long last_record = 0;

    // ---- TÁI SỬ DỤNG BOX (Box Reuse): Tăng tốc gấp đôi chu kỳ nhận diện ----
    FaceBox cached_box{}; 
    bool have_cached_box = false; 
    int recon_cycles = 0;
    int unknown_streak = 0;
    bool last_match_ok = false; 
    const int MAX_UNKNOWN_PAUSE = 2;

    for (;;) {
        esp_task_wdt_reset(); // Feed watchdog đầu vòng lặp

        bool has_frame = false;
        xSemaphoreTake(image_mutex, portMAX_DELAY);
        if (is_new_frame_available) { 
            is_new_frame_available = false; 
            has_frame = true; 
        }
        xSemaphoreGive(image_mutex);

        if (has_frame) {
            bool prepared = false;
            // CHÍNH SÁCH BOX REUSE: Chỉ tái sử dụng box khi lượt trước khớp đúng người
            bool can_reuse_box = have_cached_box &&
                                 (last_match_ok || (match_count > 0 && unknown_streak <= 1)) &&
                                 (recon_cycles < 10);
            bool need_detect = !can_reuse_box;
            unsigned long us_det = 0, us_prep = 0, us_rec = 0;

            xSemaphoreTake(image_mutex, portMAX_DELAY);
            if (need_detect) {
                unsigned long t_perf = micros();
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
                // Chu kỳ recognize nhanh: dùng lại box đã detect
                unsigned long t_perf = micros();
                prepared = preprocess_face(cached_box, face_tensor.data());
                us_prep = micros() - t_perf;
                recon_cycles++;
            }
            xSemaphoreGive(image_mutex);

            vTaskDelay(1);
            esp_task_wdt_reset();

            if (prepared) {
                unsigned long t_perf = micros();
                extract_face_embedding(face_tensor.data(), embedding);
                us_rec = micros() - t_perf;

                Serial.printf("[PERF] det:%lu.%02lums bil:%luus he:%luus rec:%lu.%02lums cy:%s\n",
                    us_det / 1000, (us_det % 1000) / 10, g_us_bilinear, g_us_he,
                    us_rec / 1000, (us_rec % 1000) / 10,
                    need_detect ? "DETECT" : "REUSE");

                const float THRESHOLD = FACE_THRESHOLD;
                const char* name = identify_face(embedding, THRESHOLD);
                bool unknown = (strcmp(name, "Unknown") == 0);
                last_match_ok = !unknown;

                // Cập nhật trạng thái AI sang Web Dashboard
                update_ai_status(name, g_last_recognized_score, !unknown);

                if (unknown) {
                    // Chính sách tạm dừng phiếu khi Unknown (Pause-on-Unknown)
                    unknown_streak++;
                    if (match_count > 0 && unknown_streak > MAX_UNKNOWN_PAUSE) {
                        match_count = 0; 
                        last_name = "";
                        have_cached_box = false;
                    }
                    if (match_count == 0) {
                        have_cached_box = false;
                    }
                    // Báo hiệu từ chối nếu liên tiếp là người lạ
                    if (unknown_streak >= TEMPORAL_VOTES && (millis() - last_record > 5000)) {
                        Serial.println("⚠️ [AI] REJECT UNKNOWN (Người lạ)");
                        last_record = millis();
                        unknown_streak = 0;
                        have_cached_box = false;

                        // 2 tiếng bíp dài + LED Đỏ bật
                        for (int i = 0; i < 2; i++) {
                            digitalWrite(BUZZER_PIN, HIGH); 
                            digitalWrite(LED_FAIL_PIN, HIGH); 
                            vTaskDelay(500 / portTICK_PERIOD_MS);
                            digitalWrite(BUZZER_PIN, LOW); 
                            digitalWrite(LED_FAIL_PIN, LOW); 
                            vTaskDelay(150 / portTICK_PERIOD_MS);
                        }
                    }
                } else {
                    unknown_streak = 0;
                    if (String(name) == last_name) {
                        match_count++; 
                    } else { 
                        match_count = 1; 
                        last_name = String(name); 
                    }

                    // Đủ 3 frame liên tiếp -> Chốt điểm danh thành công
                    if (match_count >= TEMPORAL_VOTES) {
                        if (millis() - last_record > 10000) { // Cooldown 10s trên MCU
                            Serial.printf("🎉 [AI] SUCCESS: %s (Độ tin cậy: %.2f)\n", last_name.c_str(), g_last_recognized_score);
                            log_attendance(last_name.c_str());
                            last_record = millis();

                            // 1 tiếng bíp ngắn + LED Xanh bật
                            digitalWrite(BUZZER_PIN, HIGH); 
                            digitalWrite(LED_SUCCESS_PIN, HIGH); 
                            vTaskDelay(100 / portTICK_PERIOD_MS); 
                            digitalWrite(BUZZER_PIN, LOW); 
                            digitalWrite(LED_SUCCESS_PIN, LOW); 
                        }
                    }
                }
            } else { 
                match_count = 0; 
                last_name = ""; 
                unknown_streak = 0; 
                last_match_ok = false; 
                have_cached_box = false; 
            }
        }
        vTaskDelay(pdMS_TO_TICKS(10));  
    }
}  

void setup() {  
    setCpuFrequencyMhz(240); // Khóa xung nhịp CPU ở mức tối đa 240MHz
    Serial.begin(115200);
    Serial.setDebugOutput(true);
    delay(1000);

    printf("\n=== ESP32-S3 Face Attendance Standalone (OV5640 Camera) ===\n");
    Serial.println("\n=== ESP32-S3 Face Attendance Standalone (OV5640 Camera) ===");
    Serial.printf("CPU Freq: %u MHz\n", getCpuFrequencyMhz());
    Serial.printf("RAM int : total=%u free=%u\n", ESP.getHeapSize(), ESP.getFreeHeap());
    Serial.printf("PSRAM   : total=%u free=%u\n", ESP.getPsramSize(), ESP.getFreePsram());
    Serial.printf("Flash   : %u bytes\n", ESP.getFlashChipSize());

    if (!psramFound()) {
        Serial.println("❌ WARN: PSRAM KHONG DUOC BAT! Hãy chọn PSRAM = OPI PSRAM trong Arduino IDE.");
    } else {
        Serial.printf("✅ PSRAM OK (OPI): %u bytes\n", ESP.getPsramSize());
    }

    // Cấu hình chân ngoại vi (Buzzer & 2 LED)
    pinMode(BUZZER_PIN, OUTPUT); digitalWrite(BUZZER_PIN, LOW);
    pinMode(LED_SUCCESS_PIN, OUTPUT); digitalWrite(LED_SUCCESS_PIN, LOW);
    pinMode(LED_FAIL_PIN, OUTPUT); digitalWrite(LED_FAIL_PIN, LOW);

    // Cấu hình Watchdog bảo vệ task
    esp_task_wdt_config_t wdt_cfg = {
        .timeout_ms = 60000,
        .idle_core_mask = 0,
        .trigger_panic = false,
    };
    if (esp_task_wdt_reconfigure(&wdt_cfg) != ESP_OK) {
        esp_task_wdt_init(&wdt_cfg);
    }

    if (!SPIFFS.begin(true)) {
        Serial.println("❌ SPIFFS fail");
    } else {
        Serial.println("✅ SPIFFS OK");
    }

    // 1. Khởi tạo Camera trực tiếp OV5640
    if (!setup_camera()) {
        Serial.println("❌ FATAL: Camera OV5640 khong the khoi tao!");
    }

    // 2. Khởi tạo Web Server HTTP nền (Tùy chọn: tự động bỏ qua nếu không có Wi-Fi)
    setup_web_server();

    // 3. Khởi tạo 2 Tasks bất đối xứng trên 2 Core
    // Core 1: Thu nhận camera liên tục (Stack 20KB an toàn tuyệt đối)
    xTaskCreatePinnedToCore(camera_capture_task, "CamTask", 20480, NULL, 3, &CameraTask, 1);
    // Core 0: Độc quyền tính toán AI
    xTaskCreatePinnedToCore(ai_processing_task, "AITask", 32768, NULL, 5, &AITask, 0);

    Serial.println("🚀 [System] He thong da khoi dong xong tat ca cac task!");
}

void loop() {
    static int sec = 0; 
    sec += 5;  
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("[HB] Web UI: http://%s | Uptime: %ds | Free PSRAM: %u | Free Heap: %u\n", 
                      WiFi.localIP().toString().c_str(), sec, ESP.getFreePsram(), ESP.getFreeHeap());
    } else {
        Serial.printf("[HB] Standalone OFFLINE | Uptime: %ds | Free PSRAM: %u | Free Heap: %u\n", 
                      sec, ESP.getFreePsram(), ESP.getFreeHeap());
    }
    vTaskDelay(pdMS_TO_TICKS(5000));
}
