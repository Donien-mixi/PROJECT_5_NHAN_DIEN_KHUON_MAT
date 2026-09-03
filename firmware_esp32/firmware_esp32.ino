#include <Arduino.h>       
#include <TensorFlowLite_ESP32.h>   
#include <SPIFFS.h>  
#include <TJpg_Decoder.h>      
#include <esp_task_wdt.h>      
#include "wifi_udp_server.h"            
#include "image_decoder.h"    
#include "ai_face_detector.h"    
#include "ai_face_recognizer.h"       
#include "ai_config.h"      

// Task handles  
TaskHandle_t NetDisplayTask;
TaskHandle_t AITask;
extern SemaphoreHandle_t image_mutex; // Khai báo trong wifi_udp_server.cpp

#define BUZZER_PIN 15
#define LED_SUCCESS_PIN 4
#define LED_FAIL_PIN 5
// Không dùng LCD — chỉ 2 LED + Buzzer + Serial (README.md:3)

void log_attendance(const char* name){
    File f = SPIFFS.open("/attendance.csv", FILE_APPEND);
    if(!f){ Serial.println("SPIFFS open fail"); return; }
    f.printf("%s,%lu\n", name, millis());
    f.close();
    Serial.printf("Saved %s\n", name);
}

void network_display_task(void *pvParameters){
    (void)pvParameters;
    setup_wifi_and_udp();
    setup_image_decoder();
    // Ủy thác cho receive_udp_stream_task (chạy trên cùng Core 1)
    receive_udp_stream_task(pvParameters);
}

void ai_processing_task(void *pvParameters){
    (void)pvParameters;
    Serial.println("[AITask] setup detectors");
    setup_face_detector();
    setup_face_recognizer();
    Serial.println("[AITask] ready");
    // Đăng ký task này với TWDT (watchdog): feed mỗi vòng lặp + sau detect.
    // IDLE không được watch (idle_core_mask=0) — xem chú thích trong setup().
    esp_task_wdt_add(NULL);
    std::vector<float> face_tensor(FACE_TARGET_SIZE*FACE_TARGET_SIZE);
    float embedding[128];
    String last_name=""; int match_count=0; unsigned long last_record=0;
    // ---- TÁI SỬ DỤNG BOX (box reuse): detect chỉ chạy khi cần thiết.
    // Detect (BlazeFace 128x128, Invoke ~20s không có ESP-NN) là phần chậm nhất;
    // Recognize (Ghost 64x64) nhẹ hơn nhiều -> chu kỳ còn ~5s thay vì ~25s.
    // Nếu lượt nhận diện gần nhất ĐẠT ngưỡng (người vẫn đứng yên) thì KHÔNG
    // detect lại định kỳ nữa -> không còn khoảng dừng 20-30s sau SUCCESS.
    // Chỉ detect lại khi: mất box, hoặc kết quả suy giảm (Unknown) cần xác minh.
    FaceBox cached_box{}; bool have_cached_box=false; int recon_cycles=0;
    int unknown_streak=0;
    bool last_match_ok=false; // lượt nhận diện gần nhất có đạt ngưỡng không
    const int BOX_REUSE_FRAMES = 4;     // khi không chắc chắn: detect lại sau tối đa 4 chu kỳ recognize
    const int MAX_UNKNOWN_PAUSE = 2;    // Unknown tạm dừng phiếu tối đa 2 chu kỳ (đồng bộ Laptop TemporalVoter)
    for(;;){
        esp_task_wdt_reset(); // feed watchdog đầu vòng
        bool has_frame=false;
        xSemaphoreTake(image_mutex, portMAX_DELAY);
        if(is_new_frame_available){ is_new_frame_available=false; has_frame=true; }
        xSemaphoreGive(image_mutex);
        if(has_frame){
            // Đọc g_frame_buffer trong MUTEX vì decoder (Core 1) ghi nó dưới cùng mutex
            // (mo_ta_project.md:107, KE 5.9) — tránh frame đang ghi dở bị đọc.
            bool prepared = false;
            // CHÍNH SÁCH BOX REUSE TỐI ƯU:
            // Chỉ reuse box khi:
            // 1. Đã có cached_box hợp lệ
            // 2. VÀ lượt trước nhận diện ĐÚNG người (last_match_ok == true),
            //    hoặc đang có phiếu mà bị chớp nhoáng 1 frame Unknown (match_count > 0 && unknown_streak <= 1).
            // 3. VÀ chưa reuse quá 10 chu kỳ (đề phòng người di chuyển nhẹ).
            // NẾU CHƯA NHẬN DIỆN ĐƯỢC AI (match_count == 0) HOẶC BỊ UNKNOWN: BẮT BUỘC DETECT LẠI NGAY!
            bool can_reuse_box = have_cached_box &&
                                 (last_match_ok || (match_count > 0 && unknown_streak <= 1)) &&
                                 (recon_cycles < 10);
            bool need_detect = !can_reuse_box;
            unsigned long us_det = 0, us_prep = 0, us_rec = 0;
            xSemaphoreTake(image_mutex, portMAX_DELAY);
            if(need_detect){
                unsigned long t_perf = micros();
                FaceBox box = detect_face();
                us_det = micros() - t_perf;
                if(box.is_valid){
                    cached_box = box;
                    have_cached_box = true;
                    recon_cycles = 0;
                    // [CHẨN ĐOÁN] box detector: score/conf + tâm + kích thước crop
                    Serial.printf("[BOX] score=%.2f cx=%.1f cy=%.1f size=%.1f -> %dx%d @(%d,%d)\n",
                        box.score, box.center_x, box.center_y, box.crop_size,
                        box.width, box.height, box.x, box.y);
                    t_perf = micros();
                    prepared = preprocess_face(box, face_tensor.data());
                    us_prep = micros() - t_perf;
                } else {
                    have_cached_box = false;
                    Serial.println("[BOX] KHONG tim thay mat (conf < 0.80)");
                }
            } else {
                // Chu kỳ recognize nhanh: dùng lại box đã detect
                unsigned long t_perf = micros();
                prepared = preprocess_face(cached_box, face_tensor.data());
                us_prep = micros() - t_perf;
                recon_cycles++;
            }
            xSemaphoreGive(image_mutex);
            // Yield 1 tick + feed watchdog (Invoke recognize có thể mất vài giây).
            vTaskDelay(1);
            esp_task_wdt_reset();
            if(prepared){
                unsigned long t_perf = micros();
                extract_face_embedding(face_tensor.data(), embedding);
                us_rec = micros() - t_perf;
                // [PERF] 4.1 — baseline từng chặng để quyết định bật ESP-NN hay không:
                // dec = JPEG decode (Core 1) | bil/he = preprocess | det/rec = TFLM Invoke (Core 0)
                Serial.printf("[PERF] det:%lu.%02lums bil:%luus he:%luus rec:%lu.%02lums dec:%luus cy:%s\n",
                    us_det/1000, (us_det%1000)/10, g_us_bilinear, g_us_he,
                    us_rec/1000, (us_rec%1000)/10, g_us_decode,
                    need_detect ? "DETECT" : "REUSE");
                const float THRESHOLD = FACE_THRESHOLD; // 0.60 (đồng bộ Laptop)
                const char* name = identify_face(embedding, THRESHOLD);
                bool unknown = strcmp(name,"Unknown")==0;
                last_match_ok = !unknown;
                if(unknown){
                    // Chính sách pause-on-Unknown (đồng bộ Laptop TemporalVoter):
                    // Unknown KHÔNG reset ngay chuỗi phiếu của người thật, chỉ tạm dừng;
                    // vượt MAX_UNKNOWN_PAUSE -> hủy chuỗi + buộc detect lại (box có thể cũ).
                    unknown_streak++;
                    if(match_count > 0 && unknown_streak > MAX_UNKNOWN_PAUSE){
                        match_count = 0; last_name = "";
                        have_cached_box = false; // buộc detect lại chu kỳ sau
                    }
                    if(match_count == 0){
                        // Chưa nhận diện được ai mà bị Unknown -> hủy ngay box cũ để frame sau detect chuẩn
                        have_cached_box = false;
                    }
                    if(unknown_streak >= TEMPORAL_VOTES && millis()-last_record>5000){
                        Serial.println("REJECT UNKNOWN");
                        last_record=millis();
                        unknown_streak = 0;
                        have_cached_box = false; // Hủy box khi reject
                        // 2 tiếng bíp dài = reject (README.md:3,186; KE 5.10)
                        for(int i=0;i<2;i++){
                            digitalWrite(BUZZER_PIN,HIGH); digitalWrite(LED_FAIL_PIN,HIGH); vTaskDelay(500/portTICK_PERIOD_MS);
                            digitalWrite(BUZZER_PIN,LOW); digitalWrite(LED_FAIL_PIN,LOW); vTaskDelay(150/portTICK_PERIOD_MS);
                        }
                    }
                } else {
                    unknown_streak = 0;
                    if(String(name)==last_name) match_count++; else { match_count=1; last_name=String(name); }
                    if(match_count >= TEMPORAL_VOTES){ // 3 frames (đồng bộ Laptop)
                        if(millis()-last_record>10000){
                            Serial.printf("SUCCESS %s\n", last_name.c_str());
                            log_attendance(last_name.c_str());
                            last_record=millis();
                            for(int i=0;i<1;i++){ digitalWrite(BUZZER_PIN,HIGH); digitalWrite(LED_SUCCESS_PIN,HIGH); vTaskDelay(100/portTICK_PERIOD_MS); digitalWrite(BUZZER_PIN,LOW); digitalWrite(LED_SUCCESS_PIN,LOW); vTaskDelay(100/portTICK_PERIOD_MS); }
                        }
                    }
                }
            } else { match_count=0; last_name=""; unknown_streak=0; last_match_ok=false; have_cached_box=false; }
        }
        vTaskDelay(pdMS_TO_TICKS(10));  
    }
}  

void setup(){  
    setCpuFrequencyMhz(240); // Khóa xung nhịp CPU ở mức tối đa 240MHz (tăng 50% hiệu năng tính toán)
    Serial.begin(115200);
    Serial.setDebugOutput(true);
    delay(1500);
    // Log qua CẢ Serial và console (printf→UART0): đảm bảo luôn thấy output bất kể
    // USB CDC On Boot = Enabled hay Disabled, vì ESP32-S3 có 2 cổng USB vật lý
    // (UART/COM qua cầu chip + native USB CDC). ROM luôn in trên UART0.
    printf("\n=== ESP32-S3 Face System 128->64 Bilinear TCP12345 (console/UART0) ===\n");
    Serial.println("\n=== ESP32-S3 Face System 128->64 Bilinear TCP12345 ===");
    Serial.printf("CPU Freq: %u MHz\n", getCpuFrequencyMhz());
    // ---- CHẨN ĐOÁN BỘ NHỚ NGAY ĐẦU BOOT (giúp xác định lỗi PSRAM/flash) ----
    Serial.printf("RAM int : total=%u free=%u\n", ESP.getHeapSize(), ESP.getFreeHeap());
    Serial.printf("PSRAM   : total=%u free=%u\n", ESP.getPsramSize(), ESP.getFreePsram());
    Serial.printf("Flash   : %u bytes\n", ESP.getFlashChipSize());
    if(!psramFound()) Serial.println("WARN: PSRAM KHONG DUOC BAT (check Tools -> PSRAM = OPI PSRAM) !");
    else Serial.printf("PSRAM OK (OPI): %u bytes\n", ESP.getPsramSize());
    Serial.printf("Heap free: %u (internal %u, PSRAM %u)\n",
        ESP.getFreeHeap(), ESP.getFreeHeap(), ESP.getFreePsram());

    pinMode(BUZZER_PIN, OUTPUT); digitalWrite(BUZZER_PIN,LOW);
    pinMode(LED_SUCCESS_PIN, OUTPUT); digitalWrite(LED_SUCCESS_PIN,LOW);
    pinMode(LED_FAIL_PIN, OUTPUT); digitalWrite(LED_FAIL_PIN,LOW);
    // Watchdog strategy (fix task_wdt IDLE0 warning triệt để):
    // - KHÔNG theo dõi IDLE tasks (idle_core_mask=0): inference 1 frame mất ~20-25s
    //   (chưa có ESP-NN), AITask chiếm CPU 0 hợp lệ khiến IDLE0 starve → nếu watch IDLE
    //   sẽ luôn có cảnh báo task_wdt.
    // - Thay vào đó ĐĂNG KÝ chính AITask (esp_task_wdt_add trong ai_processing_task) và
    //   feed nó mỗi vòng + sau detect. AI treo/deadlock → không feed → cảnh báo → biết lỗi.
    // - trigger_panic=false: chỉ cảnh báo, không reboot (backpressure đã giữ hệ sống).
    esp_task_wdt_config_t wdt_cfg = {
        .timeout_ms = 60000,
        .idle_core_mask = 0,
        .trigger_panic = false,
    };
    if (esp_task_wdt_reconfigure(&wdt_cfg) != ESP_OK) {
        esp_task_wdt_init(&wdt_cfg); // TWDT chưa init thì init luôn
    }
    if(!SPIFFS.begin(true)) Serial.println("SPIFFS fail"); else Serial.println("SPIFFS OK");
    if(!image_mutex) image_mutex = xSemaphoreCreateMutex();
    // BOOT SEQUENCE ĐÚNG (KE_HOACH 5.10, mo_ta_project.md:12):
    // network_display_task sẽ gọi setup_wifi_and_udp() → setup_image_decoder() rồi mới
    // vào vòng nhận frame. Không tạo receive_udp_stream_task trước lifecycle này.
    xTaskCreatePinnedToCore(network_display_task, "NetTask", 8192, NULL, 4, &NetDisplayTask, 1);
    xTaskCreatePinnedToCore(ai_processing_task, "AITask", 32768, NULL, 5, &AITask, 0);
    Serial.println("Tasks created");
}
void loop(){
    static int sec=0; sec+=2;  
    if(WiFi.status()==WL_CONNECTED) Serial.printf("[HB] IP %s TCP %d Uptime %ds\n", WiFi.localIP().toString().c_str(), tcp_port, sec);
    else Serial.printf("[HB] WiFi connecting %ds\n", sec);
    vTaskDelay(pdMS_TO_TICKS(2000));
}
