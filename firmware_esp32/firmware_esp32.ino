#include <Arduino.h>                
#include <TensorFlowLite_ESP32.h>       
#include <SPIFFS.h> // Thư viện ghi file lưu lịch sử
#include <TJpg_Decoder.h>     
#include "wifi_udp_server.h"     
#include "image_decoder.h"    
#include "ai_face_detector.h"
#include "ai_face_recognizer.h"      
#include "ai_config.h"

// Task handles     
TaskHandle_t NetworkDisplayTask;               
TaskHandle_t AITask;
      
SemaphoreHandle_t image_mutex; 

#define BUZZER_PIN 15
#define LED_SUCCESS_PIN 4
#define LED_FAIL_PIN 5

// Hàm ghi log điểm danh vào SPIFFS
void log_attendance(const char* name) {
    File file = SPIFFS.open("/attendance.csv", FILE_APPEND);
    if (!file) {
        Serial.println("❌ Lỗi: Không thể mở file attendance.csv để ghi!");
        return;
    }
    // Ghi tên và thời gian (Uptime)  
    file.printf("%s, Uptime: %lu ms\n", name, millis());
    file.close();
    Serial.printf("💾 Đã lưu lịch sử điểm danh của %s vào bộ nhớ Flash!\n", name);
}

void network_display_task(void *pvParameters) {  
    Serial.println("[NetDisplayTask] Bắt đầu setup_wifi_and_udp()...");
    setup_wifi_and_udp();
    
    Serial.println("[NetDisplayTask] Bắt đầu setup_image_decoder()...");
    setup_image_decoder();
    
    Serial.println("[NetDisplayTask] Vào vòng lặp chính...");
    
    while (true) {
        // Kiểm tra và duy trì kết nối Client an toàn hơn
        if (server.hasClient()) {
            if (client) {
                Serial.println("[NetTask] Đóng kết nối cũ để nhận kết nối mới...");
                client.stop();
            }
            client = server.available();
            if (client) {
                Serial.println("[NetTask] Client mới đã kết nối thành công!");
            }
        }
        
        // Nếu client cũ đã ngắt kết nối đột ngột   
        if (client && !client.connected()) {  
            client.stop();  
        }
        
        // Nhận ảnh từ TCP
        if (client && client.available() >= 4) {
            uint32_t length = 0;
            client.readBytes((uint8_t*)&length, 4); // Đọc 4 bytes header (Little Endian)
            // Serial.printf("[NetTask] Có Header báo độ dài gói tin: %d bytes\n", length);
            
            if (length > 0 && length < MAX_UDP_PACKET_SIZE) {
                int received = 0;
                unsigned long start_time = millis();
                
                // Serial.println("[NetTask] Bắt đầu đọc payload...");
                // Đọc toàn bộ gói tin JPEG
                while (received < length && client.connected()) {
                    int avail = client.available();
                    if (avail > 0) {
                        int to_read = (avail < (length - received)) ? avail : (length - received);
                        int len = client.read(packet_buffer + received, to_read);
                        if (len > 0) received += len;
                    } else {
                        vTaskDelay(pdMS_TO_TICKS(1)); // Chờ dữ liệu đến
                    }
                    // Timeout chống kẹt
                    if (millis() - start_time > 2000) {
                        Serial.println("⚠️ [NetTask] Timeout khi đọc TCP! Mạng chập chờn...");
                        break;
                    }
                }
                
                // Serial.printf("[NetTask] Đã đọc xong payload. Nhận được: %d/%d bytes\n", received, length);
                
                if (received == length) {
                    // Serial.printf("[NetTask] Đã nhận xong khung hình %d bytes. Bắt đầu giải mã...\n", length);
                    decode_jpeg_frame(packet_buffer, length);
                    // Serial.println("[NetTask] Giải mã xong. Báo cho AITask...");
                    
                    // Kích hoạt AITask xử lý sau khi đã giải mã xong
                    xSemaphoreTake(image_mutex, portMAX_DELAY);
                    is_new_frame_available = true;
                    xSemaphoreGive(image_mutex);
                    // Serial.println("[NetTask] Đã đẩy frame vào hàng đợi AI!");
                } else {
                    Serial.println("⚠️ [NetTask] Gói tin bị thiếu dữ liệu, bỏ qua!");
                    // Đọc bỏ các byte rác còn lại trong buffer để tránh lệch frame sau
                    while(client.available()) client.read();
                }
            } else {
                Serial.println("[NetTask] LỖI: Độ dài gói tin không hợp lệ!");
                while(client.available()) client.read();
            }
        }
        vTaskDelay(pdMS_TO_TICKS(5)); // Tránh WDT (Watchdog Timer)
    }
}

void ai_processing_task(void *pvParameters) {
    Serial.println("[AITask] Initializing models...");
    setup_face_detector();
    setup_face_recognizer();
    Serial.println("[AITask] Models ready.");
    
    // Cấp phát trên HEAP để tránh tràn STACK
    std::vector<float> face_tensor(FACE_TARGET_SIZE * FACE_TARGET_SIZE);
    float face_embedding[128];
    
    // Biến cho Temporal Voting
    String last_matched_name = "";
    int match_count = 0;
    unsigned long last_record_time = 0;
    unsigned long last_ping = millis();
    
    while (true) {
        bool has_new_image = false;
        
        if (millis() - last_ping > 5000) {
            Serial.println("[AITask] Alive...");
            last_ping = millis();
        }
        
        xSemaphoreTake(image_mutex, portMAX_DELAY);
        if (is_new_frame_available) {
            Serial.println("[AITask] Đã bắt được cờ is_new_frame_available = true!");
            is_new_frame_available = false;
            has_new_image = true;
        }
        xSemaphoreGive(image_mutex);
        
        if (has_new_image) {
            Serial.println("==========================================");
            Serial.println("[AITask] Đã nhận tín hiệu ảnh mới. Bắt đầu chạy detect_face()...");
            Serial.println("==========================================");
            Serial.flush(); // Đảm bảo in ra ngay lập tức
            
            unsigned long t0 = millis();
            // Bước 1: Dò khuôn mặt
            FaceBox box = detect_face();
            Serial.printf("[AITask] detect_face() chạy xong! Thời gian: %lu ms\n", millis() - t0);
            Serial.flush();  
            
            if (box.is_valid) {
                // Serial.println("[AITask] Đang tiền xử lý (Crop & Grayscale)...");
                if (preprocess_face(box, face_tensor.data())) {
                    // Bước 3: Rút trích Vector 128-D
                    extract_face_embedding(face_tensor.data(), face_embedding);
                    
                    // Bước 4: So khớp danh tính (Threshold 0.93 theo đề xuất)
                    const float THRESHOLD = 0.93f;
                    const char* name = identify_face(face_embedding, THRESHOLD);
                    bool is_unknown = (strcmp(name, "Unknown") == 0);
                    
                    // -------- THUẬT TOÁN TEMPORAL VOTING --------
                    if (String(name) == last_matched_name) {
                        match_count++;
                    } else {
                        match_count = 1;
                        last_matched_name = name;
                    }
                    
                    // Nếu trùng khớp 3 lần liên tiếp (3 frames)
                    if (match_count >= 3) {
                        if (!is_unknown) {
                            // Người quen: Ghi log, chống spam (Cooldown 10 giây)
                            if (millis() - last_record_time > 10000) {
                                Serial.printf("\n=========================================\n");
                                Serial.printf("🎉 ĐIỂM DANH THÀNH CÔNG: %s\n", last_matched_name.c_str());
                                Serial.printf("=========================================\n\n");
                                
                                log_attendance(last_matched_name.c_str());
                                last_record_time = millis();
                                
                                // Buzzer & Green LED Success (2 tiếng bíp)
                                for(int i=0; i<2; i++) {
                                    digitalWrite(BUZZER_PIN, HIGH);
                                    digitalWrite(LED_SUCCESS_PIN, HIGH);
                                    delay(100);
                                    digitalWrite(BUZZER_PIN, LOW);
                                    digitalWrite(LED_SUCCESS_PIN, LOW);
                                    delay(100);
                                }
                            }
                        } else {
                            // Người lạ: Cảnh báo, cooldown 5 giây
                            if (millis() - last_record_time > 5000) {
                                Serial.printf("\n=========================================\n");
                                Serial.printf("🚨 TỪ CHỐI: PHÁT HIỆN NGƯỜI LẠ!\n");
                                Serial.printf("=========================================\n\n");
                                
                                last_record_time = millis();
                                
                                // Buzzer & Red LED Warning (Bíp dài)
                                digitalWrite(BUZZER_PIN, HIGH);
                                digitalWrite(LED_FAIL_PIN, HIGH);
                                delay(1000);
                                digitalWrite(BUZZER_PIN, LOW);
                                digitalWrite(LED_FAIL_PIN, LOW);
                            }
                        }
                    }
                }
            } else {
                match_count = 0; // Tắt reset nếu không có mặt để kết quả ổn định hơn, nhưng thôi cứ reset cho chắc
            }
        }
        
        vTaskDelay(pdMS_TO_TICKS(10));    
    }
}

#include <esp_task_wdt.h>

void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("\n\n--- HỆ THỐNG KHỞI ĐỘNG ---");
    
    // Khởi tạo Buzzer và LEDs
    pinMode(BUZZER_PIN, OUTPUT);
    digitalWrite(BUZZER_PIN, LOW);
    
    pinMode(LED_SUCCESS_PIN, OUTPUT);
    digitalWrite(LED_SUCCESS_PIN, LOW);
    
    pinMode(LED_FAIL_PIN, OUTPUT);
    digitalWrite(LED_FAIL_PIN, LOW);
    
    // Cấu hình lại WDT cho Arduino Core v3.x (ESP-IDF v5)
    esp_task_wdt_config_t twdt_config = {
        .timeout_ms = 30000,
        .idle_core_mask = (1 << portNUM_PROCESSORS) - 1,
        .trigger_panic = false,
    };
    esp_task_wdt_reconfigure(&twdt_config);

    // Khởi tạo SPIFFS để ghi lịch sử  
    if (!SPIFFS.begin(true)) {
        Serial.println("❌ Lỗi: Không thể khởi tạo SPIFFS!");
    } else {
        Serial.println("✅ Đã khởi tạo bộ nhớ Flash (SPIFFS) thành công.");
    }
    
    // Tạo Mutex để bảo vệ biến dùng chung jpeg_buffer
    image_mutex = xSemaphoreCreateMutex();
    if (image_mutex == NULL) {
        Serial.println("Lỗi: Không thể tạo Mutex!");
    }
    
    Serial.println("Đang khởi tạo Core 1 (Network & Image Decoding)...");
    // Khởi chạy Core 1 cho việc Nhận ảnh Wi-Fi và vẽ màn hình (tránh lag UI)
    BaseType_t resNet = xTaskCreatePinnedToCore(
        network_display_task,
        "NetDisplayTask",
        8192,
        NULL,
        1,
        &NetworkDisplayTask,
        1 // Core 1
    );
    if (resNet != pdPASS) Serial.println("❌ LỖI: Không thể tạo NetDisplayTask!");
    
    // Khởi tạo AI Task (Core 0)
    BaseType_t resAI = xTaskCreatePinnedToCore(
        ai_processing_task,
        "AITask",
        8192, // Giảm xuống 8KB vì stack lớn có thể khiến ESP32 hết RAM nội bộ và không tạo được Task
        NULL,
        1,
        &AITask,
        0 // Chạy trên Core 0
    );
    if (resAI != pdPASS) {
        Serial.println("❌ LỖI: Không thể tạo AITask! (Hết RAM?)");
    } else {
        Serial.println("✅ Đã tạo AITask trên Core 0 thành công!");
    }
}

void loop() {
    static int sec_count = 0;
    sec_count += 2;
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("[Heartbeat] ESP32 dang chay | IP: %s | UDP Port: 12345 | Uptime: %d s\n", 
                      WiFi.localIP().toString().c_str(), sec_count);
    } else {
        Serial.printf("[Heartbeat] Dang cho ket noi Wi-Fi... (Uptime: %d s)\n", sec_count);
    }
    delay(2000);
}
