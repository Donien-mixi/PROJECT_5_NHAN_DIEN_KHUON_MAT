#include "wifi_udp_server.h"
#include "ai_config.h"
#include "image_decoder.h"
#include <Arduino.h>

// TODO: Đổi WiFi của bạn
const char *ssid = "Phong 2";  
const char *password = "Kimhoang";  
const int tcp_port = 12345;

WiFiServer server(tcp_port);
WiFiClient client;

volatile bool is_new_frame_available = false;
uint8_t* packet_buffer = nullptr;
static uint8_t* jpeg_data = nullptr;
static uint32_t jpeg_len = 0;
SemaphoreHandle_t image_mutex = nullptr; // Dùng chung với .ino
// [PERF] 4.1 — thời gian giải mã JPEG gần nhất (micros), đọc từ AITask
volatile unsigned long g_us_decode = 0;

void setup_wifi_and_udp(){
    packet_buffer = (uint8_t*)heap_caps_malloc(PACKET_BUFFER_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if(!packet_buffer){ Serial.println("FATAL: PSRAM packet_buffer fail"); return; }
    Serial.printf("packet_buffer %d PSRAM OK\n", PACKET_BUFFER_SIZE);

    Serial.printf("\n[WiFi] Connecting to %s\n", ssid);
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);
    while(WiFi.status()!=WL_CONNECTED){ delay(500); Serial.print("."); }
    Serial.println("\n[WiFi] Connected");
    Serial.println(WiFi.localIP());
    server.begin();
    Serial.printf("TCP 12345 listening (packet %d)\n", PACKET_BUFFER_SIZE);
    if(!image_mutex) image_mutex = xSemaphoreCreateMutex();
    jpeg_data = (uint8_t*)heap_caps_malloc(PACKET_BUFFER_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
}

bool is_new_frame_available_flag(){ return is_new_frame_available; }
uint32_t get_last_jpeg_size(){ return jpeg_len; }
uint8_t* get_last_jpeg_data(){ return jpeg_data; }

// Task Core 1: nhận JPEG qua TCP (gọi sau setup_wifi_and_udp + setup_image_decoder)
//
// BACKPRESSURE FIX (quan trọng): việc ĐỌC socket phải tách rời việc DECODE.
// Inference trên Core 0 mất vài giây/frame (chưa có ESP-NN). Nếu NetTask chờ
// image_mutex trước khi đọc tiếp, TCP backlog đầy → streamer bị block trong
// sendall → frame lệch nửa chừng → disconnect → hệ hết frame ("chết lặng").
// Giải pháp: LUÔN đọc header+payload từ socket (nhanh), sau đó CHỈ decode khi
// AITask đã xử lý xong frame trước (flag=false) và lấy được mutex NGAY
// (non-blocking). Nếu AI đang bận → DROP frame này, socket vẫn được xả sạch.
void receive_udp_stream_task(void *pvParameters){
    (void)pvParameters;
    for(;;){
        client = server.available();
        if(client){
            client.setTimeout(2000); // 2s timeout cho readBytes — tránh treo task
            Serial.println("[Net] Client connected");
            while(client.connected()){
                // Đọc ĐỦ 4 bytes length (little endian) bằng readBytes, không giả định
                // một read() luôn đủ 4 byte (mo_ta_project.md:16, README.md:57)
                uint32_t len=0;
                if(client.readBytes((uint8_t*)&len, 4) != 4){
                    Serial.println("[Net] Header under-read, disconnect");
                    break;
                }
                if(len==0 || len>PACKET_BUFFER_SIZE){
                    Serial.printf("Bad len %u, disconnect\n", len);
                    break; // dữ liệu lệch frame → đóng kết nối để streamer reconnect
                }
                uint32_t received=0;
                unsigned long start=millis();
                while(received < len){
                    if(!client.connected()){
                        Serial.println("[Net] Payload: client lost");
                        break;
                    }
                    int chunk = client.read(packet_buffer + received, len - received);
                    if(chunk>0){
                        received += chunk;
                    } else {
                        // chunk<=0: không có dữ liệu (timeout tối đa 2s trong read)
                        if(millis()-start > 3000){
                            Serial.println("[Net] Payload timeout");
                            break;
                        }
                        vTaskDelay(2/portTICK_PERIOD_MS);
                    }
                }
                if(received != len){
                    // Đọc bỏ dữ liệu rác còn lại để không lệch frame sau
                    while(client.available()) client.read();
                    continue;
                }

                // LUÔN giải mã frame mới nhất vào g_frame_buffer mỗi khi lấy được mutex.
                // Loại bỏ điều kiện (!is_new_frame_available) để Core 1 liên tục cập nhật
                // frame mới nhất trong suốt 4.7s Core 0 đang chạy AI. Nhờ đó khi Core 0
                // kết thúc chu kỳ cũ và bắt đầu chu kỳ mới, frame trong g_frame_buffer
                // luôn là frame "nóng hổi" của vài chục miligiây trước, KHÔNG PHẢI frame cũ của 5s trước!
                if(image_mutex && xSemaphoreTake(image_mutex, 0) == pdTRUE){
                    memcpy(jpeg_data, packet_buffer, len);
                    jpeg_len = len;
                    // Giải mã ngay trên Core 1 để Core 0 chỉ việc detect (README.md:151-152)
                    unsigned long t_perf = micros();
                    decode_jpeg_frame(jpeg_data, jpeg_len);
                    g_us_decode = micros() - t_perf;
                    is_new_frame_available = true;
                    xSemaphoreGive(image_mutex);
                }
                // Ngược lại: frame bị drop (AI đang crop/detect) — không sao, socket vẫn được xả sạch.
            }
            client.stop();
            Serial.println("[Net] Client disconnected");
        }
        vTaskDelay(10/portTICK_PERIOD_MS);
    }
}
