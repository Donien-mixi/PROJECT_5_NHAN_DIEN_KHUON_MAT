#ifndef WIFI_UDP_SERVER_H
#define WIFI_UDP_SERVER_H

#include <WiFi.h>
#include <vector>
#include "ai_config.h"

// WiFi config
extern const char* ssid;
extern const char* password;
extern const int tcp_port;
extern WiFiServer server;
extern WiFiClient client;

extern volatile bool is_new_frame_available;
extern uint8_t* packet_buffer;
extern SemaphoreHandle_t image_mutex; // Mutex dùng chung với firmware_esp32.ino (mo_ta_project.md:107)
extern volatile unsigned long g_us_decode; // [PERF] 4.1 — thời gian giải mã JPEG gần nhất (micros)
#define MAX_PACKET_SIZE PACKET_BUFFER_SIZE // 32KB = 128*128*2

void setup_wifi_and_udp();
void receive_udp_stream_task(void *pvParameters);
bool is_new_frame_available_flag();
uint32_t get_last_jpeg_size();
uint8_t* get_last_jpeg_data();

#endif
