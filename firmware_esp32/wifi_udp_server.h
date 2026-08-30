#ifndef WIFI_UDP_SERVER_H
#define WIFI_UDP_SERVER_H

#include <WiFi.h>
#include <vector>

// Cấu hình Wi-Fi
extern const char* ssid;
extern const char* password;
extern const int server_port;

extern WiFiServer server;
extern WiFiClient client;

extern std::vector<uint8_t> jpeg_buffer;
extern volatile bool is_new_frame_available;
extern uint8_t* packet_buffer;
extern const int MAX_UDP_PACKET_SIZE;

void setup_wifi_and_udp();
void receive_udp_stream_task(void *pvParameters);

#endif
