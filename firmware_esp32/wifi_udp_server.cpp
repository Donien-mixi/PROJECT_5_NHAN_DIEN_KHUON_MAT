#include "wifi_udp_server.h"

// TODO: Đổi thành thông tin mạng của người dùng
const char *ssid = "Nghi";
const char *password = "12345678";
const int tcp_port = 12345;

WiFiServer server(tcp_port);
WiFiClient client;

std::vector<uint8_t> jpeg_buffer;
volatile bool is_new_frame_available = false;   

// Kích thước tối đa của gói tin (đệm bộ nhớ)
uint8_t* packet_buffer = nullptr;   
const int MAX_UDP_PACKET_SIZE = 65507;

void setup_wifi_and_udp() {
  packet_buffer = (uint8_t*)heap_caps_malloc(MAX_UDP_PACKET_SIZE, MALLOC_CAP_SPIRAM);
  if (!packet_buffer) {
    Serial.println("PSRAM allocation failed for Buffer! Falling back to SRAM...");
    packet_buffer = (uint8_t*)malloc(MAX_UDP_PACKET_SIZE);
  }

  Serial.printf("\n[Wi-Fi] Bat dau ket noi toi SSID: %s\n", ssid);
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  int count = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    count++;
    if (count % 20 == 0) {
      Serial.printf("\n[Wi-Fi] Dang cho ket noi toi '%s'...", ssid);
    }
  }
  Serial.println("\n[Wi-Fi] >>> KET NOI THANH CONG! <<<");
  Serial.print("[Wi-Fi] IP cua ESP32: ");
  Serial.println(WiFi.localIP());

  server.begin();
  Serial.printf("Listening on TCP port %d\n", tcp_port);

  // Cấp phát trước vector buffer trên PSRAM nếu có thể
  jpeg_buffer.reserve(30000);
}
