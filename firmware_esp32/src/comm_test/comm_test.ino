#include <Arduino.h>

// ==============================================================================
// CẤU HÌNH TỐC ĐỘ BAUD VÀ HEADER GIAO THỨC
// ==============================================================================
const int BAUD_RATE = 921600;
const uint8_t HEADER_1 = 0xFF;
const uint8_t HEADER_2 = 0xAA;

// Bộ đệm tĩnh 64KB (Tránh cấp phát malloc trong loop gây phân mảnh RAM)
#define MAX_PAYLOAD_SIZE 65536
uint8_t payloadBuffer[MAX_PAYLOAD_SIZE];

void setup() {
  // 1. CẤU HÌNH BỘ ĐỆM RX TRƯỚC KHI GỌI Serial.begin() (BẮT BUỘC TRÊN ESP32)
  Serial.setRxBufferSize(65536); 
  
  // 2. Khởi tạo Serial tốc độ cao
  Serial.begin(BAUD_RATE);
  
  // Chờ cổng Serial ổn định
  delay(1000); 
  
  // In thông báo khởi động kèm thông số phần cứng ESP32-S3 N16R8
  Serial.println();
  Serial.print("{\"status\":\"ready\",\"chip\":\"ESP32-S3\",\"psram_size_kb\":");
  Serial.print(ESP.getPsramSize() / 1024);
  Serial.print(",\"free_psram_kb\":");
  Serial.print(ESP.getFreePsram() / 1024);
  Serial.print(",\"free_heap_kb\":");
  Serial.print(ESP.getFreeHeap() / 1024);
  Serial.println("}");
}

// Máy trạng thái đọc gói tin (Non-blocking Finite State Machine)
enum State { WAIT_H1, WAIT_H2, READ_CMD, READ_LEN, READ_PAYLOAD, READ_CHECKSUM };
State currentState = WAIT_H1;

uint32_t payloadLength = 0;
uint8_t currentCmd = 0;
uint32_t bytesRead = 0;
uint32_t lenBytesRead = 0;
uint8_t calculatedChecksum = 0;
unsigned long lastByteReceivedTime = 0;

void loop() {
  // Timeout bảo vệ: Nếu đang đọc dở gói tin mà mất kết nối quá 150ms -> Reset về đầu
  if (currentState != WAIT_H1 && (millis() - lastByteReceivedTime > 150)) {
    currentState = WAIT_H1;
  }

  while (Serial.available() > 0) {
    uint8_t incomingByte = Serial.read();
    lastByteReceivedTime = millis();

    switch (currentState) {
      case WAIT_H1:
        if (incomingByte == HEADER_1) {
          currentState = WAIT_H2;
        }
        break;

      case WAIT_H2:
        if (incomingByte == HEADER_2) {
          currentState = READ_CMD;
        } else {
          currentState = (incomingByte == HEADER_1) ? WAIT_H2 : WAIT_H1;
        }
        break;

      case READ_CMD:
        currentCmd = incomingByte; 
        currentState = READ_LEN;
        payloadLength = 0;
        lenBytesRead = 0;
        break;

      case READ_LEN:
        // Đọc 4 bytes độ dài (Little-Endian)
        payloadLength |= ((uint32_t)incomingByte << (8 * lenBytesRead));
        lenBytesRead++;
        if (lenBytesRead == 4) {
          if (payloadLength > MAX_PAYLOAD_SIZE) { 
            // Gói tin bất thường -> Reset
            currentState = WAIT_H1;
          } else {
            bytesRead = 0;
            calculatedChecksum = 0;
            if (payloadLength == 0) {
              currentState = READ_CHECKSUM;
            } else {
              currentState = READ_PAYLOAD;
            }
          }
        }
        break;

      case READ_PAYLOAD:
        payloadBuffer[bytesRead] = incomingByte;
        calculatedChecksum ^= incomingByte;
        bytesRead++;
        
        if (bytesRead == payloadLength) {
          currentState = READ_CHECKSUM;
        }
        break;

      case READ_CHECKSUM:
        if (incomingByte == calculatedChecksum) {
          unsigned long processStartTime = micros();

          // ========================================================
          // XỬ LÝ LỆNH TỪ LAPTOP
          // ========================================================
          if (currentCmd == 0x00) {
            // LỆNH 0x00: PING TEST (Kiểm tra phản hồi nhanh)
            Serial.println("{\"status\":\"success\",\"cmd\":\"PING\",\"message\":\"PONG\"}");
          } 
          else if (currentCmd == 0x01) {
            // LỆNH 0x01: NHẬN ẢNH KHUÔN MẶT
            // (Sau này là nơi gọi TFLite Micro trên Core 1)
            delay(10); // Giả lập thời gian suy luận AI 10ms
            
            unsigned long processEndTime = micros();
            unsigned long durationMs = (processEndTime - processStartTime) / 1000;
            
            // Trả về kết quả JSON
            Serial.print("{\"status\":\"success\",");
            Serial.print("\"cmd\":\"IMAGE\",");
            Serial.print("\"received_bytes\":");
            Serial.print(payloadLength);
            Serial.print(",");
            Serial.print("\"simulated_infer_ms\":");
            Serial.print(durationMs);
            Serial.print(",");
            Serial.print("\"message\":\"Image received correctly!\"}");
            Serial.println();
          }
        } else {
          // Lỗi checksum
          Serial.println("{\"status\":\"error\",\"message\":\"Checksum mismatch\"}");
        }

        // Reset trạng thái sẵn sàng đón gói tin tiếp theo
        currentState = WAIT_H1;
        break;
    }
  }
}
