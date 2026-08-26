#include <Arduino.h>
#include <LiquidCrystal.h>

// ==============================================================================
// 1. CẤU HÌNH CHÂN MÀN HÌNH LCD1602 (CHẾ ĐỘ 4-BIT) VÀ CÒI CHÍP BUZZER
// ==============================================================================
// Các chân GPIO an toàn trên ESP32-S3 (không trùng PSRAM/Flash/Serial)
const int LCD_RS = 4;
const int LCD_EN = 5;
const int LCD_D4 = 6;
const int LCD_D5 = 7;
const int LCD_D6 = 15;
const int LCD_D7 = 16;
const int BUZZER_PIN = 17; // Còi chíp 5V/3.3V (Báo bíp khi điểm danh)

// Khởi tạo đối tượng màn hình LCD 16x2
LiquidCrystal lcd(LCD_RS, LCD_EN, LCD_D4, LCD_D5, LCD_D6, LCD_D7);

// ==============================================================================
// 2. NHÚNG MÔ HÌNH AI INT8 VÀ CƠ SỞ DỮ LIỆU KHUÔN MẶT
// ==============================================================================
#include "model_data.h"
#include "face_database.h"

// ==============================================================================
// 3. THƯ VIỆN TENSORFLOW LITE FOR MICROCONTROLLERS (TFLITE MICRO)
// ==============================================================================
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Thư viện FreeRTOS
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

// ==============================================================================
// 4. CẤU HÌNH PHẦN CỨNG & THUẬT TOÁN NHẬN DIỆN
// ==============================================================================
const int BAUD_RATE = 921600;
const uint8_t HEADER_1 = 0xFF;
const uint8_t HEADER_2 = 0xAA;

const uint8_t CMD_PING = 0x00;
const uint8_t CMD_INFO = 0x01;
const uint8_t CMD_INFERENCE = 0x02;

const int INPUT_WIDTH = 64;
const int INPUT_HEIGHT = 64;
const int INPUT_CHANNELS = 1;
const int EXPECTED_PAYLOAD_SIZE = INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS; // 4096 bytes
const int EMBEDDING_DIM = 128;
const float MATCH_THRESHOLD = 0.40f; // Ngưỡng Cosine Similarity (hạ xuống 0.40 phù hợp với mạng Ghost siêu nhẹ 64x64 INT8)

// ==============================================================================
// TỐI ƯU HÓA BỘ NHỚ SRAM:
// - Tensor Arena: 200KB trong Internal SRAM (siêu tốc, đủ cho Ghost network activation maps)
// - PayloadBuffer: Chỉ 8KB (ảnh 64x64 chỉ cần 4096 bytes + dư cho header/checksum)
// - Giải phóng ~57KB so với cấu hình cũ (payloadBuffer 64KB + arena 136KB)
// ==============================================================================
const size_t kTensorArenaSize = 200 * 1024;
uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));

// Các đối tượng quản lý TFLite Micro
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

// Cờ trạng thái setup thành công (bảo vệ loop() khỏi crash khi setup thất bại)
volatile bool setup_ok = false;

// Bộ đệm nhận gói tin Serial (chỉ cần 8KB cho ảnh 64x64 = 4096 bytes)
#define MAX_PAYLOAD_SIZE 8192
uint8_t payloadBuffer[MAX_PAYLOAD_SIZE];

// Máy trạng thái nhận gói tin (FSM)
enum State { WAIT_H1, WAIT_H2, READ_CMD, READ_LEN, READ_PAYLOAD, READ_CHECKSUM };
State currentState = WAIT_H1;

uint32_t payloadLength = 0;
uint8_t currentCmd = 0;
uint32_t bytesRead = 0;
uint32_t lenBytesRead = 0;
uint8_t calculatedChecksum = 0;
unsigned long lastByteReceivedTime = 0;

// Signal cho FreeRTOS Task
volatile bool inference_requested = false;

// FreeRTOS Queue để truyền ảnh từ Core 1 sang Core 0
QueueHandle_t imageQueue;

// ==============================================================================
// TASK FREERTOS: XỬ LÝ AI TRÊN CORE 0
// ==============================================================================
void InferenceTask(void *pvParameters) {
  uint8_t taskPayload[EXPECTED_PAYLOAD_SIZE];
  while (true) {
    // Block vĩnh viễn cho đến khi nhận được khung hình từ Core 1
    if (xQueueReceive(imageQueue, &taskPayload, portMAX_DELAY) == pdPASS) {
      processInference(taskPayload, EXPECTED_PAYLOAD_SIZE);
    }
  }
}

// ==============================================================================
// HÀM ĐIỀU KHIỂN MÀN HÌNH LCD 1602 VÀ CÒI CHÍP
// ==============================================================================
void lcd_show(const char* line1, const char* line2) {
  lcd.setCursor(0, 0);
  char buf1[17];
  snprintf(buf1, sizeof(buf1), "%-16s", line1);
  lcd.print(buf1);

  lcd.setCursor(0, 1);
  char buf2[17];
  snprintf(buf2, sizeof(buf2), "%-16s", line2);
  lcd.print(buf2);
}

void buzzer_beep(int times, int duration_ms) {
  for (int i = 0; i < times; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(duration_ms);
    digitalWrite(BUZZER_PIN, LOW);
    if (i < times - 1) delay(60);
  }
}

unsigned long last_inference_time = 0;
bool is_idle_displayed = false;

// ==============================================================================
// 5. KHỞI TẠO HỆ THỐNG (SETUP)
// ==============================================================================
void setup() {
  // 0. Khởi tạo Còi chíp & Màn hình LCD1602 (16 cột x 2 dòng)
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  lcd.begin(16, 2);
  lcd_show("TINYML EDGE AI", "Khoi dong ESP...");
  buzzer_beep(1, 100);

  // Cấu hình bộ đệm RX 16KB cho Serial tốc độ cao (Bắt buộc trước Serial.begin)
  Serial.setRxBufferSize(16384);
  Serial.begin(BAUD_RATE);
  delay(500);

  // 1. Kiểm tra bộ nhớ 8MB Octal PSRAM
  if (!psramFound()) {
    Serial.println("{\"status\":\"error\",\"message\":\"PSRAM not found! Please select OPI PSRAM in Tools menu\"}");
    lcd_show("LOI HE THONG!", "Khong thay PSRAM");
    return;
  }

  // 2. Nạp mô hình AI từ mảng Byte g_model_data
  model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("{\"status\":\"error\",\"message\":\"Model schema version mismatch!\"}");
    lcd_show("LOI MO HINH!", "Schema mismatch");
    return;
  }

  // 3. Khởi tạo Op Resolver với đúng 5 Ops
  static tflite::MicroErrorReporter micro_error_reporter;
  static tflite::ErrorReporter* error_reporter = &micro_error_reporter;
  
  static tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddConcatenation();
  resolver.AddAdd();
  resolver.AddFullyConnected();

  // 4. Build Interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter
  );
  interpreter = &static_interpreter;

  // 5. Cấp phát bộ nhớ cho các tầng nơ-ron (Allocate Tensors)
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.print("{\"status\":\"error\",\"message\":\"AllocateTensors() failed! Arena=");
    Serial.print(kTensorArenaSize / 1024);
    Serial.print("KB, Model=");
    Serial.print(g_model_data_len / 1024);
    Serial.println("KB\"}");
    lcd_show("LOI ALLOCATE!", "Arena tran RAM");
    return;
  }

  // 6. Lấy con trỏ Input và Output Tensor
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  // Đánh dấu setup thành công & hiển thị sẵn sàng lên LCD
  setup_ok = true;
  lcd_show("TINYML EDGE AI", "ESP32-S3 READY");
  buzzer_beep(2, 50);

  // In thông báo sẵn sàng ra Serial dạng JSON chuẩn
  Serial.println();
  Serial.print("{\"status\":\"ready\",\"chip\":\"ESP32-S3\",\"model\":\"TinyFaceNet-Ghost-INT8\"");
  Serial.print(",\"model_size_kb\":");
  Serial.print(g_model_data_len / 1024.0, 2);
  Serial.print(",\"arena_size_kb\":");
  Serial.print(kTensorArenaSize / 1024);
  Serial.print(",\"arena_used_kb\":");
  Serial.print(interpreter->arena_used_bytes() / 1024);
  Serial.print(",\"registered_faces\":");
  Serial.print(NUM_REGISTERED_FACES);
  Serial.print(",\"psram_size_kb\":");
  Serial.print(ESP.getPsramSize() / 1024);
  Serial.print(",\"free_heap_kb\":");
  Serial.print(ESP.getFreeHeap() / 1024);
  Serial.println("}");
  
  // Khởi tạo FreeRTOS Queue (độ dài 1 để nhận khung hình mới nhất)
  imageQueue = xQueueCreate(1, EXPECTED_PAYLOAD_SIZE);
  if (imageQueue == NULL) {
    Serial.println("{\"status\":\"error\",\"message\":\"Failed to create FreeRTOS Queue\"}");
    return;
  }

  // Khởi tạo FreeRTOS Task cho Inference để tách biệt khỏi Serial loop
  // Chạy trên Core 0 (Pro CPU) để dành Core 1 (App CPU) cho Serial và WiFi
  xTaskCreatePinnedToCore(InferenceTask, "InferenceTask", 32768, NULL, 1, NULL, 0);
}

// ==============================================================================
// 6. HÀM THỰC THI SUY LUẬN AI & SO KHỚP COSINE (INFERENCE & MATCHING)
// ==============================================================================
void processInference(uint8_t* raw_image_bytes, uint32_t length) {
  last_inference_time = millis();
  is_idle_displayed = false;

  if (length != EXPECTED_PAYLOAD_SIZE) {
    Serial.print("{\"status\":\"error\",\"message\":\"Invalid image length: ");
    Serial.print(length);
    Serial.println(", expected 4096 bytes\"}");
    return;
  }

  if (!interpreter || !model_input || !model_output) {
    Serial.println("{\"status\":\"error\",\"message\":\"Interpreter or tensors not initialized!\"}");
    return;
  }

  // 1. Nạp và lượng tử hóa ảnh vào Input Tensor (Grayscale 64x64 -> INT8)
  float in_scale = model_input->params.scale;
  int32_t in_zero_point = model_input->params.zero_point;

  for (int i = 0; i < EXPECTED_PAYLOAD_SIZE; i++) {
    // Chuẩn hóa pixel uint8 [0..255] về dải Float [-1.0, 1.0]
    float norm_float = ((float)raw_image_bytes[i] - 127.5f) / 128.0f;
    // Quantize sang INT8 [-128..127] theo đúng Scale & ZeroPoint của mô hình
    int32_t q_val = (int32_t)round(norm_float / in_scale) + in_zero_point;
    if (q_val < -128) q_val = -128;
    if (q_val > 127) q_val = 127;
    model_input->data.int8[i] = (int8_t)q_val;
  }

  // 2. Bấm giờ và thực thi suy luận mạng nơ-ron
  unsigned long t_start = micros();
  TfLiteStatus invoke_status = interpreter->Invoke();
  unsigned long t_duration = micros() - t_start;
  float inference_ms = t_duration / 1000.0f;

  if (invoke_status != kTfLiteOk) {
    Serial.println("{\"status\":\"error\",\"message\":\"interpreter->Invoke() failed!\"}");
    return;
  }

  // 3. Trích xuất Output Vector 128D, Dequantize và Chuẩn hóa L2
  float out_scale = model_output->params.scale;
  int32_t out_zero_point = model_output->params.zero_point;
  float live_embedding[EMBEDDING_DIM];
  float norm_sum = 0.0f;

  for (int i = 0; i < EMBEDDING_DIM; i++) {
    float val = ((float)model_output->data.int8[i] - (float)out_zero_point) * out_scale;
    live_embedding[i] = val;
    norm_sum += val * val;
  }

  // Chuẩn hóa L2 về bán kính đơn vị 1.0
  float norm = sqrt(norm_sum) + 1e-7f;
  for (int i = 0; i < EMBEDDING_DIM; i++) {
    live_embedding[i] /= norm;
  }

  // 4. So khớp khoảng cách Cosine với CSDL (face_database.h)
  int best_id = -1;
  const char* best_name = "Unknown";
  float best_similarity = -1.0f;

  for (int k = 0; k < NUM_REGISTERED_FACES; k++) {
    float sim = 0.0f;
    for (int i = 0; i < EMBEDDING_DIM; i++) {
      sim += live_embedding[i] * FACE_DATABASE[k].embedding[i];
    }
    if (sim > best_similarity) {
      best_similarity = sim;
      best_id = FACE_DATABASE[k].id;
      best_name = FACE_DATABASE[k].name;
    }
  }

  // 5. Kiểm tra với ngưỡng xác thực MATCH_THRESHOLD
  bool is_matched = (best_similarity >= MATCH_THRESHOLD);

  // 6. CẬP NHẬT HIỂN THỊ LÊN MÀN HÌNH LCD1602 VÀ PHÁT ÂM THANH
  if (is_matched) {
    // Trích xuất tên ngắn gọn hiển thị vừa khít dòng 1 (16 ký tự)
    char clean_name[12];
    strncpy(clean_name, best_name, sizeof(clean_name) - 1);
    clean_name[sizeof(clean_name) - 1] = '\0';
    for (int i = 0; clean_name[i]; i++) {
      if (clean_name[i] == '_') clean_name[i] = ' ';
    }

    char line1_buf[17];
    snprintf(line1_buf, sizeof(line1_buf), "%-9s %.0f%%", clean_name, best_similarity * 100.0f);
    
    char line2_buf[17];
    snprintf(line2_buf, sizeof(line2_buf), "DIEM DANH XONG!");

    lcd_show(line1_buf, line2_buf);
    buzzer_beep(2, 60); // Bíp 2 tiếng xác nhận thành công
  } else {
    char line2_buf[17];
    snprintf(line2_buf, sizeof(line2_buf), "Tu choi! (%.0f%%)", best_similarity > 0 ? best_similarity * 100.0f : 0.0f);
    lcd_show("CANH BAO NGUOI LA", line2_buf);
    buzzer_beep(1, 150); // Bíp 1 tiếng dài cảnh báo
  }

  // 7. Phản hồi kết quả nhận diện về Laptop qua Serial dạng JSON
  Serial.print("{\"status\":\"success\",\"cmd\":\"INFERENCE\",\"matched\":");
  Serial.print(is_matched ? "true" : "false");
  Serial.print(",\"name\":\"");
  Serial.print(is_matched ? best_name : "Unknown");
  Serial.print("\",\"id\":");
  Serial.print(is_matched ? best_id : 0);
  Serial.print(",\"similarity\":");
  Serial.print(best_similarity, 4);
  Serial.print(",\"inference_ms\":");
  Serial.print(inference_ms, 2);
  Serial.print(",\"free_psram_kb\":");
  Serial.print(ESP.getFreePsram() / 1024);
  Serial.println("}");
}

// ==============================================================================
// 7. VÒNG LẶP ĐỌC GÓI TIN SERIAL KHÔNG CHẶN (LOOP)
// ==============================================================================
void loop() {
  // Tự động chuyển LCD về trạng thái chờ khi không có ai đứng trước camera sau 3.5s
  if (setup_ok && !is_idle_displayed && (millis() - last_inference_time > 3500)) {
    lcd_show("DIEM DANH TU DONG", "Dua mat vao cam ");
    is_idle_displayed = true;
  }

  // Timeout bảo vệ nếu mất kết nối dở chừng
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
        } else if (incomingByte != HEADER_1) {
          currentState = WAIT_H1;
        }
        break;

      case READ_CMD:
        currentCmd = incomingByte;
        payloadLength = 0;
        lenBytesRead = 0;
        currentState = READ_LEN;
        break;

      case READ_LEN:
        payloadLength |= ((uint32_t)incomingByte << (8 * (3 - lenBytesRead)));
        lenBytesRead++;
        if (lenBytesRead == 4) {
          if (payloadLength > MAX_PAYLOAD_SIZE) {
            Serial.println("{\"status\":\"error\",\"message\":\"Payload exceeds MAX_PAYLOAD_SIZE\"}");
            currentState = WAIT_H1;
          } else if (payloadLength == 0) {
            currentState = READ_CHECKSUM;
            calculatedChecksum = 0;
          } else {
            bytesRead = 0;
            calculatedChecksum = 0;
            currentState = READ_PAYLOAD;
          }
        }
        break;

      case READ_PAYLOAD:
        payloadBuffer[bytesRead] = incomingByte;
        calculatedChecksum ^= incomingByte;
        bytesRead++;
        if (bytesRead >= payloadLength) {
          currentState = READ_CHECKSUM;
        }
        break;

      case READ_CHECKSUM:
        if (incomingByte == calculatedChecksum) {
          // XỬ LÝ LỆNH HỢP LỆ
          if (currentCmd == CMD_PING) {
            Serial.println("{\"status\":\"success\",\"cmd\":\"PING\",\"message\":\"PONG\"}");
          } else if (currentCmd == CMD_INFO) {
            Serial.print("{\"status\":\"success\",\"cmd\":\"INFO\",\"chip\":\"ESP32-S3\",\"psram_size_kb\":");
            Serial.print(ESP.getPsramSize() / 1024);
            Serial.print(",\"free_psram_kb\":");
            Serial.print(ESP.getFreePsram() / 1024);
            Serial.print(",\"free_heap_kb\":");
            Serial.print(ESP.getFreeHeap() / 1024);
            Serial.println("}");
          } else if (currentCmd == CMD_INFERENCE) {
            // Bảo vệ: Chỉ xử lý nếu setup thành công (tránh crash khi imageQueue == NULL)
            if (setup_ok && imageQueue != NULL) {
              xQueueOverwrite(imageQueue, payloadBuffer);
            } else {
              Serial.println("{\"status\":\"error\",\"message\":\"AllocateTensors() failed!\"}");
            }
          } else {
            Serial.print("{\"status\":\"error\",\"message\":\"Unknown command 0x");
            Serial.print(currentCmd, HEX);
            Serial.println("\"}");
          }
        } else {
          Serial.print("{\"status\":\"error\",\"message\":\"Checksum error (expected 0x");
          Serial.print(calculatedChecksum, HEX);
          Serial.print(", got 0x");
          Serial.print(incomingByte, HEX);
          Serial.println(")\"}");
        }
        currentState = WAIT_H1;
        break;
    }
  }
}
