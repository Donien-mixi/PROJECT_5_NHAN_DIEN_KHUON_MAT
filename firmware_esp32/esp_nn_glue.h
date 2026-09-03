#ifndef ESP_NN_GLUE_H
#define ESP_NN_GLUE_H

#include "tensorflow/lite/c/common.h"

// =====================================================================
// [4.1 REALTIME] ESP-NN SIMD kernels cho CONV_2D + DEPTHWISE_CONV_2D
// (Espressif esp-nn, kernel assembly Xtensa LX7 cho ESP32-S3).
//
// HIỆU QUẢ ĐO THỰC TẾ (2025): det 20.3s -> 1.3s, rec 4.7s -> 0.21s.
//
// CÔNG TẮC BISECT (khi chẩn đoán điểm số):
//   1/1 = bật cả hai (mặc định, realtime)
//   0/1 = chỉ recognizer dùng esp-nn -> nếu điểm vẫn ~0.35 => lỗi nằm ở
//         recognizer; nếu điểm hồi 0.6-0.8 => lỗi nằm ở detector.
//   1/0 = chỉ detector dùng esp-nn (kiểm tra ngược lại)
//   0/0 = kernel stock TFLM (hành vi cũ, chậm nhưng đúng)
// Self-test số học lúc boot (EspNnSelfTest) in maxdiff esp-nn vs reference.
// =====================================================================
#define AI_ESP_NN_CONV_DET 1  // BẬT ESP-NN SIMD cho detector: thời gian dò mặt ~1.3s - 1.9s
#define AI_ESP_NN_CONV_REC 1  // BẬT ESP-NN SIMD cho recognizer: giảm từ 5000ms xuống ~200ms - 300ms (Realtime 5 FPS)
#define RUN_ESPNN_SELFTEST 1  // Tự động kiểm tra số học bit-exact lúc boot

#if AI_ESP_NN_CONV_DET || AI_ESP_NN_CONV_REC
namespace ai_esp_nn {

// Thay thế Register_CONV_2D() — truyền vào resolver.AddConv2D(...) vì
// MicroMutableOpResolver::AddConv2D/AddDepthwiseConv2D nhận custom registration
// (micro_mutable_op_resolver.h:191,211).
TfLiteRegistration Register_CONV_2D_ESPNN();
TfLiteRegistration Register_DEPTHWISE_CONV_2D_ESPNN();

// Self-test: so esp-nn conv/dwconv với reference integer ops trên buffer
// ngẫu nhiên — in "[ESPNN-TEST] conv maxdiff=X dwconv maxdiff=Y" (0 = bit-exact).
void EspNnSelfTest();

}  // namespace ai_esp_nn
#endif

#endif  // ESP_NN_GLUE_H
