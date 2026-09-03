#include "ai_face_recognizer.h"
#include <Arduino.h>
#include <math.h>

// TFLite Micro Headers
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "esp_nn_glue.h"
#include "model_data.h"
#include "face_database.h"
#include "ai_config.h"

// Biến toàn cục cho TFLite
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
// Kích thước Arena được quản lý bởi ai_config.h  
uint8_t* tensor_arena = nullptr;

const int EMBEDDING_SIZE = 128;

#include "tensorflow/lite/micro/micro_error_reporter.h"

void setup_face_recognizer() {
    Serial.println("Khoi tao TFLite Micro Face Recognizer...");
    // [4.1 REALTIME] Arena PSRAM lớn (1.75MB) vì esp-nn cần scratch buffer = bản sao
    // filter đã align (~bằng tổng weights model) ngoài 187KB activations.
    // SRAM nội 512KB không đủ chứa arena này — PSRAM 8MB dư nhiều.
    uint8_t* raw_arena = (uint8_t*)heap_caps_malloc(RECOGNIZER_ARENA_SIZE + 16, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

    if (!raw_arena) {
        Serial.println("FATAL: Failed to allocate tensor arena!");
        return;
    }
    
    // Căn lề 16 bytes
    tensor_arena = (uint8_t*)(((uintptr_t)raw_arena + 15) & ~15);
    model = tflite::GetModel(g_model_recognizer);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model version mismatch!");
        return;
    }

    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;

    // Resolver theo audit model thật (Recognizer ops: CONV_2D, DEPTHWISE_CONV_2D, ADD,
    // CONCATENATION, FULLY_CONNECTED — không có RESHAPE; ReLU/BatchNorm đã fused).
    static tflite::MicroMutableOpResolver<5> resolver;
#if AI_ESP_NN_CONV_REC
    // [4.1 REALTIME] CONV_2D + DEPTHWISE_CONV_2D chạy kernel SIMD esp-nn
    resolver.AddConv2D(ai_esp_nn::Register_CONV_2D_ESPNN());
    resolver.AddDepthwiseConv2D(ai_esp_nn::Register_DEPTHWISE_CONV_2D_ESPNN());
#else
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
#endif
    resolver.AddAdd();
    resolver.AddConcatenation();
    resolver.AddFullyConnected();
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, RECOGNIZER_ARENA_SIZE, error_reporter);
    
    interpreter = &static_interpreter;
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[Recognizer] AllocateTensors() FAILED");
        Serial.printf("[Recognizer] Need | Arena: %d / %d\n", interpreter->arena_used_bytes(), RECOGNIZER_ARENA_SIZE);
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
    Serial.printf("[Recognizer] Arena used %d / %d\n", interpreter->arena_used_bytes(), RECOGNIZER_ARENA_SIZE);
    Serial.println(">>> TFLite Micro Model da nap thanh cong vao PSRAM!");
#if AI_ESP_NN_CONV_REC
    Serial.println(">>> ESP-NN SIMD (recognizer): BAT — realtime mode");
#else
    Serial.println(">>> ESP-NN SIMD (recognizer): TAT (kernel stock)");
#endif
#if (AI_ESP_NN_CONV_DET || AI_ESP_NN_CONV_REC) && RUN_ESPNN_SELFTEST
    // [CHẨN ĐOÁN] Self-test esp-nn vs reference — phải PASS (maxdiff=0)
    ai_esp_nn::EspNnSelfTest();
#endif
}

void extract_face_embedding(const float* input_tensor, float* output_embedding) {
    
    // Copy ảnh vào input tensor với Quantization
    if (input->type == kTfLiteInt8) {
        float input_scale = input->params.scale;
        int input_zero_point = input->params.zero_point;
        
        for (int i = 0; i < 64 * 64; i++) {
            // Chuyển đổi từ float [-1.0, 1.0] sang int8
            int val = round(input_tensor[i] / input_scale) + input_zero_point;
            // Kẹp giới hạn (Clamp)
            if (val > 127) val = 127;
            if (val < -128) val = -128;
            input->data.int8[i] = (int8_t)val;
        }
    } else if (input->type == kTfLiteUInt8) {
        float input_scale = input->params.scale;
        int input_zero_point = input->params.zero_point;
        
        for (int i = 0; i < 64 * 64; i++) {
            int val = round(input_tensor[i] / input_scale) + input_zero_point;
            if (val > 255) val = 255;
            if (val < 0) val = 0;
            input->data.uint8[i] = (uint8_t)val;
        }
    } else {
        // Fallback cho Float32
        for (int i = 0; i < 64 * 64; i++) {
            input->data.f[i] = input_tensor[i];
        }
    }
    
    // Chạy suy luận
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Invoke failed!");
        return;
    }
    
    // Lấy kết quả 128-D với Dequantization
    if (output->type == kTfLiteInt8) {
        float output_scale = output->params.scale;
        int output_zero_point = output->params.zero_point;
        
        for (int i = 0; i < EMBEDDING_SIZE; i++) {
            // Chuyển đổi từ int8 về float
            output_embedding[i] = (output->data.int8[i] - output_zero_point) * output_scale;
        }
    } else if (output->type == kTfLiteUInt8) {
        float output_scale = output->params.scale;
        int output_zero_point = output->params.zero_point;
        
        for (int i = 0; i < EMBEDDING_SIZE; i++) {
            output_embedding[i] = (output->data.uint8[i] - output_zero_point) * output_scale;
        }
    } else {
        for (int i = 0; i < EMBEDDING_SIZE; i++) {
            output_embedding[i] = output->data.f[i];
        }
    }
}

// Tính Cosine Similarity
float cosine_similarity(const float* a, const float* b, int size) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < size; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a == 0 || norm_b == 0) return 0.0f;
    return dot / (sqrt(norm_a) * sqrt(norm_b));
}

const char* identify_face(const float* face_embedding, float threshold) {
    // Multi-template MAX-SIM (đồng bộ Laptop face_recognizer.py):
    // 1) Tìm template có điểm cao nhất qua TẤT CẢ người (argmax trước).
    // 2) Áp ngưỡng per-identity (4.2, Verheyen ARES 2023): người dễ nhầm với người khác
    //    có ngưỡng riêng cao hơn (được generate_embeddings.py tính và lưu trong struct),
    //    hiệu lực = max(ngưỡng global 0.60, ngưỡng riêng) → FAR ổn định khi DB dày lên.
    int best_i = -1;
    float best_score = -1.0f;

    for (int i = 0; i < NUM_REGISTERED_FACES; i++) {
        int n = FACE_DATABASE[i].num_templates;
        if (n <= 0) n = 1; // an toàn cho DB cũ centroid
        for (int t = 0; t < n; t++) {
            float score = cosine_similarity(face_embedding, FACE_DATABASE[i].embeddings[t], EMBEDDING_SIZE);
            if (score > best_score) {
                best_score = score;
                best_i = i;
            }
        }
    }

    if (best_i >= 0) {
        float eff_threshold = threshold;
        if (FACE_DATABASE[best_i].threshold > eff_threshold) {
            eff_threshold = FACE_DATABASE[best_i].threshold;
        }
        // Log chi tiết để chẩn đoán ngưỡng thấp: luôn in best_match dù Unknown
        if (best_score >= eff_threshold) {
            Serial.printf("🔍 [AI] Frame hiện tại: %s (Độ tin cậy: %.2f, ngưỡng %.2f)\n",
                          FACE_DATABASE[best_i].name, best_score, eff_threshold);
            return FACE_DATABASE[best_i].name;
        } else {
            Serial.printf("🔍 [AI] Frame hiện tại: Unknown (best %s %.2f < ngưỡng %.2f)\n",
                          FACE_DATABASE[best_i].name, best_score, eff_threshold);
            return "Unknown";
        }
    }

    Serial.printf("🔍 [AI] Frame hiện tại: Unknown (Độ tin cậy: %.2f)\n", best_score);
    return "Unknown";
}
