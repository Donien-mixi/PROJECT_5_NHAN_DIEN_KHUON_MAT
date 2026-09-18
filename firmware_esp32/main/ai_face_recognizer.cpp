#include "ai_face_recognizer.h"
#include <math.h>
#include <string.h>
#include "esp_log.h"
#include "esp_heap_caps.h"

// TFLite Micro Headers
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"
#include "face_database.h"
#include "ai_config.h"

static const char* TAG = "FACE_RECOGNIZER";

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
uint8_t* tensor_arena = nullptr;

const int EMBEDDING_SIZE = 128;
float g_last_recognized_score = 0.0f;

void setup_face_recognizer() {
    ESP_LOGI(TAG, "Khởi tạo TFLite Micro Face Recognizer (Ghost-TinyFace V3 INT8)...");
    uint8_t* raw_arena = (uint8_t*)heap_caps_malloc(RECOGNIZER_ARENA_SIZE + 16, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!raw_arena) {
        ESP_LOGE(TAG, "FATAL: Failed to allocate tensor arena in PSRAM!");
        return;
    }
    
    tensor_arena = (uint8_t*)(((uintptr_t)raw_arena + 15) & ~15);
    model = tflite::GetModel(g_model_recognizer);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model version mismatch!");
        return;
    }

    // Resolver: CONV_2D, DEPTHWISE_CONV_2D (esp-nn SIMD), ADD, CONCATENATION, FULLY_CONNECTED
    static tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddAdd();
    resolver.AddConcatenation();
    resolver.AddFullyConnected();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, RECOGNIZER_ARENA_SIZE);
    
    interpreter = &static_interpreter;
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "[Recognizer] AllocateTensors() FAILED");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
    ESP_LOGI(TAG, "[Recognizer] Arena used %d / %d bytes (PSRAM, esp-nn SIMD: BẬT)",
             (int)interpreter->arena_used_bytes(), RECOGNIZER_ARENA_SIZE);
}

void extract_face_embedding(const float* input_tensor, float* output_embedding) {
    if (!interpreter || !input || !output || !input_tensor || !output_embedding) return;

    // Copy ảnh vào input tensor với Quantization
    if (input->type == kTfLiteInt8) {
        float input_scale = input->params.scale;
        int input_zero_point = input->params.zero_point;
        
        for (int i = 0; i < 64 * 64; i++) {
            int val = round(input_tensor[i] / input_scale) + input_zero_point;
            if (val > 127)  val = 127;
            if (val < -128) val = -128;
            input->data.int8[i] = (int8_t)val;
        }
    } else if (input->type == kTfLiteUInt8) {
        float input_scale = input->params.scale;
        int input_zero_point = input->params.zero_point;
        
        for (int i = 0; i < 64 * 64; i++) {
            int val = round(input_tensor[i] / input_scale) + input_zero_point;
            if (val > 255) val = 255;
            if (val < 0)   val = 0;
            input->data.uint8[i] = (uint8_t)val;
        }
    } else {
        // Fallback Float32
        for (int i = 0; i < 64 * 64; i++) {
            input->data.f[i] = input_tensor[i];
        }
    }
    
    // Chạy suy luận SIMD
    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed!");
        return;
    }
    
    // Lấy kết quả 128-D với Dequantization
    if (output->type == kTfLiteInt8) {
        float output_scale = output->params.scale;
        int output_zero_point = output->params.zero_point;
        
        for (int i = 0; i < EMBEDDING_SIZE; i++) {
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

float cosine_similarity(const float* a, const float* b, int size) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < size; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

const char* identify_face(const float* face_embedding, float threshold) {
    int best_i = -1;
    float best_score = -1.0f;

    for (int i = 0; i < NUM_REGISTERED_FACES; i++) {
        int n = FACE_DATABASE[i].num_templates;
        if (n <= 0) n = 1;
        for (int t = 0; t < n; t++) {
            float score = cosine_similarity(face_embedding, FACE_DATABASE[i].embeddings[t], EMBEDDING_SIZE);
            if (score > best_score) {
                best_score = score;
                best_i = i;
            }
        }
    }

    g_last_recognized_score = best_score;

    if (best_i >= 0) {
        float eff_threshold = threshold;
        if (FACE_DATABASE[best_i].threshold > eff_threshold) {
            eff_threshold = FACE_DATABASE[best_i].threshold;
        }
        if (best_score >= eff_threshold) {
            ESP_LOGI(TAG, "🔍 [AI] Frame hiện tại: %s (Độ tin cậy: %.2f, ngưỡng %.2f)",
                     FACE_DATABASE[best_i].name, best_score, eff_threshold);
            return FACE_DATABASE[best_i].name;
        } else {
            ESP_LOGI(TAG, "🔍 [AI] Frame hiện tại: Unknown (best %s %.2f < ngưỡng %.2f)",
                     FACE_DATABASE[best_i].name, best_score, eff_threshold);
            return "Unknown";
        }
    }

    ESP_LOGI(TAG, "🔍 [AI] Frame hiện tại: Unknown (Độ tin cậy: %.2f)", best_score);
    return "Unknown";
}
