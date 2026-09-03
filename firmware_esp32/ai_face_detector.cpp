#include "ai_face_detector.h"
#include <Arduino.h>
#include <math.h>
#include <vector>
#include "ai_config.h"
#include "esp_nn_glue.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "detector_model_data.h"

uint16_t* g_frame_buffer = nullptr;
// [PERF] 4.1 — thời gian từng chặng preprocess (micros), cập nhật mỗi lần gọi
unsigned long g_us_bilinear = 0;
unsigned long g_us_he = 0;

const tflite::Model* detector_model = nullptr;
tflite::MicroInterpreter* detector_interpreter = nullptr;
TfLiteTensor* detector_input = nullptr;
TfLiteTensor* detector_output_scores = nullptr;
TfLiteTensor* detector_output_boxes = nullptr;
uint8_t* g_detector_tensor_arena = nullptr;
std::vector<std::pair<float,float>> g_anchors;
static float g_ema_cx = -1.0f;
static float g_ema_cy = -1.0f;
static float g_ema_w = -1.0f;
static float g_ema_h = -1.0f;
static constexpr float kEmaAlpha = 0.35f;

static float sigmoid_f(float x){
    x = max(-80.0f, min(80.0f, x));
    return 1.0f / (1.0f + expf(-x));
}

static float tensor_value(const TfLiteTensor* tensor, int index){
    switch(tensor->type){
        case kTfLiteFloat32:
            return tensor->data.f[index];
        case kTfLiteInt8:
            return (tensor->data.int8[index] - tensor->params.zero_point) * tensor->params.scale;
        case kTfLiteUInt8:
            return (tensor->data.uint8[index] - tensor->params.zero_point) * tensor->params.scale;
        default:
            return NAN;
    }
}

static void reset_ema(){
    g_ema_cx = g_ema_cy = g_ema_w = g_ema_h = -1.0f;
}

void setup_face_detector(){
    Serial.println("Khoi tao BlazeFace 128...");
    g_frame_buffer = (uint16_t*)heap_caps_malloc(FRAME_BUFFER_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if(!g_frame_buffer){ Serial.println("FATAL: Frame buffer PSRAM fail"); return; }
    Serial.println("Frame buffer 32KB PSRAM OK");

    uint8_t* raw_arena = (uint8_t*)heap_caps_malloc(DETECTOR_ARENA_SIZE + 16, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if(!raw_arena){ Serial.println("FATAL: Detector arena PSRAM fail"); return; }
    g_detector_tensor_arena = (uint8_t*)(((uintptr_t)raw_arena + 15) & ~15);
    detector_model = tflite::GetModel(g_detector_model);
    if(detector_model->version()!=TFLITE_SCHEMA_VERSION){ Serial.println("Model version mismatch"); return; }
    // Resolver đăng ký ĐỦ ops theo audit model thật (quantize_detector_int8.py):
    // ADD, CONCATENATION, CONV_2D, DEPTHWISE_CONV_2D, MAX_POOL_2D, PAD, RESHAPE.
    // Model BlazeFace hiện là FULL INT8 (ReLU đã fused, không còn DEQUANTIZE).
    static tflite::MicroErrorReporter reporter;
    static tflite::MicroMutableOpResolver<7> resolver;
#if AI_ESP_NN_CONV_DET
    // [4.1 REALTIME] CONV_2D + DEPTHWISE_CONV_2D chạy kernel SIMD esp-nn (Xtensa LX7)
    resolver.AddConv2D(ai_esp_nn::Register_CONV_2D_ESPNN());
    resolver.AddDepthwiseConv2D(ai_esp_nn::Register_DEPTHWISE_CONV_2D_ESPNN());
#else
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
#endif
    resolver.AddAdd();
    resolver.AddConcatenation();
    resolver.AddMaxPool2D();
    resolver.AddPad();
    resolver.AddReshape();
    static tflite::MicroInterpreter interp(detector_model, resolver, g_detector_tensor_arena, DETECTOR_ARENA_SIZE, &reporter);
    detector_interpreter = &interp;
    if(detector_interpreter->AllocateTensors()!=kTfLiteOk){
        Serial.println("[Detector] AllocateTensors FAILED");
    Serial.printf("[Detector] Arena used %d / %d\n", detector_interpreter->arena_used_bytes(), DETECTOR_ARENA_SIZE);
#if AI_ESP_NN_CONV_DET
    Serial.println("[Detector] ESP-NN SIMD: BAT");
#else
    Serial.println("[Detector] ESP-NN SIMD: TAT (kernel stock)");
#endif
        return;
    }
    detector_input = detector_interpreter->input(0);
    // TFLite có thể xếp regressors trước classificators hoặc ngược lại.
    TfLiteTensor* output0 = detector_interpreter->output(0);
    TfLiteTensor* output1 = detector_interpreter->output(1);
    int output0_last_dim = output0->dims->data[output0->dims->size - 1];
    if(output0_last_dim == 1){
        detector_output_scores = output0;
        detector_output_boxes = output1;
    } else {
        detector_output_scores = output1;
        detector_output_boxes = output0;
    }
    Serial.printf("[Detector] Arena used %d / %d\n", detector_interpreter->arena_used_bytes(), DETECTOR_ARENA_SIZE);

    // 896 anchors 16x16*2 + 8x8*6
    g_anchors.clear();
    for(int y=0;y<16;++y) for(int x=0;x<16;++x){ float cx=(x+0.5f)/16, cy=(y+0.5f)/16; g_anchors.push_back({cx,cy}); g_anchors.push_back({cx,cy}); }
    for(int y=0;y<8;++y) for(int x=0;x<8;++x){ float cx=(x+0.5f)/8, cy=(y+0.5f)/8; for(int i=0;i<6;++i) g_anchors.push_back({cx,cy}); }
}

FaceBox detect_face(){
    FaceBox box{};
    box.is_valid = false;
    if(!g_frame_buffer || !detector_interpreter || !detector_input ||
       !detector_output_scores || !detector_output_boxes) return box;

    // Chuẩn bị input RAW 128x128 RGB, normalize [(pixel - 127.5) / 128].
    // Với model quantized, lượng tử hóa tensor đã normalize giống emulator Python.
    const float input_scale = detector_input->params.scale;
    const int input_zero_point = detector_input->params.zero_point;
    for(int y = 0; y < RAW_FRAME_SIZE; ++y){
        for(int x = 0; x < RAW_FRAME_SIZE; ++x){
            uint16_t c = g_frame_buffer[y * RAW_FRAME_SIZE + x];
            uint8_t r = ((c>>11)&0x1F)<<3;
            uint8_t g = ((c>>5)&0x3F)<<2;
            uint8_t b = (c&0x1F)<<3;
            int idx = (y * RAW_FRAME_SIZE + x) * 3;
            float rf = (r - 127.5f)/128.0f;
            float gf = (g - 127.5f)/128.0f;
            float bf = (b - 127.5f)/128.0f;
            if(detector_input->type == kTfLiteFloat32){
                detector_input->data.f[idx + 0] = rf;
                detector_input->data.f[idx + 1] = gf;
                detector_input->data.f[idx + 2] = bf;
            } else if(detector_input->type == kTfLiteInt8 && input_scale > 0.0f){
                detector_input->data.int8[idx + 0] = (int8_t)max(-128, min(127, (int)roundf(rf / input_scale + input_zero_point)));
                detector_input->data.int8[idx + 1] = (int8_t)max(-128, min(127, (int)roundf(gf / input_scale + input_zero_point)));
                detector_input->data.int8[idx + 2] = (int8_t)max(-128, min(127, (int)roundf(bf / input_scale + input_zero_point)));
            } else if(detector_input->type == kTfLiteUInt8 && input_scale > 0.0f){
                detector_input->data.uint8[idx + 0] = (uint8_t)max(0, min(255, (int)roundf(rf / input_scale + input_zero_point)));
                detector_input->data.uint8[idx + 1] = (uint8_t)max(0, min(255, (int)roundf(gf / input_scale + input_zero_point)));
                detector_input->data.uint8[idx + 2] = (uint8_t)max(0, min(255, (int)roundf(bf / input_scale + input_zero_point)));
            } else {
                return box;
            }
        }
    }
    if(detector_interpreter->Invoke()!=kTfLiteOk){ return box; }

    // Decode score cao nhất. Regressor có stride 16, nhưng chỉ 4 giá trị đầu
    // là (x_center, y_center, width, height) theo pixel BlazeFace.
    float best_score = -1.0f;
    int best_idx = -1;
    for(int i = 0; i < (int)g_anchors.size(); ++i){
        float score = sigmoid_f(tensor_value(detector_output_scores, i));
        if(score > best_score){
            best_score = score;
            best_idx = i;
        }
    }
    if(best_score < DETECTOR_CONF_THRESH || best_idx < 0){
        reset_ema();
        return box;
    }

    const int box_stride = detector_output_boxes->dims->data[detector_output_boxes->dims->size - 1];
    if(box_stride < 4){
        reset_ema();
        return box;
    }
    float dx = tensor_value(detector_output_boxes, best_idx * box_stride + 0);
    float dy = tensor_value(detector_output_boxes, best_idx * box_stride + 1);
    float dw = tensor_value(detector_output_boxes, best_idx * box_stride + 2);
    float dh = tensor_value(detector_output_boxes, best_idx * box_stride + 3);
    float cx = dx + g_anchors[best_idx].first * RAW_FRAME_SIZE;
    float cy = dy + g_anchors[best_idx].second * RAW_FRAME_SIZE;
    if(!isfinite(cx) || !isfinite(cy) || !isfinite(dw) || !isfinite(dh) || dw <= 0.0f || dh <= 0.0f){
        reset_ema();
        return box;
    }
    if(cx + dw / 2.0f <= 0.0f || cy + dh / 2.0f <= 0.0f ||
       cx - dw / 2.0f >= RAW_FRAME_SIZE || cy - dh / 2.0f >= RAW_FRAME_SIZE){
        reset_ema();
        return box;
    }

    if(g_ema_cx < 0.0f){
        g_ema_cx = cx; g_ema_cy = cy; g_ema_w = dw; g_ema_h = dh;
    } else {
        g_ema_cx = kEmaAlpha * cx + (1.0f - kEmaAlpha) * g_ema_cx;
        g_ema_cy = kEmaAlpha * cy + (1.0f - kEmaAlpha) * g_ema_cy;
        g_ema_w = kEmaAlpha * dw + (1.0f - kEmaAlpha) * g_ema_w;
        g_ema_h = kEmaAlpha * dh + (1.0f - kEmaAlpha) * g_ema_h;
    }

    const float crop_size = max(g_ema_w, g_ema_h);
    const float left = max(0.0f, g_ema_cx - crop_size / 2.0f);
    const float top = max(0.0f, g_ema_cy - crop_size / 2.0f);
    const float right = min((float)RAW_FRAME_SIZE, g_ema_cx + crop_size / 2.0f);
    const float bottom = min((float)RAW_FRAME_SIZE, g_ema_cy + crop_size / 2.0f);
    if(right <= left || bottom <= top){
        reset_ema();
        return box;
    }
    box.x = (int)left;
    box.y = (int)top;
    box.width = (int)right - box.x;
    box.height = (int)bottom - box.y;
    if(box.width <= 0 || box.height <= 0){
        reset_ema();
        return box;
    }
    box.center_x = g_ema_cx;
    box.center_y = g_ema_cy;
    box.crop_size = crop_size;
    box.score = best_score;
    box.is_valid = true;
    return box;
}

// Histogram Equalization LUT — ĐỒNG BỘ bit-exact với equalize_gray_256()
// trong host_laptop/core/vision_utils.py (cùng công thức số nguyên floor).
// Tham khảo: Shan et al., "Illumination normalization for robust face recognition
// against varying lighting conditions", AMFG 2003 — HE cải thiện độ chính xác
// nhận diện khi ánh sáng thay đổi giữa lúc enroll và lúc chấm công.
static void equalize_gray_u8(const uint8_t* src, uint8_t* dst, int n){
    int hist[256] = {0};
    for(int i = 0; i < n; ++i) hist[src[i]]++;
    int cdf[256];
    int run = 0;
    for(int i = 0; i < 256; ++i){ run += hist[i]; cdf[i] = run; }
    int v0 = 0;
    while(v0 < 256 && hist[v0] == 0) ++v0;
    if(v0 >= 256){ memcpy(dst, src, n); return; }
    int cdf_min = cdf[v0];
    int den = n - cdf_min;
    uint8_t lut[256];
    for(int v = 0; v < 256; ++v){
        if(den <= 0){ lut[v] = (uint8_t)v; continue; }       // ảnh đơn điệu → identity
        int num = cdf[v] - cdf_min;
        int e = (num <= 0) ? 0 : (int)(((int64_t)num * 255) / den);
        if(e > 255) e = 255;
        lut[v] = (uint8_t)e;
    }
    for(int i = 0; i < n; ++i) dst[i] = lut[src[i]];
}

bool preprocess_face(const FaceBox& box, float* out_tensor){
    if(!box.is_valid || !g_frame_buffer || !out_tensor ||
       !isfinite(box.center_x) || !isfinite(box.center_y) ||
       !isfinite(box.crop_size) || box.crop_size <= 0.0f) return false;
    const float cx = box.center_x;
    const float cy = box.center_y;
    const float box_size = box.crop_size * 1.1f; // mở rộng 10%
    float half = box_size/2;
    float x1_box = cx - half;
    float y1_box = cy - half;
    float scale = box_size / FACE_TARGET_SIZE;
    // Bilinear thủ công đồng bộ với host_laptop/core/vision_utils.py
    // Bước 1: crop + grayscale uint8 (truncated — khớp int() của Python)
    uint8_t gray_u8[FACE_TARGET_SIZE * FACE_TARGET_SIZE];
    unsigned long t_perf = micros();
    for(int ty=0; ty<FACE_TARGET_SIZE; ++ty){
        for(int tx=0; tx<FACE_TARGET_SIZE; ++tx){
            float sx = x1_box + (tx + 0.5f)*scale - 0.5f;
            float sy = y1_box + (ty + 0.5f)*scale - 0.5f;
            // Clamp tọa độ thực trước khi tính x1/y1. Nếu clamp sau đó,
            // box vượt biên sẽ khiến x1/y1 âm giống lỗi host trước đây.
            sx = max(0.0f, min(sx, (float)RAW_FRAME_SIZE - 1.0f));
            sy = max(0.0f, min(sy, (float)RAW_FRAME_SIZE - 1.0f));
            int x0 = (int)floorf(sx);
            int y0 = (int)floorf(sy);
            int x1 = min(x0 + 1, RAW_FRAME_SIZE - 1);
            int y1 = min(y0 + 1, RAW_FRAME_SIZE - 1);
            float wx = sx - x0;
            float wy = sy - y0;
            // Lấy 4 điểm RGB565
            auto getGray = [&](int x,int y)->float{
                uint16_t c = g_frame_buffer[y * RAW_FRAME_SIZE + x];
                uint8_t r = ((c>>11)&0x1F)<<3;
                uint8_t g = ((c>>5)&0x3F)<<2;
                uint8_t b = (c&0x1F)<<3;
                return 0.299f*r + 0.587f*g + 0.114f*b;
            };
            float g00 = getGray(x0,y0);
            float g01 = getGray(x1,y0);
            float g10 = getGray(x0,y1);
            float g11 = getGray(x1,y1);
            float top = (1-wx)*g00 + wx*g01;
            float bot = (1-wx)*g10 + wx*g11;
            float gray = (1-wy)*top + wy*bot;
            // Truncate về uint8 như int(np.clip(...)) phía Python — đầu vào của LUT HE
            int gi = (int)max(0.0f, min(255.0f, gray));
            gray_u8[ty * FACE_TARGET_SIZE + tx] = (uint8_t)gi;
        }
    }
    g_us_bilinear = micros() - t_perf;
    // Bước 2: Histogram Equalization khử nhạy ánh sáng (đồng bộ Python)
    uint8_t gray_eq[FACE_TARGET_SIZE * FACE_TARGET_SIZE];
    t_perf = micros();
    equalize_gray_u8(gray_u8, gray_eq, FACE_TARGET_SIZE * FACE_TARGET_SIZE);
    g_us_he = micros() - t_perf;
    // Bước 3: chuẩn hóa (gray - 127.5)/128.0
    for(int i = 0; i < FACE_TARGET_SIZE * FACE_TARGET_SIZE; ++i){
        out_tensor[i] = ((float)gray_eq[i] - 127.5f)/128.0f;
    }
    // [CHẨN ĐOÁN 4.1] log thống kê crop để kiểm tra cắt ảnh: mean/var trước/sau HE
    // (giúp phát hiện crop lệch, tối/cháy, hoặc HE sai)
    {
        long sum = 0, sum2 = 0;
        for(int i=0;i<FACE_TARGET_SIZE*FACE_TARGET_SIZE;i++){ sum += gray_u8[i]; sum2 += gray_u8[i]*gray_u8[i]; }
        float mean = sum / 4096.0f;
        float var = sum2/4096.0f - mean*mean;
        long sum_eq=0, sum2_eq=0;
        for(int i=0;i<FACE_TARGET_SIZE*FACE_TARGET_SIZE;i++){ sum_eq += gray_eq[i]; sum2_eq += gray_eq[i]*gray_eq[i]; }
        float mean_eq = sum_eq/4096.0f;
        float var_eq = sum2_eq/4096.0f - mean_eq*mean_eq;
        Serial.printf("[CROP] mean=%.1f var=%.1f -> HE mean=%.1f var=%.1f box=(%.1f,%.1f) s=%.1f\n",
            mean, var, mean_eq, var_eq, box.center_x, box.center_y, box.crop_size);
    }
    return true;
}
