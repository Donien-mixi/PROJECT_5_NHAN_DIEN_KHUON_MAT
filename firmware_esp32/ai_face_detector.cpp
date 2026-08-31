#include "ai_face_detector.h"
#include <Arduino.h>
#include <math.h>
#include <TJpg_Decoder.h>
#include "ai_config.h"

// Thêm TFLite headers
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "detector_model_data.h"

// Bộ đệm lưu trữ toàn bộ ảnh sau khi giải mã từ JPEG (240x240 RGB565)
// Kích thước: 240 * 240 * 2 = 115,200 bytes (~112KB)
uint16_t* g_frame_buffer = nullptr;
static const int FRAME_WIDTH = 240;
static const int FRAME_HEIGHT = 240;

// Variables cho Face Detector (BlazeFace 128x128)
const tflite::Model* detector_model = nullptr;
tflite::MicroInterpreter* detector_interpreter = nullptr;
TfLiteTensor* detector_input = nullptr;

// Mô hình Float32 Weight Quantized BlazeFace cần nhiều Arena hơn so với model UINT8.
// Kích thước Arena được quản lý động bởi ai_config.h từ AI Module.
uint8_t* g_detector_tensor_arena = nullptr;

struct Anchor { float x, y; };
std::vector<Anchor> g_anchors;

void setup_face_detector() {
    Serial.println("Khoi tao Face Preprocessing & Detector (BlazeFace)...");
    
    // Cấp phát Frame Buffer trên PSRAM
    g_frame_buffer = (uint16_t*)heap_caps_malloc(FRAME_WIDTH * FRAME_HEIGHT * 2, MALLOC_CAP_SPIRAM);
    if (!g_frame_buffer) {
        Serial.println("FATAL ERROR: Không thể cấp phát Frame Buffer trên PSRAM!");
    } else {
        Serial.println("Đã cấp phát 112KB PSRAM cho Frame Buffer.");
    }
    
    // Khởi tạo Tensor Arena cho Detector
    // Cấp phát trong bộ nhớ nội bộ (Internal SRAM) nếu có thể, nếu không đủ sẽ chuyển qua PSRAM.
    uint8_t* raw_arena = (uint8_t*)heap_caps_malloc(DETECTOR_ARENA_SIZE + 16, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!raw_arena) {
        Serial.println("⚠️ Cảnh báo: Không đủ Internal SRAM cho BlazeFace! Đang chuyển sang PSRAM (chậm hơn)...");
        raw_arena = (uint8_t*)heap_caps_malloc(DETECTOR_ARENA_SIZE + 16, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    }
    
    if (raw_arena) {
        g_detector_tensor_arena = (uint8_t*)(((uintptr_t)raw_arena + 15) & ~15);
        detector_model = tflite::GetModel(g_detector_model);
        
        static tflite::MicroErrorReporter micro_error_reporter;
        static tflite::AllOpsResolver resolver;
        static tflite::MicroInterpreter static_interpreter(
            detector_model, resolver, g_detector_tensor_arena, DETECTOR_ARENA_SIZE, &micro_error_reporter);
            
        detector_interpreter = &static_interpreter;
        if (detector_interpreter->AllocateTensors() == kTfLiteOk) {
            detector_input = detector_interpreter->input(0);
            Serial.println("Detector AllocateTensors() successful");
            Serial.printf("[BlazeFace] Arena size used: %d bytes\n", detector_interpreter->arena_used_bytes());
        } else {
            Serial.println("Detector AllocateTensors() failed");
        }
    }
    
    // Tạo 896 Anchors cho BlazeFace
    g_anchors.clear();
    // 16x16 feature map
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            float cx = (x + 0.5f) / 16.0f;
            float cy = (y + 0.5f) / 16.0f;
            g_anchors.push_back({cx, cy});
            g_anchors.push_back({cx, cy});
        }
    }
    // 8x8 feature map
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            float cx = (x + 0.5f) / 8.0f;
            float cy = (y + 0.5f) / 8.0f;
            for (int i = 0; i < 6; ++i) g_anchors.push_back({cx, cy});
        }
    }
}

// Callback của TJpgDec: Chép từng block ảnh đã giải mã vào Frame Buffer tổng
bool tjpg_decode_callback(int16_t x, int16_t y, uint16_t w, uint16_t h, uint16_t* bitmap) {
    if (!g_frame_buffer) return false;
    
    for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) {
            int px = x + i;
            int py = y + j;
            if (px < FRAME_WIDTH && py < FRAME_HEIGHT) {
                g_frame_buffer[py * FRAME_WIDTH + px] = bitmap[j * w + i];
            }
        }
    }
    return true;
}

// Hàm chạy suy luận TFLite để tìm khuôn mặt
FaceBox detect_face() {
    FaceBox box;
    box.is_valid = false;
    
    if (!g_frame_buffer) {
        Serial.println("[AI LỖI] Frame Buffer chưa sẵn sàng!");
        return box;
    }
    
    Serial.println("[AITask] Đã vào detect_face(). Chuẩn bị Bilinear Interpolation...");
    
    // Ảnh ĐÃ ĐƯỢC GIẢI MÃ BỞI NetTask và lưu vào g_frame_buffer.
    // Chúng ta không gọi TJpgDec.drawJpg ở đây nữa để tránh xung đột đa luồng!
    
    if (!detector_interpreter || !detector_input) {
        Serial.println("[AI LỖI] BlazeFace chưa được khởi tạo!");
        return box;
    }
    
    // Bước 2: Chuẩn bị input 128x128 cho BlazeFace bằng Bilinear (từ 240x240)
    const int TARGET_SIZE = 128;
    float scale = (float)FRAME_WIDTH / TARGET_SIZE;
    
    for (int y = 0; y < TARGET_SIZE; y++) {
        for (int x = 0; x < TARGET_SIZE; x++) {
            float src_x = x * scale;
            float src_y = y * scale;
            int x1 = (int)src_x;
            int y1 = (int)src_y;
            int x2 = min(x1 + 1, FRAME_WIDTH - 1);
            int y2 = min(y1 + 1, FRAME_HEIGHT - 1);
            float wx = src_x - x1;
            float wy = src_y - y1;
            
            auto get_rgb = [](uint16_t c, uint8_t& r, uint8_t& g, uint8_t& b) {
                r = (c >> 11) << 3;
                g = ((c >> 5) & 0x3F) << 2;
                b = (c & 0x1F) << 3;
            };
            
            uint8_t r11, g11, b11, r21, g21, b21, r12, g12, b12, r22, g22, b22;
            get_rgb(g_frame_buffer[y1 * FRAME_WIDTH + x1], r11, g11, b11);
            get_rgb(g_frame_buffer[y1 * FRAME_WIDTH + x2], r21, g21, b21);
            get_rgb(g_frame_buffer[y2 * FRAME_WIDTH + x1], r12, g12, b12);
            get_rgb(g_frame_buffer[y2 * FRAME_WIDTH + x2], r22, g22, b22);
            
            float rf = r11*(1-wx)*(1-wy) + r21*wx*(1-wy) + r12*(1-wx)*wy + r22*wx*wy;
            float gf = g11*(1-wx)*(1-wy) + g21*wx*(1-wy) + g12*(1-wx)*wy + g22*wx*wy;
            float bf = b11*(1-wx)*(1-wy) + b21*wx*(1-wy) + b12*(1-wx)*wy + b22*wx*wy;
            
            // Normalize [-1.0, 1.0]
            int idx = (y * TARGET_SIZE + x) * 3;
            if (detector_input->type == kTfLiteFloat32) {
                detector_input->data.f[idx + 0] = (rf - 127.5f) / 128.0f;
                detector_input->data.f[idx + 1] = (gf - 127.5f) / 128.0f;
                detector_input->data.f[idx + 2] = (bf - 127.5f) / 128.0f;
            } else if (detector_input->type == kTfLiteInt8) {
                detector_input->data.int8[idx + 0] = (int8_t)(((rf - 127.5f) / 128.0f) / detector_input->params.scale + detector_input->params.zero_point);
                detector_input->data.int8[idx + 1] = (int8_t)(((gf - 127.5f) / 128.0f) / detector_input->params.scale + detector_input->params.zero_point);
                detector_input->data.int8[idx + 2] = (int8_t)(((bf - 127.5f) / 128.0f) / detector_input->params.scale + detector_input->params.zero_point);
            } else if (detector_input->type == kTfLiteUInt8) {
                detector_input->data.uint8[idx + 0] = (uint8_t)(((rf - 127.5f) / 128.0f) / detector_input->params.scale + detector_input->params.zero_point);
                detector_input->data.uint8[idx + 1] = (uint8_t)(((gf - 127.5f) / 128.0f) / detector_input->params.scale + detector_input->params.zero_point);
                detector_input->data.uint8[idx + 2] = (uint8_t)(((bf - 127.5f) / 128.0f) / detector_input->params.scale + detector_input->params.zero_point);
            } else {
                Serial.println("[AI LỖI] Model mới yêu cầu kTfLiteFloat32, kTfLiteInt8 hoặc kTfLiteUInt8 input!");
            }
        }
    }
    
    // Bước 3: Chạy Suy Luận
    Serial.println("[AITask] Bắt đầu gọi BlazeFace Invoke()...");
    if (detector_interpreter->Invoke() != kTfLiteOk) {
        Serial.println("[AI LỖI] BlazeFace Invoke thất bại!");
        return box;
    }
    Serial.println("[AITask] BlazeFace Invoke() thành công!");
    
    // Bước 4: Giải mã các Tensor (Hỗ trợ model 2 tensor Float32 chuẩn hoặc 4 tensor Quantized)
    int num_outputs = detector_interpreter->outputs_size();
    float max_score = 0.0f;
    int best_anchor = -1;
    bool is_8x8 = false;
    float best_dx = 0, best_dy = 0, best_w = 0, best_h = 0;
    
    if (num_outputs == 4) {
        // Mô hình chia thành 4 Tensors
        TfLiteTensor* reg_16 = detector_interpreter->output(0);
        TfLiteTensor* cls_16 = detector_interpreter->output(1);
        TfLiteTensor* reg_8  = detector_interpreter->output(2);
        TfLiteTensor* cls_8  = detector_interpreter->output(3);
        
        for (int i = 0; i < 512; i++) {
            float score = 0.0f;
            if (cls_16->type == kTfLiteFloat32) { score = 1.0f / (1.0f + exp(-cls_16->data.f[i])); }
            else if (cls_16->type == kTfLiteUInt8) { 
                if (cls_16->data.uint8[i] == 255) score = 0.95f; 
                else score = 1.0f / (1.0f + exp(-((cls_16->data.uint8[i] - cls_16->params.zero_point) * cls_16->params.scale))); 
            }
            else if (cls_16->type == kTfLiteInt8) { score = 1.0f / (1.0f + exp(-((cls_16->data.int8[i] - cls_16->params.zero_point) * cls_16->params.scale))); }
            if (score > max_score) { max_score = score; best_anchor = i; is_8x8 = false; }
        }
        for (int i = 0; i < 384; i++) {
            float score = 0.0f;
            if (cls_8->type == kTfLiteFloat32) { score = 1.0f / (1.0f + exp(-cls_8->data.f[i])); }
            else if (cls_8->type == kTfLiteUInt8) { 
                if (cls_8->data.uint8[i] == 255) score = 0.95f; 
                else score = 1.0f / (1.0f + exp(-((cls_8->data.uint8[i] - cls_8->params.zero_point) * cls_8->params.scale))); 
            }
            else if (cls_8->type == kTfLiteInt8) { score = 1.0f / (1.0f + exp(-((cls_8->data.int8[i] - cls_8->params.zero_point) * cls_8->params.scale))); }
            if (score > max_score) { max_score = score; best_anchor = i; is_8x8 = true; }
        }
        
        if (max_score >= 0.60f && best_anchor >= 0) {
            TfLiteTensor* reg = is_8x8 ? reg_8 : reg_16;
            if (reg->type == kTfLiteFloat32) {
                best_dx = reg->data.f[best_anchor * 16 + 0];
                best_dy = reg->data.f[best_anchor * 16 + 1];
                best_w  = reg->data.f[best_anchor * 16 + 2];
                best_h  = reg->data.f[best_anchor * 16 + 3];
            } else if (reg->type == kTfLiteUInt8) {
                best_dx = (reg->data.uint8[best_anchor * 16 + 0] - reg->params.zero_point) * reg->params.scale;
                best_dy = (reg->data.uint8[best_anchor * 16 + 1] - reg->params.zero_point) * reg->params.scale;
                best_w  = (reg->data.uint8[best_anchor * 16 + 2] - reg->params.zero_point) * reg->params.scale;
                best_h  = (reg->data.uint8[best_anchor * 16 + 3] - reg->params.zero_point) * reg->params.scale;
            } else if (reg->type == kTfLiteInt8) {
                best_dx = (reg->data.int8[best_anchor * 16 + 0] - reg->params.zero_point) * reg->params.scale;
                best_dy = (reg->data.int8[best_anchor * 16 + 1] - reg->params.zero_point) * reg->params.scale;
                best_w  = (reg->data.int8[best_anchor * 16 + 2] - reg->params.zero_point) * reg->params.scale;
                best_h  = (reg->data.int8[best_anchor * 16 + 3] - reg->params.zero_point) * reg->params.scale;
            }
            if (is_8x8) best_anchor += 512; // Cập nhật global anchor idx
        }
    } else if (num_outputs == 2) {
        // Mô hình gộp thành 2 Tensors (Chuẩn)
        TfLiteTensor* reg = detector_interpreter->output(0);
        TfLiteTensor* cls = detector_interpreter->output(1);
        
        // Tự động hoán đổi nếu model xuất cls ở index 0
        if (reg->dims->data[reg->dims->size - 1] == 1) { 
            TfLiteTensor* temp = reg;
            reg = cls;
            cls = temp;
        }
        
        for (int i = 0; i < 896; i++) {
            float score = 0.0f;
            if (cls->type == kTfLiteFloat32) {
                score = 1.0f / (1.0f + exp(-cls->data.f[i]));
            } else if (cls->type == kTfLiteUInt8) {
                if (cls->data.uint8[i] == 255) score = 0.95f;
                else score = 1.0f / (1.0f + exp(-((cls->data.uint8[i] - cls->params.zero_point) * cls->params.scale)));
            } else if (cls->type == kTfLiteInt8) {
                score = 1.0f / (1.0f + exp(-((cls->data.int8[i] - cls->params.zero_point) * cls->params.scale)));
            }
            if (score > max_score) { max_score = score; best_anchor = i; }
        }
        
        if (max_score >= 0.60f && best_anchor >= 0) {
            if (reg->type == kTfLiteFloat32) {
                best_dx = reg->data.f[best_anchor * 16 + 0];
                best_dy = reg->data.f[best_anchor * 16 + 1];
                best_w  = reg->data.f[best_anchor * 16 + 2];
                best_h  = reg->data.f[best_anchor * 16 + 3];
            } else if (reg->type == kTfLiteUInt8) {
                best_dx = (reg->data.uint8[best_anchor * 16 + 0] - reg->params.zero_point) * reg->params.scale;
                best_dy = (reg->data.uint8[best_anchor * 16 + 1] - reg->params.zero_point) * reg->params.scale;
                best_w  = (reg->data.uint8[best_anchor * 16 + 2] - reg->params.zero_point) * reg->params.scale;
                best_h  = (reg->data.uint8[best_anchor * 16 + 3] - reg->params.zero_point) * reg->params.scale;
            } else if (reg->type == kTfLiteInt8) {
                best_dx = (reg->data.int8[best_anchor * 16 + 0] - reg->params.zero_point) * reg->params.scale;
                best_dy = (reg->data.int8[best_anchor * 16 + 1] - reg->params.zero_point) * reg->params.scale;
                best_w  = (reg->data.int8[best_anchor * 16 + 2] - reg->params.zero_point) * reg->params.scale;
                best_h  = (reg->data.int8[best_anchor * 16 + 3] - reg->params.zero_point) * reg->params.scale;
            }
        }
    } else {
        Serial.printf("[AI LỖI] Số lượng output không hợp lệ: %d\n", num_outputs);
        return box;
    }
    
    if (max_score >= 0.65f && best_anchor >= 0) {
        int global_anchor_idx = best_anchor;
        
        float raw_cx = best_dx / 128.0f + g_anchors[global_anchor_idx].x;
        float raw_cy = best_dy / 128.0f + g_anchors[global_anchor_idx].y;
        float raw_w = best_w / 128.0f;
        float raw_h = best_h / 128.0f;
        
        // --- BỘ LỌC EMA (Exponential Moving Average) ---
        // Giúp ổn định khung mặt, chống rung lắc (jitter) để Face Recognition chính xác hơn
        static float ema_cx = -1.0f, ema_cy = -1.0f, ema_w = -1.0f, ema_h = -1.0f;
        const float alpha = 0.35f; // Hệ số mượt (Càng nhỏ càng mượt nhưng phản hồi chậm)
        
        if (ema_cx < 0.0f) { // Khởi tạo lần đầu
            ema_cx = raw_cx; ema_cy = raw_cy; ema_w = raw_w; ema_h = raw_h;
        } else {
            // Nếu khung mặt di chuyển quá xa (sai số > 20%), reset EMA để bám theo ngay lập tức
            if (abs(raw_cx - ema_cx) > 0.2f || abs(raw_cy - ema_cy) > 0.2f) {
                ema_cx = raw_cx; ema_cy = raw_cy; ema_w = raw_w; ema_h = raw_h;
            } else {
                ema_cx = alpha * raw_cx + (1.0f - alpha) * ema_cx;
                ema_cy = alpha * raw_cy + (1.0f - alpha) * ema_cy;
                ema_w = alpha * raw_w + (1.0f - alpha) * ema_w;
                ema_h = alpha * raw_h + (1.0f - alpha) * ema_h;
            }
        }
        
        float cx = ema_cx;
        float cy = ema_cy;
        float w = ema_w;
        float h = ema_h;
        
        // Scale lại kích thước gốc 240x240
        float x_center = cx * FRAME_WIDTH;
        float y_center = cy * FRAME_HEIGHT;
        float box_w = w * FRAME_WIDTH;
        float box_h = h * FRAME_HEIGHT;
        
        box.x = max(0, (int)(x_center - box_w / 2));
        box.y = max(0, (int)(y_center - box_h / 2));
        box.width = min((int)box_w, FRAME_WIDTH - box.x);
        box.height = min((int)box_h, FRAME_HEIGHT - box.y);
        
        Serial.printf("[BlazeFace RAW] dx:%.2f dy:%.2f w:%.2f h:%.2f -> cx:%.3f cy:%.3f box_w:%.2f box_h:%.2f\n", 
                      best_dx, best_dy, best_w, best_h, raw_cx, raw_cy, box_w, box_h);
        
        // Cắt cho vuông để tương thích với GhostFaceNet
        int size = max(box.width, box.height);
        
        // Căn giữa hình vuông
        int x_adj = box.x - (size - box.width) / 2;
        int y_adj = box.y - (size - box.height) / 2;
        
        box.x = max(0, x_adj);
        box.y = max(0, y_adj);
        box.width = min(size, FRAME_WIDTH - box.x);
        box.height = min(size, FRAME_HEIGHT - box.y);
        
        box.is_valid = true;
        Serial.printf("[BlazeFace] Tìm thấy mặt! Score: %.2f | Tọa độ: [%d, %d, %d, %d]\n", max_score, box.x, box.y, box.width, box.height);
    } else {
        Serial.printf("[BlazeFace] Không tìm thấy mặt. Max score: %.2f\n", max_score);
    }
    
    return box;
}

// Hàm nội suy song tuyến (Bilinear Interpolation) và chuyển Grayscale
bool preprocess_face(const FaceBox& box, float* out_tensor) {
    if (!box.is_valid || !g_frame_buffer) return false;
    
    // Kích thước đích của mô hình nhận diện
    const int TARGET_SIZE = 64;
    
    // Tính toán tỉ lệ scale
    float scale_x = (float)box.width / TARGET_SIZE;
    float scale_y = (float)box.height / TARGET_SIZE;
    
    for (int dst_y = 0; dst_y < TARGET_SIZE; dst_y++) {
        for (int dst_x = 0; dst_x < TARGET_SIZE; dst_x++) {
            
            // Tìm tọa độ tương ứng trên ảnh gốc (Bilinear Interpolation)
            float src_x = dst_x * scale_x + box.x;
            float src_y = dst_y * scale_y + box.y;
            
            // Lấy 4 điểm lân cận
            int x1 = (int)src_x;
            int y1 = (int)src_y;
            int x2 = min(x1 + 1, FRAME_WIDTH - 1);
            int y2 = min(y1 + 1, FRAME_HEIGHT - 1);
            
            // Trọng số nội suy
            float wx = src_x - x1;
            float wy = src_y - y1;
            
            // Lấy màu 4 điểm (RGB565 -> tách kênh)
            auto get_gray = [](uint16_t c) {
                uint8_t r = (c >> 11) << 3;
                uint8_t g = ((c >> 5) & 0x3F) << 2;
                uint8_t b = (c & 0x1F) << 3;
                return 0.299f * r + 0.587f * g + 0.114f * b;
            };
            
            float q11 = get_gray(g_frame_buffer[y1 * FRAME_WIDTH + x1]);
            float q21 = get_gray(g_frame_buffer[y1 * FRAME_WIDTH + x2]);
            float q12 = get_gray(g_frame_buffer[y2 * FRAME_WIDTH + x1]);
            float q22 = get_gray(g_frame_buffer[y2 * FRAME_WIDTH + x2]);
            
            // Tính giá trị xám cuối cùng
            float gray = q11 * (1 - wx) * (1 - wy) + 
                         q21 * wx * (1 - wy) + 
                         q12 * (1 - wx) * wy + 
                         q22 * wx * wy;
                         
            // Chuẩn hóa vào khoảng [-1.0, 1.0] cho GhostFaceNet
            out_tensor[dst_y * TARGET_SIZE + dst_x] = (gray - 127.5f) / 128.0f;
        }
    }
    
    return true;
}
