#ifndef AI_FACE_RECOGNIZER_H
#define AI_FACE_RECOGNIZER_H

#include <stdint.h>

void setup_face_recognizer();

// Chạy mô hình phân tích khuôn mặt để trích xuất ra vector đặc trưng 128 chiều
// input_tensor: Mảng 64x64 float (4096 phần tử)
// output_embedding: Mảng 128 float
void extract_face_embedding(const float* input_tensor, float* output_embedding);

// So khớp với Cơ sở dữ liệu khuôn mặt. 
// Trả về tên người dùng khớp nhất (có độ tương đồng Cosine Similarity > threshold)
// Hoặc "Unknown" nếu không khớp ai.
const char* identify_face(const float* face_embedding, float threshold = 0.85f);

#endif
