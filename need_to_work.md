# Những việc cần làm

## 1. Train lại model với dữ liệu lớn hơn
Train lại model bằng tập dữ liệu lớn hơn như **CASIA-WebFace, MS1M hoặc VGGFace** để tăng độ tin cậy, thay vì tập LFW (LFW vốn chỉ dùng để kiểm tra, không phải để train).

## 2. Nâng cấp firmware
- [x] Chuyển toàn bộ từ Arduino IDE sang **ESP-IDF 5.3**, dùng bộ tăng tốc SIMD `esp-nn` và RAM ngoài PSRAM. Đã build sạch và chạy thử đạt yêu cầu.
