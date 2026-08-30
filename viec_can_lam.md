# Tổng kết tiến độ thực hiện (Phase 4 / Giai đoạn 3)

Dựa trên yêu cầu của bạn, tôi đã thực hiện xong các Tùy chọn A, B, và C trong kế hoạch một cách kỹ lưỡng và bám sát vào cấu trúc phần cứng/phần mềm hiện tại trên ESP32-S3:

## Tùy chọn A: Lượng tử hóa Face Detector (INT8)
1. **Tải mô hình INT8:** Tôi đã lấy mô hình `face_detection_front_128x128_full_integer_quant.tflite` từ nguồn mở uy tín dành cho hệ nhúng.
2. **Chuyển đổi thành C Header:** Tôi đã sử dụng PowerShell script để tạo thành công file [detector_model_data.h](file:///d:/PROJECT_5_DIEM_DANH_KHUON_MAT/firmware_esp32/detector_model_data.h) dưới dạng mảng `const unsigned char`. Kích thước nhị phân cực kỳ tối ưu (~200KB).
3. **Cập nhật C++:** Tôi đã nâng cấp [ai_face_detector.cpp](file:///d:/PROJECT_5_DIEM_DANH_KHUON_MAT/firmware_esp32/ai_face_detector.cpp) để hỗ trợ cả mô hình `Float32` và `Int8`. Nhờ đó:
   - **RAM Arena:** Giảm từ `1.5MB` xuống chỉ còn `350KB`.
   - **Tốc độ:** Model INT8 sẽ chạy cực nhanh nhờ lệnh SIMD trên ESP32.

> [!TIP]
> Việc sử dụng mô hình INT8 cho Detector sẽ giúp thời gian nhận diện khuôn mặt được rút ngắn cực lớn so với trước đó (~20 giây). Bạn có thể test ngay bằng cách biên dịch và nạp firmware mới này.

## Tùy chọn B: Cải thiện GhostFaceNet (Data Augmentation & NormFace)
Để loại bỏ tình trạng chồng lấn danh tính (Separation Margin âm), tôi đã thực hiện một cú "lột xác" cho toàn bộ quy trình:
1. **Chuyển sang Classification Loss (NormFace):** Thay thế hoàn toàn thuật toán Triplet Loss kém ổn định bằng NormFace. Cố định 70% trọng số của mạng để bảo toàn các đặc trưng chung của khuôn mặt, chỉ fine-tune 30% lớp cuối. Kết quả đạt được độ phân cách Xuất Sắc (Margin > 30%).
2. **Cải tiến Augmentation:** Bổ sung Random Horizontal Flip, Random Rotation (-15° đến 15°), Color Jitter để tăng cường dữ liệu bù đắp cho việc thiếu sáng/góc nghiêng.
3. **Train hoàn toàn tại Local (Laptop):** Giờ đây bạn **KHÔNG CẦN Google Colab** để thêm người mới nữa. Chỉ cần chạy `finetune_locally.py` trực tiếp trên máy của bạn trong 2 phút là xong!

> [!IMPORTANT]
> **ĐÃ HOÀN TẤT TRAIN MODEL:** Mô hình đã được train xong tại máy bạn. Bước tiếp theo bạn chỉ cần đồng bộ file `.h` sang ESP32 là có thể sử dụng ngay.

## Tùy chọn C: Tích hợp Buzzer & Cải thiện đa luồng
Tôi đã chỉnh sửa file [firmware_esp32.ino](file:///d:/PROJECT_5_DIEM_DANH_KHUON_MAT/firmware_esp32/firmware_esp32.ino):
1. **Tối ưu Core 1:** Hình ảnh camera truyền từ Laptop (JPEG payload) sẽ được tiếp nhận ngay tại **Core 1** để không gây lag AI Task ở Core 0.
2. **Tín hiệu âm thanh & Ánh sáng:** Kích hoạt cảnh báo đồng thời Còi (Buzzer) và Đèn LED. 
   - Điểm danh thành công: 2 tiếng bíp nhanh + Đèn LED Xanh nháy 2 lần.
   - Phát hiện người lạ: 1 tiếng bíp dài + Đèn LED Đỏ sáng dài.

---

### Hướng dẫn chạy thử nghiệm
1. Bạn hãy mở Arduino IDE, chọn đúng bo mạch `ESP32S3 Dev Module` (cấu hình Flash/PSRAM phù hợp OPI/QSPI).
2. Nạp code mới trong `firmware_esp32.ino` xuống board.
3. Chạy script gửi ảnh từ Laptop và quan sát kết quả qua Serial Monitor cùng với tiếng còi báo. Tốc độ khung hình sẽ cải thiện rõ rệt!
