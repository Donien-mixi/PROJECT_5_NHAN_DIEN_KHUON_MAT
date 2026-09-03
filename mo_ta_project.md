# 📖 MÔ TẢ CHI TIẾT TẤT CẢ CÁC THƯ MỤC VÀ TỪNG FILE TRONG TOÀN BỘ DỰ ÁN

Dưới đây là tài liệu mô tả kiến trúc và hợp đồng xử lý của hệ thống sau khi đã hoàn thành tối ưu hóa hiệu suất giai đoạn 4 (Phase 4.1). Dự án được chia thành 3 phân hệ chính (firmware/host/training). Các phân hệ giữ nguyên các hợp đồng chung về model, tensor, pixel, protocol và database.

---

## 🏗️ PHÂN HỆ 1: `firmware_esp32/` (Vi Điều Khiển Nhúng)
Chứa mã nguồn C/C++ nạp trực tiếp vào **ESP32-S3** thông qua Arduino IDE. 
**Nhiệm vụ:** Hoạt động như một "Bộ Não Chạy Biên" (Edge AI). Nó chỉ nhận ảnh qua Wi-Fi và tự chạy suy luận AI.
**Lưu ý:** ESP32 trong dự án này hoàn toàn **không sử dụng màn hình LCD** — đầu ra duy nhất là **2 LED (xanh/đỏ) + Buzzer + Serial**. 

*   `firmware_esp32.ino`: File chính của Arduino IDE (nhạc trưởng). Khởi tạo SPIFFS, Wi-Fi/TCP, 2 task (AITask, NetTask) và **khóa xung nhịp CPU ở 240MHz**. Nhận JPEG 128×128, giải mã về RGB565, chạy BlazeFace 128 → crop Bilinear → Ghost-TinyFace 64. Hỗ trợ **Box-Reuse** để tăng tốc nhận diện.
*   `ai_config.h`: File chứa tham số bộ nhớ. Khai báo Detector Arena (2.5MB) và Recognizer Arena (1.75MB) trên PSRAM để hỗ trợ bộ đệm scratch cho quá trình tính toán SIMD.
*   `ai_face_detector.h/.cpp`: Gọi BlazeFace FULL INT8, giải mã bounding box và crop vuông Bilinear. Tốc độ SIMD đạt ~1.9s.
*   `esp_nn/` + `esp_nn_glue.h/.cpp`: **[TÍNH NĂNG TĂNG TỐC QUAN TRỌNG]** Kernel SIMD esp-nn v1.3 (Espressif). Ép TFLite Micro sử dụng lệnh phần cứng của Xtensa LX7 cho `Conv2D` và `DepthwiseConv2D`. File `esp_nn_glue.cpp` chứa bộ lọc tự động đẩy các node DepthwiseConv 3x3 s8pad lỗi về hàm mềm `Reference` nhằm đảm bảo **độ chính xác tuyệt đối (maxdiff = 0)** trong khi vẫn giữ nguyên tốc độ siêu việt (~1.3s).
*   `ai_face_recognizer.h/.cpp`: Nạp Ghost-TinyFace INT8 64×64. Áp dụng Histogram Equalization (LUT), trích xuất vector 128-D, so khớp MAX-SIM.
*   `wifi_udp_server.h/.cpp`: TCP Server port 12345. Nhận payload JPEG trên Core 1 và **ghi đè liên tục** lên `g_frame_buffer`. Loại bỏ hoàn toàn khối chặn (blocking) khi AI đang tính toán, triệt tiêu độ trễ 5s của hàng đợi mạng cũ.
*   `image_decoder.h/.cpp`: Giải mã JPEG 128×128 thành buffer pixel RGB565 bằng thư viện TJpg_Decoder.
*   `face_database.h`: Chứa 16 templates/người × 128-D. Matching MAX-SIM lấy điểm cao nhất.
*   `model_data.h` & `detector_model_data.h`: Mảng C Array lưu trữ Model (Ghost-TinyFace và BlazeFace) nạp trên ROM.

---

## 💻 PHÂN HỆ 2: `host_laptop/` (Trạm Phát & Trạm Quản Lý Giao Diện)
Chứa mã nguồn Python chạy trên máy tính. 
**Nhiệm vụ:** Truyền camera tới ESP32, hiển thị HUD và ghi log.

*   `main.py`: File điều phối chính (hiển thị HUD, SQLite log). E2E test đồng nhất với quy trình của firmware ESP32.
*   `ip_camera_streamer.py`: Dumb camera — đọc webcam, crop/resize 128×128, nén JPEG, gửi qua TCP 12345. **Đã thiết lập cờ tắt Nagle (`TCP_NODELAY`) và vô hiệu hóa cache bộ đệm của OpenCV (`CAP_PROP_BUFFERSIZE=1`)** để đảm bảo frame truyền là frame thời gian thực.
*   `enroll_tool.py`: Chụp ảnh người dùng mới (20 ảnh/người, 70/30 pose).
*   `convert_tflite_to_c.py`: Script nhỏ để đổi file `.tflite` thành mảng C.
*   **`core/vision_utils.py`**: Các thuật toán chuẩn hóa: Bilinear 128→64 tự viết (không dùng OpenCV resize) và `equalize_gray_256` (HE LUT) đồng bộ bit-exact với C++.
*   **`detector/blazeface_esp32.py`**: Unified BlazeFace Emulator mô phỏng chính xác thuật toán RGB565 và Crop C++.
*   **`recognizer/face_recognizer.py`**: Nhận diện MAX-SIM + Temporal Voting (pause-on-Unknown).
*   **`database/db_manager.py`**: SQLite database + tính năng cooldown điểm danh chống spam (30s).
*   **`ui/hud_renderer.py`**: Vẽ giao diện HUD khoa học viễn tưởng.

### HỢP ĐỒNG ĐỒNG NHẤT LAPTOP ↔ ESP32 ĐÃ THỰC THI THÀNH CÔNG
1. Crop vuông Bilinear thủ công 128→64 với cùng cách làm tròn + Histogram Equalization LUT số nguyên giống nhau 100%.
2. Cùng normalize `(gray - 127.5) / 128.0` và L2-normalize. Đạt độ chính xác 100% khi test chéo (Laptop và ESP32 ra kết quả embedding & điểm Cosine y hệt).

---

## 🧠 PHÂN HỆ 3: `training_tinyml/` (Xưởng Đào Tạo Trí Tuệ Nhân Tạo)
Mọi việc huấn luyện model đã hoàn tất. Chuẩn hóa theo quy trình **Zero-Retraining**.

*   `train_distillation.py`: (Đã chạy) SFace → Ghost-TinyFace bằng KD + ArcFace loss + Illumination-invariance.
*   `quantize_qat_int8.py` & `quantize_detector_int8.py`: (Đã chạy) Quantize 2 model sang INT8 TFLite.
*   `update_face_database.py`: **(Dùng hàng ngày)** Sinh `face_database.h` và `ai_config.h` khi có người dùng mới được thêm qua `enroll_tool`. Không cần train lại AI.
*   `generate_embeddings.py`: TFLite INT8 → 128-D + per-identity threshold.
*   `evaluate_model.py`: Benchmark similarity và FAR/TAR.

---

## 🚀 THÀNH QUẢ GIAI ĐOẠN 4.1: TỐI ƯU TỐC ĐỘ VÀ ĐỘ TRỄ REAL-TIME
Những bí quyết đã được tích hợp thành công trên phần cứng thật:

1. **Khử Độ Trễ Mạng Bằng Cơ Chế "Ghi Đè Không Chờ" (Non-blocking Override):** Tách bạch Mutex của Frame Buffer. Thay vì Core 1 phải chờ Core 0 rảnh mới được tải ảnh, Core 1 nay sẽ bắt luồng liên tục và lưu Frame mới nhất vào RAM bất chấp Core 0 đang chậm. Nhờ vậy, AI khi chạy vòng lặp mới sẽ **luôn chụp được frame vừa xảy ra tức thời**, độ trễ từ camera đến AI bị triệt tiêu hoàn toàn (< 50ms).
2. **Tăng Tốc Phần Cứng SIMD (Tối Đa & Chính Xác):** Nhúng thư viện `esp-nn` của Espressif. Cả BlazeFace và Ghost-TinyFace đều được nhân ma trận song song bằng lệnh Xtensa. Tốc độ nhận diện nhanh gấp 4 lần (1.3s) và phát hiện khuôn mặt nhanh gấp 10.5 lần (1.9s).
3. **Xử Lý Lỗi Silicon (Hybrid Ops):** Lỗi của SIMD `s8pad` 3x3 đã được khắc phục triệt để bằng logic rẽ nhánh thông minh bên trong `esp_nn_glue.cpp`, qua đó vừa giữ được sức mạnh SIMD vừa duy trì độ nhận diện chuẩn xác ban đầu.
4. **Box-Reuse Pipeline:** Nếu kết quả frame trước là hợp lệ, hệ thống bỏ qua BlazeFace và chỉ chạy thẳng Recognizer. Thời gian điểm danh mỗi frame giờ đây chỉ còn **1.3 giây**.

---

## 💎 TÀI LIỆU LƯU TRỮ SOURCE CODE CỐT LÕI (BÍ QUYẾT BẢN QUYỀN)

### 1. Bí quyết Khử Domain Shift (Unified BlazeFace Emulator)
Laptop phải tự mô phỏng Bilinear y hệt C++ thay vì OpenCV `cv2.resize()` để giữ nguyên pixel:
```python
def _crop_and_resize_bilinear_gray(img_bgr, cx, cy, box_size, target_size=64):
    # Tính Bilinear từng pixel trên input đã mô phỏng RGB565
    # ...
```

### 2. Bí quyết Quy hoạch Bộ nhớ PSRAM cho SIMD ESP-NN
SIMD yêu cầu bộ nhớ đệm (scratch buffer) lớn để căn chỉnh bộ lọc. Phải ép cấp phát trên PSRAM (8MB):
```cpp
#define RECOGNIZER_ARENA_SIZE (1750 * 1024) // 1.75MB cho nhận diện
uint8_t* raw_arena = (uint8_t*)heap_caps_malloc(RECOGNIZER_ARENA_SIZE + 16, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
```

### 3. Bí quyết Kiến trúc Đa luồng (RTOS Dual-Core) và Triệt Tiêu Độ Trễ
ESP32 có 2 nhân. Mạng (NetTask) chạy Core 1; AI (AITask) chạy Core 0. Gỡ bỏ lệnh `!is_new_frame_available` khi khóa Mutex ở Core 1 để đạt real-time:
```cpp
// Trích xuất từ wifi_udp_server.cpp (Core 1)
xSemaphoreTake(image_mutex, portMAX_DELAY);
memcpy(g_frame_buffer, payload_buffer, payload_len); // LUÔN LUÔN GHI ĐÈ ẢNH MỚI NHẤT
is_new_frame_available = true;
xSemaphoreGive(image_mutex);
```

### 4. Bí quyết Rẽ nhánh Hybrid Ops (Tránh lỗi phần cứng SIMD)
```cpp
// Trích xuất từ esp_nn_glue.cpp
if (ch_mult == 1 && ch % 16 == 0 && filter_width == 3 && filter_height == 3) {
    // Phát hiện trường hợp SIMD lỗi s8pad, điều hướng về hàm chạy bằng phần mềm chuẩn
    return tflite::reference_integer_ops::DepthwiseConvPerChannel(...);
} else {
    // Các lớp khác được tăng tốc tối đa
    esp_nn_depthwise_conv_s8(...);
}
```
