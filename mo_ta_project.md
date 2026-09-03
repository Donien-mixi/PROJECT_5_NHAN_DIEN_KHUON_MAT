# 📖 MÔ TẢ CHI TIẾT TẤT CẢ CÁC THƯ MỤC VÀ TỪNG FILE TRONG TOÀN BỘ DỰ ÁN

Dưới đây là tài liệu mô tả kiến trúc và hợp đồng xử lý của hệ thống. Dự án được chia thành 3 phân hệ chính (firmware/host/training) + 1 giai đoạn vận hành. Các phân hệ có thể phát triển riêng, nhưng phải giữ nguyên các hợp đồng chung về model, tensor, pixel, protocol và database để tránh lệch khi chuyển từ Laptop xuống ESP32.

---

## 🏗️ PHÂN HỆ 1: `firmware_esp32/` (Vi Điều Khiển Nhúng)
Chứa mã nguồn C/C++ nạp trực tiếp vào **ESP32-S3** thông qua Arduino IDE. 
**Nhiệm vụ:** Hoạt động như một "Bộ Não Chạy Biên" (Edge AI). Nó chỉ nhận ảnh qua Wi-Fi và tự chạy suy luận AI.
**Lưu ý:** ESP32 trong dự án này hoàn toàn **không sử dụng màn hình LCD** — đầu ra duy nhất là **2 LED (xanh = đã nhận diện, đỏ = người lạ) + Buzzer (1 bip ngắn = success, 2 bip dài = reject) + Serial**. Toàn bộ giao diện hiển thị được chuyển hết về máy tính.

*   `firmware_esp32.ino`: File chính của Arduino IDE (nhạc trưởng). Khởi tạo SPIFFS, Wi-Fi/TCP và các task; **khóa xung nhịp CPU 2 nhân ở 240MHz** (`setCpuFrequencyMhz(240)`); nhận JPEG 128×128 từ `host_laptop`, giải mã về pixel RGB565, sau đó chạy BlazeFace 128 → crop Bilinear → Ghost-TinyFace 64. Hỗ trợ cơ chế **Box-Reuse** (cy: REUSE đạt 1.32s/frame khi khuôn mặt hợp lệ).
*   `ai_config.h`: File chứa các tham số bộ nhớ (Arena Size) và cấu hình để ESP32 tự động cấp phát PSRAM/SRAM khi biên dịch (detector 2.5MB + recognizer 1.75MB PSRAM — gồm esp-nn scratch buffer để căn chỉnh bộ lọc — + packet buffer 32KB).
*   `ai_face_detector.h` & `ai_face_detector.cpp`: Chuẩn bị input RGB 128×128 từ buffer RGB565, gọi BlazeFace FULL INT8, giải mã bounding box và crop vuông Bilinear. Tốc độ suy luận đạt ~1.95s nhờ tăng tốc phần cứng SIMD.
*   `esp_nn/` + `esp_nn_glue.h/.cpp`: **[4.1 REALTIME - ĐÃ HOÀN THÀNH & NGHIỆM THU]** Vendor kernel SIMD `esp-nn v1.3` (Espressif, Apache-2.0) cho CONV_2D/DEPTHWISE_CONV_2D trên Xtensa LX7 — đăng ký qua `resolver.AddConv2D(reg)`. Tích hợp bộ lọc rẽ nhánh thông minh `DwUseEspNn()` để bypass lỗi phần cứng assembly `s8pad` 3x3 của chip ESP32-S3, tự động chuyển về `tflite::reference_integer_ops::DepthwiseConvPerChannel` khi gặp cấu hình lỗi. Nhờ đó đạt **maxdiff = 0** trên `EspNnSelfTest()`, bảo toàn 100% độ chính xác trong khi tăng tốc detector gấp 10.5 lần (~1.95s) và recognizer gấp 4 lần (~1.30s).
*   `ai_face_recognizer.h` & `ai_face_recognizer.cpp`: Nạp mô hình nhận diện Ghost-TinyFace INT8 64×64, trích xuất vector 128 chiều, so khớp Cosine MAX-SIM với `face_database.h`. `identify_face` áp per-identity threshold (`max(0.60, ngưỡng riêng)` — argmax trước, ngưỡng sau).
*   `wifi_udp_server.h` & `wifi_udp_server.cpp`: Tên file được giữ để tương thích, nhưng giao thức thực tế là TCP port 12345. Module có trách nhiệm cấp phát buffer PSRAM, đọc đủ header độ dài và payload, đồng thời phải có timeout/reconnect an toàn. **Cập nhật mới (Phase 4.1):** Xóa bỏ ràng buộc `!is_new_frame_available`, cho phép Core 1 liên tục giải mã và ghi đè JPEG mới nhất vào `g_frame_buffer`. Loại bỏ hoàn toàn độ trễ hàng đợi 5s của pipeline cũ; Core 0 luôn nhận được frame tức thời (<50ms delay).
*   `image_decoder.h` & `image_decoder.cpp`: Giải mã JPEG 128×128 thành buffer pixel RGB565 128×128 cho AI thông qua `TJpg_Decoder`.
*   `face_database.h`: Chứa tới 16 embedding 128-D (templates) cho mỗi người dùng; matching trên ESP32 lấy MAX-SIM đồng bộ với Laptop. Ảnh đăng ký vẫn nằm trong `data/registered_faces/` và được dùng để regenerate database.
*   `model_data.h`: C array của model Ghost-TinyFace INT8 64×64; kích thước artifact thực tế ~1MB.
*   `detector_model_data.h`: C array của model BlazeFace FULL INT8 128×128; kích thước artifact thực tế ~1.1MB.
*   ~~`platformio.ini`~~: Đã loại bỏ — project hiện build 100% bằng **Arduino IDE** (ESP32 Arduino core 3.x). Không còn PlatformIO/`.pio`.

---

## 💻 PHÂN HỆ 2: `host_laptop/` (Trạm Phát & Trạm Quản Lý Giao Diện)
Chứa mã nguồn Python chạy trên máy tính. 
**Nhiệm vụ:** Truyền camera tới ESP32, hiển thị giao diện HUD công nghệ cao, quản lý thêm người dùng và ghi log SQL. Không can thiệp vào thuật toán nhận diện bên trong ESP32.

*   `main.py`: File điều phối chính, HUD và SQLite. Khi test E2E phải dùng cùng frame contract với ESP32: center crop/resize RAW 128 → JPEG decode → mô phỏng RGB565 → BlazeFace → Bilinear 128→64 → Ghost-TinyFace → Cosine, cùng normalize và voting.
*   `ip_camera_streamer.py`: Đọc webcam, center-crop vuông, resize một lần về 128×128, encode JPEG quality 80 và truyền frame có header độ dài qua TCP port 12345. Đây là dumb camera, không nhận diện. **Cập nhật mới (Phase 4.1):** Đã bổ sung `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)` và `sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)` để loại bỏ triệt để độ trễ buffer driver webcam và thuật toán Nagle. Bỏ 15 frame warm-up đầu để tránh điểm thấp do auto-exposure.
*   `enroll_tool.py`: Thu ảnh người dùng mới theo quy tắc 15-20 ảnh/người và khoảng 70/30 pose. Frame được đưa qua đường JPEG và detector/crop trước khi lưu ảnh grayscale 64×64; Laptop mô phỏng RGB565 để khớp tuyệt đối với ESP32.
*   `convert_tflite_to_c.py`: Script nhỏ để đổi file `.tflite` INT8 thành `.h` (dùng cho detector).
*   **`core/vision_utils.py`**: `prepare_esp32_frame`, `center_crop_to_raw`, Bilinear 128→64, `equalize_gray_256` (HE LUT), `rgb565_roundtrip` — hợp đồng pixel đồng bộ ESP32.
*   **`detector/blazeface_esp32.py`** & `face_detection_short_range*.tflite`: `UnifiedFaceDetector` INT8 128×128 — mô phỏng RGB565, Bilinear `scale=box_size/64`, HE, normalize như C++.
*   **`recognizer/face_recognizer.py`**: `FaceRecognizer` (TFLite INT8 → 128-D, L2-norm, cosine MAX-SIM 16 templates) + `TemporalVoter` (3 frame, pause-on-Unknown max 2).
*   **`database/db_manager.py`**: SQLite `attendance.db` + cooldown 30s chống spam.
*   **`ui/hud_renderer.py`**: Vẽ HUD OpenCV trên **màn hình Laptop/PC**, không phải LCD gắn ESP32 — ESP32 chỉ dùng 2 LED + Buzzer.

### HỢP ĐỒNG ĐỒNG NHẤT LAPTOP ↔ ESP32

Hai bên phải dùng cùng một hợp đồng xử lý, không chỉ cùng kích thước ảnh:

1. Webcam được center-crop vuông và resize một lần về RAW 128×128.
2. Streamer encode JPEG quality 80, bảo đảm payload không vượt buffer 32KB, thêm length header 4 byte little-endian và gửi qua TCP 12345.
3. Laptop test và ESP32 phải dùng cùng JPEG. Laptop phải mô phỏng bước ESP32 giải mã vào RGB565 trước detector/crop, hoặc firmware phải đổi sang một biểu diễn pixel chung đã được xác minh.
4. Detector phải dùng cùng input quantization, anchor/decode box và confidence threshold.
5. Crop mặt phải là crop vuông Bilinear thủ công 128→64 với cùng half-pixel formula, clamp và cách làm tròn, sau đó áp **Histogram Equalization (LUT số nguyên)** khử nhạy ánh sáng — cài đặt bit-exact ở cả Python (`equalize_gray_256`) và C++ (`equalize_gray_u8`). Hiện chưa triển khai Affine 5 điểm; không được ghi là có Affine nếu chưa có cả Python và C++.
6. Recognizer phải dùng cùng normalize `(gray - 127.5) / 128.0`, dequantize output và L2-normalize trước matching. Matching phải là MAX-SIM trên nhiều templates/người ở CẢ Laptop và ESP32 (không dùng centroid đơn) để đồng bộ độ chính xác.
7. Một golden set JPEG cố định phải được chạy trên Laptop và ESP32, so sánh pixel/crop/embedding/danh tính trong sai số đã đo.

---

## 🧠 PHÂN HỆ 3: `training_tinyml/` (Xưởng Đào Tạo Trí Tuệ Nhân Tạo)
Đây là phân hệ huấn luyện/export, tách khỏi runtime nhưng cung cấp artifact và contract mà host/firmware cùng sử dụng.
Hệ thống hướng tới chuẩn **Zero-Retraining**: AI học đặc trưng chung (Universal) từ tập LFW quốc tế. Sau khi model đã được kiểm chứng, thêm người mới chỉ cần tạo embedding database, không train lại; điều này không loại bỏ yêu cầu kiểm thử lại dữ liệu mới.

*   `download_lfw_dataset.py`: Tự động tải tập LFW 13.000 ảnh chuẩn quốc tế.
*   `train_distillation.py`: Huấn luyện trên Colab: Teacher SFace (38MB) dạy Student Ghost-TinyFace 64×64 bằng **KD + ArcFace loss** (ép margin giữa danh tính) + **illumination-invariance** (teacher nhúng ảnh sạch, student nhận ảnh augment sáng/tối + HE) — giải quyết bài toán 2 người bị nhầm nhau khi ánh sáng chụp khác nhau.
*   `quantize_qat_int8.py`: Export Ghost-TinyFace sang INT8; representative dataset **có HE** đồng bộ inference.
*   `quantize_detector_int8.py`: PTQ BlazeFace (float16-hybrid) → FULL INT8 giữ nguyên topo 2-output; validate maxdiff < 1e-3 rồi mới quantize. Model BlazeFace được quantize bằng script này.
*   `generate_embeddings.py`: TFLite INT8 → 128-D, L2-norm, Trimmed 80% (MAX-SIM 16 templates/người) + **per-identity threshold** (`cross_max + 0.02`, floor 0.60, cap 0.80) ghi vào JSON và struct C++.
*   `update_face_database.py`: (Công cụ hàng ngày) Quét `registered_faces` → `face_database.json` + `firmware_esp32/face_database.h` + regenerate `ai_config.h`. **Thêm người mới chỉ cần enroll 20 ảnh → chạy file này → flash ESP32, không train lại (Zero-Retraining).**
*   `export_config.py`: Sinh `ai_config.h` (Arena PSRAM 2.5MB+1.75MB+32KB, RAW 128, FACE 64, 0.60/0.80/3).
*   `evaluate_model.py`: Benchmark similarity, TAR/FAR, threshold tối ưu, **Identification MAX-SIM 16 templates** (mục 5); probe/impostor độc lập, đo cả detector khi E2E.
*   `colab_exports/`: Bản copy `model_data.h` / `face_database.h` xuất từ Colab (nếu train trên Colab).

---

## 📂 CÁC THƯ MỤC LƯU TRỮ VÀ TÀI LIỆU (DATA & DOCS)

### Thư mục `data/` (Kho lưu trữ dữ liệu tĩnh)
*   `registered_faces/nhien|thao|toan/`: Mỗi người **20 PNG 64×64 grayscale** (hiện tại 20/20/20) — HE và Bilinear đã áp ở enroll; kiểm tra tỷ lệ pose 70/30 trước khi tạo database.
*   `face_database.json`: JSON trung gian — **16 templates/người × 128-D** (MAX-SIM, không phải centroid đơn).
*   `attendance.db`: SQLite log điểm danh.

### CÁC FILE QUÁ HẠN (OBSOLETE / ĐÃ LOẠI BỎ)
*Hệ thống cũ có một số file, nay do chuyển sang Zero-Retraining nên không bao giờ dùng tới nữa (bạn có thể xóa tự do để sạch project):*
*   🗑️ `training_tinyml/finetune_locally.py`
*   🗑️ `training_tinyml/dataset_loader.py`
*   🗑️ `training_tinyml/fix_and_evaluate.py`
*   🗑️ `training_tinyml/find_bad_photos.py`

### Tài liệu hướng dẫn ở gốc (Root Directory)
*   `HUONG_DAN_COLAB_TRAIN_V2.md` + `colab.zip`: Gói 1-lệnh train v2 trên Colab (ArcFace + illumination-invariance).
*   `mo_ta_project.md`: Chính là file bạn đang đọc.
*   `README.md`: Roadmap + trạng thái tiến độ.
*   `kich_hoat_moi_truong.md`: Lệnh nhanh conda + flash + stream (nguồn chân lý vận hành).
*   `ket_qua_esp32.txt`: Log test thực tế ESP32-S3 mới nhất đo trên phần cứng thật (chứng minh tốc độ và độ chính xác).
*   ~~`DANH_SACH_LOI_TEST_ARDUINO_IDE.md`~~ / ~~`HUONG_DAN_TRAIN_COLAB.md`~~ / ~~`colab_training/`~~: Đã loại bỏ hoặc thay thế.

---

## 🚀 GIAI ĐOẠN 4: TỐI ƯU & VẬN HÀNH THỰC TẾ (khả thi với kiến trúc hiện tại)

GĐ4 không thêm model lớn, chỉ **đo được + hiệu chuẩn + cứng hóa** trên đúng phần cứng N16R8, `ai_config.h` (Arena 2.5MB+1.75MB), TCP JPEG 128, HE bit-exact:

*   **Tốc độ (Phase 4.1 - ĐÃ HOÀN THÀNH & NGHIỆM THU):**
    * Baseline ban đầu: `det: 20.3s`, `rec: 4.7s`.
    * Tích hợp thành công kernel SIMD `esp-nn v1.3` chính thức của Espressif + xử lý bypass lỗi `s8pad` bằng `DwUseEspNn()`.
    * Khóa xung nhịp CPU 240MHz.
    * Triệt tiêu độ trễ hàng đợi mạng: Core 1 ghi đè ảnh liên tục, Core 0 lấy frame <50ms.
    * Kết quả thực tế: `det` đạt **~1.95s** (nhanh 10.5x), `rec` đạt **~1.30s** (nhanh 4x), chu kỳ Box-Reuse (`cy: REUSE`) đạt **1.32s/frame**.
*   **Chính xác (Phase 4.2 - ĐÃ HOÀN THÀNH):** Ngưỡng 0.60 + per-identity threshold (`max inter + 0.02`, floor 0.60, cap 0.80) trong `face_database.h`. Đo được FAR=0 trên 120 mẫu impostor độc lập.
*   **Chống giả mạo nhẹ:** không dùng blink (thiếu landmark `blazeface_esp32.py:216`). Chọn LBP/variance trên `gray_eq` 64×64 + active quay đầu (EMA `cx` drift); ToF VL53L5C 8×8 là tuỳ chọn phần cứng, không tốn arena.
*   **Vận hành:** NTP `configTime()` + SPIFFS epoch, enroll tại chỗ ghi SPIFFS JSON (16 templates MAX-SIM, không trung bình), OTA, leak test 24h `getFreeHeap`.
*   **Kiểm thử:** matrix ≥10 người × 3 sáng × 3 khoảng cách + golden frame `README.md:274`; chuyển OV2640 trực tiếp là tuỳ chọn, giữ `ip_camera_streamer.py` cho debug.

> Chi tiết Action Items xem `README.md:218-249` — GĐ4 đã chia thành 4.1 (tốc độ đo được - đã xong), 4.2 (hiệu chuẩn - đã xong), 4.3 (liveness nhẹ), 4.4 (NTP/enroll/OTA), 4.5 (field test + vỏ).

---

## 💎 TÀI LIỆU LƯU TRỮ SOURCE CODE CỐT LÕI (BÍ QUYẾT BẢN QUYỀN)

Vì phần lớn mã nguồn đã được dọn dẹp theo yêu cầu, đây là những bí quyết công nghệ và các đoạn code cốt lõi (secret sauce) đã làm nên thành công của hệ thống này để bạn có thể tham khảo lại khi viết lại code từ đầu:

### 1. Bí quyết Khử Domain Shift (Unified BlazeFace Emulator)
Bí quyết ở đây là Laptop phải mô phỏng đúng đường đi của ESP32: cùng JPEG, cùng biểu diễn RGB565 sau decode, cùng thuật toán Bilinear Interpolation trên ảnh 128x128 để cắt vùng mặt 64x64. Nếu Python dùng OpenCV `cv2.resize()` cho crop mặt thì có thể lệch pixel. Do đó, phải tự viết lại thuật toán crop trên Python và xác minh bằng golden frame:
```python
# Trích xuất từ blazeface_esp32.py (Python Emulator)
def _crop_and_resize_bilinear_gray(img_bgr, cx, cy, box_size, target_size=64):
    half_box = box_size / 2.0
    x1_box, y1_box = cx - half_box, cy - half_box
    scale = box_size / target_size
    # Tính Bilinear từng pixel trên input đã mô phỏng RGB565, y hệt C++ thay vì dùng OpenCV resize
    # ... (Chi tiết nội suy song tuyến)
```

### 2. Bí quyết Quy hoạch Bộ nhớ PSRAM ESP32-S3 (Hỗ trợ SIMD Scratch Buffer)
Không bao giờ để TFLite tự cấp phát RAM (vì SRAM chỉ có 512KB). Phải ép cấp phát trên PSRAM (8MB) thông qua cờ `MALLOC_CAP_SPIRAM` với mức phân bổ chính xác đo được từ `export_config.py`. Với kernel SIMD esp-nn, Arena cần mở rộng để chứa scratch buffer căn chỉnh filter:
```cpp
// Trích xuất từ ai_face_recognizer.cpp
// 1.75MB PSRAM cho Ghost-TinyFace INT8 để chứa scratch filter alignment của SIMD
#define RECOGNIZER_ARENA_SIZE (1750 * 1024)
uint8_t* raw_arena = (uint8_t*)heap_caps_malloc(RECOGNIZER_ARENA_SIZE + 16, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
```

### 3. Bí quyết Kiến trúc Đa luồng (RTOS Dual-Core) và Triệt Tiêu Độ Trễ Luồng Ảnh
ESP32 có 2 nhân. Tránh để TCP/IP (mạng) và AI chạy chung nhân vì AI sẽ block toàn bộ mạng gây mất kết nối. Đồng thời, Core 1 phải liên tục giải mã và ghi đè ảnh mới nhất vào buffer thay vì chờ Core 0 rảnh:
```cpp
// Phân chia Task sau khi pipeline single-task đã được kiểm chứng
xTaskCreatePinnedToCore(network_display_task, "NetTask", 8192, NULL, 4, &NetTaskHandle, 1); // Core 1: Wi-Fi/TCP/JPEG
xTaskCreatePinnedToCore(ai_processing_task, "AITask", 32768, NULL, 5, &AITaskHandle, 0); // Core 0: TFLite

// Trích xuất từ wifi_udp_server.cpp (Core 1): Luôn ghi đè ảnh mới nhất, khử trễ 5s cũ
xSemaphoreTake(image_mutex, portMAX_DELAY);
memcpy(g_frame_buffer, payload_buffer, payload_len);
is_new_frame_available = true;
xSemaphoreGive(image_mutex);
```

### 4. Bí quyết Loss Function (Knowledge Distillation)
Trong file `train_distillation.py`, student Ghost-TinyFace học embedding từ teacher SFace bằng cosine loss, MSE trên vector đã normalize và hard-negative loss trong batch. Model không tự đảm bảo L2 norm ở output; bước inference phải L2-normalize trước matching:
```python
def distillation_loss(teacher_embeddings, student_embeddings):
    # 1. Cosine Distance Loss (Ép góc vector giống nhau)
    cos_loss = tf.reduce_mean(1.0 - tf.reduce_sum(teacher_embeddings * student_embeddings, axis=1))
    # 2. Mean Squared Error (Ép độ lớn giống nhau)
    mse_loss = tf.keras.losses.mean_squared_error(teacher_embeddings, student_embeddings)
    # Hard-negative loss được tính riêng trong train_step và cộng với trọng số gamma.
    return cos_loss + 0.5 * mse_loss
```

### 5. Bí quyết Rẽ nhánh Hybrid Ops (Khắc phục lỗi assembly s8pad của ESP-NN)
Lớp `DepthwiseConv` 3x3 với `ch_mult == 1 && ch % 16 == 0` có lỗi sai số trên tập lệnh vector assembly `esp-nn v1.3`. Nhờ cơ chế rẽ nhánh kiểm tra cấu hình, ta cho cấu hình lỗi này chạy bằng kernel chuẩn TFLM Reference Op còn tất cả các lớp khác vẫn chạy SIMD cực nhanh:
```cpp
// Trích xuất từ firmware_esp32/esp_nn_glue.cpp
static inline bool DwUseEspNn(const TfLiteDepthwiseConvParams* params, const TfLiteTensor* input, const TfLiteTensor* filter) {
    int ch_mult = params->depth_multiplier;
    int ch = filter->dims->data[3];
    int filter_width = filter->dims->data[2];
    int filter_height = filter->dims->data[1];
    // Nhận diện chính xác cấu hình dính lỗi s8pad
    if (ch_mult == 1 && (ch % 16) == 0 && filter_width == 3 && filter_height == 3) {
        return false; // Fallback về Reference Op chuẩn -> maxdiff = 0
    }
    return true; // Dùng SIMD esp-nn tăng tốc tối đa
}
```
