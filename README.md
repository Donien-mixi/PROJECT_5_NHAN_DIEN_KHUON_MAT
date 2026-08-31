# 🚀 ROADMAP: HỆ THỐNG ĐIỂM DANH NHẬN DIỆN KHUÔN MẶT TRÊN ESP32-S3

> **Mục tiêu cuối cùng:** Một hệ thống nhận diện khuôn mặt chạy **toàn bộ thuật toán (Phát hiện + Nhận diện) trên MCU ESP32-S3 N16R8**. Do giới hạn kinh phí, Laptop sẽ đóng vai trò như một IP Camera, truyền luồng video thô qua Wi-Fi xuống ESP32. ESP32 sẽ tự thực hiện phát hiện khuôn mặt, so khớp danh tính và hiển thị kết quả (báo Buzzer hoặc in ra Serial).

> **Phương châm cốt lõi:** Phát triển và hoàn thiện 100% trên Laptop trước → Khi đạt độ chính xác mong muốn → Triển khai xuống ESP32-S3.

---

## 📊 TỔNG QUAN BỘ KHUNG HIỆN TẠI

### Cấu trúc thư mục

```
PROJECT_5_DIEM_DANH_KHUON_MAT/
├── host_laptop/                    # 🖥️ Ứng dụng chạy trên Laptop (đang phát triển)
│   ├── main.py                     #    Điều phối chính (Camera → Detect → Recognize → DB)
│   ├── enroll_tool.py              #    Thu thập ảnh đăng ký người dùng
│   ├── detector/
│   │   ├── yunet_detector.py       #    Phát hiện khuôn mặt (MediaPipe / YuNet)
│   │   └── *.onnx                  #    Model YuNet ONNX
│   ├── recognizer/
│   │   └── face_recognizer.py      #    Trích xuất embedding + so khớp + Temporal Voting
│   ├── database/
│   │   └── db_manager.py           #    Ghi log điểm danh SQLite
│   └── ui/
│       └── hud_renderer.py         #    Vẽ giao diện HUD trên OpenCV
│
├── training_tinyml/                # 🧠 Huấn luyện mô hình AI
│   ├── models/
│   │   └── ghost_tinyface.py       #    Kiến trúc mạng Ghost-TinyFace (128-D, 64x64 grayscale)
│   ├── train_distillation.py       #    Huấn luyện Knowledge Distillation (SFace → Ghost-TinyFace)
│   ├── dataset_loader.py           #    Xử lý dữ liệu huấn luyện
│   ├── download_lfw_dataset.py     #    Tải tập LFW (5000+ ảnh)
│   ├── evaluate_model.py           #    Đánh giá độ chính xác mô hình
│   ├── generate_embeddings.py      #    Tạo embedding cho người dùng đăng ký
│   ├── update_face_database.py     #    Cập nhật face_database.json
│   ├── quantize_qat_int8.py        #    Chuyển Keras → TFLite INT8 (cho ESP32)
│   └── weights/
│       ├── tinyface_backbone.keras #    Trọng số mô hình Keras (đang dùng trên Laptop)
│       ├── tinyface_int8.tflite    #    Model INT8 TFLite (sẽ nạp lên ESP32)
│       └── face_recognition_sface_2021dec.onnx  # Teacher model SFace
│
├── data/                           # 💾 Dữ liệu
│   ├── registered_faces/           #    Ảnh khuôn mặt đã đăng ký (3 người)
│   ├── face_database.json          #    Embedding 128-D của từng người
│   └── attendance.db               #    CSDL SQLite log điểm danh
│
├── colab_training/                 # ☁️ Gói huấn luyện cho Google Colab
└── requirements.txt
```

### Thông số kỹ thuật hiện tại

| Thành phần | Giá trị |
|---|---|
| **Mô hình** | Ghost-TinyFace (GhostNet Bottleneck) |
| **Input** | 64×64 Grayscale (1 channel) |
| **Output** | 128-D L2-normalized embedding |
| **Kích thước model** | ~2MB (Keras), ~294KB (TFLite INT8) |
| **Face Detector** | BlazeFace (TFLite INT8 trên ESP32) / Unified BlazeFace Emulator (trên Laptop) |
| **Alignment** | Affine Similarity Transform (chuẩn InsightFace) |
| **Matching** | Cosine Similarity (dot product trên L2-normalized vectors) |
| **Threshold** | 0.75 (YuNet conf: 0.85) |
| **Chống nhiễu** | Temporal Voting (5 frame liên tiếp) |
| **Training** | Knowledge Distillation (Teacher: SFace → Student: Ghost-TinyFace) |
| **Dataset** | LFW (~5,000 ảnh, ~1,600 danh tính) |
| **CSDL người dùng** | 3 người (JSON, mỗi người 1 embedding 128-D) |

---

## 🗺️ ROADMAP CHI TIẾT TỪNG GIAI ĐOẠN

```
  ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
  │   GIAI ĐOẠN 1   │  ──►  │   GIAI ĐOẠN 2   │  ──►  │   GIAI ĐOẠN 3   │  ──►  │   GIAI ĐOẠN 4   │
  │  HOÀN THIỆN     │       │  QUANTIZE INT8  │       │  FIRMWARE ESP32 │       │  TỐI ƯU & HOÀN  │
  │  TRÊN LAPTOP    │       │  CHO ESP32-S3   │       │  CHẠY ĐỘC LẬP   │       │  THIỆN SẢN PHẨM │
  └─────────────────┘       └─────────────────┘       └─────────────────┘       └─────────────────┘
```

---

### GIAI ĐOẠN 1: HOÀN THIỆN TRÊN LAPTOP ĐẠT ĐỘ CHÍNH XÁC CAO
> **Mục tiêu:** Hệ thống chạy trên Laptop nhận diện đúng 100% người đăng ký, không nhận nhầm người lạ.

#### 💡 Chiến lược áp dụng trong Giai đoạn 1:
1. **Bí quyết 1 — Chưng cất tri thức (Knowledge Distillation):** Dùng mạng thầy SFace (38MB, >99.5% accuracy) truyền dạy biểu diễn đặc trưng cho mạng học sinh Ghost-TinyFace (160KB) để mạng nhỏ đạt độ khái quát hóa cao mà không bị học vẹt.
2. **Bí quyết 2 — Canh chỉnh chuẩn hóa khuôn mặt (Face Alignment):** Luôn dùng ma trận Affine 5 điểm mốc vàng để xoay ngang 2 mắt về tọa độ chuẩn 64x64, triệt tiêu góc nghiêng trước khi đưa vào mạng nhận diện.
3. **Chiến lược Dataset 2 bước:**
   * *Bước 1 (Hiện tại - Fast Prototyping):* Huấn luyện trên tập **LFW (~13.000 ảnh / ~5.700 người)** trên Colab GPU chỉ mất 15–20 phút để xác thực nhanh toàn bộ luồng.
   * *Bước 2 (Nâng cấp - Production-Ready):* Khi pipeline đã ổn định, mở rộng sang tập **CASIA-WebFace (~500.000 ảnh / ~10.500 người)** để đạt độ chính xác thương mại cao nhất.
4. **Quy tắc thu thập ảnh đăng ký (Tỷ lệ vàng 70/30):** Chụp 15–20 ảnh/người với 70% góc nhìn thẳng tự nhiên và 30% góc nghiêng nhẹ 10–15° để vector đại diện bao quát mọi trường hợp đời thực.

#### 📋 Các đầu việc thực hiện (Action Items):
- [x] **1.1 — Huấn luyện mô hình trên Google Colab:**
  - [x] Chạy `train_distillation.py` với tập LFW trên Colab GPU.
  - [x] Tải trọng số `tinyface_backbone.keras` về thư mục `training_tinyml/weights/`.
  - [x] Chạy `evaluate_model.py` để đo độ chính xác (mục tiêu: **Accuracy ≥ 90%** trên tập test).
  - [ ] *(Tùy chọn mở rộng)*: Chạy train trên tập CASIA-WebFace nếu cần độ tách biệt danh tính cao hơn.
- [x] **1.2 — Vận hành và tinh chỉnh pipeline nhận diện trên Laptop:**
  - [x] Tạo lại CSDL vector bằng `generate_embeddings.py` dựa trên trọng số mới.
  - [x] Chạy `host_laptop/main.py` kiểm tra nhận diện trực tiếp thời gian thực qua webcam.
  - [x] Tinh chỉnh ngưỡng `threshold` (đã tinh chỉnh kỹ lưỡng: conf=0.8, threshold=0.88) và số phiếu `Temporal Voting` (5 frame) để triệt tiêu False Accepts.
  - [x] Đo đạc tỷ lệ nhận đúng (True Positive) và tỷ lệ từ chối người lạ (True Negative) (Đã test thành công trường hợp Unknown).
- [x] **1.3 — Chuẩn hóa dữ liệu khuôn mặt đăng ký:**
  - [x] Dùng `enroll_tool.py` chụp lại bộ ảnh 15-20 tấm theo tỷ lệ vàng cho tất cả người dùng trong hệ thống.
  - [x] Tái tạo lại `face_database.json` và kiểm thử so khớp.

> [!IMPORTANT]
> **Điều kiện tiên quyết chuyển sang Giai đoạn 2:** Hệ thống chạy trên Laptop phải phân biệt chính xác tuyệt đối các thành viên đã đăng ký, và khi có người lạ đứng trước camera phải hiện `UNKNOWN` (không được nhận nhầm).

---

### GIAI ĐOẠN 2: CHUYỂN ĐỔI MODEL SANG ESP32-S3 (QUANTIZE)
> **Mục tiêu:** Chuyển model Keras (`float32`) → TFLite INT8 (`int8`) mà không làm suy giảm độ chính xác.

#### 💡 Chiến lược áp dụng trong Giai đoạn 2:
1. **Bí quyết 3 — Lượng tử hóa số nguyên 8-bit (Full INT8 Quantization):** Ép toàn bộ Weights và Activations từ 32-bit float về 8-bit int. Giảm dung lượng 75% (700KB → 160KB) và tăng tốc độ tính toán gấp 3–4 lần trên vi điều khiển.
2. **Kiểm thử mô phỏng MCU trên Laptop (Offline Simulation):** Viết script chạy bộ suy luận TFLite INT8 ngay trên Laptop để đối chiếu sai số với Keras trước khi nạp xuống mạch phần cứng.

#### 📋 Các đầu việc thực hiện (Action Items):
- [x] **2.1 — Thực hiện lượng tử hóa INT8:**
  - [x] Chạy `quantize_qat_int8.py` với Representative Dataset lấy từ ảnh khuôn mặt thực tế để hiệu chuẩn phân phối giá trị activation.
  - [x] Xuất ra file `training_tinyml/weights/tinyface_int8.tflite`.
  - [x] Kiểm tra độ tương đồng vector embedding giữa Keras và INT8 TFLite (đã đạt: **Cosine Similarity = 99.87%**).
  - [x] Nếu sai lệch > 5%, kích hoạt Quantization-Aware Training (QAT) trong quá trình train lại.
- [x] **2.2 — Kiểm thử mô hình INT8 trên Laptop:**
  - [x] Tích hợp bộ đọc TFLite Interpreter vào module `FaceRecognizer` trên Laptop.
  - [x] So sánh kết quả nhận diện trực tiếp bằng TFLite INT8 so với Keras float32.
  - [x] Đảm bảo file `.tflite` hoàn toàn sẵn sàng và chuẩn xác trước khi bước vào lập trình nhúng.

> [!IMPORTANT]
> **Ràng buộc kỹ thuật trên ESP32-S3:**
> - Tensor Arena memory ≤ 136KB (bắt buộc vừa khít trong Internal SRAM).
> - Kích thước file model ≤ 160KB.
> - Chỉ chứa 5 Ops chuẩn: `CONV_2D`, `DEPTHWISE_CONV_2D`, `ADD`, `FULLY_CONNECTED`, `RESHAPE`.

---

### GIAI ĐOẠN 3: TRIỂN KHAI FIRMWARE ESP32-S3
> **Mục tiêu:** Port toàn bộ pipeline nhận diện (Face Detector + Face Recognizer) xuống ESP32-S3 N16R8. Sử dụng luồng video thô nhận từ Laptop qua Wi-Fi để thay thế Camera vật lý, mọi tính toán AI diễn ra trên chip.

#### 💡 Chiến lược áp dụng trong Giai đoạn 3:
1. **Bí quyết 4 — Tăng tốc phần cứng AI (ESP-NN / Vector Instructions):** Tận dụng tập lệnh vector SIMD chuyên dụng trên nhân Xtensa LX7 của ESP32-S3 để xử lý các phép nhân ma trận INT8 song song cho cả 2 mô hình (Dò mặt + Nhận diện).
2. **Chiến lược phân bổ bộ nhớ kép (Dual-Memory Allocation Strategy):**
   * *Internal SRAM (~512KB, tốc độ cao):* Dành riêng cho Stack/Heap hệ thống, FreeRTOS và các task khắt khe về độ trễ.
   * *External PSRAM (8MB, tốc độ cao Octal SPIRAM):* Cấp phát 2 vùng Tensor Arena riêng biệt: `detector_arena` (~1.5MB) và `recognizer_arena` (~768KB). Đồng thời chứa bộ đệm nhận gói tin TCP/JPEG (`packet_buffer` ~64KB).
3. **Chiến lược Face Detector gọn nhẹ On-Device (Dual-Model TinyML Pipeline):** 
   * Sử dụng mô hình Face Detector INT8 siêu nhẹ (BlazeFace / Ultra-Light INT8 input 128x128) chạy trực tiếp trên TFLite Micro ESP32 để tìm tọa độ Bounding Box.
   * Viết thuật toán C++ Fast Crop & Bilinear Interpolation trên PSRAM để cắt và thu nhỏ vùng mặt về đúng chuẩn $64 \times 64$ Grayscale nạp sang mô hình Ghost-TinyFace mà không phụ thuộc vào thư viện OpenCV. Đặc biệt, xây dựng bộ giả lập Unified BlazeFace Emulator trên Laptop bằng Python để đồng bộ tuyệt đối thuật toán cắt ảnh, giúp triệt tiêu hoàn toàn lệch pha miền dữ liệu (Domain Shift).
4. **Kiến trúc xử lý đa luồng (Dual-Core Asymmetric Processing):**
   * **Core 0:** Chuyên chạy suy luận 2 mô hình AI tuần tự (Detector $128 \times 128 \rightarrow$ Fast Crop $64 \times 64 \rightarrow$ Recognizer $64 \times 64 \rightarrow$ Cosine Matching).
   * **Core 1:** Chuyên xử lý ngoại vi (Nhận luồng ảnh JPEG thô qua TCP Socket, điều khiển còi Buzzer, in ra Serial, ghi log).

#### 📋 Các đầu việc thực hiện (Action Items):
- [x] **3.1 — Khởi tạo dự án Firmware C++ (ESP-IDF / PlatformIO):**
  - [x] Tạo thư mục `firmware_esp32/` cấu hình PlatformIO với framework ESP-IDF/Arduino.
  - [x] Chuyển đổi `tinyface_int8.tflite` thành mảng C array (`model_data.h`) và nạp vào Flash.
  - [x] Khởi tạo TFLite Micro với `MicroMutableOpResolver` (chỉ đăng ký đúng 5 ops cần thiết).
  - [x] Nhúng CSDL vector đặc trưng các thành viên vào file `face_database.h`.
- [x] **3.2 — Thiết lập hạ tầng truyền nhận video TCP & Bộ đệm đa nhân:**
  - [x] Viết script `ip_camera_streamer.py` trên Laptop để stream luồng webcam qua TCP Socket (Port 12345).
  - [x] ESP32 cấu hình Wi-Fi STA, thiết lập TCP Server, liên tục nhận payload JPEG vào PSRAM (`packet_buffer`).
  - [x] Thiết lập Mutex (`image_mutex`) đồng bộ khung hình an toàn giữa Core 1 (Mạng) và Core 0 (AI).
- [x] **3.3 — Triển khai Mô hình Face Detector On-Device xuống ESP32 (Chuyển giao 100% từ Laptop sang ESP32):**
  - [x] **3.3a — Chuẩn bị & Lượng tử hóa Model Detector (BlazeFace / Ultra-Light INT8):**
    - [x] Lựa chọn/xuất mô hình Face Detector INT8 siêu nhẹ (kích thước input $128 \times 128$, dung lượng Flash $\le 200\text{KB}$).
    - [x] Dùng công cụ Python chuyển đổi thành mảng C array `detector_model_data.h` để nhúng vào ROM ESP32. (Đã dùng bản INT8)
  - [x] **3.3b — Tích hợp Dual-Interpreter TFLite Micro trên Core 0:**
    - [x] Cấp phát vùng nhớ `detector_tensor_arena` (~350KB) trên PSRAM.
    - [x] Khởi tạo Interpreter thứ 2 cho Detector chạy trước mô hình Recognizer.
    - [x] Cài đặt thuật toán giải mã Bounding Box & NMS (Non-Maximum Suppression) siêu nhẹ bằng C++ thuần.
  - [x] **3.3c — Xây dựng Module C++ Fast Crop & Grayscale Normalization:**
    - [x] Lấy tọa độ Bounding Box từ Detector $\rightarrow$ tự động trích xuất vùng mặt từ buffer ảnh thô.
    - [x] Thực hiện nội suy song tuyến (Bilinear Interpolation) thu phóng về đúng kích thước chuẩn $64 \times 64$ Grayscale.
    - [x] Chuẩn hóa pixel `(gray - 127.5) / 128.0` nạp trực tiếp vào mảng `face_tensor` của Model Recognizer.
  - [x] **3.3d — Tinh gọn Laptop Streamer thành Dumb IP Camera:**
    - [x] Gỡ bỏ hoàn toàn OpenCV Detector / MediaPipe trên `ip_camera_streamer.py`.
    - [x] Chỉ đọc webcam máy tính và gửi thẳng luồng JPEG thô $240 \times 240$ xuống ESP32.
- [x] **3.4 — Triển khai luồng nhận diện và so khớp (Recognizer Pipeline):**
  - [x] Đưa ảnh 64x64 vào Tensor Arena → chạy `Invoke()` để sinh ra vector 128-D.
  - [x] Tính Cosine Similarity (phép tính tích vô hướng dot product) giữa vector vừa tạo với `face_database.h`.
  - [x] Cài đặt bộ đếm `Temporal Voting` (chốt kết quả khi trùng khớp 3 frames liên tiếp).
  - [x] Lưu lịch sử điểm danh vào bộ nhớ Flash (SPIFFS: file `/attendance.csv`) kèm cơ chế Cooldown chống spam.
- [x] **3.5 — Kết nối thiết bị ngoại vi (Buzzer, Serial, LED):**
  - [x] Điều khiển còi Buzzer phát âm thanh (1 tiếng bip ngắn = thành công, 2 tiếng bip dài = từ chối).
  - [x] Điều khiển LED báo hiệu trạng thái (Xanh = Đã nhận diện, Đỏ = Người lạ).
- [x] **3.6 — Tối ưu đa nhân FreeRTOS (Dual-Core Asymmetric):**
  - [x] Tách task AI sang Core 0 (`AITask` Stack 32KB) và task Ngoại vi/Mạng sang Core 1 (`NetDisplayTask` Stack 8KB).
  - [x] Đồng bộ truyền nhận dữ liệu giữa 2 Core bằng Mutex Semaphore (`image_mutex`).

---

### GIAI ĐOẠN 4: TỐI ƯU VÀ HOÀN THIỆN SẢN PHẨM
> **Mục tiêu:** Hệ thống đạt độ ổn định cao, tốc độ suy luận nhanh, chống gian lận và sẵn sàng vận hành liên tục.

#### 💡 Chiến lược áp dụng trong Giai đoạn 4:
1. **Chiến lược chống giả mạo (Anti-Spoofing / Liveness Detection nhẹ):** Yêu cầu người dùng chớp mắt hoặc phát hiện chuyển động vi mô của khuôn mặt trực tiếp trên thuật toán của ESP32.
2. **Chiến lược đăng ký tại chỗ (On-Device Enrollment):** Cho phép bấm nút vật lý trên thiết bị ESP32 để chụp luồng ảnh từ Wi-Fi, trích xuất vector người mới và lưu thẳng vào bộ nhớ Flash.

#### 📋 Các đầu việc thực hiện (Action Items):
- `[ ]` **4.1 — Tối ưu hóa hiệu năng và tốc độ:**
  - `[ ]` Đo thời gian suy luận trên chip — mục tiêu: **Thời gian inference (Detector + Recognizer) ≤ 400ms/khuôn mặt**.
  - `[ ]` Bật cờ tối ưu hóa biên dịch `-O3` và thư viện tăng tốc `esp-nn`.
  - `[ ]` Đo đạc rò rỉ bộ nhớ (Memory Leak), đảm bảo Heap và PSRAM ổn định sau 24h chạy liên tục.
- `[ ]` **4.2 — Tích hợp tính năng chống giả mạo (Liveness Detection):**
  - `[ ]` Cài đặt thuật toán kiểm tra chớp mắt (Eye Blink Detection) nhẹ trên ESP32.
  - `[ ]` Từ chối điểm danh nếu phát hiện khuôn mặt là ảnh phẳng cố định.
- `[ ]` **4.3 — Xây dựng tính năng Đăng ký người dùng mới trực tiếp trên ESP32:**
  - `[ ]` Bổ sung nút nhấn "Enroll Mode".
  - `[ ]` Khi nhấn nút: Bắt 10 ảnh liên tiếp từ luồng Wi-Fi → ESP32 tự trích xuất 10 vector → tính vector trung bình → lưu vào SPIFFS.
- `[ ]` **4.4 — Kiểm thử thực tế toàn diện (Field Testing):**
  - `[ ]` Kiểm thử với ≥ 10 người dùng khác nhau trong các điều kiện ánh sáng (phòng tối, đèn tuýp, ngoài trời).
  - `[ ]` Kiểm thử với người lạ để xác nhận tỷ lệ False Accept Rate = 0%.
  - `[ ]` Đóng vỏ hộp thiết bị hoàn chỉnh (3D Print case) và bàn giao sản phẩm.

---

## 📍 TRẠNG THÁI TIẾN ĐỘ HIỆN TẠI

| Giai đoạn | Nội dung chính | Trạng thái |
|---|---|---|
| **1.1** | Huấn luyện Knowledge Distillation trên Colab | ✅ **Hoàn thành** |
| **1.2** | Kiểm thử và tinh chỉnh nhận diện trên Laptop | ✅ **Hoàn thành (Đã test thành công với Unknown)** |
| **1.3** | Thu thập dữ liệu đăng ký theo tỷ lệ vàng | ✅ **Hoàn thành** |
| **2.x** | Lượng tử hóa INT8 và kiểm thử mô phỏng | ✅ **Hoàn thành (Đã tạo `.tflite`, Cosine Sim 99.87%)** |
| **3.1** | Cấu hình PlatformIO, nạp model_data.h & face_database.h | ✅ **Hoàn thành** |
| **3.2** | Hạ tầng mạng TCP & truyền nhận ảnh đa nhân | ✅ **Hoàn thành** |
| **3.3** | Đưa Face Detector INT8 xuống ESP32 (Chạy 100% On-Device) | ✅ **Hoàn thành (Đã khử Domain Shift bằng Python Emulator)** |
| **3.4** | Pipeline Nhận diện, Cosine Matching, Temporal Voting & SPIFFS Log | ✅ **Hoàn thành (Đã quy hoạch 768KB PSRAM cho Recognizer)** |
| **3.5** | Tích hợp LCD ILI9341, Buzzer & LED trạng thái | 🔄 **Đang thực hiện** |
| **3.6** | Phân tách đa nhân FreeRTOS (Dual-Core Asymmetric) | ✅ **Hoàn thành** |
| **4.x** | Tối ưu tốc độ, Anti-Spoofing và kiểm thử 24h | ⏳ **Chờ thực hiện** |

---

## 🔑 NGUYÊN TẮC BẤT BIẾN

1. **Laptop-First, ESP32-Final:** Mọi thay đổi về thuật toán và mô hình phải được kiểm chứng đạt độ chính xác cao trên Laptop trước khi nạp xuống ESP32.
2. **Giữ nghiêm ngặt ràng buộc phần cứng MCU:**
   * Kích thước ảnh đầu vào luôn cố định **64×64 Grayscale** (1 kênh).
   * Vector đặc trưng luôn là **128 chiều** (128-D L2-normalized).
   * Chuẩn hóa pixel: `(pixel - 127.5) / 128.0`.
   * So khớp danh tính: `Cosine Similarity` (tích vô hướng dot product).
3. **Đo lường bằng số liệu thực tế:** Mọi đánh giá phải dựa trên tỷ lệ Accuracy, FAR/FRR, thời gian suy luận (ms) và mức tiêu thụ RAM thay vì cảm tính.
