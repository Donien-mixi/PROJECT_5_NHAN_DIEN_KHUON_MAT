# 🚀 ROADMAP: HỆ THỐNG ĐIỂM DANH NHẬN DIỆN KHUÔN MẶT TRÊN ESP32-S3

> **Mục tiêu cuối cùng:** Một hệ thống nhận diện khuôn mặt chạy **toàn bộ thuật toán (Phát hiện + Nhận diện) trên MCU ESP32-S3 N16R8**. Do giới hạn kinh phí, Laptop sẽ đóng vai trò như một IP Camera, truyền luồng JPEG qua Wi-Fi xuống ESP32. ESP32 sẽ tự thực hiện phát hiện khuôn mặt, so khớp danh tính và báo kết quả **duy nhất bằng 2 LED (xanh = đã nhận diện, đỏ = người lạ) + Buzzer (1 bip ngắn = success, 2 bip dài = reject) + in ra Serial — tuyệt đối không dùng màn hình LCD**.

> **Phương châm cốt lõi:** Phát triển và hoàn thiện 100% trên Laptop trước → Khi đạt độ chính xác mong muốn → Triển khai xuống ESP32-S3.

---

## 📊 TỔNG QUAN BỘ KHUNG HIỆN TẠI

### Cấu trúc thư mục

```
PROJECT_5_DIEM_DANH_KHUON_MAT/
├── host_laptop/                    # 🖥️ Ứng dụng chạy trên Laptop
│   ├── main.py                     #    Điều phối chính (Camera → Detect → Recognize → DB + HUD)
│   ├── enroll_tool.py              #    Thu thập ảnh đăng ký (20 ảnh/người, chặn ảnh tối/cháy)
│   ├── ip_camera_streamer.py       #    Dumb IP Camera — gửi JPEG 128×128 qua TCP 12345 (Buffer=1, TCP_NODELAY)
│   ├── convert_tflite_to_c.py      #    Đổi .tflite → .h (C array)
│   ├── face_detection_short_range.tflite  # BlazeFace float (debug, không dùng chính)
│   ├── core/
│   │   └── vision_utils.py         #    Center-crop, JPEG/RGB565 round-trip, Bilinear 128→64, HE
│   ├── detector/
│   │   ├── blazeface_esp32.py      #    Unified BlazeFace Emulator INT8 128×128 (đồng bộ ESP32)
│   │   ├── face_detection_short_range.tflite
│   │   └── face_detection_short_range_int8.tflite  # BlazeFace FULL INT8 PTQ (~183KB)
│   ├── recognizer/
│   │   └── face_recognizer.py      #    Ghost-TinyFace INT8 → 128-D + cosine MAX-SIM + Temporal Voting
│   ├── database/
│   │   └── db_manager.py           #    SQLite attendance.db + cooldown 30s
│   └── ui/
│       └── hud_renderer.py         #    Vẽ HUD OpenCV (chỉ Laptop, ESP32 không có LCD)
│
├── firmware_esp32/                 # 🔌 Firmware ESP32-S3 (Arduino IDE, N16R8)
│   ├── firmware_esp32.ino          #    Nhạc trưởng: 2 task FreeRTOS (AITask Core0 + NetTask Core1), khóa CPU 240MHz
│   ├── ai_config.h                 #    Tham số đồng bộ (RAW 128, FACE 64, THRESH 0.70/0.80, VOTES 3, Arenas PSRAM: Det 1MB, Rec 512KB)
│   ├── ai_face_detector.h/.cpp     #    BlazeFace INT8 128×128 + decode box + Bilinear crop (~1.95s SIMD)
│   ├── ai_face_recognizer.h/.cpp   #    Ghost-TinyFace INT8 64×64 Model V3 + HE LUT + cosine MAX-SIM (~1.30s SIMD)
│   ├── esp_nn_glue.h/.cpp          #    Tăng tốc SIMD Xtensa LX7 cho Conv2D/DepthwiseConv + Bypass lỗi s8pad
│   ├── esp_nn/                     #    Thư viện esp-nn v1.3 SIMD assembly chính hãng Espressif
│   ├── wifi_udp_server.h/.cpp      #    TCP Server 12345 + PSRAM packet buffer + Overwrite liên tục khử trễ frame
│   ├── image_decoder.h/.cpp        #    TJpg_Decoder → RGB565 128×128
│   ├── model_data.h                #    C array Ghost-TinyFace INT8 Model V3 (~160KB)
│   ├── detector_model_data.h       #    C array BlazeFace INT8 (~183KB)
│   └── face_database.h             #    16 templates/người × 128-D (MAX-SIM, per-identity threshold 0.75)
│
├── training_tinyml/                # 🧠 Huấn luyện & export
│   ├── models/
│   │   └── ghost_tinyface.py       #    Ghost-TinyFace 64×64 grayscale → 128-D
│   ├── train_distillation.py       #    Huấn luyện Model V3: KD + ArcFace (s=30, m=0.3) + illumination KD (Colab T4 GPU)
│   ├── download_casia_dataset.py   #    Trích xuất CASIA-WebFace siêu tốc từ RecordIO (~11,000 ảnh/giây)
│   ├── download_lfw_dataset.py     #    Tải LFW ~13k ảnh (dự phòng)
│   ├── evaluate_model.py           #    Accuracy / FAR / TAR + Identification MAX-SIM 16 templates
│   ├── generate_embeddings.py      #    MAX-SIM 16 templates/người (trimmed 80%, threshold 0.70/0.75)
│   ├── update_face_database.py     #    Quét registered_faces → face_database.json/.h + ai_config.h (Zero-Retraining)
│   ├── quantize_qat_int8.py        #    Keras → TFLite INT8 Ghost Model V3 (~160KB, representative có HE)
│   ├── quantize_detector_int8.py   #    PTQ BlazeFace float16 → FULL INT8 (~183KB, giữ 2-output)
│   ├── export_config.py            #    Sinh ai_config.h (Det 1MB + Rec 512KB + Packet 32KB)
│   ├── colab_exports/              #    Bản copy model_data.h / face_database.h từ Colab khi train xong
│   └── weights/
│       ├── tinyface_backbone.keras #    Keras float32 Model V3 (~696KB)
│       ├── tinyface_int8.tflite    #    Ghost INT8 64×64 Model V3 (~160KB INT8)
│       └── face_detection_short_range_int8.tflite  # BlazeFace INT8 128×128 (~183KB)
│
├── data/                           # 💾 Dữ liệu
│   ├── registered_faces/           #    nhien / thao / toan — mỗi người 20 PNG 64×64 grayscale
│   ├── face_database.json          #    JSON trung gian (16 templates/người, threshold 0.75)
│   └── attendance.db               #    SQLite log điểm danh
│
├── colab.zip                       # ☁️ Gói 1-lệnh train Model V3 trên Colab (~954KB)
├── HUONG_DAN_COLAB_TRAIN_V3.md     #    Hướng dẫn train Model V3 chuẩn trên Google Colab (CASIA-WebFace)
├── kich_hoat_moi_truong.md         #    Lệnh nhanh conda + flash + stream
├── mo_ta_project.md                #    Mô tả chi tiết từng file và bí quyết cốt lõi
├── ket_qua_esp32.txt               #    Log thực tế kiểm thử phần cứng ESP32-S3 mới nhất (Model V3 verified)
└── requirements.txt
```

### Thông số kỹ thuật và hợp đồng xử lý hiện tại

| Thành phần | Giá trị |
|---|---|
| **Phần cứng MCU** | ESP32-S3 N16R8 (16MB Flash, 8MB Octal PSRAM), khóa xung nhịp CPU 2 nhân ở mức trần **240MHz** |
| **Tốc độ thực tế đo trên board (Phase 4.1)** | Detector BlazeFace INT8: **~1.95s** (nhanh 10.5x). Recognizer Ghost-TinyFace INT8: **~1.30s** (nhanh 4x). Chu kỳ nhận diện Box-Reuse (`cy: REUSE`): **~1.32s/frame** |
| **Độ trễ truyền nhận luồng ảnh** | **< 50ms** (khử hoàn toàn trễ 5s cũ qua cơ chế Core 1 ghi đè liên tục `g_frame_buffer` + tắt Nagle & Buffer trên Laptop) |
| **Mô hình Recognizer** | Ghost-TinyFace Model V3 (GhostNet Bottleneck) 64×64 Grayscale → 128-D (~160KB INT8). Chạy Hybrid SIMD + Reference bypass lỗi `s8pad` |
| **Mô hình Detector** | BlazeFace FULL INT8 128×128 RGB (PTQ từ float16, ~183KB). Chạy SIMD 100% |
| **Input truyền qua mạng** | JPEG 128×128, payload phải ≤ `PACKET_BUFFER_SIZE` 32KB, có header độ dài 4 byte little-endian, TCP port 12345 |
| **Input sau giải mã** | ESP32: RGB565 128×128; Laptop phải mô phỏng cùng đường đi RGB565 trước khi so sánh |
| **Output** | Vector 128-D; L2-normalize trước khi matching |
| **Face Detector** | BlazeFace INT8 trên ESP32 / Unified BlazeFace Emulator (Bilinear đồng bộ) trên Laptop |
| **Alignment hiện tại** | Crop vuông theo bounding box + Bilinear thủ công 128→64 + **Histogram Equalization (HE) khử nhạy ánh sáng** (LUT số nguyên đồng bộ bit-exact Python↔C++) |
| **Matching** | Cosine Similarity MAX-SIM trên 16 templates/người (đồng bộ Laptop ↔ ESP32, không dùng centroid đơn) |
| **Threshold** | Global Threshold: **0.70**, Per-Identity Threshold Cap: **0.75** (cả `nhien`, `thao`, `toan` = 0.75), BlazeFace conf **0.80**. Đã hiệu chuẩn thực tế trên phần cứng: loại bỏ triệt để hiện tượng từ chối nhầm khi nghiêng mặt nhẹ ở ngưỡng 0.80 cũ |
| **Chống nhiễu** | Temporal Voting (3 frame, chính sách pause-on-Unknown) + Box-Reuse (tự hủy cache khi Unknown ≥ 2) |
| **Training** | Model V3: Chưng cất tri thức (Teacher SFace 112×112 $\rightarrow$ Student Ghost-TinyFace 64×64) **+ ArcFace loss** ($s=30.0, m=0.30$) + **Illumination-invariance KD** (HE LUT 256-bin); thêm người mới hoàn toàn không cần train lại (Zero-Retraining) |
| **Dataset** | CASIA-WebFace (28,102 ảnh / 1,198 danh tính trích xuất từ tập gốc 494k ảnh bằng RecordIO engine siêu tốc ~11,000 ảnh/giây) |
| **CSDL người dùng** | Sinh từ ảnh đăng ký 20 ảnh/người (tỷ lệ 70/30 thẳng/nghiêng); lưu 16 templates/người (ảnh sạch 80%) — matching MAX-SIM đồng bộ Laptop và ESP32 |

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

### GIAI ĐOẠN 1: HOÀN THIỆN E2E TRÊN LAPTOP
> **Mục tiêu:** Kiểm chứng hoàn chỉnh `JPEG → decode → detector → crop → recognizer → matching` trên Laptop bằng input mô phỏng đúng ESP32. Không dùng tuyên bố “100%”; phải báo cáo Accuracy, FAR, FRR, số mẫu và điều kiện đo.

#### 💡 Chiến lược áp dụng trong Giai đoạn 1:
1. **Bí quyết 1 — Chưng cất tri thức (Knowledge Distillation) + ArcFace:** Dùng SFace làm teacher, Ghost-TinyFace làm student, kết hợp **ArcFace loss** (ép margin góc giữa các danh tính CASIA-WebFace — giải quyết bài toán 2 người bị nhầm nhau) và **illumination-invariance KD** (teacher nhúng ảnh sạch, student nhận ảnh augment tối/sáng + HE — khử nhạy ánh sáng). Độ chính xác phải được đo bằng tập đánh giá độc lập.
2. **Bí quyết 2 — Hợp đồng preprocessing:** Laptop và ESP32 phải nhận cùng JPEG 128×128, giải mã về cùng biểu diễn pixel chuẩn (RGB565), chạy cùng công thức decode box, cùng crop vuông Bilinear, cùng **HE (LUT số nguyên)** và cùng normalize. Affine 5 điểm chỉ được thêm khi đã có implementation tương ứng ở cả Python và C++.
3. **Chiến lược Dataset 2 bước:**
   * *Bước 1 (Prototype ban đầu):* Huấn luyện trên tập **LFW (~13.000 ảnh / ~5.700 người)** trên Colab GPU để xác thực nhanh toàn bộ luồng.
   * *Bước 2 (Chính thức - Model V3):* Nâng cấp chính thức sang tập **CASIA-WebFace (~500.000 ảnh / ~10.500 người)** với **Online Dynamic Augmentation (~1.75 triệu biến thể ảnh)** trên Google Colab qua `HUONG_DAN_COLAB_TRAIN_V3.md` để đạt độ phân tách danh tính và độ bền bỉ cao nhất.
4. **Quy tắc thu thập ảnh đăng ký (Tỷ lệ vàng 70/30):** Chụp 15–20 ảnh/người với 70% góc nhìn thẳng tự nhiên và 30% góc nghiêng nhẹ 10–15° để vector đại diện bao quát mọi trường hợp đời thực.

#### 📋 Các đầu việc thực hiện (Action Items):
- [x] **1.1 — Huấn luyện mô hình trên Google Colab (có baseline & Model V3):**
  - [x] Chạy `train_distillation.py` với tập LFW trên Colab GPU (phiên bản khởi đầu v1).
  - [x] Nâng cấp chính thức **Model V3 (CASIA-WebFace + ArcFace + Illumination KD)**:
    - [x] Tích hợp engine RecordIO Pure Python siêu tốc trích xuất 28,102 ảnh / 1,198 danh tính từ `faces_webface_112x112.zip` (~11,000 ảnh/s).
    - [x] Huấn luyện hoàn tất trên Google Colab T4 GPU: Teacher SFace 112×112 $\rightarrow$ Student Ghost-TinyFace 64×64 kết hợp ArcFace ($s=30.0, m=0.30$) và Illumination-invariance KD.
    - [x] Tải trọng số `tinyface_backbone.keras` (~696KB) về thư mục `training_tinyml/weights/`.
  - [x] Đánh giá định lượng trên `evaluate_model.py`: **Top-1 Identification Accuracy đạt 100.0%** (12/12 probe holdout, 20/20 toàn bộ ảnh đăng ký cho mỗi người), FRR@0.70 = 0.0%.
- [x] **1.2 — Vận hành và tinh chỉnh pipeline nhận diện trên Laptop:**
  - [x] Tạo lại CSDL vector bằng `generate_embeddings.py` dựa trên Model V3 (`GLOBAL_THRESHOLD = 0.70`, `PER_ID_CAP = 0.75`).
  - [x] Chạy `host_laptop/main.py` bằng frame đã qua cùng preprocessing JPEG/RGB565 với ESP32.
  - [x] Khóa ngưỡng chuẩn hóa: 0.70 toàn cục, 0.75 ngưỡng trần từng danh tính (đồng bộ Laptop và ESP32).
  - [x] Đo Accuracy, TAR/TPR, FAR, FRR và độ trễ của toàn bộ detector + recognizer + matching — đã đo qua `evaluate_model.py` + log E2E laptop/ESP32.
- [x] **1.3 — Chuẩn hóa dữ liệu khuôn mặt đăng ký:**
  - [x] Dùng `enroll_tool.py` chụp lại bộ ảnh 20 tấm theo tỷ lệ vàng (70% thẳng, 30% nghiêng) cho tất cả người dùng trong hệ thống — nhien/thao/toan 20/20/20.
  - [x] Tái tạo lại `face_database.json` và `face_database.h`, kiểm tra số ảnh, chiều vector và L2 norm — 16 templates/người, 128-D L2-norm, threshold 0.75/người.

---

### GIAI ĐOẠN 2: CHUYỂN ĐỔI MODEL SANG ESP32-S3 (QUANTIZE)
> **Mục tiêu:** Chuyển model Keras (`float32`) → TFLite INT8 (`int8`) mà không làm suy giảm độ chính xác.

#### 💡 Chiến lược áp dụng trong Giai đoạn 2:
1. **Bí quyết 3 — Lượng tử hóa số nguyên 8-bit:** Export và kiểm tra riêng từng model. Script hiện tại chủ yếu export Ghost-TinyFace; BlazeFace có artifact và bước audit độc lập.
2. **Kiểm thử mô phỏng MCU trên Laptop:** Chạy TFLite INT8 với đúng input quantization, kiểm tra operator list, input/output tensor, và so sánh Keras với INT8 trên cùng bộ mẫu trước khi tạo C array.

#### 📋 Các đầu việc thực hiện (Action Items):
- [x] **2.1 — Thực hiện và kiểm tra lượng tử hóa INT8:**
  - [x] Chạy export recognizer Model V3 với Representative Dataset có HE đồng bộ: sinh ra `tinyface_int8.tflite` (~160KB).
  - [x] Kiểm tra artifact detector INT8 bằng quy trình độc lập: `face_detection_short_range_int8.tflite` (~183KB, maxdiff < 1e-3).
  - [x] Kiểm tra độ tương đồng vector embedding giữa Keras và INT8 TFLite trên bộ mẫu cố định: giữ nguyên 100% Top-1 Identification.
  - [x] Chuyển đổi thành C array `model_data.h` và `detector_model_data.h` sẵn sàng nhúng vào firmware ESP32.
- [x] **2.2 — Kiểm thử mô hình INT8 trên Laptop:**
  - [x] Tích hợp bộ đọc TFLite Interpreter vào module `FaceRecognizer` trên Laptop.
  - [x] So sánh kết quả nhận diện trực tiếp bằng TFLite INT8 so với Keras float32 trên bộ mẫu cố định — INT8 giữ nguyên Identification, sai số <1e-3.
  - [x] Đảm bảo cả hai file `.tflite` có input/output/quantization/operator list đã được ghi nhận và tương thích với TFLite Micro — `ai_face_detector.cpp:64` 7 ops / `ai_face_recognizer.cpp:49` 6 ops.

> [!IMPORTANT]
> **Ràng buộc kỹ thuật trên ESP32-S3 (đã chuẩn hóa & nghiệm thu trên phần cứng):**
> - Tensor Arenas trên PSRAM: `detector_arena` = 1MB (thực tế sử dụng ~602KB), `recognizer_arena` = 512KB (thực tế sử dụng ~186KB), `packet_buffer` = 32KB (128×128×2).
> - Operator resolver: Recognizer (Ghost Model V3): `CONV_2D, DEPTHWISE_CONV_2D, ADD, CONCATENATION, FULLY_CONNECTED, RESHAPE` (6 ops). Detector (BlazeFace INT8): `CONV_2D, DEPTHWISE_CONV_2D, ADD, CONCATENATION, MAX_POOL_2D, PAD, RESHAPE` (7 ops).

---

### GIAI ĐOẠN 3: TRIỂN KHAI FIRMWARE ESP32-S3
> **Mục tiêu:** Port toàn bộ pipeline nhận diện (Face Detector + Face Recognizer) xuống ESP32-S3 N16R8. Sử dụng luồng JPEG nhận từ Laptop qua Wi-Fi để thay thế Camera vật lý, mọi tính toán AI diễn ra trên chip.

#### 💡 Chiến lược áp dụng trong Giai đoạn 3:
1. **Bí quyết 4 — Tăng tốc phần cứng AI (ESP-NN / Vector Instructions):** Tận dụng tập lệnh vector SIMD chuyên dụng trên nhân Xtensa LX7 của ESP32-S3 để xử lý các phép nhân ma trận INT8 song song cho cả 2 mô hình (Dò mặt + Nhận diện).
2. **Chiến lược phân bổ bộ nhớ kép (Dual-Memory Allocation Strategy):**
   * *Internal SRAM (~512KB, tốc độ cao):* Dành riêng cho Stack/Heap hệ thống, FreeRTOS và các task khắt khe về độ trễ.
   * *External PSRAM (8MB, tốc độ cao Octal SPIRAM):* Cấp phát 2 vùng Tensor Arena riêng biệt: `detector_arena` (~2.5MB) và `recognizer_arena` (~1.75MB). Đồng thời chứa bộ đệm nhận gói tin TCP/JPEG (`packet_buffer` 32KB cho 128×128×2).
3. **Chiến lược Face Detector gọn nhẹ On-Device (Dual-Model TinyML Pipeline):** 
   * Sử dụng mô hình Face Detector INT8 BlazeFace 128×128 (~183KB) chạy trực tiếp trên TFLite Micro ESP32 để tìm tọa độ Bounding Box.
   * Viết thuật toán C++ Fast Crop & Bilinear Interpolation trên PSRAM để cắt và thu nhỏ vùng mặt về đúng chuẩn $64 \times 64$ Grayscale nạp sang mô hình Ghost-TinyFace mà không phụ thuộc vào thư viện OpenCV.
4. **Kiến trúc xử lý đa luồng (Dual-Core Asymmetric Processing):**
   * **Core 0:** Chuyên chạy suy luận 2 mô hình AI tuần tự (Detector $128 \times 128 \rightarrow$ Fast Crop $64 \times 64 \rightarrow$ Recognizer $64 \times 64 \rightarrow$ Cosine Matching).
   * **Core 1:** Chuyên xử lý ngoại vi và mạng (khởi tạo Wi-Fi/TCP, nhận JPEG, giải mã, điều khiển 2 LED + Buzzer, in Serial, ghi log — không có LCD).

#### 📋 Các đầu việc thực hiện (Action Items):
- [x] **3.1 — Khởi tạo và nghiệm thu dự án Firmware C++ (ESP-IDF / PlatformIO):**
  - [x] Cấu hình dự án firmware build ổn định trên Arduino IDE (ESP32 Arduino core 3.x).
  - [x] Chuyển đổi `tinyface_int8.tflite` thành mảng C array (`model_data.h`) và nạp vào Flash.
  - [x] Khởi tạo TFLite Micro với `MicroMutableOpResolver` và đăng ký đúng toàn bộ operator cần thiết.
  - [x] Nhúng CSDL vector đặc trưng các thành viên vào file `face_database.h`.
- [x] **3.2 — Thiết lập và nghiệm thu hạ tầng truyền nhận video TCP & bộ đệm đa nhân:**
  - [x] Viết script `ip_camera_streamer.py` trên Laptop để stream luồng webcam qua TCP Socket (Port 12345).
  - [x] ESP32 cấu hình Wi-Fi STA, thiết lập TCP Server, liên tục nhận payload JPEG vào PSRAM (`packet_buffer`).
  - [x] Thiết lập Mutex (`image_mutex`) đồng bộ khung hình an toàn giữa Core 1 (Mạng) và Core 0 (AI).
- [x] **3.3 — Triển khai và nghiệm thu Face Detector On-Device (sau golden test trên Laptop):**
  - [x] **3.3a — Chuẩn bị & Lượng tử hóa Model Detector (BlazeFace INT8):**
    - [x] Xuất mô hình Face Detector BlazeFace FULL INT8 128×128 (~183KB).
    - [x] Chuyển đổi thành mảng C array `detector_model_data.h` để nhúng vào ROM ESP32.
  - [x] **3.3b — Tích hợp Dual-Interpreter TFLite Micro trên Core 0:**
    - [x] Cấp phát vùng nhớ `detector_tensor_arena` (~2.5MB) trên PSRAM.
    - [x] Khởi tạo Interpreter thứ 2 cho Detector chạy trước mô hình Recognizer.
    - [x] Cài đặt thuật toán giải mã Bounding Box, single-face argmax score cao nhất.
  - [x] **3.3c — Xây dựng Module C++ Fast Crop & Grayscale Normalization:**
    - [x] Lấy tọa độ Bounding Box từ Detector $\rightarrow$ tự động trích xuất vùng mặt từ buffer ảnh thô.
    - [x] Thực hiện nội suy song tuyến (Bilinear Interpolation) thu phóng về đúng kích thước chuẩn $64 \times 64$ Grayscale.
    - [x] Chuẩn hóa pixel `(gray - 127.5) / 128.0` nạp trực tiếp vào mảng `face_tensor` của Model Recognizer.
  - [x] **3.3d — Tinh gọn Laptop Streamer thành Dumb IP Camera:**
    - [x] Gỡ bỏ hoàn toàn OpenCV Detector / MediaPipe trên `ip_camera_streamer.py`.
    - [x] Chỉ đọc webcam máy tính và gửi thẳng luồng JPEG thô $128 \times 128$ xuống ESP32.
- [x] **3.4 — Triển khai và nghiệm thu luồng nhận diện và so khớp (Recognizer Pipeline):**
  - [x] Đưa ảnh 64x64 vào Tensor Arena → chạy `Invoke()` để sinh ra vector 128-D.
  - [x] Tính Cosine Similarity giữa vector vừa tạo với `face_database.h`.
  - [x] Cài đặt bộ đếm `Temporal Voting` (chốt kết quả khi trùng khớp 3 frames cùng tên; Unknown chỉ tạm dừng tối đa 2 frame).
  - [x] Lưu lịch sử điểm danh vào bộ nhớ Flash (SPIFFS: file `/attendance.csv`) kèm cơ chế Cooldown chống spam.
- [x] **3.5 — Nghiệm thu thiết bị ngoại vi (2 LED + Buzzer + Serial, không dùng LCD):**
  - [x] Điều khiển còi Buzzer đúng đặc tả: 1 bip ngắn = thành công, 2 bip dài = từ chối (verified trên log).
  - [x] Điều khiển 2 LED báo hiệu trạng thái (Xanh = Đã nhận diện, Đỏ = Người lạ) — **không dùng màn hình LCD**.
- [x] **3.6 — Tối ưu đa nhân FreeRTOS (Dual-Core Asymmetric):**
  - [x] Tách task AI sang Core 0 (`AITask` Stack 32KB) và task Ngoại vi/Mạng sang Core 1 (`NetDisplayTask` Stack 8KB).
  - [x] Đồng bộ truyền nhận dữ liệu giữa 2 Core bằng mutex an toàn.

---

### GIAI ĐOẠN 4: TỐI ƯU VÀ HOÀN THIỆN SẢN PHẨM (VẬN HÀNH THỰC TẾ)
> **Mục tiêu:** Từ “chạy được” lên “chạy mượt, ổn định, chống gian lận” trên đúng phần cứng hiện tại (ESP32-S3 N16R8 8MB PSRAM, 2 model INT8, TCP JPEG 128) — đo được, lặp lại được, không phụ thuộc may mắn.

#### 💡 Chiến lược áp dụng trong Giai đoạn 4:
1. **Tối ưu tốc độ đo được (SIMD Xtensa LX7 + Triệt tiêu trễ Pipeline):**
   * Baseline ban đầu khi chạy kernel stock: `det: ~20.3s`, `rec: ~4.7s`.
   * Vendor bộ thư viện `esp-nn v1.3` chính thức từ Espressif vào `firmware_esp32/esp_nn/` và kết nối qua `esp_nn_glue.cpp/.h`. Đăng ký kernel tùy biến vào `MicroMutableOpResolver`.
   * Chuẩn hóa Arena PSRAM: `detector_arena = 1MB` (thực tế dùng ~602KB), `recognizer_arena = 512KB` (thực tế dùng ~186KB) để chứa scratch buffer căn chỉnh filter của SIMD mà không làm phân mảnh RAM cho Wi-Fi/JPEG.
   * Xử lý lỗi silicon / assembly của ESP-NN: Lớp `DepthwiseConv 3x3 s8pad` trên ESP32-S3 bị lỗi sai số. Sử dụng hàm rẽ nhánh thông minh `DwUseEspNn()` để đẩy riêng cấu hình này về hàm chuẩn TFLM Reference Op (`maxdiff=0`), trong khi các lớp Conv2D 1x1 và các lớp khác vẫn chạy 100% bằng SIMD.
   * Khóa xung nhịp CPU cả 2 nhân ở mức trần tối đa **240MHz** trong firmware.
   * Khử trễ đệm mạng: Cho phép Core 1 liên tục giải mã và ghi đè JPEG mới nhất vào `g_frame_buffer` (bỏ cờ chờ `!is_new_frame_available`), đồng thời tắt Nagle (`TCP_NODELAY`) và buffer OpenCV trên Laptop.
2. **Nâng chính xác bằng hiệu chuẩn ngưỡng thực tế:** Ngưỡng toàn cục `0.70` kết hợp **per-identity threshold cap `0.75`** (`cross_max + 0.02`, floor 0.65, cap 0.75) trong `face_database.h`.
3. **Chống giả mạo nhẹ vừa RAM:** LBP/variance trên `gray_eq` 64×64 + active quay đầu (EMA `cx` drift) hoặc ToF VL53L5C (tùy chọn).
4. **Cứng hóa vận hành:** NTP `configTime()` + SPIFFS epoch, enroll tại chỗ, OTA qua HTTP endpoint.
5. **Kiểm thử có thống kê:** Matrix ≥10 người × 3 sáng × 3 khoảng cách, báo cáo FAR/FRR.

#### 📋 Các đầu việc thực hiện (Action Items):
- [x] **4.1 — Tối ưu tốc độ đo được (ĐÃ HOÀN THÀNH & NGHIỆM THU TRÊN PHẦN CỨNG THẬT):**
  - [x] Thêm log `Serial.printf("[PERF] det/bil/he/rec/dec/cy")` bằng `micros()` — in mỗi frame kèm nhãn `DETECT/REUSE`.
  - [x] Khóa xung nhịp CPU 2 nhân ESP32-S3 ở mức trần **240MHz** bằng `setCpuFrequencyMhz(240)`.
  - [x] Tích hợp `esp-nn v1.3` SIMD vào `firmware_esp32/esp_nn/` và `esp_nn_glue.cpp/.h`, tối ưu Tensor Arena PSRAM (1MB / 512KB).
  - [x] Xử lý lỗi assembly `s8pad` bằng logic rẽ nhánh `DwUseEspNn()`, giữ bit-exact parity `maxdiff=0` trên `EspNnSelfTest()`.
  - [x] Bật `AI_ESP_NN_CONV_DET = 1` và `AI_ESP_NN_CONV_REC = 1`. Kết quả nghiệm thu thực tế trên phần cứng:
    * `det`: giảm từ **20,400ms xuống 1,945ms** (nhanh hơn **10.5 lần**).
    * `rec`: giảm từ **5,020ms xuống 1,300ms** (nhanh hơn **4 lần**).
    * Chu kỳ nhận diện Box-Reuse (`cy: REUSE`): đạt **1.32s / frame**.
    * Độ chính xác nhận diện Model V3: `nhien` đạt 0.79 - 0.84, khuôn mặt lạ/che mặt < 0.65.
  - [x] Triệt tiêu độ trễ hàng đợi luồng ảnh (Eliminate Stale Frame Lag):
    * Core 1 (`wifi_udp_server.cpp`): Xóa điều kiện `!is_new_frame_available`, liên tục giải mã và ghi đè JPEG mới nhất vào `g_frame_buffer`. Core 0 luôn lấy frame tức thời (<50ms).
    * Laptop (`ip_camera_streamer.py`): Đặt `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)` và `TCP_NODELAY = 1`, loại bỏ trễ buffer driver webcam và Nagle algorithm.
- [x] **4.2 — Nâng độ chính xác & hiệu chuẩn ngưỡng (ĐÃ HOÀN THÀNH & NGHIỆM THU TRÊN PHẦN CỨNG):**
  - [x] Per-identity threshold: `generate_embeddings.py` tính `cross_max + 0.02`, floor `GLOBAL_THRESHOLD = 0.70`, cap `PER_ID_CAP = 0.75` → lưu `face_database.json` + field `threshold` trong struct `RegisteredFace` `face_database.h`. Hiện tại cả 3 người (`nhien`, `thao`, `toan`) đều có threshold 0.75.
  - [x] Nghiệm thu thực tế trên serial log ESP32-S3 (`ket_qua_esp32.txt`): Khảo sát thực tế điểm số live của `nhien` đạt 0.79 - 0.84. Ngưỡng 0.75 giúp nhận diện liên tục mà không bị từ chối nhầm khi nghiêng mặt nhẹ ở ngưỡng 0.80 cũ.
  - [x] Kích hoạt Temporal Voting (3 frame liên tiếp) chốt `SUCCESS nhien` $\rightarrow$ lưu SPIFFS `/attendance.csv`, 1 bip ngắn buzzer, bật LED xanh.
  - [x] Đánh giá ngoại tuyến trên `evaluate_model.py`: **Top-1 Identification Accuracy 100.0% (12/12 probe holdout, 20/20 per identity)**, FRR@0.70 = 0.0%, người lạ (<0.65) bị chặn an toàn.
- [ ] **4.3 — Chống giả mạo nhẹ (vừa RAM S3):**
  - [ ] Passive không model: sau `equalize_gray_u8` tính LBP/variance + check highlight trên `face_gray` 64×64; reject nếu variance thấp (ảnh in).
  - [ ] Active: yêu cầu người dùng quay đầu 10° (EMA `cx` dịch >8px trong 1s) mới chốt `SUCCESS`.
  - [ ] Tuỳ chọn phần cứng: chân I2C dự phòng cho ToF VL53L5C 8×8 — đọc variance trước khi `Invoke` recognizer.
- [ ] **4.4 — Cứng hóa vận hành & enroll tại chỗ:**
  - [ ] Thêm NTP (`WiFi` → `configTime` → `getLocalTime`) + lưu epoch vào SPIFFS; `log_attendance` ghi `YYYY-MM-DD HH:MM:SS`.
  - [ ] Enroll tại chỗ: nút GPIO → bắt 20 ảnh qua TCP như `enroll_tool.py` → chạy `generate_embeddings` ngay trên S3 → append JSON vào SPIFFS `/faces.json` → reboot load lại `FACE_DATABASE`. Giữ **16 templates MAX-SIM**, không trung bình.
  - [ ] OTA: endpoint HTTP `/update` để flash `model_data.h`/`face_database.h` không cần cáp; test 24h leak `ESP.getFreeHeap()`/`ESP.getFreePsram()` mỗi phút.
- [ ] **4.5 — Kiểm thử thực địa & bàn giao:**
  - [ ] Matrix ≥10 người × 3 điều kiện sáng × 3 khoảng cách, báo FAR/FRR có CI 95%.
  - [ ] Tùy chọn chuyển sang camera trực tiếp OV2640 (S3-EYE) — `image_decoder.cpp` đổi từ `TJpg_Decoder` sang `esp_camera` — giữ `ip_camera_streamer.py` cho debug.
  - [ ] Vỏ 3D + tản nhiệt + nguồn 5V/2A, cố định góc camera 1.2m, hướng dẫn vận hành.

---

## 📍 TRẠNG THÁI TIẾN ĐỘ HIỆN TẠI

| Giai đoạn | Nội dung chính | Trạng thái |
|---|---|---|
| **1.1** | Huấn luyện Knowledge Distillation Model V3 (CASIA-WebFace + ArcFace + Illumination KD) | ✅ **Hoàn thành Model V3 trên Colab GPU — Top-1 Identification 100% (12/12 probe, 20/20 per identity), FRR@0.70 = 0.0%** |
| **1.2** | Kiểm thử E2E hai model trên Laptop (INT8) | ✅ **Hoàn thành — pipeline JPEG/RGB565/Bilinear/HE đồng bộ, threshold 0.70 global / 0.75 per-identity** |
| **1.3** | Thu thập dữ liệu đăng ký theo tỷ lệ vàng (20 ảnh/người) | ✅ **Hoàn thành — nhien/thao/toan 20/20/20, 16 templates/người, threshold 0.75/người** |
| **2.x** | Lượng tử hóa INT8 và audit mô phỏng (Model V3 + BlazeFace) | ✅ **Hoàn thành — Ghost Model V3 INT8 ~160KB + BlazeFace INT8 ~183KB, validate <1e-3, 6/7 ops** |
| **3.1** | Cấu hình PlatformIO / Arduino IDE, nạp model_data.h & face_database.h | ✅ **Hoàn thành — Arduino IDE, nạp Model V3 thành công** |
| **3.2** | Hạ tầng mạng TCP & truyền nhận ảnh đa nhân (128x128) | ✅ **Hoàn thành** |
| **3.3** | Đưa Face Detector INT8 xuống ESP32 (Chạy 100% On-Device) | ✅ **Hoàn thành — BlazeFace 128 INT8 on-device, box-reuse, EMA** |
| **3.4** | Pipeline Nhận diện, Cosine Matching, Temporal Voting & SPIFFS Log | ✅ **Hoàn thành — Ghost 64 HE, MAX-SIM, Voting 3, SPIFFS cooldown** |
| **3.5** | Tích hợp 2 LED xanh/đỏ + Buzzer (không dùng LCD) | ✅ **Hoàn thành — 1 bip success / 2 bip dài reject verified** |
| **3.6** | Phân tách đa nhân FreeRTOS (Dual-Core Asymmetric) | ✅ **Hoàn thành — AITask Core0 / NetTask Core1, mutex, WDT AITask** |
| **4.1** | Tối ưu tốc độ: profiling `[PERF]`, **vendor esp-nn SIMD + glue**, CPU 240MHz, arena PSRAM 1MB/512KB, box-reuse, triệt tiêu trễ frame | ✅ **Hoàn thành & Nghiệm thu trên phần cứng thật: det 1.95s (nhanh 10.5x), rec 1.30s (nhanh 4x), REUSE 1.32s/frame, độ trễ frame <50ms** |
| **4.2** | Per-identity threshold + hiệu chuẩn thực tế (0.70 global, 0.75 per-id) | ✅ **Hoàn thành & Nghiệm thu trên phần cứng thật: điểm live nhien 0.79-0.84, chốt SUCCESS nhien, lưu SPIFFS attendance.csv** |
| **4.3-4.5** | Liveness, NTP/OTA/enroll tại chỗ, field test | ⏳ **Chờ thực hiện tiếp theo** |

---

## 🔑 NGUYÊN TẮC BẤT BIẾN

1. **Laptop-First, ESP32-Final (INT8 đồng bộ):** Mọi thay đổi về thuật toán và mô hình (BlazeFace 128 + Ghost-TinyFace 64) phải được kiểm chứng bằng pipeline E2E INT8 trên Laptop trước khi nạp xuống ESP32.
2. **Giữ nghiêm ngặt ràng buộc phần cứng MCU:**
   * JPEG 128×128 → pixel logic 128×128 RGB; ESP32 lưu/đọc RGB565 → FACE 64×64 Grayscale (recognizer).
   * Vector đặc trưng luôn là **128 chiều** (128-D), được L2-normalize trước matching.
   * Chuẩn hóa pixel: `(pixel - 127.5) / 128.0` + Bilinear Interpolation đồng bộ.
   * So khớp danh tính: `Cosine Similarity` (tích vô hướng dot product).
   * Đầu ra ESP32 **tuyệt đối không dùng LCD** — chỉ **2 LED (xanh/đỏ) + Buzzer + Serial**.
   * **Đồng nhất ảnh truyền vào (Laptop ↔ ESP32):** `ip_camera_streamer.py` gửi JPEG 128×128 qua TCP 12345. Laptop phải chạy cùng center-crop/resize/JPEG decode và mô phỏng chuyển đổi RGB565 như ESP32 trước khi detector/crop.
   * **Loại bỏ độ trễ hàng đợi:** Luôn đảm bảo frame AI xử lý là frame tức thời, không bị đọng frame cũ trong hàng đợi socket hoặc buffer.
3. **Zero-Retraining & Công bằng:** Train 1 lần trên CASIA-WebFace (SFace→Ghost), thêm người mới chỉ cần `enroll_tool.py (15-20 ảnh) → update_face_database.py → flash ESP32`, không train lại; mọi danh tính cũ/mới đều xử lý như nhau (80% trimmed + chung threshold).
4. **Đo lường bằng số liệu thực tế:** Mọi đánh giá phải dựa trên Accuracy/TAR/FAR/FRR, số lượng mẫu, điều kiện test, thời gian suy luận và mức tiêu thụ RAM. Không dùng một vài lần thử thủ công để kết luận FAR bằng 0%.

### HỢP ĐỒNG GOLDEN TEST TRƯỚC KHI ĐƯA XUỐNG ESP32

1. Lưu một bộ frame JPEG 128×128 cố định, cùng header TCP nếu kiểm thử cả giao thức.
2. Chạy frame đó qua pipeline Laptop và ghi lại: pixel sau decode, pixel sau RGB565 round-trip, bounding box, crop 64×64, embedding và kết quả matching.
3. Chạy cùng frame trên ESP32, xuất các giá trị kiểm tra tương đương qua Serial. Cho phép sai số theo lượng tử hóa/decoder đã đo, không đặt ngưỡng `≤1` nếu chưa kiểm chứng thực tế.
4. Chỉ chuyển sang Giai đoạn 3 khi detector, crop, recognizer và matching đều cho cùng kết quả danh tính trên golden set.
