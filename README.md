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
│   ├── ip_camera_streamer.py       #    Dumb IP Camera — gửi JPEG 128×128 qua TCP 12345 (Đã tắt Nagle/Buffer delay)
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
│   ├── firmware_esp32.ino          #    Nhạc trưởng: 2 task FreeRTOS (AITask Core0 + NetTask Core1), khóa 240MHz
│   ├── ai_config.h                 #    Tham số đồng bộ (RAW 128, FACE 64, THRESH 0.60/0.80, VOTES 3)
│   ├── ai_face_detector.h/.cpp     #    BlazeFace INT8 128×128 + decode box + Bilinear crop (~1.9s SIMD)
│   ├── ai_face_recognizer.h/.cpp   #    Ghost-TinyFace INT8 64×64 + HE LUT + cosine MAX-SIM (~1.3s SIMD)
│   ├── esp_nn_glue.h/.cpp          #    Tích hợp ESP-NN SIMD Vector Acceleration
│   ├── wifi_udp_server.h/.cpp      #    TCP Server 12345 + Không delay cập nhật frame liên tục
│   ├── image_decoder.h/.cpp        #    TJpg_Decoder → RGB565 128×128
│   ├── model_data.h                #    C array Ghost-TinyFace INT8 (~1MB)
│   ├── detector_model_data.h       #    C array BlazeFace INT8 (~1.1MB)
│   └── face_database.h             #    16 templates/người × 128-D (MAX-SIM)
│
├── training_tinyml/                # 🧠 Huấn luyện & export
│   ├── models/
│   │   └── ghost_tinyface.py       #    Ghost-TinyFace 64×64 grayscale → 128-D
│   ├── train_distillation.py       #    KD + ArcFace + illumination-invariance (Colab)
│   ├── download_lfw_dataset.py     #    Tải LFW ~13k ảnh
│   ├── evaluate_model.py           #    Accuracy / FAR / TAR + Identification MAX-SIM
│   ├── generate_embeddings.py      #    MAX-SIM 16 templates/người (trimmed 80%)
│   ├── update_face_database.py     #    Quét registered_faces → face_database.json/.h + ai_config.h
│   ├── quantize_qat_int8.py        #    Keras → TFLite INT8 Ghost (representative có HE)
│   ├── quantize_detector_int8.py   #    PTQ BlazeFace float16 → FULL INT8 (giữ 2-output)
│   ├── export_config.py            #    Sinh ai_config.h (512KB+512KB+32KB)
│   └── weights/
│       ├── tinyface_backbone.keras #    Keras float32 (~2MB, sau train)
│       ├── tinyface_int8.tflite    #    Ghost INT8 64×64 (~160KB INT8)
│       └── face_detection_short_range_int8.tflite  # BlazeFace INT8 128×128
│
├── data/                           # 💾 Dữ liệu
│   ├── registered_faces/           #    nhien / thao / toan — mỗi người 20 PNG 64×64 grayscale
│   ├── face_database.json          #    JSON trung gian (templates 128-D)
│   └── attendance.db               #    SQLite log điểm danh
│
├── colab.zip                       # ☁️ Gói 1-lệnh train trên Colab
├── HUONG_DAN_COLAB_TRAIN_V2.md     #    Hướng dẫn train v2 (ArcFace + illumination KD)
├── kich_hoat_moi_truong.md         #    Lệnh nhanh conda + flash + stream
├── mo_ta_project.md                #    Mô tả chi tiết từng file (Đã đồng bộ)
└── requirements.txt
```

### Thông số kỹ thuật và hợp đồng xử lý hiện tại

| Thành phần | Giá trị |
|---|---|
| **Vi điều khiển** | ESP32-S3 N16R8, khóa xung nhịp CPU 240MHz. |
| **Giao thức Mạng** | TCP socket truyền JPEG liên tục, tắt TCP Nagle (TCP_NODELAY), loại bỏ buffer trễ. Core 1 liên tục cập nhật frame mới nhất. |
| **Hiệu năng AI (Thực tế)**| Phát hiện khuôn mặt (BlazeFace): **~1.9 giây**. Nhận diện (Ghost-TinyFace): **~1.3 giây**. Tốc độ điểm danh khi Box-Reuse (cy: REUSE): **~1.3s/frame**. |
| **Mô hình Recognizer** | Ghost-TinyFace (GhostNet Bottleneck) 64×64 Grayscale → 128-D. Chạy Hybrid SIMD/Reference (tránh lỗi s8pad). |
| **Mô hình Detector** | BlazeFace FULL INT8 128×128 RGB (~183KB). Chạy SIMD 100%. |
| **Alignment hiện tại** | Crop vuông theo bounding box + Bilinear thủ công 128→64 + **Histogram Equalization (HE)** khử nhạy sáng. |
| **Matching** | Cosine Similarity MAX-SIM trên nhiều templates/người (đồng bộ Laptop ↔ ESP32). |
| **Threshold** | Recognizer 0.60, BlazeFace conf 0.80. |
| **Chống nhiễu** | Temporal Voting (3 frame, chính sách pause-on-Unknown). Box-reuse. |

---

## 🗺️ ROADMAP CHI TIẾT TỪNG GIAI ĐOẠN

```
  ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
  │   GIAI ĐOẠN 1   │  ──►  │   GIAI ĐOẠN 2   │  ──►  │   GIAI ĐOẠN 3   │  ──►  │   GIAI ĐOẠN 4   │
  │  HOÀN THIỆN     │       │  QUANTIZE INT8  │       │  FIRMWARE ESP32 │       │  TỐI ƯU ĐỘ TRỄ  │
  │  TRÊN LAPTOP    │       │  CHO ESP32-S3   │       │  CHẠY ĐỘC LẬP   │       │  & TĂNG TỐC AI  │
  └─────────────────┘       └─────────────────┘       └─────────────────┘       └─────────────────┘
```

---

### GIAI ĐOẠN 1: HOÀN THIỆN E2E TRÊN LAPTOP
> **Mục tiêu:** Kiểm chứng hoàn chỉnh `JPEG → decode → detector → crop → recognizer → matching` trên Laptop bằng input mô phỏng đúng ESP32. Đảm bảo đạt độ chính xác FAR=0 tại threshold 0.60.

- [x] **1.1 — Huấn luyện mô hình:** Train LFW với ArcFace + Illumination KD. Đạt Identification 100%.
- [x] **1.2 — Vận hành pipeline E2E:** Tích hợp preprocessing đồng nhất với C++ (Bilinear + HE LUT số nguyên).
- [x] **1.3 — Chuẩn hóa CSDL:** Enroll theo tỷ lệ vàng (70% thẳng / 30% nghiêng), tạo DB 16 templates/người.

---

### GIAI ĐOẠN 2: CHUYỂN ĐỔI MODEL SANG ESP32-S3 (QUANTIZE)
> **Mục tiêu:** Chuyển model Keras → TFLite INT8 mà không làm suy giảm độ chính xác.

- [x] **2.1 — Lượng tử hóa INT8:** Ghost-TinyFace có Representative Dataset (đã qua HE). BlazeFace PTQ FLOAT16 → FULL INT8.
- [x] **2.2 — Kiểm thử TFLite Micro trên Laptop:** Audit operator list, đảm bảo maxdiff < 1e-3, độ chính xác đồng nhất với Keras.

---

### GIAI ĐOẠN 3: TRIỂN KHAI FIRMWARE ESP32-S3
> **Mục tiêu:** Port toàn bộ luồng nhận diện xuống ESP32-S3, đa nhân, cấp phát bộ nhớ PSRAM an toàn.

- [x] **3.1 — Cấp phát bộ nhớ PSRAM:** Phân vùng Arena (Detector 2.5MB, Recognizer 1.75MB để chạy SIMD).
- [x] **3.2 — Đa nhân (Dual-Core):** Core 1 lo nhận Wi-Fi JPEG/Giải mã; Core 0 lo chạy mô hình AI. Mutex an toàn.
- [x] **3.3 — BlazeFace On-device:** Bounding box decode + Bilinear Crop + LUT HE độc lập không dùng OpenCV.
- [x] **3.4 — Ghost-TinyFace & Matching:** Vector 128-D L2-norm, MAX-SIM, SPIFFS Log.
- [x] **3.5 — Ngoại vi:** 2 LED và còi Buzzer (1 bip = success, 2 bip = reject).

---

### GIAI ĐOẠN 4: TỐI ƯU VÀ HOÀN THIỆN SẢN PHẨM (VẬN HÀNH THỰC TẾ)
> **Mục tiêu:** Tăng tốc thuật toán bằng lệnh phần cứng (SIMD), khóa xung nhịp, loại bỏ độ trễ hàng đợi mạng. Đưa tốc độ từ "vài giây" xuống "real-time" có thể chấp nhận được (~1s).

- [x] **4.1 — Tối ưu tốc độ (SIMD ESP-NN & Khóa Clock & Zero-Delay Pipeline):**
  - [x] Khóa xung nhịp CPU ESP32-S3 ở mức trần **240MHz**.
  - [x] Bật thư viện `esp-nn` tăng tốc Vector SIMD Xtensa cho TFLite Micro. Tốc độ Detector giảm từ 20.3s xuống **1.9s**, Recognizer giảm từ 4.7s xuống **1.3s**.
  - [x] Viết Glue Code xử lý ngoại lệ hàm DepthwiseConv 3x3 s8pad của `esp-nn` bị lỗi. Trả về reference TFLM để giữ độ chính xác tuyệt đối.
  - [x] Sửa lỗi hàng đợi mạng: Cho phép Core 1 liên tục cập nhật frame mới đè lên `g_frame_buffer`, loại bỏ hoàn toàn độ trễ 5s giữa Camera và AI. Độ trễ thực tế hiện tại là < 50ms giữa lúc AI bắt đầu chạy và frame được chụp.
  - [x] Giữ tính năng **Box-Reuse** (cy: REUSE). Chỉ tốn 1.3s/frame khi người dùng đang đứng yên trước camera.
- [x] **4.2 — Nâng độ chính xác & hiệu chuẩn ngưỡng:**
  - [x] Per-identity threshold (max inter + margin) đã triển khai. Ngưỡng mặc định `0.60`. FAR=0.
- [ ] **4.3 — Chống giả mạo nhẹ (Liveness/Spoofing):**
  - [ ] Check Variance hoặc LBP trên ảnh Grayscale để từ chối ảnh in.
  - [ ] Yêu cầu người dùng quay đầu nhẹ (EMA drift).
- [ ] **4.4 — NTP & OTA & Enroll tại chỗ:**
  - [ ] Đồng bộ thời gian NTP + lưu log thời gian thực. Cập nhật Model qua HTTP OTA.
- [ ] **4.5 — Đóng vỏ phần cứng:**
  - [ ] Tích hợp camera OV2640 trực tiếp nếu cần thiết, thiết kế vỏ tản nhiệt.

---

## 📍 TRẠNG THÁI TIẾN ĐỘ HIỆN TẠI

| Giai đoạn | Nội dung chính | Trạng thái |
|---|---|---|
| **1.x** | Hoàn thiện luồng đào tạo & E2E trên Laptop (ArcFace+KD) | ✅ **Hoàn thành (Identification 100%, FAR=0)** |
| **2.x** | Lượng tử hóa INT8 2 model và audit | ✅ **Hoàn thành (Ghost 160KB, Blaze 183KB)** |
| **3.x** | Firmware ESP32-S3 (PSRAM, Dual-Core, TCP, No-LCD) | ✅ **Hoàn thành** |
| **4.1** | Tối ưu SIMD ESP-NN, Khóa 240MHz CPU, Fix TCP Lag | ✅ **Hoàn thành — Tốc độ xử lý: det 1.9s, rec 1.3s, độ trễ frame <50ms** |
| **4.2** | Hiệu chuẩn Threshold | ✅ **Hoàn thành** |
| **4.3-4.5** | Liveness, NTP/OTA, Field Test | ⏳ **Kế hoạch tiếp theo** |

---

## 🔑 NGUYÊN TẮC BẤT BIẾN

1. **Laptop-First, ESP32-Final (INT8 đồng bộ):** Mọi thay đổi về thuật toán và mô hình phải được kiểm chứng bằng pipeline E2E INT8 trên Laptop trước khi nạp xuống ESP32.
2. **Giữ nghiêm ngặt ràng buộc phần cứng MCU:**
   * ESP32 lưu/đọc RGB565 → FACE 64×64 Grayscale (recognizer).
   * Chuẩn hóa pixel: `(pixel - 127.5) / 128.0` + Bilinear Interpolation đồng bộ.
   * Đầu ra ESP32 **tuyệt đối không dùng LCD** — chỉ **2 LED + Buzzer + Serial**.
   * Loại bỏ độ trễ: Frame ảnh AI sử dụng **luôn là frame mới nhất**, không có hàng đợi (buffer=1).
3. **Độ chính xác là số một:** Việc tối ưu tốc độ (SIMD/ESP-NN) không được phép làm thay đổi output đầu ra so với code nguyên bản của Google (maxdiff = 0). Nếu phát hiện lỗi phần cứng, phải linh hoạt chuyển về hàm phần mềm (Reference Ops) để đảm bảo độ chính xác thay vì chấp nhận sai số.
4. **Zero-Retraining:** Thêm người mới chỉ cần enroll ảnh → update database → flash ESP32, không train lại AI.
