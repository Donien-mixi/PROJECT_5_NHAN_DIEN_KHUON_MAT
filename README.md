# 🚀 ROADMAP: HỆ THỐNG ĐIỂM DANH NHẬN DIỆN KHUÔN MẶT TRÊN ESP32-S3

> **Mục tiêu cuối cùng:** Một hệ thống nhận diện khuôn mặt chạy kết hợp **(Laptop + ESP32-S3 N16R8)** qua mạng Wi-Fi. Laptop đảm nhiệm Camera & Face Detection. ESP32-S3 đảm nhiệm trích xuất đặc trưng (TFLite INT8), so khớp danh tính, ghi log và hiển thị giao diện trên màn hình LCD TFT ILI9341.

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
| **Kích thước model** | ~700KB (Keras), ~160KB (TFLite INT8) |
| **Face Detector** | MediaPipe Face Mesh (468 điểm) / YuNet (5 điểm) |
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
  - [ ] Chạy `evaluate_model.py` để đo độ chính xác (mục tiêu: **Accuracy ≥ 90%** trên tập test).
  - [ ] *(Tùy chọn mở rộng)*: Chạy train trên tập CASIA-WebFace nếu cần độ tách biệt danh tính cao hơn.
- [x] **1.2 — Vận hành và tinh chỉnh pipeline nhận diện trên Laptop:**
  - [x] Tạo lại CSDL vector bằng `generate_embeddings.py` dựa trên trọng số mới.
  - [x] Chạy `host_laptop/main.py` kiểm tra nhận diện trực tiếp thời gian thực qua webcam.
  - [x] Tinh chỉnh ngưỡng `threshold` (đã nâng lên 0.75) và số phiếu `Temporal Voting` (5 frame) để triệt tiêu False Accepts.
  - `[ ]` Đo đạc tỷ lệ nhận đúng (True Positive) và tỷ lệ từ chối người lạ (True Negative).
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
- `[ ]` **2.1 — Thực hiện lượng tử hóa INT8:**
  - `[ ]` Chạy `quantize_qat_int8.py` với Representative Dataset lấy từ ảnh khuôn mặt thực tế để hiệu chuẩn phân phối giá trị activation.
  - `[ ]` Xuất ra file `training_tinyml/weights/tinyface_int8.tflite`.
  - `[ ]` Kiểm tra độ tương đồng vector embedding giữa Keras và INT8 TFLite (mục tiêu: **Cosine Similarity ≥ 0.95** trên cùng 1 ảnh).
  - `[ ]` Nếu sai lệch > 5%, kích hoạt Quantization-Aware Training (QAT) trong quá trình train lại.
- `[ ]` **2.2 — Kiểm thử mô hình INT8 trên Laptop:**
  - `[ ]` Tích hợp bộ đọc TFLite Interpreter vào module `FaceRecognizer` trên Laptop.
  - `[ ]` So sánh kết quả nhận diện trực tiếp bằng TFLite INT8 so với Keras float32.
  - `[ ]` Đảm bảo file `.tflite` hoàn toàn sẵn sàng và chuẩn xác trước khi bước vào lập trình nhúng.

> [!IMPORTANT]
> **Ràng buộc kỹ thuật trên ESP32-S3:**
> - Tensor Arena memory ≤ 136KB (bắt buộc vừa khít trong Internal SRAM).
> - Kích thước file model ≤ 160KB.
> - Chỉ chứa 5 Ops chuẩn: `CONV_2D`, `DEPTHWISE_CONV_2D`, `ADD`, `FULLY_CONNECTED`, `RESHAPE`.

---

### GIAI ĐOẠN 3: TRIỂN KHAI FIRMWARE ESP32-S3
> **Mục tiêu:** Port module Face Recognizer xuống ESP32-S3 N16R8. Thiết lập kiến trúc phân tán: Laptop truyền ảnh khuôn mặt đã cắt qua Wi-Fi, ESP32 chạy suy luận TFLite và hiển thị kết quả lên LCD ILI9341.

#### 💡 Chiến lược áp dụng trong Giai đoạn 3:
1. **Bí quyết 4 — Tăng tốc phần cứng AI (ESP-NN / Vector Instructions):** Tận dụng tập lệnh vector SIMD chuyên dụng trên nhân Xtensa LX7 của ESP32-S3 để xử lý các phép nhân ma trận INT8 song song.
2. **Chiến lược phân bổ bộ nhớ kép (Dual-Memory Allocation Strategy):**
   * *Internal SRAM (~512KB, tốc độ cao):* Dành riêng cho Tensor Arena của TFLite Micro và Stack/Heap hệ thống để suy luận nhanh nhất, không bị nghẽn bus.
   * *External PSRAM (8MB, tốc độ chậm):* Dành cho bộ đệm nhận dữ liệu Wi-Fi và Frame Buffer cho màn hình LCD ILI9341 (240x320).
3. **Chiến lược phân tán tính toán (Distributed Computing):** Tận dụng sức mạnh của Laptop để chạy Face Detector (MediaPipe) và cắt/xoay ảnh về 64x64. ESP32 chỉ tập trung chạy nhận diện, giúp MCU hoạt động cực kỳ nhẹ nhàng và ổn định.
4. **Kiến trúc xử lý đa luồng (Dual-Core Asymmetric Processing):**
   * **Core 0:** Chuyên chạy suy luận mô hình AI TFLite Micro (tính toán nặng).
   * **Core 1:** Chuyên xử lý ngoại vi (Nhận ảnh 64x64 qua TCP/UDP Socket, xuất màn hình LCD ILI9341 SPI, còi Buzzer, ghi log).

#### 📋 Các đầu việc thực hiện (Action Items):
- `[ ]` **3.1 — Khởi tạo dự án Firmware C++ (ESP-IDF / PlatformIO):**
  - `[ ]` Tạo thư mục `firmware_esp32/` cấu hình PlatformIO với framework ESP-IDF/Arduino.
  - `[ ]` Chuyển đổi `tinyface_int8.tflite` thành mảng C array (`model_data.h`) và nạp vào Flash.
  - `[ ]` Khởi tạo TFLite Micro với `MicroMutableOpResolver` (chỉ đăng ký đúng 5 ops cần thiết).
  - `[ ]` Nhúng CSDL vector đặc trưng các thành viên vào file `face_database.h`.
- `[ ]` **3.2 — Tích hợp Nhận dữ liệu Camera qua Wi-Fi:**
  - `[ ]` Cấu hình ESP32 kết nối cùng mạng Wi-Fi với Laptop, tạo TCP/UDP Server.
  - `[ ]` Viết script trên Laptop để liên tục cắt ảnh khuôn mặt (64x64 grayscale) và truyền qua socket xuống ESP32.
  - `[ ]` ESP32 nhận buffer ảnh và đưa trực tiếp vào Tensor Arena.
- `[ ]` **3.3 — Triển khai luồng nhận diện và so khớp:**
  - `[ ]` Chạy TFLite `Invoke()` để sinh ra vector 128-D từ ảnh nhận qua Wi-Fi.
  - `[ ]` Tính Cosine Similarity (phép tính tích vô hướng dot product) giữa vector vừa tạo với `face_database.h`.
  - `[ ]` Cài đặt bộ đếm `Temporal Voting` (chốt kết quả khi trùng khớp 3 lần liên tiếp).
  - `[ ]` Lưu lịch sử điểm danh vào bộ nhớ Flash (SPIFFS / LittleFS) hoặc thẻ nhớ SD Card.
- `[ ]` **3.4 — Kết nối thiết bị ngoại vi và giao diện (LCD ILI9341 SPI):**
  - `[ ]` Tích hợp thư viện điều khiển màn hình TFT ILI9341 (240x320) qua giao tiếp SPI (8 chân).
  - `[ ]` Thiết kế UI: Hiển thị tên người điểm danh, trạng thái, và vẽ lại luồng ảnh khuôn mặt 64x64 nhận từ Laptop.
  - `[ ]` Điều khiển còi Buzzer phát âm thanh (1 tiếng bip ngắn = thành công, 2 tiếng bip dài = từ chối).
  - `[ ]` Điều khiển LED báo hiệu trạng thái (Xanh = Đã nhận diện, Đỏ = Người lạ).
- `[ ]` **3.5 — Tối ưu đa nhân FreeRTOS (Dual-Core):**
  - `[ ]` Tách task AI sang Core 0 và task Ngoại vi/Camera sang Core 1.
  - `[ ]` Đồng bộ truyền nhận dữ liệu giữa 2 Core bằng FreeRTOS Queue.

---

### GIAI ĐOẠN 4: TỐI ƯU VÀ HOÀN THIỆN SẢN PHẨM
> **Mục tiêu:** Hệ thống đạt độ ổn định cao, tốc độ suy luận nhanh, chống gian lận và sẵn sàng vận hành liên tục.

#### 💡 Chiến lược áp dụng trong Giai đoạn 4:
1. **Chiến lược chống giả mạo (Anti-Spoofing / Liveness Detection):** Triển khai nhận diện chớp mắt hoặc xoay đầu trên Laptop trước khi gửi ảnh xuống ESP32, đảm bảo tính chống giả mạo cao.
2. **Đăng ký linh hoạt qua Socket:** Laptop đảm nhiệm công việc thêm người dùng mới vào hệ thống, sau đó truyền trực tiếp file CSDL mới qua Wi-Fi để ESP32 cập nhật vào Flash.

#### 📋 Các đầu việc thực hiện (Action Items):
- `[ ]` **4.1 — Tối ưu hóa hiệu năng và tốc độ:**
  - `[ ]` Đo thời gian suy luận trên chip — mục tiêu: **Thời gian inference ≤ 300ms/khuôn mặt**.
  - `[ ]` Bật cờ tối ưu hóa biên dịch `-O3` và thư viện tăng tốc `esp-nn`.
  - `[ ]` Đo đạc rò rỉ bộ nhớ (Memory Leak), đảm bảo Heap và PSRAM ổn định sau 24h chạy liên tục.
- `[ ]` **4.2 — Tích hợp tính năng chống giả mạo (Liveness Detection):**
  - `[ ]` Cài đặt thuật toán kiểm tra chớp mắt (Eye Blink Detection) hoặc phát hiện góc nghiêng động.
  - `[ ]` Từ chối điểm danh nếu phát hiện khuôn mặt là ảnh phẳng cố định.
- `[ ]` **4.3 — Xây dựng cơ chế Cập nhật Database OTA (Over-The-Air):**
  - `[ ]` Bổ sung chế độ "Update DB Mode" trên ESP32.
  - `[ ]` Khi có người đăng ký mới trên Laptop, Laptop gửi file `face_database.json` mới qua Wi-Fi để ESP32 đồng bộ vào SPIFFS.
- `[ ]` **4.4 — Kiểm thử thực tế toàn diện (Field Testing):**
  - `[ ]` Kiểm thử với ≥ 10 người dùng khác nhau trong các điều kiện ánh sáng (phòng tối, đèn tuýp, ngoài trời).
  - `[ ]` Kiểm thử với người lạ để xác nhận tỷ lệ False Accept Rate = 0%.
  - `[ ]` Đóng vỏ hộp thiết bị hoàn chỉnh (3D Print case) và bàn giao sản phẩm.

---

## 📍 TRẠNG THÁI TIẾN ĐỘ HIỆN TẠI

| Giai đoạn | Nội dung chính | Trạng thái |
|---|---|---|
| **1.1** | Huấn luyện Knowledge Distillation trên Colab | 🔄 **Đang thực hiện (Chờ tải file `.keras`)** |
| **1.2** | Kiểm thử và tinh chỉnh nhận diện trên Laptop | 🔄 **Đã sẵn sàng code `main.py`** |
| **1.3** | Thu thập dữ liệu đăng ký theo tỷ lệ vàng | ✅ Đã có dữ liệu 3 người mẫu |
| **2.x** | Lượng tử hóa INT8 và kiểm thử mô phỏng | ⏳ Chờ hoàn thành GĐ 1 |
| **3.x** | Triển khai Firmware C++ trên ESP32-S3 | ⏳ Chờ hoàn thành GĐ 2 |
| **4.x** | Tối ưu tốc độ, Anti-Spoofing và kiểm thử 24h | ⏳ Chờ hoàn thành GĐ 3 |

---

## 🔑 NGUYÊN TẮC BẤT BIẾN

1. **Laptop-First, ESP32-Final:** Mọi thay đổi về thuật toán và mô hình phải được kiểm chứng đạt độ chính xác cao trên Laptop trước khi nạp xuống ESP32.
2. **Giữ nghiêm ngặt ràng buộc phần cứng MCU:**
   * Kích thước ảnh đầu vào luôn cố định **64×64 Grayscale** (1 kênh).
   * Vector đặc trưng luôn là **128 chiều** (128-D L2-normalized).
   * Chuẩn hóa pixel: `(pixel - 127.5) / 128.0`.
   * So khớp danh tính: `Cosine Similarity` (tích vô hướng dot product).
3. **Đo lường bằng số liệu thực tế:** Mọi đánh giá phải dựa trên tỷ lệ Accuracy, FAR/FRR, thời gian suy luận (ms) và mức tiêu thụ RAM thay vì cảm tính.
