# Kích hoạt môi trường & chạy dự án (đồng bộ README/mo_ta_project/KE_HOACH)

> Chuẩn: `Webcam → center crop/RAW 128 (resize 1 lần) → JPEG q80 → RGB565 → BlazeFace INT8 128 → Bilinear 128→64 → HE (equalize) → Ghost-TinyFace INT8` — `TCP 12345`, `0.60/0.80/3`, `2 LED+Buzzer` — `README.md:57,62,133,161,240` + `mo_ta_project.md:29-48`
>
> - **Detector (BlazeFace)** và **Recognizer (Ghost-TinyFace)** là HAI model dùng đồng thời lúc hoạt động. Cả hai ĐÃ có sẵn dạng INT8 trong repo, nên phần lớn trường hợp **không cần chạy bước 4**.
> - Chỉ chạy bước 4 khi bạn **retrain/thay model** → khi đó phải quantize lại CẢ HAI (4a + 4b).
> - **Model v2 (ArcFace + illumination-invariance):** nếu hệ thống còn nhầm lẫn giữa các
>   người đã đăng ký (kiểm tra bằng mục 5 IDENTIFICATION của evaluate_model.py), hãy
>   huấn luyện lại model trên Colab bằng `train_distillation.py` đã nâng cấp (ArcFace
>   loss + illumination-invariance KD), rồi chạy lại bước 4a → B → C → D. KHÔNG cần
>   chụp lại ảnh đăng ký — DB regenerate từ chính ảnh hiện có.

> **Lưu ý: chạy các lệnh Python trong môi trường `projet_5`.**
> - Mở **Anaconda Prompt** (có conda trong PATH) rồi `conda activate projet_5`.
> - **Di chuyển đến thư mục dự án:** `cd /d d:\PROJECT_5_DIEM_DANH_KHUON_MAT`
> - Hoặc dùng trực tiếp python của env: `C:\Users\DONG NHIEN\.conda\envs\projet_5\python.exe ...` khi shell không có `conda`.
> - Nếu gặp lỗi `UnicodeEncodeError` khi chạy (console Windows cp1252 không hiển thị tiếng Việt), đặt biến môi trường: `set PYTHONIOENCODING=utf-8` trước khi chạy.

```bat
:: 1. Cài thư viện Python (lần đầu hoặc khi requirements đổi)
pip install -r requirements.txt

:: Kiểm tra môi trường & audit model (TensorFlow):
python -c "import tensorflow; print('TF', tensorflow.__version__)"

:: 2. Thu thập ảnh đăng ký 20 ảnh/người (14 thẳng + 6 nghiêng 10-15°)
::    Tool lưu PNG grayscale 64x64, CHẶN ở 20 ảnh/người, TỪ CHỐI ảnh tối (<65)
::    hoặc cháy sáng (>200) — nhìn chỉ báo "Do sang" trên màn hình để chỉnh ánh sáng
python host_laptop/enroll_tool.py
:: → nhập tên (vd: thao), nhấn 'C' chụp, 'Q' thoát
:: → Kiểm tra: directory data\registered_faces\<tên>\ có đúng 20 file .png 64x64 GRAYSCALE

:: 3. Sinh CSDL Zero-Retraining (không cần train lại)
::    → Matching MULTI-TEMPLATE MAX-SIM (tới 16 ảnh/người) ĐỒNG BỘ Laptop và ESP32.
python training_tinyml/generate_embeddings.py
:: → data/face_database.json + firmware_esp32/face_database.h (16 templates/người, Trimmed 80%)
python training_tinyml/update_face_database.py
:: → regenerate firmware_esp32/ai_config.h (512KB+512KB+32KB, 128, 64, 0.60, 0.80, 3)

:: 4. [CHỈ KHI ĐỔI MODEL] Quantize lại CẢ HAI model — KHÔNG train lại
::    Nếu KHÔNG đổi model (đang dùng bản INT8 có sẵn) => BỎ QUA hoàn toàn bước này.
:: 4a. Recognizer Ghost-TinyFace:
python training_tinyml/quantize_qat_int8.py
:: → training_tinyml/weights/tinyface_int8.tflite + firmware_esp32/model_data.h
:: 4b. Detector BlazeFace (PTQ float16 -> FULL INT8, tự validate trước khi xuất):
python training_tinyml/quantize_detector_int8.py
:: → training_tinyml/weights/face_detection_short_range_int8.tflite (~183KB)
python host_laptop/convert_tflite_to_c.py --model host_laptop\detector\face_detection_short_range_int8.tflite --out firmware_esp32\detector_model_data.h --symbol g_detector_model
copy training_tinyml\weights\face_detection_short_range_int8.tflite host_laptop\detector\face_detection_short_range_int8.tflite

:: Kiểm tra cú pháp Python toàn bộ dự án (dùng glob, tránh lỗi wildcard trên Windows):
python -c "import py_compile,glob; [py_compile.compile(f,doraise=True) for f in glob.glob('host_laptop/*.py')+glob.glob('host_laptop/core/*.py')+glob.glob('host_laptop/detector/*.py')+glob.glob('host_laptop/recognizer/*.py')+glob.glob('training_tinyml/*.py')+glob.glob('training_tinyml/models/*.py')]; print('PY-OK')"

:: 5. Đánh giá model + TAR/FAR với gallery/probe/impostor (cần người lạ: thư mục Impostor_*)
python training_tinyml/evaluate_model.py
:: → In margin, threshold tối ưu (max TAR-FAR), TAR/FAR tại 0.60 (đang dùng) và 0.88

:: 6. Test Laptop-First E2E (bắt buộc trước khi flash ESP32)
python host_laptop/main.py
:: → Cùng pipeline ESP32 (crop 128 + JPEG q80 + RGB565 + detector INT8), HUD BLAZEFACE_INT8 128
:: → probe đạt mục tiêu, impostor đo FAR, Temporal Voting 3 frame (pause-on-Unknown)
```

---

## 7. BIÊN DỊCH VÀ NẠP FIRMWARE BẰNG ESP-IDF 5.3 + SERIAL MONITOR

### 7.1 Sơ đồ đấu nối phần cứng ngoại vi (ESP32-S3 WROOM-1 N16R8 CAM)
> **Lưu ý:** Module Camera OV5640 cắm trực tiếp vào socket Freenove/S3-EYE trên bo mạch.
> Các chân ngoại vi được đấu nối vào Header chân tự do như sau:
*   **Còi Buzzer:** Chân tín hiệu vào **GPIO 1** (GND nối GND bo mạch).
*   **LED Xanh Lá (Điểm danh thành công):** Anode (+) qua điện trở 220Ω vào **GPIO 2**, Cathode (-) vào GND.
*   **LED Đỏ (Người lạ / Từ chối):** Anode (+) qua điện trở 220Ω vào **GPIO 3**, Cathode (-) vào GND.

### 7.2 Nạp Firmware lên ESP32-S3 (ESP-IDF 5.3)

> **LƯU Ý:** Đóng mọi cửa sổ Serial Monitor (hoặc Arduino IDE) đang chiếm cổng COM trước khi nạp.

#### Cách 1: Nạp một chạm bằng script PowerShell (Khuyên dùng)
Mở cửa sổ PowerShell (hoặc ESP-IDF PowerShell) và chạy:
```powershell
powershell -ExecutionPolicy Bypass -File D:\PROJECT_5_DIEM_DANH_KHUON_MAT\tools\flash_project_5.ps1 -Port COM3
```
*(Thay `COM3` bằng cổng thực tế của bạn nếu khác. Script sẽ nạp Bootloader, Partition Table và Firmware chính với tốc độ cao 460800 baud, sau đó tự động bật Serial Monitor)*.

#### Cách 2: Nạp trực tiếp qua lệnh idf.py trong cửa sổ ESP-IDF 5.3 PowerShell
```powershell
cd D:\PROJECT_5_DIEM_DANH_KHUON_MAT\firmware_esp32
idf.py -p COM3 flash monitor
```
*(Bấm tổ hợp phím `Ctrl + ]` để thoát Serial Monitor).*

### 7.3 Log khởi động mẫu trên Serial Monitor
- Khởi động thành công sẽ in:
  - `🚀 PROJECT 5: STANDALONE FACE ATTENDANCE SYSTEM (OV5640 CAM)`
  - `⚡ 100% EDGE AI ON ESP32-S3 N16R8 (ESP-IDF 5.3 + SIMD esp-nn)`
  - `CPU Freq: 240 MHz`
  - `Bộ nhớ Octal PSRAM : 8 MB`
  - `✅ SPIFFS đã mount thành công tại /spiffs`
  - `📷 [CameraTask] Bắt đầu thu thập ảnh từ OV5640 trên Core 1...`
  - `🧠 [AITask] Đang khởi tạo mô hình AI trên Core 0 (SIMD esp-nn)...`
  - `[Detector] Arena used ... (PSRAM, esp-nn SIMD: BẬT)`
  - `[Recognizer] Arena used ... (PSRAM, esp-nn SIMD: BẬT)`
  - `[WiFi] Connected kèm IP (ví dụ: http://192.168.1.150)`
  - `🌐 [Web Server] HTTP Server da khoi dong tren port 80!`
  - `[PERF] det: ...ms bil: ...us he: ...us rec: ...ms cy: ...` (~0.8s / chu kỳ nhận diện)
- Nếu Wi-Fi không có sẵn, ESP32 sẽ tự động chạy chế độ **100% OFFLINE độc lập**, nhận diện và báo LED/Buzzer bình thường!

---

## 8. XEM GÓC MÁY CAMERA VÀ LỊCH SỬ ĐIỂM DANH QUA WEB (LỰA CHỌN A)

Sau khi ESP32 khởi động và kết nối Wi-Fi, bạn có thể dùng điện thoại hoặc laptop cùng mạng Wi-Fi truy cập:
👉 `http://<ĐỊA_CHỈ_IP_CỦA_ESP32>/` (ví dụ `http://192.168.1.150`)

Giao diện Web Dashboard cung cấp:
1. **Live Camera Preview:** Xem trực tiếp góc máy camera OV5640 với tốc độ cao.
2. **AI Realtime Status:** Hiển thị tên người đứng trước camera, độ tương đồng Cosine (ví dụ: `nhien - 84.5%`) và trạng thái khớp/người lạ.
3. **Điều khiển góc máy:** Nút **Lật Dọc (V-Flip)** và **Lật Gương (H-Mirror)** để xoay camera đúng chiều bạn đặt.
4. **Nhật ký điểm danh:** Xem bảng danh sách điểm danh thời gian thực, nút tải file `/attendance.csv` và nút xóa lịch sử.

> **HOÀN TOÀN ĐỘC LẬP:** Bạn KHÔNG cần mở laptop chạy `ip_camera_streamer.py` nữa! Toàn bộ hệ thống giờ chạy 100% bằng ESP32-S3 và camera OV5640.

> **Lưu ý hiệu năng ESP32 ([4.1 REALTIME] đã bật ESP-NN SIMD):**
> - **esp-nn vendored** trong `firmware_esp32/esp_nn/` + glue `esp_nn_glue.cpp/.h`.
>   Boot log phải in `>>> ESP-NN SIMD kernels BAT (conv/dwconv Xtensa LX7) — realtime mode`.
>   Nếu build lỗi: mở `esp_nn_glue.h` đổi `#define AI_ESP_NN_CONV 0` → quay về kernel stock (chậm như cũ nhưng chạy đúng).
> - **[PERF]** mỗi frame: `det:... rec:...` — sau ESP-NN kỳ vọng det ~0.5s, rec ~0.15s
>   → đưa mặt vào, SUCCESS sau 3 phiếu trong <1s (realtime).
> - **Arena PSRAM mới**: detector 2.5MB, recognizer 1.75MB (esp-nn cần scratch = bản sao
>   filter). Boot log `Arena used ~470KB/2.6MB` và `~187KB/1.8MB` là bình thường.
> - **[4.2] Per-identity threshold:** `face_database.h` mới có field `threshold`; matching hiệu lực
>   `max(0.60, ngưỡng riêng)` — **bắt buộc Verify+Upload lại firmware sau khi regenerate DB**.
> - **Watchdog (đã fix triệt để):** TWDT KHÔNG theo dõi IDLE tasks (IDLE starve là hợp lệ
>   khi inference dài), thay vào đó **đăng ký AITask** và feed mỗi vòng + sau detect
>   (timeout 60s, chỉ cảnh báo không reboot). Nhờ đó không còn cảnh báo
>   `task_wdt: IDLE0` trong Serial.
> - **Backpressure (đã fix):** NetTask LUÔN đọc/xả socket, chỉ decode khi AI rảnh; frame dư
>   bị DROP. Nhờ đó streamer không bao giờ bị block và **tự kết nối lại** khi ESP32 restart.
> - Với stream 15fps, ESP32 xử lý chậm hơn nguồn gửi — đó là hành vi đúng; cần 3 frame
>   cùng tên (Temporal Voting, Unknown chỉ tạm dừng tối đa 2 chu kỳ) mới chốt.
> - **Box reuse (tối ưu tốc độ):** AITask chỉ chạy detect khi (a) chưa có box, hoặc
>   (b) lượt nhận diện gần nhất KHÔNG đạt ngưỡng và đã qua tối đa 4 chu kỳ recognize.
>   Khi người vẫn đứng yên được nhận diện đạt ngưỡng → mọi chu kỳ chỉ recognize (~5s),
>   không còn khoảng dừng detect 20-30s chen giữa sau SUCCESS. Unknown kéo dài sẽ
>   tự buộc detect lại (box có thể cũ). Tốc độ chỉ đạt mục tiêu ≤400ms/frame khi bật
>   ESP-NN ở GĐ4.
> - **Laptop chống lag cửa sổ:** recognizer chạy mỗi 3 frame (~10 lần/s), kết quả gần
>   nhất được giữ lại cho HUD → cửa sổ mượt, so khớp vẫn nhanh và đúng.

> **Ghi chú:**
> - `192.168.1.XXX` là ví dụ — xem IP thực tại Serial Monitor sau khi ESP32 kết nối WiFi (`ssid/password` trong `firmware_esp32/wifi_udp_server.cpp`, nhớ sửa cho đúng mạng của bạn).
> - `data/registered_faces/nhien/` hiện đã có **20 ảnh PNG grayscale 64×64 chuẩn** — chỉ dùng khi enroll người mới, `nhien` giữ nguyên.
> - Không dùng LCD — ESP32 chỉ báo bằng 2 LED xanh/đỏ + Buzzer + Serial (`README.md:3`).
> - Golden test Python↔C++: ghi 1 frame JPEG 128 cố định, so sánh box/crop/embedding giữa `main.py` và Serial ESP32 trước khi kết luận đồng nhất (`README.md:250-255`). Cho phép sai số theo quantization đã đo (detector INT8 có thể lệch bbox vài pixel).
> - Nếu shell không có `conda` trong PATH, chạy tool bằng đường dẫn tuyệt đối: `C:\Users\DONG NHIEN\.conda\envs\projet_5\python.exe ...`
> - Nhớ `set PYTHONIOENCODING=utf-8` trước khi chạy Python nếu gặp `UnicodeEncodeError`.

---

## 9. THÊM NGƯỜI MỚI (ZERO-RETRAINING, KHÔNG CẦN TRAIN LẠI)

Hệ thống hỗ trợ Zero-Retraining: thêm người mới **chỉ cần enroll 20 ảnh, không train lại**, cả Laptop và ESP32 tự động nhận diện được. Quy trình:

```bat
:: A. Chụp 20 ảnh cho người mới (đặt tên KHÔNG DẤU, viết liền — vd: thao)
python host_laptop/enroll_tool.py
:: → nhập tên (vd: thao), nhấn 'C' chụp đủ 20 ảnh, 'Q' thoát
:: → Tool CHẶN ở 20 ảnh (không chụp được quá 20 — đủ cho 16 templates)
:: → Tool hiển thị "Do sang" trực tiếp và TỪ CHỐI lưu ảnh:
::     • quá tối (<65)  → thêm đèn/chỉnh ánh sáng rồi chụp lại
::     • cháy sáng (>200) → giảm ánh sáng
::     • KHÔNG chụp ngược sáng, KHÔNG đổi ánh sáng giữa các ảnh ( embedding lệch)
:: → Kiểm tra: dir data\registered_faces\thao\ có đúng 20 file .png 64x64 GRAYSCALE

:: B. Sinh lại database (quét TẤT CẢ thư mục người dùng, tự thêm người mới)
python training_tinyml/update_face_database.py
:: → Bước 1: regenerate data/face_database.json + firmware_esp32/face_database.h
::   (multi-template: mỗi người tới 16 templates, có cả người cũ lẫn người mới)
:: → Bước 2: regenerate firmware_esp32/ai_config.h (đồng bộ tham số)
:: Kiểm tra: mở firmware_esp32\face_database.h → NUM_REGISTERED_FACES = 3
::   và thấy entry "nhien", "thao", "toan" (num_templates=16 mỗi người)

:: C. (BẮT BUỘC với ≥2 người) Kiểm tra Identification trên Laptop TRƯỚC khi flash
python training_tinyml/evaluate_model.py
:: → Xem mục 5 IDENTIFICATION ACCURACY (argmax MAX-SIM — đúng logic hệ thống):
::     • Top-1 đúng người ≥ 90% → OK, sang bước D
::     • Nếu có cặp NHẦM LẪN (vd A → B) → ảnh 2 người đó quá giống nhau hoặc
::       có ảnh tối/lệch: xóa thư mục người đó → enroll LẠI với ánh sáng đủ và
::       đa dạng góc → chạy lại bước B → C cho tới khi sạch
:: → Chạy main.py — NHỚ KHỞI ĐỘNG LẠI nếu đang mở (main.py load DB lúc khởi động):
python host_laptop/main.py
:: → Người mới phải được nhận đúng tên; người lạ (không đăng ký) → UNKNOWN

:: D. Nạp lại firmware lên ESP32 (BẮT BUỘC khi face_database.h thay đổi)
powershell -ExecutionPolicy Bypass -File D:\PROJECT_5_DIEM_DANH_KHUON_MAT\tools\flash_project_5.ps1 -Port COM3
:: hoặc:
cd D:\PROJECT_5_DIEM_DANH_KHUON_MAT\firmware_esp32 && idf.py -p COM3 flash monitor

:: E. Đứng trước Camera OV5640 để kiểm tra trực tiếp (100% Standalone)
:: → Serial Monitor in: 🔍 [AI] Frame hiện tại: <tên-người-mới> (Độ tin cậy: ... >= ngưỡng)
:: → Đủ 3 frame cùng tên → SUCCESS <tên-người-mới> + LED xanh sáng + 1 tiếng bíp ngắn!
```

> **⚠️ BÀI HỌC THỰC TẾ (đã gặp):** ảnh chụp TỐI (độ sáng mean ~42-89 thay vì ~100-155)
> khiến embedding kém tin cậy → 2 người bị nhầm lẫn nhau (Identification chỉ 66.7%).
> enroll_tool giờ chặn ảnh tối (<65) / cháy sáng (>200) ngay lúc chụp. Nếu đã lỡ enroll
> dữ liệu xấu: **xóa thư mục người đó → enroll lại với ánh sáng đầy đủ → chạy lại B → C**.

> **Lưu ý về đa người dùng:**
> - Mỗi người giữ tới **16 templates** trong `face_database.h` (từ tối đa 20 ảnh, trim 80%).
>   enroll_tool **chặn cứng ở 20 ảnh/người** — chụp nhiều hơn sẽ hỏng header C++ (đã fix).
> - Bộ nhớ: 1 người ≈ 16×128×4 = **8KB Flash** (const, không tốn RAM động). 10 người ≈ 80KB,
>   100 người ≈ 800KB — vẫn nằm trong phân vùng app 3MB.
> - Nếu dùng rất nhiều người (>200) mới cần giảm `MAX_TEMPLATES` (vd 16→8) trong
>   `training_tinyml/generate_embeddings.py` VÀ `training_tinyml/export_config.py`
>   (giữ 2 file này bằng nhau), rồi chạy lại update_face_database.py.
> - **Không** sao chép thư mục `nhien` rồi đổi tên — ảnh trong đó là khuôn mặt nhien,
>   người khác phải enroll riêng bằng enroll_tool.py.
> - Không bao giờ xóa ảnh gốc trong `data/registered_faces/<tên>/` — chúng là nguồn tái tạo
>   database. Muốn XÓA người: xóa thư mục của họ → chạy lại update_face_database.py → flash.
> - Tên người mới chỉ dùng ký tự không dấu, viết liền (vd `thao`, `van_a`) — tên nằm trong
>   chuỗi C++ của face_database.h, dấu tiếng Việt sẽ hiển thị sai trên Serial.
