# Hướng dẫn chạy project (các lệnh nhanh)

> **Quy trình chuẩn:** Webcam → cắt vuông / thu nhỏ 128x128 → nén JPEG → chuyển màu như ESP32 → model tìm mặt (BlazeFace) → cắt khuôn mặt về 64x64 → chỉnh sáng đều (HE) → model nhận diện (Ghost-TinyFace).
> Dùng cổng TCP 12345, ngưỡng 0.70/0.80, chạy 3 ảnh mới chốt, báo bằng 2 LED + còi.

> **Lưu ý quan trọng:**
> - **Tìm mặt (BlazeFace)** và **nhận diện (Ghost-TinyFace)** là 2 model dùng cùng lúc. Cả hai **đã có sẵn** dạng INT8 trong project, nên bình thường **không cần chạy bước 4**.
> - Chỉ chạy bước 4 khi bạn **train lại / đổi model**.
> - Nếu hệ thống còn nhận nhầm giữa những người đã đăng ký (xem mục 5 của `evaluate_model.py`), hãy train lại model trên Colab rồi chạy lại bước 4. **Không cần chụp lại ảnh** — dữ liệu sẽ được tạo lại từ ảnh sẵn có.

> **Cách chạy lệnh Python:** dùng môi trường `projet_5`.
> - Mở **Anaconda Prompt** rồi gõ `conda activate projet_5`.
> - Vào thư mục project: `cd /d d:\PROJECT_5_DIEM_DANH_KHUON_MAT`
> - Nếu máy không nhận `conda`, dùng đường dẫn đầy đủ: `C:\Users\DONG NHIEN\.conda\envs\projet_5\python.exe ...`
> - Nếu gặp lỗi hiển thị tiếng Việt (`UnicodeEncodeError`), gõ trước: `set PYTHONIOENCODING=utf-8`

```bat
:: 1. Cài thư viện Python (chỉ lần đầu)
pip install -r requirements.txt

:: Kiểm tra TensorFlow đã cài chưa
python -c "import tensorflow; print('TF', tensorflow.__version__)"

:: 2. Thu ảnh đăng ký: 20 ảnh/người (14 thẳng + 6 nghiêng nhẹ)
::    Tool lưu ảnh xám 64x64, chặn ở 20 ảnh/người, từ chối ảnh tối (<65) hoặc cháy sáng (>200)
python host_laptop/enroll_tool.py
:: → nhập tên (ví dụ: thao), nhấn 'C' để chụp, 'Q' để thoát
:: → Kiểm tra thư mục data\registered_faces\<tên>\ có đúng 20 file .png

:: 3. Tạo lại dữ liệu khuôn mặt (không cần train lại)
python training_tinyml/generate_embeddings.py
:: → sinh data/face_database.json + firmware_esp32/face_database.h
python training_tinyml/update_face_database.py
:: → sinh lại firmware_esp32/ai_config.h

:: 4. CHỈ KHI ĐỔI MODEL: nén lại cả 2 model (không train lại)
::    Nếu vẫn dùng model có sẵn thì BỎ QUA bước này.
:: 4a. Model nhận diện
python training_tinyml/quantize_qat_int8.py
:: → training_tinyml/weights/tinyface_int8.tflite + firmware_esp32/model_data.h
:: 4b. Model tìm mặt
python training_tinyml/quantize_detector_int8.py
:: → training_tinyml/weights/face_detection_short_range_int8.tflite
python host_laptop/convert_tflite_to_c.py --model host_laptop\detector\face_detection_short_range_int8.tflite --out firmware_esp32\detector_model_data.h --symbol g_detector_model
copy training_tinyml\weights\face_detection_short_range_int8.tflite host_laptop\detector\face_detection_short_range_int8.tflite

:: 5. Kiểm tra và đánh giá model (độ đúng, ngưỡng, nhầm lẫn)
::    (muốn đo người lạ thì thêm ảnh vào data/registered_faces/Impostor_XX/)
python training_tinyml/evaluate_model.py

:: 6. Test trên laptop trước khi nạp chip (bắt buộc)
python host_laptop/main.py
:: → cùng quy trình như ESP32, hiện cửa sổ HUD
```

---

## 7. Nạp firmware lên ESP32-S3 (ESP-IDF 5.3)

### 7.1 Cách nối dây phần cứng
> Camera OV5640 cắm trực tiếp vào khe trên board (chuẩn Freenove / S3-EYE).
> Các chân báo kết quả cắm vào header tự do:
- **Còi:** chân tín hiệu vào **GPIO 1** (GND nối GND).
- **LED xanh (thành công):** qua điện trở 220Ω vào **GPIO 2**, chân còn lại vào GND.
- **LED đỏ (người lạ):** qua điện trở 220Ω vào **GPIO 3**, chân còn lại vào GND.

### 7.2 Nạp firmware

> **Lưu ý:** đóng mọi cửa sổ đang dùng cổng COM (Serial Monitor, Arduino IDE) trước khi nạp.

**Cách 1 (khuyên dùng) — script 1 chạm:**
```powershell
powershell -ExecutionPolicy Bypass -File D:\PROJECT_5_DIEM_DANH_KHUON_MAT\tools\flash_project_5.ps1 -Port COM3
```
*(Đổi `COM3` cho đúng cổng thật. Script tự build, nạp và mở Serial Monitor.)*

**Cách 2 — lệnh trực tiếp:**
```powershell
cd D:\PROJECT_5_DIEM_DANH_KHUON_MAT\firmware_esp32
idf.py -p COM3 flash monitor
```
*(Bấm `Ctrl + ]` để thoát Serial Monitor.)*

### 7.3 Log khởi động mong đợi
Khi khởi động thành công sẽ thấy các dòng kiểu:
- `🚀 PROJECT 5: STANDALONE FACE ATTENDANCE SYSTEM (OV5640 CAM)`
- `CPU Freq: 240 MHz`
- `Bộ nhớ Octal PSRAM : 8 MB`
- `✅ SPIFFS đã mount thành công tại /spiffs`
- `📷 [CameraTask] Bắt đầu thu thập ảnh...`
- `🧠 [AITask] Đang khởi tạo mô hình AI...`
- `[Detector] Arena used ...`, `[Recognizer] Arena used ...`
- `[WiFi] Connected` kèm IP, ví dụ `http://192.168.1.150`
- `🌐 [Web Server] HTTP Server da khoi dong tren port 80!`
- `[PERF] det: ...ms rec: ...ms cy: ...`

Nếu không có Wi-Fi, chip vẫn tự chạy độc lập, nhận diện và báo LED/còi bình thường.

---

## 8. Xem camera và lịch sử điểm danh qua web

Sau khi chip kết nối Wi-Fi, dùng điện thoại/laptop cùng mạng mở:
👉 `http://<IP-của-ESP32>/`

Web có:
1. **Xem camera trực tiếp.**
2. **Trạng thái AI:** tên người đang đứng trước camera + độ giống (ví dụ `nhien - 84.5%`).
3. **Đổi chiều ảnh:** nút Lật dọc và Lật gương.
4. **Lịch sử điểm danh:** xem bảng, tải file `/attendance.csv`, xóa lịch sử.

> **Độc lập hoàn toàn:** không cần mở laptop chạy `ip_camera_streamer.py` nữa.

> **Hiệu năng:** bộ tăng tốc trong `firmware_esp32/esp_nn/` giúp AI nhanh hơn nhiều. Nếu build lỗi, mở `esp_nn_glue.h` đổi `AI_ESP_NN_CONV 0` để quay về bản chậm hơn nhưng chạy đúng.
> **Ngưỡng từng người:** sau khi tạo lại dữ liệu, phải nạp lại firmware thì ngưỡng mới có tác dụng.

> **Ghi chú:**
> - `192.168.1.XXX` chỉ là ví dụ. Sửa `ssid/password` trong `firmware_esp32/wifi_udp_server.cpp` (hoặc `web_server.cpp`) cho đúng mạng của bạn.
> - Không dùng màn hình LCD — chỉ 2 LED + còi + Serial.
> - Nếu shell không nhận `conda`, dùng đường dẫn đầy đủ tới `python.exe` của môi trường.
> - Nhớ `set PYTHONIOENCODING=utf-8` nếu gặp lỗi tiếng Việt.

---

## 9. Thêm người mới (không cần train lại)

Thêm người mới chỉ cần **thu 20 ảnh**, không train lại. Cả laptop và chip tự nhận ra.

```bat
:: A. Thu 20 ảnh cho người mới (tên không dấu, viết liền — ví dụ: thao)
python host_laptop/enroll_tool.py
:: → nhập tên, nhấn 'C' chụp đủ 20 ảnh, 'Q' thoát
:: → Tool chặn ở 20 ảnh, từ chối ảnh quá tối (<65) / cháy sáng (>200)
:: → Kiểm tra: đúng 20 file .png trong data\registered_faces\thao\

:: B. Tạo lại dữ liệu (quét tất cả mọi người, tự thêm người mới)
python training_tinyml/update_face_database.py
:: → sinh lại data/face_database.json + firmware_esp32/face_database.h + ai_config.h
:: Kiểm tra: mở firmware_esp32\face_database.h xem đủ tên mọi người

:: C. Kiểm tra trên laptop trước khi nạp chip (bắt buộc khi ≥2 người)
python training_tinyml/evaluate_model.py
:: → xem mục 5 IDENTIFICATION ACCURACY: đúng người ≥ 90% là tốt
:: → nếu có cặp nhầm lẫn: xóa thư mục người đó, enroll lại, chạy lại B → C
python host_laptop/main.py

:: D. Nạp lại firmware (bắt buộc khi face_database.h thay đổi)
powershell -ExecutionPolicy Bypass -File D:\PROJECT_5_DIEM_DANH_KHUON_MAT\tools\flash_project_5.ps1 -Port COM3

:: E. Đứng trước camera để kiểm tra thật
:: → Serial in: 🔍 [AI] ... (Độ tin cậy: ...)
:: → Đủ 3 ảnh cùng tên → SUCCESS + LED xanh + 1 tiếng bíp!
```

> **Bài học thực tế:** ảnh chụp quá tối (độ sáng trung bình ~42-89 thay vì ~100-155) sẽ khiến dữ liệu kém tin cậy và dễ nhầm người. `enroll_tool.py` đã chặn ảnh tối/cháy sáng ngay lúc chụp. Nếu lỡ thu ảnh xấu: xóa thư mục người đó, thu lại với đủ ánh sáng rồi chạy lại B → C.

> **Vài lưu ý khi dùng nhiều người:**
> - Mỗi người giữ tối đa **16 mẫu** (từ 20 ảnh, bỏ 20% ảnh kém nhất).
> - Mỗi người tốn khoảng **8KB** bộ nhớ flash. 100 người khoảng 800KB, vẫn vừa.
> - **Không** copy thư mục `nhien` rồi đổi tên — ảnh bên trong là mặt nhien, người khác phải thu riêng.
> - Không xóa ảnh gốc trong `data/registered_faces/<tên>/` — đó là nguồn để tạo lại dữ liệu.
> - Tên người mới dùng chữ không dấu, viết liền (ví dụ `thao`, `van_a`).
