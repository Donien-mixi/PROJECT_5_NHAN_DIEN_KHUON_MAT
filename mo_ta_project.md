# Mô tả chi tiết các thư mục và file trong project

Project chia làm 3 phần chính, cộng thêm phần tài liệu. Ba phần phải "nói chuyện" khớp với nhau (cùng model, cùng cách xử lý ảnh, cùng dữ liệu) để khi chuyển từ laptop xuống chip không bị lệch.

---

## Phần 1: `firmware_esp32/` — Code chạy trên chip ESP32-S3

Đây là code nạp vào chip **ESP32-S3 WROOM-1 N16R8 CAM** bằng **ESP-IDF 5.3**.

**Nhiệm vụ:** Chạy như một máy điểm danh khuôn mặt hoàn chỉnh, tự chụp ảnh, tự tìm mặt, tự nhận diện, không cần máy tính.

**Thiết bị báo kết quả:**
- **LED xanh** (GPIO 2): điểm danh thành công.
- **LED đỏ** (GPIO 3): người lạ, bị từ chối.
- **Còi** (GPIO 1): 1 tiếng bíp ngắn khi thành công, 2 tiếng bíp dài khi người lạ.
- **Bộ nhớ trong chip:** tự lưu danh sách điểm danh vào file `/attendance.csv`.
- **Web server:** xem camera trực tiếp, xem tên đang nhận diện, đổi chiều ảnh, tải danh sách điểm danh qua trình duyệt điện thoại/laptop. Không có Wi-Fi thì vẫn chạy bình thường.

**Các file:**

* `CMakeLists.txt` và `sdkconfig.defaults`: cấu hình build, cho chip chạy 240MHz, bật RAM ngoài PSRAM 8MB, tối ưu tốc độ, bật tăng tốc tính toán SIMD.
* `partitions.csv`: chia bộ nhớ flash 16MB (4MB cho app, 2MB cho dữ liệu SPIFFS).
* `main/main.cpp`: file khởi động chính. Khởi tạo bộ nhớ, camera, web server, và 2 việc chạy song song trên 2 nhân:
  - **Nhân 1 (CameraTask):** liên tục chụp ảnh từ camera đưa vào bộ đệm.
  - **Nhân 0 (AITask):** chạy toàn bộ AI: tìm mặt → cắt và chỉnh sáng → nhận diện → so khớp → đếm 3 ảnh → bật LED/còi và ghi điểm danh.
* `firmware_esp32.ino`: bản Arduino cũ, giữ lại để tham khảo.
* `camera_pins.h`: khai báo chân cắm camera OV5640 và chân còi/LED.
* `camera_driver.h/.cpp`: điều khiển camera OV5640, chụp ảnh 240x240, thu nhỏ về 128x128 cho AI. Đồng thời chia sẻ ảnh cho web server.
* `web_server.h/.cpp`: web server chạy nền, có giao diện web xem camera trực tiếp, xem tên và độ tin cậy, nút lật ảnh, tải/xóa danh sách điểm danh.
* `ai_config.h`: chứa các thông số chung (kích thước ảnh, ngưỡng, kích thước bộ nhớ). File này được sinh tự động, đừng sửa tay.
* `ai_face_detector.h/.cpp`: tìm khuôn mặt bằng BlazeFace, cắt vuông khuôn mặt và cắt về 64x64.
* `esp_nn/` và `esp_nn_glue.h/.cpp`: bộ tăng tốc tính toán của nhà sản xuất chip, giúp AI chạy nhanh hơn nhiều lần. Có xử lý một lỗi nhỏ của bộ tăng tốc để giữ kết quả chính xác.
* `ai_face_recognizer.h/.cpp`: model nhận diện Ghost-TinyFace, tạo ra 128 số đặc trưng, so khớp với dữ liệu và trả về tên người khớp nhất.
* `face_database.h`: dữ liệu khuôn mặt 3 người (nhien, thao, toan), mỗi người 16 mẫu, mỗi mẫu 128 số.
* `model_data.h`: model nhận diện dưới dạng mảng số, khoảng 160KB.
* `detector_model_data.h`: model tìm mặt dưới dạng mảng số, khoảng 183KB.
* `wifi_udp_server.cpp` và `image_decoder.cpp`: code cũ (nhận ảnh từ laptop qua Wi-Fi), đã tắt vì không dùng nữa.

---

## Phần 2: `host_laptop/` — Code chạy trên laptop

Code Python dùng trên máy tính để **test** và **quản lý**. Phần này không tham gia vào lúc chạy thật trên chip.

* `main.py`: chạy thử toàn bộ quy trình trên laptop và hiện cửa sổ HUD (bảng thông tin). Dùng đúng cách xử lý ảnh như ESP32 để kết quả khớp.
* `ip_camera_streamer.py`: biến laptop thành "camera ngố" — chỉ đọc webcam, thu nhỏ ảnh 128x128, nén JPEG và gửi qua TCP cổng 12345. Không nhận diện gì cả.
* `enroll_tool.py`: thu 20 ảnh cho một người mới. Từ chối ảnh quá tối hoặc quá sáng để dữ liệu tốt.
* `convert_tflite_to_c.py`: đổi file model `.tflite` thành file `.h` để nhúng vào firmware.
* `core/vision_utils.py`: các hàm xử lý ảnh dùng chung — cắt vuông, chuyển màu giống ESP32, chỉnh sáng đều (HE), cắt và thu nhỏ khuôn mặt về 64x64.
* `detector/blazeface_esp32.py`: mô phỏng model tìm mặt BlazeFace trên laptop, giống hệt trên ESP32.
* `recognizer/face_recognizer.py`: model nhận diện, tạo 128 số, so khớp lấy điểm cao nhất, và bộ đếm 3 ảnh chống nhiễu.
* `database/db_manager.py`: lưu lịch sử điểm danh vào file SQLite, có chống ghi trùng trong 30 giây.
* `ui/hud_renderer.py`: vẽ khung, tên, độ tin cậy lên cửa sổ laptop. **Không phải màn hình gắn trên ESP32.**

**Quy tắc để laptop và ESP32 giống nhau:**

1. Webcam được cắt vuông giữa rồi thu nhỏ về 128x128 (chỉ 1 lần).
2. Nén JPEG chất lượng 80, gửi kèm độ dài, qua TCP cổng 12345.
3. Laptop phải mô phỏng đúng bước chuyển màu RGB565 giống ESP32 trước khi xử lý.
4. Model tìm mặt dùng cùng cách và cùng ngưỡng (0.80).
5. Cắt khuôn mặt bằng đúng thuật toán, rồi chỉnh sáng đều (HE) giống hệt trên chip.
6. Model nhận diện dùng cùng cách chuẩn hóa và cùng cách so khớp (lấy điểm cao nhất trong 16 mẫu).
7. Ngưỡng: 0.70 chung, riêng từng người tối đa 0.75.

---

## Phần 3: `training_tinyml/` — Nơi huấn luyện AI và tạo dữ liệu

Đây là nơi "dạy" AI và tạo ra model/dữ liệu mà laptop và chip cùng dùng.

**Cách hoạt động:** AI học đặc trưng khuôn mặt chung từ tập dữ liệu lớn **CASIA-WebFace** (hàng trăm nghìn ảnh). Có thầy (SFace) dạy trò (Ghost-TinyFace nhỏ gọn). Thêm người mới chỉ cần thu 20 ảnh, **không cần train lại**.

* `download_casia_dataset.py`: đọc dữ liệu CASIA-WebFace trực tiếp từ file zip, tách ra ảnh cho cả thầy và trò.
* `train_distillation.py`: huấn luyện model trên Google Colab. Trò học bắt chước thầy, kèm kỹ thuật giúp phân biệt rõ người này với người khác và không bị ảnh hưởng bởi ánh sáng.
* `quantize_qat_int8.py`: nén model nhận diện xuống INT8 (nhẹ, khoảng 160KB) và sinh file `model_data.h`.
* `quantize_detector_int8.py`: nén model tìm mặt xuống INT8 (khoảng 183KB).
* `generate_embeddings.py`: từ ảnh đã thu, tạo dữ liệu 128 số cho mỗi người (giữ 16 mẫu tốt nhất) và ghi ra `data/face_database.json` + `firmware_esp32/face_database.h`.
* `update_face_database.py`: tự động chạy lại bước tạo dữ liệu khi thêm/xóa người. **Đây là cách thêm người mới, không cần train lại.**
* `export_config.py`: sinh file `ai_config.h` (các thông số chung).
* `evaluate_model.py`: kiểm tra, in ra các số liệu đánh giá (độ đúng, nhầm lẫn, ngưỡng).
* `colab_exports/`: bản copy model/dữ liệu sinh ra trên Colab.

---

## Phần 4: `data/` và tài liệu

**Thư mục `data/`:**
- `registered_faces/nhien|thao|toan/`: mỗi người 20 ảnh xám 64x64.
- `face_database.json`: dữ liệu 128 số của 3 người.
- `attendance.db`: lịch sử điểm danh (trên laptop).

**Tài liệu ở gốc project:**
- `HUONG_DAN_COLAB_TRAIN_V3.md`: hướng dẫn train model V3 trên Colab.
- `colab.zip`: gói code gửi lên Colab.
- `mo_ta_project.md`: file bạn đang đọc.
- `README.md`: giới thiệu tổng quan.
- `kich_hoat_moi_truong.md`: các lệnh chạy nhanh.
- `ket_qua_esp32.txt`: log chạy thật trên chip.
- `HUONG_DAN_COLAB_TRAIN_V2.md`: bản cũ, đã thay bằng V3.

---

## Giai đoạn 4 (tối ưu và vận hành)

Không thêm model mới, chỉ **đo lại, chỉnh lại cho tốt hơn** trên đúng phần cứng hiện có:

- **Tốc độ (đã xong):** thêm bộ tăng tốc, khóa chip ở 240MHz, giảm độ trễ. Kết quả đo thật: tìm mặt ~1.95s, nhận diện ~1.30s.
- **Độ chính xác và chọn ngưỡng (đã xong):** ngưỡng 0.70 chung, riêng từng người 0.75. Đo thật thấy nhận diện đúng liên tục.
- **Còn làm:** chống giả mạo bằng ảnh in/màn hình; thêm giờ thật (NTP); nạp firmware không cần cáp (OTA), thu ảnh ngay trên chip; test thực tế với nhiều người và nhiều điều kiện sáng.
