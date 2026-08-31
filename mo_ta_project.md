# 📖 MÔ TẢ CHI TIẾT TẤT CẢ CÁC THƯ MỤC VÀ TỪNG FILE TRONG TOÀN BỘ DỰ ÁN

Dưới đây là tài liệu mô tả đầy đủ, chi tiết và toàn diện 100% về kiến trúc của hệ thống. Dự án được thiết kế theo tiêu chuẩn **Mô-đun hóa (Modular)** chia làm 3 phân hệ độc lập hoàn toàn. Khi bạn sửa một phân hệ, các phân hệ khác sẽ không bị ảnh hưởng, tránh rủi ro "chỉnh file này hỏng file kia".

---

## 🏗️ PHÂN HỆ 1: `firmware_esp32/` (Vi Điều Khiển Nhúng)
Chứa mã nguồn C/C++ nạp trực tiếp vào **ESP32-S3** thông qua Arduino IDE. 
**Nhiệm vụ:** Hoạt động như một "Bộ Não Chạy Biên" (Edge AI). Nó chỉ nhận ảnh qua Wi-Fi và tự chạy suy luận AI.
**Lưu ý:** ESP32 trong dự án này hoàn toàn **không sử dụng màn hình LCD**. Giao diện được chuyển hết về máy tính.

*   `firmware_esp32.ino`: File chính của Arduino IDE (nhạc trưởng). Khởi tạo Wi-Fi, nhận gói tin UDP ảnh JPEG từ `host_laptop`, giải mã, sau đó đẩy qua mô-đun AI để dò mặt và nhận diện, cuối cùng gửi lại kết quả.
*   `ai_config.h`: File chứa các tham số bộ nhớ (Arena Size) và cấu hình để ESP32 tự động cấp phát PSRAM/SRAM khi biên dịch.
*   `ai_face_detector.h` & `ai_face_detector.cpp`: Cắt ảnh và gọi mô hình phát hiện khuôn mặt (BlazeFace). 
*   `ai_face_recognizer.h` & `ai_face_recognizer.cpp`: Nạp mô hình nhận diện (Ghost-TinyFace), trích xuất vector 128 chiều, so khớp Cosine Similarity với cơ sở dữ liệu `face_database.h`.
*   `wifi_udp_server.h` & `wifi_udp_server.cpp`: Quản lý kết nối Wi-Fi, mở Socket UDP và nhận các mảng byte hình ảnh JPEG.
*   `image_decoder.h` & `image_decoder.cpp`: Giải mã gói dữ liệu ảnh JPEG thành mảng điểm ảnh (pixel RGB) đẩy vào AI.
*   `face_database.h`: Chứa mảng C++ dữ liệu khuôn mặt 128 chiều của những người dùng đã đăng ký.
*   `model_data.h`: File chứa trí tuệ của mô hình nhận diện khuôn mặt (Ghost-TinyFace) đã ép kiểu INT8 (kích thước siêu nhỏ ~294KB).
*   `detector_model_data.h`: File chứa trí tuệ của mô hình dò tìm khuôn mặt (BlazeFace) đã ép kiểu INT8. Mọi thuật toán C++ ở đây đều được đồng bộ chặt chẽ với Python trên máy tính.
*   `platformio.ini`: File dự phòng cấu hình cho PlatformIO.

---

## 💻 PHÂN HỆ 2: `host_laptop/` (Trạm Phát & Trạm Quản Lý Giao Diện)
Chứa mã nguồn Python chạy trên máy tính. 
**Nhiệm vụ:** Truyền camera tới ESP32, hiển thị giao diện HUD công nghệ cao, quản lý thêm người dùng và ghi log SQL. Không can thiệp vào thuật toán nhận diện bên trong ESP32.

*   `main.py`: File điều phối chính. Chạy luồng giao diện HUD và ghi log điểm danh vào CSDL SQLite.
*   `ip_camera_streamer.py`: Công cụ đọc luồng webcam máy tính, nén ảnh JPEG và liên tục truyền các gói tin UDP datagram qua Wi-Fi tới địa chỉ IP của ESP32.
*   `enroll_tool.py`: Bật camera chụp ảnh người dùng mới và lưu vào `data/registered_faces/`.
*   `convert_tflite_to_c.py`: Script nhỏ để đổi file `.tflite` thành `.h`.
*   **`core/`** & **`detector/`**: Chứa công cụ giả lập `UnifiedFaceDetector` (BlazeFace Emulator). Đây là một bước đột phá của hệ thống khi dùng Python mô phỏng chính xác 100% thuật toán C++ cắt ảnh (Fast Crop & Bilinear Interpolation) trên ESP32. Điều này giúp triệt tiêu hoàn toàn độ lệch pha miền dữ liệu (Domain Shift), đảm bảo ảnh lúc train và ảnh lúc thực thi là y hệt nhau.
*   **`recognizer/`**: Chứa các file giả lập thuật toán AI nhận diện chạy trên PC để test trước độ chính xác.
*   **`database/`**: Chứa `db_manager.py` quản lý file SQLite, ghi lại giờ giấc điểm danh.
*   **`ui/`**: Chứa `hud_renderer.py` chuyên vẽ giao diện HUD như trong phim viễn tưởng lên màn hình PC.

---

## 🧠 PHÂN HỆ 3: `training_tinyml/` (Xưởng Đào Tạo Trí Tuệ Nhân Tạo)
Đây là phân hệ tách biệt hoàn toàn dùng để nâng cấp "độ thông minh" cho AI.
Hệ thống nay đã dùng chuẩn **Zero-Retraining**: AI học đặc trưng chung (Universal) từ tập LFW quốc tế, nên bạn chỉ cần chạy train trên Colab 1 lần duy nhất, sau này thêm người mới KHÔNG CẦN train lại.

*   `download_lfw_dataset.py`: Tự động tải tập dữ liệu 13.000 khuôn mặt chuẩn LFW quốc tế từ internet.
*   `train_distillation.py`: Script huấn luyện trên Colab. Học cách chắt lọc tinh hoa từ LFW để phân biệt người này với người khác.
*   `quantize_qat_int8.py`: Chuyển đổi mô hình thành số nguyên int8, giảm dung lượng siêu nhỏ để đưa vào ESP32. Nó lấy ngẫu nhiên ảnh LFW để chống nhiễu hạt.
*   `update_face_database.py`: (Công cụ xài hàng ngày) Dùng để quét thư mục `registered_faces`, tạo file `face_database.h`. **Mỗi lần thêm người mới bạn chỉ cần chạy file này**.
*   `evaluate_model.py`: Chạy biểu đồ để tự bạn đánh giá xem mô hình mới đã xuất sắc chưa.
*   `generate_embeddings.py`: Được gọi tự động ngầm bởi update_face_database để tính toán ra vector 128 chiều.
*   `export_config.py`: Tự động tính toán Arena Size của RAM và xuất ra `ai_config.h`. (Đã được cấu hình cứng cấp phát 768KB trên PSRAM để nhường chỗ cho các phép toán nội suy khổng lồ của TFLite, giúp tránh lỗi Core Panic).

---

## 📂 CÁC THƯ MỤC LƯU TRỮ VÀ TÀI LIỆU (DATA & DOCS)

### Thư mục `data/` (Kho lưu trữ dữ liệu tĩnh)
*   `registered_faces/`: Các thư mục con chứa ảnh gốc khuôn mặt chụp của bạn và nhân viên công ty.
*   `face_database.json`: File dữ liệu vector đặc trưng 128 chiều (dạng text trung gian).
*   `attendance.db`: File CSDL SQLite ghi lại toàn bộ nhật ký người nào đã điểm danh giờ nào.

### CÁC FILE QUÁ HẠN (OBSOLETE / ĐÃ LOẠI BỎ)
*Hệ thống cũ có một số file, nay do chuyển sang Zero-Retraining nên không bao giờ dùng tới nữa (bạn có thể xóa tự do để sạch project):*
*   🗑️ `training_tinyml/finetune_locally.py`
*   🗑️ `training_tinyml/dataset_loader.py`
*   🗑️ `training_tinyml/fix_and_evaluate.py`
*   🗑️ `training_tinyml/find_bad_photos.py`

### Tài liệu hướng dẫn ở gốc (Root Directory)
*   `HUONG_DAN_TRAIN_COLAB.md`: Quy trình mới nhất vứt bỏ hoàn toàn sự rườm rà, đưa code lên Colab chạy 1 lần.
*   `mo_ta_project.md`: Chính là file bạn đang đọc.
*   `README.md`: Hướng dẫn chung.
*   `kich_hoat_moi_truong.md`: Lệnh nhanh mở conda.
*   `DANH_SACH_LOI_TEST_ARDUINO_IDE.md`: Tổng hợp gỡ lỗi C++.