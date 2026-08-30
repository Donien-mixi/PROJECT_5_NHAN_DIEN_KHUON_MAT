# 📖 MÔ TẢ CHI TIẾT TẤT CẢ CÁC THƯ MỤC VÀ TỪNG FILE TRONG TOÀN BỘ DỰ ÁN

Dưới đây là tài liệu mô tả đầy đủ, chi tiết và toàn diện 100% từng thư mục, từng thư mục con và từng file trong toàn bộ dự án **Hệ Thống Điểm Danh Nhận Diện Khuôn Mặt Trên ESP32-S3 (TinyML Edge AI)**:

---

### 1. Thư mục `host_laptop/` (Ứng dụng chạy trên máy tính)
Chứa toàn bộ mã nguồn của hệ thống nhận diện điểm danh chạy giả lập và kiểm thử độ chính xác trên Laptop (Giai đoạn 1 & 2), đồng thời đóng vai trò là trạm phát video IP Camera không dây xuống vi điều khiển ESP32 (Giai đoạn 3).
*   `main.py`: File điều phối chính trên Laptop. Tự động mở Webcam -> Gọi bộ phát hiện khuôn mặt (Detector) -> Trích xuất đặc trưng và so khớp (Recognizer) -> Lọc chống nhiễu (Temporal Voting) -> Hiển thị giao diện công nghệ cao (HUD) và ghi log điểm danh vào CSDL SQLite.
*   `ip_camera_streamer.py`: Công cụ biến Laptop thành một IP Camera không dây. Đọc luồng webcam máy tính, cắt ảnh vuông 240x240, nén ảnh JPEG và liên tục truyền các gói tin UDP datagram (5-10KB) qua Wi-Fi tới địa chỉ IP của ESP32.
*   `enroll_tool.py`: Công cụ chụp ảnh thu thập dữ liệu khuôn mặt người dùng mới. Tự động căn chỉnh 5 điểm mốc vàng (mắt, mũi, khóe miệng) theo chuẩn Affine InsightFace và cắt lưu ảnh $64 \times 64$ pixels.
*   **Thư mục con `bridge/` (Cầu nối giao tiếp):**
    *   Thư mục dự phòng chứa các module giao tiếp dữ liệu nối tiếp (Serial/UART hoặc Socket) giữa Laptop và phần cứng ESP32.
*   **Thư mục con `detector/` (Phát hiện khuôn mặt):**
    *   `yunet_detector.py`: Thuật toán phát hiện khuôn mặt đa chế độ (hỗ trợ Google MediaPipe Mesh 468 điểm mốc hoặc OpenCV YuNet 5 điểm) kèm bộ lọc Primary Face Selector để loại bỏ 100% vật thể nhiễu nền (bao bì, tranh ảnh, quần áo).
    *   `face_detection_yunet_2023mar.onnx`: Trọng số mô hình AI YuNet chạy trên OpenCV DNN để phát hiện vị trí khuôn mặt.
*   **Thư mục con `recognizer/` (Xác thực danh tính):**
    *   `face_recognizer.py`: Nạp mô hình AI (Keras Float32 hoặc TFLite INT8) -> Trích xuất vector đặc trưng 128 chiều (128-D Embedding) từ ảnh $64 \times 64$ Grayscale -> So khớp khoảng cách Cosine Similarity với CSDL. Chứa lớp `TemporalVoter` yêu cầu trùng khớp liên tiếp 3 frame mới chốt kết quả.
*   **Thư mục con `database/` (Cơ sở dữ liệu):**
    *   `db_manager.py`: Quản lý file SQLite `attendance.db`, tự động khởi tạo bảng `attendance_logs` và ghi nhận lịch sử điểm danh kèm cơ chế Cooldown 30s chống ghi spam.
*   **Thư mục con `ui/` (Giao diện hiển thị):**
    *   `hud_renderer.py`: Chuyên vẽ giao diện HUD công nghệ cao lên khung hình OpenCV: Thanh Header (FPS, chế độ AI, thời gian suy luận Infer ms), ô ảnh thu nhỏ PiP (Picture-in-Picture) và khung Bounding Box Xanh/Đỏ.

---

### 2. Thư mục `firmware_esp32/` (Mã nguồn C++ nhúng trên ESP32-S3)
Chứa toàn bộ mã nguồn C++ chạy trực tiếp trên vi điều khiển **ESP32-S3 N16R8 (16MB Flash, 8MB PSRAM)** để xử lý nhận diện AI độc lập trên chip.
*   `platformio.ini`: File cấu hình PlatformIO. Khai báo vi điều khiển ESP32-S3, xung nhịp 240MHz, bật bộ nhớ PSRAM ngoài, cấu hình phân vùng nhớ `huge_app.csv`, thiết lập chân SPI cho màn hình LCD ILI9341 và tải các thư viện `TFT_eSPI`, `TJpg_Decoder`, `TensorFlowLite_ESP32`.
*   **Thư mục con `src/` (Mã nguồn C++ chính):**
    *   `main.cpp`: Nhạc trưởng điều phối đa nhân FreeRTOS (Dual-Core Asymmetric). Giao **Core 1** chuyên nhận ảnh JPEG qua Wi-Fi UDP và vẽ lên màn hình LCD; giao **Core 0** chuyên chạy các tác vụ AI tính toán nặng (Dò mặt, chạy TFLite Micro, so khớp Cosine Similarity). Đồng bộ luồng bằng Mutex Semaphore.
    *   `wifi_udp_server.h` & `wifi_udp_server.cpp`: Quản lý kết nối Wi-Fi và mở Socket UDP lắng nghe tại port 12345 để nhận các frame ảnh JPEG từ Laptop gửi xuống và lưu vào bộ đệm PSRAM.
    *   `display_ili9341.h` & `display_ili9341.cpp`: Điều khiển màn hình màu LCD TFT ILI9341 (240x320) qua giao tiếp SPI 8 chân. Giải mã JPEG vẽ luồng video trực tiếp và vẽ khung chữ nhật Xanh/Đỏ kèm tên người điểm danh.
    *   `ai_face_detector.h` & `ai_face_detector.cpp`: Định nghĩa cấu trúc `FaceBox` (x, y, w, h) và hàm tiền xử lý cắt ảnh, chuyển sang đen trắng (Grayscale), co giãn về đúng $64 \times 64$ pixels và chuẩn hóa pixel về dải `[-1.0, 1.0]`.
    *   `ai_face_recognizer.h` & `ai_face_recognizer.cpp`: Quản lý mô hình TFLite Micro trên ESP32. Chịu trách nhiệm nạp Tensor Arena (100KB-120KB), chạy `Invoke()` để rút trích vector 128 chiều, tính tích vô hướng Cosine Similarity với CSDL và trả về tên người điểm danh (hoặc `Unknown`).
    *   `face_database.h`: Chứa mảng C++ hằng số `FACE_DATABASE` lưu trữ sẵn các vector đặc trưng 128-D của 3 người dùng đã đăng ký để so khớp trực tiếp trên RAM.
    *   **Thư mục con `tinyml_recognizer/`:**
        *   `model_data.h`: File header chứa mảng byte Hex `const unsigned char g_model[]` của toàn bộ mô hình AI INT8 (~160KB) được nhúng thẳng vào bộ nhớ Flash ROM của ESP32.
        *   `face_database.h`: File CSDL cấu trúc `RegisteredUser` cho module `tinyml_recognizer`.

---

### 3. Thư mục `training_tinyml/` (Khu vực huấn luyện AI & Lượng tử hóa)
Chứa toàn bộ các script huấn luyện mô hình học sâu (Deep Learning) nhận diện khuôn mặt sao cho mô hình đủ nhỏ gọn để chạy mượt mà trên vi điều khiển.
*   `train_distillation.py`: Script quan trọng nhất dùng để huấn luyện mô hình. Dạy mô hình nhỏ (Student) học theo mô hình lớn (Teacher - SFace) qua phương pháp Knowledge Distillation (Chưng cất tri thức).
*   `dataset_loader.py`: Đọc, tải và tiền xử lý tăng cường dữ liệu ảnh (Data Augmentation) trước khi đưa vào mô hình huấn luyện.
*   `download_lfw_dataset.py`: Script tự động tải bộ dữ liệu khuôn mặt chuẩn LFW (hơn 5.000 ảnh) từ Internet về.
*   `evaluate_model.py`: Script chạy kiểm thử đo độ chính xác (Accuracy, ROC curve) của mô hình sau khi huấn luyện.
*   `generate_embeddings.py`: Quét ảnh các thành viên đã đăng ký, áp dụng thuật toán Trimmed Centroid (lọc bỏ 20% ảnh mờ/lệch) để tạo vector đại diện 128-D sạch và cập nhật vào file `face_database.json`.
*   `update_face_database.py`: Script hỗ trợ cập nhật nhanh CSDL khuôn mặt khi có thay đổi.
*   `quantize_qat_int8.py`: Script thực hiện "Lượng tử hóa" - ép kiểu mô hình từ số thực float32 (chạy trên PC) xuống số nguyên int8. Giảm dung lượng model từ 700KB xuống 160KB để vừa khít bộ nhớ SRAM của ESP32.
*   **Thư mục con `models/` (Kiến trúc mạng nơ-ron):**
    *   `ghost_tinyface.py`: File định nghĩa kiến trúc mạng nơ-ron **Ghost-TinyFace** siêu nhẹ sử dụng Ghost Bottleneck, Depthwise Separable Conv, ReLU6 và L2 Normalization (Input: ảnh 64x64 đen trắng -> Output: vector 128 chiều).
*   **Thư mục con `weights/` (Lưu trữ trọng số/Mô hình AI):**
    *   `tinyface_backbone.keras`: File mô hình chuẩn float32 chạy trên Laptop (~700KB).
    *   `tinyface_int8.tflite`: File mô hình lượng tử hóa INT8 để nạp xuống ESP32 (~160KB).
    *   `face_recognition_sface_2021dec.onnx`: Mô hình người Thầy (SFace 38MB).

---

### 4. Thư mục `data/` (Kho lưu trữ dữ liệu)
Nơi chứa toàn bộ dữ liệu "cứng" phục vụ cho vận hành hệ thống.
*   **Thư mục con `registered_faces/`**: Chứa các thư mục ảnh gốc (chụp bằng `enroll_tool.py`) của 3 người dùng đã đăng ký:
    *   `Mai_Thi_Thu_Thao_07_01_1975/`: Chứa bộ ảnh khuôn mặt đăng ký của người dùng 1.
    *   `Trinh_Dong_Nhien_17_01_06/`: Chứa bộ ảnh khuôn mặt đăng ký của người dùng 2.
    *   `Trinh_Minh_Toan/`: Chứa bộ ảnh khuôn mặt đăng ký của người dùng 3.
*   `face_database.json`: Bộ não tra cứu của hệ thống. Chứa các vector 128-D đại diện cho từng người để so khớp nhận diện.
*   `face_database_backup.json`: File sao lưu dự phòng của cơ sở dữ liệu vector.
*   `attendance.db`: File CSDL SQLite ghi lại toàn bộ nhật ký (Tên người, thời gian điểm danh, độ tương đồng Sim và thời gian infer).

---

### 5. Thư mục `tools/` (Công cụ chuyển đổi sang C/C++)
Chứa các script tiện ích tự động hóa việc đưa mô hình AI và CSDL từ Python sang nhúng trong mã nguồn C++ của ESP32:
*   `convert_tflite_to_c.py`: Đọc file mô hình `tinyface_int8.tflite` và xuất ra mảng byte C `model_data.h` để nạp vào Flash ESP32.
*   `convert_db_to_c.py`: Đọc file CSDL `face_database.json` và xuất ra file header C `face_database.h` chứa mảng hằng số `RegisteredUser face_database[]`.

---

### 6. Thư mục `colab_training/` (Gói huấn luyện đám mây Google Colab)
*   **Thư mục `colab_training/`**: Bản sao độc lập của thư mục `training_tinyml` kèm dữ liệu mẫu được đóng gói gọn gàng để tải lên Google Colab nhằm tận dụng GPU miễn phí để huấn luyện mô hình nhanh chóng.
*   `colab_training.zip`: File nén của gói huấn luyện Colab dùng để tải lên Google Drive / Google Colab chỉ với 1 thao tác.

---

### 7. Các file kịch bản & Tài liệu hướng dẫn ở thư mục gốc (Root Directory)
*   `README.md`: Tài liệu tổng quan toàn bộ dự án, bảng thông số kỹ thuật, lộ trình 4 giai đoạn phát triển và trạng thái tiến độ hiện tại.
*   `ROADMAP_TINYML_ESP32S3.md`: Bản thiết kế kiến trúc chi tiết chuyên sâu cho ESP32-S3, phân tích tài nguyên phần cứng, sơ đồ luồng dữ liệu Dual-Core và nguyên tắc tối ưu SRAM.
*   `DANH_SACH_LOI_TEST_ARDUINO_IDE.md`: Tài liệu tổng hợp toàn bộ các lỗi tồn đọng, thiếu sót trong firmware ESP32 và checklist cấu hình chuẩn trước khi nạp bằng Arduino IDE.
*   `full_tune_nhan_dien.md`: Cẩm nang hướng dẫn chi tiết cách tự tay tinh chỉnh 3 con số quan trọng nhất: Ngưỡng nhận diện (`threshold`), Ngưỡng dò mặt (`conf_threshold`) và Ngưỡng chống nhiễu (`required_votes`).
*   `kich_hoat_moi_truong.md`: Ghi chú ngắn gọn các câu lệnh chuyển ổ đĩa và kích hoạt môi trường ảo Anaconda (`conda activate projet_5`).
*   `verify_quantization.py`: Script kiểm thử tự động so sánh sai số Cosine Similarity giữa mô hình Keras Float32 và TFLite INT8 trên toàn bộ ảnh đăng ký (đã đạt độ tương đồng 99.87%).
*   `requirements.txt`: Danh sách tất cả các thư viện Python (OpenCV, TensorFlow, Mediapipe, v.v.) cần cài đặt để chạy dự án.
*   `.gitignore`: File cấu hình Git loại trừ các file tạm thời, file rác cache (`__pycache__`, file `.zip`, file `.db`).