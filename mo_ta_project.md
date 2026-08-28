Viewed README.md:1-224

Dựa vào tài liệu `README.md` của dự án, hệ thống của bạn được chia thành 4 phần chính tương ứng với các thư mục. Dưới đây là chức năng cụ thể của từng thư mục và các file bên trong:

### 1. Thư mục `host_laptop/` (Ứng dụng chạy trên máy tính)
Đây là nơi chứa toàn bộ mã nguồn của hệ thống điểm danh để chạy giả lập và kiểm thử trên Laptop trước khi đưa xuống mạch ESP32.
*   `main.py`: File điều phối chính. Nó gọi Camera, lấy ảnh truyền cho bộ phận Detect (tìm mặt), sau đó truyền cho Recognize (nhận diện) và cuối cùng ghi kết quả xuống CSDL (DB).
*   `enroll_tool.py`: Công cụ dùng để thu thập (chụp) ảnh khuôn mặt của những người dùng mới muốn đăng ký vào hệ thống.
*   **Thư mục `detector/` (Phát hiện khuôn mặt):**
    *   `yunet_detector.py`: Chứa thuật toán phát hiện vị trí khuôn mặt trong khung hình (dựa trên YuNet/MediaPipe).
    *   `*.onnx`: Chứa trọng số (file mô hình) của YuNet.
*   **Thư mục `recognizer/` (Xác thực danh tính):**
    *   `face_recognizer.py`: Làm 3 nhiệm vụ: Cắt và chuẩn hóa khuôn mặt -> Đưa vào AI trích xuất vector 128 chiều (embedding) -> So sánh vector đó với CSDL xem là ai. Tích hợp cơ chế "Temporal Voting" (phải khớp 3 lần liên tiếp mới xác nhận).
*   **Thư mục `database/` (Cơ sở dữ liệu):**
    *   `db_manager.py`: Chịu trách nhiệm tương tác (thêm, đọc) với file SQLite để ghi lại lịch sử điểm danh.
*   **Thư mục `ui/` (Giao diện):**
    *   `hud_renderer.py`: Đoạn code vẽ các khung chữ nhật, hiển thị tên, màu sắc (Xanh/Đỏ) lên cửa sổ Camera (OpenCV).

### 2. Thư mục `training_tinyml/` (Khu vực huấn luyện AI)
Chứa tất cả các kịch bản (script) để huấn luyện mô hình học sâu (Deep Learning) nhận diện khuôn mặt sao cho mô hình đủ nhỏ gọn để chạy được trên vi điều khiển.
*   `models/ghost_tinyface.py`: File định nghĩa cấu trúc (kiến trúc) mạng nơ-ron của mô hình Ghost-TinyFace siêu nhẹ (Input: ảnh 64x64 đen trắng -> Output: vector 128 chiều).
*   `train_distillation.py`: Script quan trọng nhất dùng để huấn luyện mô hình. Dạy mô hình nhỏ (Student) học theo mô hình lớn (Teacher - SFace) qua phương pháp Knowledge Distillation.
*   `dataset_loader.py`: Code dùng để đọc, tải và xử lý (augmentation) dữ liệu ảnh trước khi ném vào mô hình để huấn luyện.
*   `download_lfw_dataset.py`: Script tự động tải bộ dữ liệu công khai LFW (hơn 5000 ảnh) từ Internet về.
*   `evaluate_model.py`: Script dùng để chạy bài kiểm tra đo độ chính xác (Accuracy) của mô hình sau khi train xong.
*   `generate_embeddings.py` & `update_face_database.py`: Chuyển đổi các bức ảnh chụp ở bước đăng ký thành chuỗi số (vector) và cập nhật vào file `face_database.json`.
*   `quantize_qat_int8.py`: Script thực hiện "Lượng tử hóa" - ép kiểu mô hình từ số thực 32-bit (chạy trên PC) xuống số nguyên 8-bit. Việc này giúp giảm dung lượng model từ 700KB xuống 160KB để vừa với bộ nhớ của ESP32.
*   **Thư mục `weights/` (Lưu trữ trọng số/Mô hình):**
    *   `tinyface_backbone.keras`: File mô hình chuẩn chạy trên Laptop.
    *   `tinyface_int8.tflite`: File mô hình siêu nhỏ (INT8) để nạp thẳng xuống mạch ESP32.
    *   `face_recognition_sface_2021dec.onnx`: Mô hình người Thầy (SFace).

### 3. Thư mục `data/` (Kho lưu trữ dữ liệu)
Nơi chứa toàn bộ dữ liệu "cứng" phục vụ cho vận hành hệ thống.
*   **Thư mục `registered_faces/`**: Chứa các file ảnh gốc (chụp bằng `enroll_tool.py`) của 3 người dùng đã đăng ký.
*   `face_database.json`: Bộ não tra cứu của hệ thống. Chứa các con số (embeddings 128-D) đại diện cho khuôn mặt của từng người. Nhận diện chính là so sánh với file này.
*   `attendance.db`: File CSDL SQLite ghi lại toàn bộ nhật ký (VD: Nguyễn Văn A điểm danh lúc 08:00 AM).

### 4. Thư mục `colab_training/` và các file khác
*   **Thư mục `colab_training/`**: Là bản sao được đóng gói của thư mục `training_tinyml` chuyên dùng để tải lên Google Colab nhằm lợi dụng sức mạnh GPU miễn phí của Google để huấn luyện mô hình cho nhanh.
*   `requirements.txt`: File liệt kê danh sách các thư viện Python (như OpenCV, TensorFlow...) cần thiết phải cài đặt bằng lệnh `pip install` để dự án có thể chạy được trên Laptop.