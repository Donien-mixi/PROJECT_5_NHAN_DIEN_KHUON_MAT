# 🚀 EDGE AI FACE RECOGNITION (TINYML ON ESP32-S3)

Hệ thống Điểm danh và Nhận diện Khuôn mặt TinyML hoạt động theo mô hình **Hardware-in-the-Loop (HIL) Edge AI Coprocessor**. Dự án đã được tối ưu hóa đặc biệt cho vi điều khiển ESP32-S3 để giải quyết triệt để các bài toán về độ trễ, giới hạn bộ nhớ (RAM/Flash) và độ chính xác của mô hình.

---

## 📌 Tổng quan Kiến trúc Hệ thống

* **🧠 Edge AI Accelerator (ESP32-S3):** 
  * Sử dụng model mạng nơ-ron tùy chỉnh **Ghost-TinyFace INT8 (64x64)**.
  * Tối ưu hóa bộ nhớ **Tensor Arena (136KB)** phân bổ trực tiếp trên **Internal SRAM** (thay vì PSRAM) để đạt tốc độ suy luận (inference) cao nhất và tránh hiện tượng phân mảnh.
  * Sử dụng **TFLite MicroMutableOpResolver** (chỉ nạp 5 Ops cần thiết) giúp tiết kiệm tối đa dung lượng bộ nhớ.
  * **Dual-Core Architecture:** Core 0 chuyên xử lý suy luận TFLite, Core 1 xử lý ngoại vi và I/O (Giao tiếp Serial, điều khiển màn hình I2C LCD 1602, Còi báo Buzzer).

* **💻 Peripheral & Host Hub (Laptop):** 
  * Cung cấp luồng dữ liệu Video (Camera) và phát hiện khuôn mặt bằng **OpenCV YuNet**.
  * Tích hợp **Biometric Filtering** (kiểm tra Tỷ lệ khung hình, cấu trúc khuôn mặt) để loại bỏ ngay lập tức các vật thể giả mạo (đồ vật, bàn tay) trước khi gửi xuống ESP32.
  * Triển khai thuật toán **Tracking Lock**, tự động khóa nhãn nhận diện khi người dùng còn đứng trong khung hình và chỉ nhận diện lại khi họ bước ra rồi quay lại. Điều này giúp tối ưu hóa luồng giao tiếp Serial và giữ trạng thái điểm danh ổn định.

* **⚡ Giao tiếp Serial:** Giao thức truyền ảnh 64x64 siêu tốc qua cổng Type-C (Baudrate 921,600 bps), có cơ chế ack đồng bộ hóa khung hình.

---

## 📂 Cấu trúc Thư mục Dự án

```text
PROJECT_5_DIEM_DANH_KHUON_MAT/
│
├── firmware_esp32/             # 🧠 Firmware C++/Arduino nạp cho ESP32-S3
│   └── src/
│       └── tinyml_recognizer/  # Source code chính (Dual-Core, TFLite, LCD, Buzzer)
│
├── host_laptop/                # 💻 Code Python xử lý trên máy tính
│   ├── main.py                 # File thực thi chính (Webcam, Tracking Lock, Biometric)
│   ├── enroll_tool.py          # Tiện ích đăng ký và thu thập dữ liệu khuôn mặt
│   ├── bridge/                 # Các module hỗ trợ (giao tiếp Serial esp32_serial.py)
│   └── detector/               # Tích hợp mô hình OpenCV YuNet
│
├── data/                       # Chứa ảnh đăng ký người dùng (registered_faces)
│
├── ngu_canh.md                 # 📖 Lịch sử chi tiết ngữ cảnh, debug và cấu trúc dữ liệu
├── ROADMAP_TINYML_ESP32S3.md   # 📄 Bản thiết kế & Lộ trình thực tế của dự án
└── requirements.txt            # Danh sách thư viện Python
```

---

## 📖 Tài Liệu Quan Trọng
* 👉 Xem kế hoạch và thiết kế kiến trúc hệ thống chuẩn tại: **[ROADMAP_TINYML_ESP32S3.md](ROADMAP_TINYML_ESP32S3.md)**.
* 👉 Xem chi tiết các vấn đề hóc búa đã giải quyết (cách xử lý lỗi TFLite, tối ưu SRAM, Tracking Lock) tại: **[ngu_canh.md](ngu_canh.md)**.
