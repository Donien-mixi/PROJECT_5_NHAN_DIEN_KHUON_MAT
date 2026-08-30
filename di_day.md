Ran command: `Select-String -Path "d:\PROJECT_5_DIEM_DANH_KHUON_MAT\firmware_esp32\firmware_esp32.ino" -Pattern "LED" -Encoding utf8`
Ran command: `cat d:\PROJECT_5_DIEM_DANH_KHUON_MAT\firmware_esp32\firmware_esp32.ino`
Ran command: `Select-String -Path "d:\PROJECT_5_DIEM_DANH_KHUON_MAT\firmware_esp32\firmware_esp32.ino" -Pattern "BUZZER_PIN" -Encoding utf8`
Edited firmware_esp32.ino
Edited viec_can_lam.md

Tuyệt vời! Trong lúc chờ Colab "luyện não" cho AI, chúng ta sẽ tranh thủ setup phần cứng trên breadboard nhé. 

Đúng như bạn nhận xét, tôi đã sơ suất chưa ghi chi tiết phần Đèn LED. Tôi vừa lập tức sửa lại file [`viec_can_lam.md`](file:///d:/PROJECT_5_DIEM_DANH_KHUON_MAT/viec_can_lam.md) và cũng **đã viết thêm code điều khiển LED vào thẳng file `firmware_esp32.ino`** cho bạn rồi! 

Hệ thống bây giờ sẽ hoạt động như sau:
- **Thành công (Người quen):** Loa kêu 2 tiếng bíp ngắn + **Đèn XANH nháy 2 lần**.
- **Thất bại (Người lạ):** Loa kêu 1 tiếng bíp dài + **Đèn ĐỎ sáng dài**.

Dưới đây là hướng dẫn đi dây chi tiết trên Breadboard dành cho mạch ESP32-S3:

### 🛠️ CHUẨN BỊ LINH KIỆN
- 1 x Còi chíp (Buzzer) loại Active 3.3V hoặc 5V.
- 1 x Đèn LED màu XANH LÁ (Báo thành công).
- 1 x Đèn LED màu ĐỎ (Báo người lạ).
- 2 x Điện trở (220Ω hoặc 330Ω) dùng để cản dòng bảo vệ đèn LED.
- Dây cắm Breadboard (Jumper wires).

---

### 🔌 HƯỚNG DẪN ĐI DÂY (WIRING)

**1. Sơ đồ cắm Đèn LED XANH (Thành công) - Dùng chân GPIO 4**
* **Chân dài (Anode +)** của LED Xanh 👉 Cắm nối tiếp qua 1 con điện trở 220Ω 👉 Nối vào chân **GPIO 4** của ESP32.
* **Chân ngắn (Cathode -)** của LED Xanh 👉 Nối vào đường **GND** trên ESP32.

**2. Sơ đồ cắm Đèn LED ĐỎ (Cảnh báo) - Dùng chân GPIO 5**
* **Chân dài (Anode +)** của LED Đỏ 👉 Cắm nối tiếp qua 1 con điện trở 220Ω 👉 Nối vào chân **GPIO 5** của ESP32.
* **Chân ngắn (Cathode -)** của LED Đỏ 👉 Nối vào đường **GND** trên ESP32.

**3. Sơ đồ cắm Loa Buzzer - Dùng chân GPIO 15**
*(Lưu ý: Nếu bạn dùng Active Buzzer thì nó thường có nhãn (+) và (-))*
* Chân **(+)** của Buzzer 👉 Nối vào chân **GPIO 15** của ESP32.
* Chân **(-)** của Buzzer 👉 Nối vào đường **GND** trên ESP32.

---

### 💡 TÓM TẮT PINOUT ĐỂ BẠN DỄ DÒ:
- `GPIO 4`  ➡️ Cực dương LED Xanh.
- `GPIO 5`  ➡️ Cực dương LED Đỏ.
- `GPIO 15` ➡️ Cực dương Loa Buzzer.
- Tất cả các cực âm còn lại đều chụm chung về cổng `GND` của ESP32.

Bạn cứ thong thả đi dây cho đẹp và gọn gàng nhé! Lát nữa khi mô hình train xong, bạn đưa xuống ESP32 là hệ thống sẽ vừa hú còi vừa chớp đèn rất xịn xò đấy!