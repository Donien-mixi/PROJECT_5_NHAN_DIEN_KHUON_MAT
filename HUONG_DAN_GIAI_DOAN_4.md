# Giai đoạn 4 — Tối ưu và vận hành thực tế

> Giai đoạn này không thêm model mới, chỉ **đo lại, chỉnh lại cho tốt hơn** trên đúng phần cứng đang có (ESP32-S3 N16R8 với 8MB PSRAM, 2 model INT8 BlazeFace ~183KB + Ghost ~160KB, ảnh 128x128).

---

## 1. Mục tiêu và ràng buộc

**Đi từ "chạy được" lên "chạy mượt, ổn định, chống gian lận"** — đo được, làm lại được, không phụ thuộc may mắn.

Ràng buộc:
- RAM trong 512KB, RAM ngoài PSRAM 8MB.
- 2 model chạy cùng lúc, đều cấp phát bộ nhớ trên PSRAM.
- Ảnh vào: 128x128, cắt khuôn mặt về 64x64 và chỉnh sáng đều (HE), giống hệt trên laptop.
- Logic "đúng 3 ảnh mới chốt" giữ nguyên khi tối ưu.
- Chỉ báo bằng 2 LED + còi + Serial.

---

## 2. Năm việc chính

| Việc | Mục tiêu | Cách làm | Đo bằng |
|---|---|---|---|
| **4.1 Tốc độ** | nhận diện nhanh | thêm bộ tăng tốc, đặt bộ nhớ đúng chỗ | log `[PERF]` |
| **4.2 Chính xác** | không nhận nhầm khi đông người | ngưỡng riêng từng người | `evaluate_model.py` |
| **4.3 Chống giả mạo** | chặn ảnh in/màn hình | cảm biến đo độ sâu 3D + kiểm tra ảnh | tỷ lệ chặn |
| **4.4 Vận hành** | không mất giờ, không mất dữ liệu, không cần cáp | NTP, thu ảnh trên chip, nạp không dây | log giờ, RAM |
| **4.5 Thực địa** | báo cáo có số liệu | test nhiều người, nhiều điều kiện sáng | FAR/FRR |

---

## 3. Việc 4.1 — Tối ưu tốc độ (đã xong)

**Cách đã làm:**
- Thêm log đo thời gian từng bước (`det`, `bil`, `he`, `rec`) in ra mỗi ảnh.
- Khóa chip ở tốc độ tối đa 240MHz.
- Thêm bộ tăng tốc `esp-nn` giúp các phép tính AI chạy nhanh hơn nhiều.
- Đặt bộ nhớ cho 2 model đúng chỗ, vừa đủ dùng, không làm phân mảnh RAM cho Wi-Fi và camera.
- Khử độ trễ: luôn xử lý ảnh mới nhất, bỏ ảnh cũ trong hàng đợi.

**Kết quả đo thật trên chip:**
- Tìm mặt: từ **20.4 giây xuống 1.95 giây** (nhanh hơn 10.5 lần).
- Nhận diện: từ **5.0 giây xuống 1.30 giây** (nhanh hơn 4 lần).
- Mỗi vòng lặp khi dùng lại khung mặt cũ: **1.32 giây/ảnh**.

Nạp firmware bằng:
```powershell
powershell -ExecutionPolicy Bypass -File D:\PROJECT_5_DIEM_DANH_KHUON_MAT\tools\flash_project_5.ps1 -Port COM3
```

---

## 4. Việc 4.2 — Nâng độ chính xác và chọn ngưỡng (đã xong)

- Ngưỡng chung: **0.70**.
- Ngưỡng riêng từng người: tính từ mức giống cao nhất với người khác, cộng thêm 0.02, tối thiểu 0.70, tối đa 0.75. Hiện cả 3 người (nhien, thao, toan) đều là **0.75**.
- Đo thật trên chip: điểm của nhien dao động **0.79-0.84**, luôn vượt ngưỡng nên nhận diện liên tục, không bị từ chối nhầm khi nghiêng mặt nhẹ.
- Đánh giá ngoại tuyến: nhận đúng **100%** trên bộ mẫu.

> **Mẹo tăng chính xác rẻ:** dùng bàn phím 4x4 hoặc thẻ RFID có sẵn để chuyển từ "tìm ai trong 3 người" sang "xác nhận đúng 1 người" — giảm nhận nhầm mạnh mà không tốn bộ nhớ.

---

## 5. Việc 4.3 — Chống giả mạo bằng cảm biến đo độ sâu 3D

Vì model tìm mặt hiện không lấy được điểm mốc khuôn mặt nên **không làm được kiểu nháy mắt**. Cách tốt nhất là dùng **cảm biến đo độ sâu ToF VL53L5CX 8x8** kết hợp với kiểm tra ảnh.

### 5.1 Nguyên lý
Cảm biến đo khoảng cách 64 điểm (lưới 8x8), đơn vị mm:
1. **Ảnh giả (ảnh in, màn hình điện thoại, video phát lại):** toàn bộ các điểm nằm gần như trên một mặt phẳng, chênh lệch rất nhỏ.
2. **Mặt người thật:** có hình dạng lồi — đỉnh mũi nhô ra trước hốc mắt/gò má/trán khoảng **20-45mm**, và vùng rìa (tóc, tai, tường phía sau) nhảy khoảng cách lớn (≥150mm).

### 5.2 Nối dây
- `VIN` → 3.3V (hoặc 5V nếu module có ổn áp).
- `GND` → GND.
- `SDA` → GPIO 8.
- `SCL` → GPIO 9.
- `LPn` → GPIO 10 (kéo HIGH khi chạy).
- `INT` → GPIO 11 (tùy chọn).
- Đặt cảm biến sát cạnh ống kính camera (khoảng cách ≤ 1.5cm), cùng hướng vào mặt người.
- Tốc độ I2C 400kHz hoặc 1MHz, đọc 64 điểm chỉ tốn dưới 2.5ms.

### 5.3 Thư viện
- Dùng thư viện chính thức của ST: `SparkFun_VL53L5CX_Arduino_Library` (hoặc port C++).
- Khởi tạo: `begin()` → đặt độ phân giải 8x8 → 15Hz → chế độ đo liên tục → `startRanging()`.
- Tài nguyên: ~85KB flash, ~3-5KB RAM, **không tốn PSRAM** (không đụng bộ nhớ của AI).

### 5.4 Ba vòng kiểm tra (chạy rất nhanh, dưới 0.05ms)
1. **Khoảng cách hợp lệ:** trung bình 4 điểm giữa mặt. Nếu < 250mm (quá sát) hoặc > 850mm (không có người) → không có người.
2. **Độ lồi của mũi:** lấy điểm gần nhất vùng mũi so với vùng gò má/trán xung quanh. Nếu chênh lệch < 18mm → nghi ngờ **ảnh phẳng 2D**.
3. **Độ phẳng bề mặt:** lấy 16 điểm trung tâm tính phương sai. Nếu quá nhỏ (bề mặt quá phẳng) → nghi ngờ **ảnh in / màn hình**. Nếu đủ lồi và đủ nhấp nhô → **người thật 3D**.

### 5.5 Cách phối hợp
- Cảm biến đọc liên tục trên nhân lo camera/mạng.
- Không có người → cho AI nghỉ để chip mát.
- Phát hiện giả mạo → **bật LED đỏ + 2 tiếng bíp dài**, in cảnh báo, và **không chạy nhận diện** để tiết kiệm thời gian.
- Người thật → chạy nhận diện bình thường.
- Vẫn giữ thêm lớp kiểm tra phần mềm (phương sai ảnh, yêu cầu quay đầu nhẹ).

### 5.6 Tiêu chuẩn đạt
- Chặn ảnh in/màn hình: **≥ 99%**.
- Không từ chối nhầm người thật: **≥ 98%** trong khoảng cách 35-75cm.
- Thời gian kiểm tra: **≤ 2.5ms**.

---

## 6. Việc 4.4 — Cứng hóa vận hành

- **Giờ thật:** dùng NTP (đồng bộ giờ qua Wi-Fi) và lưu vào bộ nhớ, ghi điểm danh kèm ngày giờ thật thay vì số giây từ lúc khởi động. Nếu offline thì dùng module đồng hồ DS3231.
- **Thu ảnh ngay trên chip:** bấm nút giữ 2 giây → chụp 20 ảnh → tạo dữ liệu ngay trên S3 → lưu vào file trên chip → khởi động lại để nạp dữ liệu.
- **Nạp firmware không cần cáp (OTA):** thêm đường dẫn web để nạp firmware mới từ xa. Kiểm tra RAM rò rỉ trong 24 giờ.

---

## 7. Việc 4.5 — Test thực địa và bàn giao

- **Bộ test:** ≥ 10 người × 3 điều kiện sáng (tối < 80 lux / đèn / ngoài trời) × 3 khoảng cách (30/50/80cm), cộng ảnh người lạ. Báo cáo FAR/FRR kèm khoảng tin cậy 95%, **không kết luận "FAR = 0%" từ vài lần thử**.
- **Ảnh chuẩn (golden frame):** lưu 1 ảnh cố định, chạy qua cả laptop và chip, so sánh khung mặt/đặc trưng.
- **Camera trực tiếp (tùy chọn):** chuyển sang OV2640 trên board S3-EYE.
- **Vỏ:** in 3D, cố định góc camera 1.2m, tản nhiệt, nguồn 5V/2A.

---

## 8. Dùng linh kiện có sẵn

| Linh kiện | Dùng cho | Cách dùng |
|---|---|---|
| Bàn phím 4x4 / thẻ RFID RC522 | 4.2 | Nhập mã/rà thẻ trước khi so mặt |
| Cảm biến ánh sáng MS-CDS05 | 4.1 | Không chạy nhận diện khi quá tối/sáng |
| Nút B3F / remote 1838T | 4.4 | Bấm để thu ảnh, đổi chế độ |
| Servo SG90 / động cơ bước | Không ưu tiên | Chỉ nếu làm gimbal xoay theo mặt |

---

## 9. Lộ trình gợi ý (2-3 tuần)

- **Tuần 1:** 4.1 + 4.2 (đã xong phần lớn).
- **Tuần 2:** 4.3 chống giả mạo + 4.4 (NTP, thu ảnh trên chip).
- **Tuần 3:** 4.5 (test nhiều người, vỏ, OTA).

---

## 10. Tiêu chí đạt Giai đoạn 4

- Nhận diện ≤ 1 giây, tổng ≤ 2 giây, chạy ổn định.
- Báo cáo đầy đủ TAR/FAR, không chỉ mỗi "độ đúng".
- Ảnh in bị từ chối ≥ 95%, người thật không bị từ chối nhầm quá 5%.
- Chạy 24 giờ không rò rỉ RAM, log đúng giờ, thu ảnh trên chip không cần cáp.
