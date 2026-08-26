# 🔌 HƯỚNG DẪN ĐẤU NỐI LCD1602 VÀ ESP32-S3 TRÊN BREADBOARD

> [!TIP]
> **Mẹo cắm dây nhanh:** Bạn chỉ cần cắm theo **4 CỤM RIÊNG BIỆT** bên dưới để không bao giờ bị rối hoặc cắm nhầm dây!

---

## 🧰 1. CHUẨN BỊ LINH KIỆN TỪ BỘ KIT

| STT | Tên linh kiện trong Kit | Hình ảnh nhận dạng | Số lượng |
|:---:|:---|:---|:---:|
| 1 | **Màn hình LCD1602** | Màn hình kính xanh lá, có 16 chân cắm | 1 cái |
| 2 | **Bo mạch ESP32-S3** | Bo mạch vi điều khiển AI chính | 1 cái |
| 3 | **Bo Test GL No.12** | Breadboard màu trắng cắm dây | 1 cái |
| 4 | **Triết áp đơn B10K** (LK #29) | Cục xoay màu xanh/bạc có 3 chân | 1 cái |
| 5 | **Điện trở 220R** (LK #21) | Thân nhỏ có 4 vạch màu: **Đỏ - Đỏ - Nâu - Vàng kim** | 1 cái |
| 6 | **Còi chíp 5V** (LK #30) | Cục trụ tròn màu đen có dán tem nhỏ, 2 chân | 1 cái |
| 7 | **Dây cắm Breadboard** | Dây cắm đực - đực và đực - cái | ~15 sợi |

---

## 🎨 2. QUY ƯỚC MÀU DÂY (KHUYÊN DÙNG ĐỂ DỄ QUAN SÁT)

* 🔴 **Dây Đỏ:** Dùng cho nguồn **5V / VIN**
* ⚫ **Dây Đen (hoặc Xanh dương đậm):** Dùng cho nguồn **GND (Mass)**
* 🟡 **Dây Vàng / Cam / Xanh lá:** Dùng cho các chân **Tín hiệu điều khiển**

---

## 🧭 3. HƯỚNG DẪN CẮM THEO 4 CỤM ĐƠN GIẢN

```text
                     CÁC CHÂN TRÊN MÀN HÌNH LCD1602 (16 CHÂN)
 ┌────────────────────────────────────────────────────────────────────────┐
 │ [1] [2] [3]   [4] [5] [6]   [7] [8] [9] [10]   [11] [12] [13] [14]   [15] [16] │
 │ VSS VDD V0    RS  RW  EN    D0  D1  D2   D3    D4   D5   D6   D7     A    K   │
 └────────────────────────────────────────────────────────────────────────┘
    ▲   ▲  ▲      ▲   ▲   ▲     ░░░ BỎ TRỐNG ░░░    ▲    ▲    ▲    ▲      ▲    ▲
    │   │  │      │   │   │                         │    │    │    │      │    │
 ┌──┴───┴──┴──┐ ┌─┴───┴───┴──┐               ┌────┴────┴────┴────┴───┐ ┌──┴────┴──┐
 │   CỤM 1    │ │   CỤM 3    │               │         CỤM 3         │ │  CỤM 2   │
 │   NGUỒN    │ │  TÍN HIỆU  │               │   DỮ LIỆU ĐIỀU KHIỂN  │ │ ĐÈN NỀN  │
 │ & TRIẾT ÁP │ │  ESP32-S3  │               │        ESP32-S3       │ │ (SÁNG LCD│
 └────────────┘ └────────────┘               └───────────────────────┘ └──────────┘
```

---

### 🔹 CỤM 1: Cấp nguồn & Chỉnh độ nét chữ (Chân 1, 2, 3)

1. **Chân 1 (VSS)** $\longrightarrow$ Cắm vào **Cột (-) GND** trên Breadboard.
2. **Chân 2 (VDD)** $\longrightarrow$ Cắm vào **Cột (+) 5V** trên Breadboard.
3. **Chân 3 (V0)** $\longrightarrow$ Cắm vào **Chân GIỮA của Triết áp B10K**.
   * *2 chân ngoài của Triết áp:* 1 chân cắm vào **(+) 5V**, 1 chân cắm vào **(-) GND**.

---

### 🔹 CỤM 2: Đèn nền màn hình (Chân 15, 16)

1. **Chân 15 (A - Cực dương đèn):** Cắm qua **Điện trở 220R** $\longrightarrow$ Nối vào **Cột (+) 5V**.
2. **Chân 16 (K - Cực âm đèn):** Cắm vào **Cột (-) GND** trên Breadboard.

> [!NOTE]
> Khi cắm xong Cụm 1 và Cụm 2, nếu cắm điện ESP32 thì **đèn màn hình sẽ sáng xanh lá**.

---

### 🔹 CỤM 3: 6 Dây điều khiển nối sang ESP32-S3 (Chân 4, 5, 6 và 11, 12, 13, 14)

*(Lưu ý: Các chân từ **7 đến 10** trên LCD để trống không cắm dây)*

| Chân trên LCD1602 | Cắm sang chân ESP32-S3 | Ghi chú |
|:---|:---|:---|
| **Chân 4 (RS)** | $\longrightarrow$ **GPIO 4** | Chọn chế độ lệnh |
| **Chân 5 (RW)** | $\longrightarrow$ **Cột (-) GND** | Nối đất để ghi dữ liệu |
| **Chân 6 (EN)** | $\longrightarrow$ **GPIO 5** | Chân kích hoạt nhận dữ liệu |
| **Chân 11 (D4)** | $\longrightarrow$ **GPIO 6** | Dữ liệu bit 4 |
| **Chân 12 (D5)** | $\longrightarrow$ **GPIO 7** | Dữ liệu bit 5 |
| **Chân 13 (D6)** | $\longrightarrow$ **GPIO 15** | Dữ liệu bit 6 |
| **Chân 14 (D7)** | $\longrightarrow$ **GPIO 16** | Dữ liệu bit 7 |

---

### 🔹 CỤM 4: Còi chíp Buzzer (Báo âm thanh khi điểm danh)

* **Chân (+) (chân dài / phía có dấu + trên nắp):** $\longrightarrow$ Cắm vào **GPIO 17** của ESP32-S3.
* **Chân (-) (chân ngắn):** $\longrightarrow$ Cắm vào **Cột (-) GND** trên Breadboard.

---

## 📋 4. BẢNG TỔNG HỢP TRA CỨU NHANH (QUICK CHECKLIST)

Hãy tích vào từng dòng sau khi cắm xong để kiểm tra:

- [ ] **Nguồn cấp ESP32:** Chân **5V (hoặc VIN)** của ESP32 $\rightarrow$ Cột (+) đỏ Breadboard.
- [ ] **Mass chung ESP32:** Chân **GND** của ESP32 $\rightarrow$ Cột (-) xanh Breadboard.
- [ ] **Chân 1 LCD (VSS)** $\rightarrow$ GND
- [ ] **Chân 2 LCD (VDD)** $\rightarrow$ 5V
- [ ] **Chân 3 LCD (V0)** $\rightarrow$ Chân GIỮA Triết áp B10K
- [ ] **Chân 4 LCD (RS)** $\rightarrow$ **GPIO 4** trên ESP32
- [ ] **Chân 5 LCD (RW)** $\rightarrow$ GND
- [ ] **Chân 6 LCD (EN)** $\rightarrow$ **GPIO 5** trên ESP32
- [ ] **Chân 11 LCD (D4)** $\rightarrow$ **GPIO 6** trên ESP32
- [ ] **Chân 12 LCD (D5)** $\rightarrow$ **GPIO 7** trên ESP32
- [ ] **Chân 13 LCD (D6)** $\rightarrow$ **GPIO 15** trên ESP32
- [ ] **Chân 14 LCD (D7)** $\rightarrow$ **GPIO 16** trên ESP32
- [ ] **Chân 15 LCD (A)** $\rightarrow$ Trở 220R $\rightarrow$ 5V
- [ ] **Chân 16 LCD (K)** $\rightarrow$ GND
- [ ] **Còi Chíp (+)** $\rightarrow$ **GPIO 17** trên ESP32

---

## 🔍 5. XỬ LÝ SỰ CỐ KHI CẮM NGUỒN LẦN ĐẦU (FAQ)

### ❓ Màn hình sáng đèn xanh nhưng không thấy chữ gì?
* **Cách khắc phục:** Lấy tay hoặc tua vít **xoay từ từ núm Triết áp B10K**. Khi xoay đúng góc, các dòng chữ màu đen `TINYML EDGE AI / ESP32-S3 READY` sẽ hiện lên rõ nét.

### ❓ Màn hình hiện 16 khối vuông màu đen ở dòng trên?
* **Nguyên nhân:** LCD đã có nguồn nhưng chưa nhận được tín hiệu khởi tạo từ ESP32.
* **Cách khắc phục:** 
  1. Kiểm tra lại chân **RS (GPIO 4)** và chân **EN (GPIO 5)** xem có bị lỏng không.
  2. Bấm nút **EN / RST** trên ESP32-S3 để khởi động lại.

### ❓ Đèn nền màn hình không sáng?
* **Cách khắc phục:** Kiểm tra lại **Chân 15** (đã qua trở 220R vào 5V chưa) và **Chân 16** (đã vào GND chưa).

