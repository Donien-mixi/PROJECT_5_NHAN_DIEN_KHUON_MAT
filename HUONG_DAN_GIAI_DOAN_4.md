# 🔧 GIAI ĐOẠN 4 — TỐI ƯU & VẬN HÀNH THỰC TẾ (KHẢ THI VỚI ESP32-S3 N16R8)

> Tài liệu build chi tiết cho `README.md:218-248` — mọi mục đều đo được trên đúng phần cứng hiện tại (`ESP32-S3 N16R8 8MB PSRAM`, `ai_config.h:5` 512KB+512KB+32KB, `BlazeFace 128 INT8` 183KB + `Ghost 64 INT8` 160KB, `TCP JPEG 128` `wifi_udp_server.cpp:43`). Không thêm model >1MB, không đòi camera mới bắt buộc.

---

## 1. Mục tiêu & ràng buộc

**Từ “chạy được” (`ket_qua_esp32.txt:102-113` SUCCESS 0.77 đúng người nhưng cách 6-32s) lên “chạy mượt, ổn định, chống gian lận”:** đo được, lặp lại được, không phụ thuộc may mắn ánh sáng.

Ràng buộc cứng:
- SRAM 512KB, PSRAM 8MB OPI — `firmware_esp32.ino:146` phải thấy `PSRAM OK 8388608`.
- 2 interpreter TFLM `ai_face_detector.cpp:64` 7 ops / `ai_face_recognizer.cpp:49` 5 ops — chung `heap_caps_malloc(MALLOC_CAP_SPIRAM)`.
- Input `RAW 128 RGB565` `image_decoder.cpp` → `Bilinear 128→64` + `HE LUT` `ai_face_detector.cpp:214` bit-exact `vision_utils.py:86`.
- `Temporal Voting 3` `face_recognizer.py:140` `firmware_esp32.ino:60` — không đổi logic khi tối ưu.
- Đầu ra chỉ `2 LED + Buzzer + Serial` `README.md:3`.

---

## 2. Tổng quan 5 trụ cột

| Trụ | Mục tiêu | Cách chính (vừa RAM) | Đo bằng |
|---|---|---|---|
| **4.1 Tốc độ** | recognize-only ≤1s, tổng ≤2s | profiling `micros()` → ESP-NN → đặt arena đúng chỗ → giữ `box-reuse` làm fallback | `[PERF]` Serial, `arena_used_bytes` |
| **4.2 Chính xác** | FAR ổn định khi 3→10+ người | per-identity threshold + `Impostor_*` + temperature scaling `evaluate_model.py:205` | `evaluate_model.py` TAR@FAR, OPIS |
| **4.3 Liveness 3D** | Chặn 100% ảnh in/màn hình/video phát lại | Cảm biến 3D ToF VL53L5CX 8×8 (0 MB PSRAM, <0.05ms CPU) + lọc kép LBP/Active | APCER ≤1%, BPCER ≤2%, trễ <2.5ms |
| **4.4 Vận hành** | Không mất giờ, không mất DB, không cần cáp | NTP/SPIFFS JSON enroll/OTA/24h leak | `HH:MM:SS` log, heap ổn định |
| **4.5 Thực địa** | Báo cáo có CI, vỏ dùng được | matrix 3 sáng×3 khoảng cách + OV2640 tuỳ chọn | FAR/FRR 95% CI |

---

## 3. 4.1 — Tối ưu tốc độ đo được

### 4.1.1 Profiling baseline (1 ngày, không đổi code AI)

Thêm `unsigned long t0=micros()` quanh:
- `decode_jpeg_frame` `wifi_udp_server.cpp:102`
- `preprocess_face` `ai_face_detector.cpp:236` (Bilinear + `equalize_gray_u8:214`)
- `detector_interpreter->Invoke()` `ai_face_detector.cpp:136`
- `interpreter->Invoke()` recognizer `ai_face_recognizer.cpp:104`

In `Serial.printf("[PERF] dec:%lu bil:%lu he:%lu det:%lu rec:%lu tot:%lu\n",...)`. Chạy trên board hiện tại, chụp 20 frame với `nhien` đứng yên 50cm đủ sáng → lấy median. Baseline hiện tại khoảng `det ~20s / rec ~5s` (`firmware_esp32.ino:51` comment). Ghi kèm `ESP.getFreeHeap()/Psram` và `arena_used_bytes` `ai_face_detector.cpp:91`.

### 4.1.2 Bật ESP-NN (2 nhánh, chọn 1)

*Hiệu quả tham khảo Espressif `esp-nn` v1.3.1: PersonDetect S3 2300ms→54ms (42×), MobileNetV3 26s→1.4s. `esp-dl` `MFN_S8_V1` S3 `5.6ms + 248ms` đạt ≤400ms.*

- **Nhánh A — Arduino (giữ IDE):** Thay `TensorFlowLite_ESP32` (hiện chỉ còn stub `src/esp_nn/README.md`) bằng `TFLiteMicro_ArduinoESP32S3` (pre-compile có ESP-NN cho S3). Không đổi `platformio.ini` (đã xóa), chỉ đổi lib trong `Arduino/libraries`.
- **Nhánh B — ESP-IDF (khuyến nghị nếu A không đủ):** Port `firmware_esp32/` sang `esp-tflite-micro` + `idf.py menuconfig → ESP-NN → Optimized` (`-DESP_NN`). `model_data.h` giữ ở PSRAM, `tensor_arena` chuyển sang SRAM nội (xem 4.1.3).

Đo lại `[PERF]` và so sánh. Nếu `det <400ms` và `rec <300ms` thì đạt mục tiêu `README.md:229` 400ms.

### 4.1.3 Đặt bộ nhớ đúng chỗ

Hiện `DET 512KB + REC 512KB + packet 32KB` `ai_config.h:5` đều PSRAM. PSRAM chậm hơn SRAM ~10 cycles. Thử:
- `g_detector_tensor_arena` / `tensor_arena` → `MALLOC_CAP_INTERNAL`
- `g_detector_model` / `g_model_recognizer` → `MALLOC_CAP_SPIRAM`

Tinh chỉnh `DETECTOR_ARENA_SIZE` theo số đo thực (hiện `ai_face_detector.cpp:91` báo ~470KB/512KB — còn 42KB dư). Giảm 10KB/lần tới khi `AllocateTensors FAILED` rồi cộng 10% headroom (ForestHub).

### 4.1.4 Fallback chính thức + gate ánh sáng

Giữ `box-reuse` `firmware_esp32.ino:71` `need_detect = !have_cached || (!last_match_ok && recon>=4)` — khi người đứng yên `last_match_ok=true` thì chỉ `rec` ~5s (sau ESP-NN ~200ms), không còn khoảng lặng 20-30s sau `SUCCESS` như `ket_qua_esp32.txt:76-112`.

Từ `components.txt:7` `MS-CDS05` : đọc ADC mỗi 500ms, nếu sáng `<65` hoặc `>200` như `enroll_tool.py:19` thì **không `Invoke`**, báo LED đỏ “thiếu sáng”. Gate này tiết kiệm 1 cycle lãng phí.

**Nghiệm thu 4.1:** `rec ≤1s`, tổng trung bình ≤2s trên 20 lần liên tiếp, `HB` 2s không trễ.

---

## 4. 4.2 — Nâng chính xác & hiệu chuẩn ngưỡng

Ngưỡng cố định `0.60` `face_recognizer.py:13` gặp threshold inconsistency (Zhang OPIS 2023, Verheyen doppelgänger 2023, Qin OTA 2022): mỗi người cần ngưỡng riêng khi DB dày lên.

- Tính **per-identity threshold** trên `data/registered_faces/`: `thresh_i = max( max_cross_sim(others) + 0.02, mean_intra - 0.05 )`. Lưu vào `face_database.json` kèm `face_database.h` (`generate_embeddings.py:122`).
- `face_recognizer.py:122` đổi `matched = sim >= max(0.60, thresh_claimed)`. Trên ESP32 `ai_face_recognizer.cpp` đọc `thresh` từ struct `RegisteredFace`.
- Mở rộng `evaluate_model.py:126` với `data/registered_faces/Impostor_*/` ≥20 ảnh người lạ thật → báo `TAR@FAR=1e-3`, `OPIS`, giữ ngưỡng đã khóa cho tập test. Đã nâng cấp chính thức sang tập quy mô lớn **CASIA-WebFace (Model V3)** với Online Dynamic Augmentation (`HUONG_DAN_COLAB_TRAIN_V3.md`) giúp phân tách cực đại các danh tính.

**Tận dụng kit `components.txt:31,32`:** `Keypad 4×4` hoặc `RFID RC522` chuyển **1:N (3×16 cosine)** sang **1:1 verification** (nhập ID/rà thẻ → chỉ so 16 template của 1 người) — FAR giảm mạnh, không tốn arena. Đây là tăng chính xác rẻ nhất, để tuỳ chọn.

---

## 5. 4.3 — Chống giả mạo sinh trắc học 3D (Anti-Spoofing / Liveness Detection) với ToF VL53L5CX 8×8

BlazeFace hiện `landmarks_5=[]` `blazeface_esp32.py:216` nên **không làm blink**. MiniFASNet-V2 600KB quantized 98.2% CelebA-Spoof vừa nhét nhưng chiếm thêm arena phải time-multiplex. Do đó, giải pháp tối ưu chuẩn công nghiệp nhất cho hệ thống là **kết hợp Cảm biến quang học đo chiều sâu 3D ToF VL53L5CX 8×8 (giải pháp phần cứng chính)** với **LBP/Variance & Active Motion (phần mềm bổ trợ)**.

### 5.1 Nguyên lý vật lý và hình học 3D cốt lõi

Cảm biến **STMicroelectronics VL53L5CX** sử dụng chùm xung laser SPAD hồng ngoại (940nm) an toàn cho mắt, thu về ma trận khoảng cách gồm **8 × 8 = 64 điểm chiều sâu** đồng thời (FoV chéo 63°, đơn vị: milimet):

1. **Trường hợp Gian lận 2D (Ảnh in phẳng A4, Màn hình Smartphone / iPad / Video phát lại):**
   * Toàn bộ 64 điểm đo nằm trên cùng một mặt phẳng nghiêng không gian: $Ax + By + Cz + D = 0$.
   * Phương sai phần dư sau khi bình phương cực tiểu (Residual Variance $\sigma^2_{\text{plane}}$) xấp xỉ bằng $0$ ($\sigma < 2.5\text{mm}$).
   * Độ chênh lệch giữa đỉnh mũi và các vùng xung quanh: $\Delta Z = Z_{\text{mũi}} - Z_{\text{mặt}} \approx 0\text{mm}$.
2. **Trường hợp Khuôn mặt người thật 3D (Bona Fide Face):**
   * Khuôn mặt người luôn có hình thái lồi đặc trưng (Convex Relief Profile).
   * **Đỉnh mũi và sống mũi** luôn nhô về phía trước so với hốc mắt, gò má và trán từ **$20\text{mm} - 45\text{mm}$**.
   * Vùng viền ngoài cùng của ma trận 8×8 (tóc, tai, nền tường phía sau) xuất hiện bước nhảy khoảng cách lớn (Drop-off to background) với độ biến thiên $\ge 150\text{mm}$.

---

### 5.2 Sơ đồ nối dây phần cứng (Wiring Diagram)

* **Kết nối I2C giữa ESP32-S3 N16R8 và Module VL53L5CX:**
  * `VIN` $\rightarrow$ `3.3V` (hoặc `5V` nếu module có sẵn IC ổn áp LDO; dòng đỉnh phát xung ~100mA).
  * `GND` $\rightarrow$ `GND`.
  * `SDA` $\rightarrow$ `GPIO 8` (Chân I2C Data).
  * `SCL` $\rightarrow$ `GPIO 9` (Chân I2C Clock).
  * `LPn` $\rightarrow$ `GPIO 10` (Chân Low Power / Enable chip, kéo HIGH khi hoạt động).
  * `INT` $\rightarrow$ `GPIO 11` (Ngắt phần cứng báo frame ToF mới sẵn sàng, tùy chọn).
* **Bố trí quang học:** Cảm biến ToF phải được đặt nằm ngang sát cạnh ống kính Camera (khoảng cách tâm quang học giữa thấu kính Camera và thấu kính ToF $\le 1.5\text{cm}$), hướng cùng trục thẳng vào khuôn mặt người dùng đứng điểm danh.
* **Tốc độ I2C:** Cấu hình `Wire.setClock(400000);` (400 kHz Fast-Mode) hoặc `1000000` (1 MHz Fast-Mode Plus) để đọc 64 điểm khoảng cách mất chưa tới **2.5 ms**, hoàn toàn không gây nghẽn mạng trên Core 1.

---

### 5.3 Tích hợp Thư viện Driver ULD (Ultra Lite Driver)

* Sử dụng thư viện chính thức từ STMicroelectronics: `SparkFun_VL53L5CX_Arduino_Library` hoặc port mã nguồn `vl53l5cx_uld` C++.
* **Quy trình khởi tạo (trong `setup()` hoặc `NetTask` Core 1):**
  1. `sensor.begin(&Wire);`
  2. Cấu hình độ phân giải: `sensor.setResolution(8 * 8);` (64 vùng đo).
  3. Cấu hình tần số đo (Ranging Frequency): `sensor.setRangingFrequency(15);` (15 Hz, đồng bộ tốc độ frame camera).
  4. Cấu hình chế độ đo liên tục: `sensor.setRangingMode(VL53L5CX_RANGING_MODE_CONTINUOUS);`
  5. Kích hoạt đo: `sensor.startRanging();`
* **Tài nguyên bộ nhớ:** 
  * Driver ULD nạp microcode firmware (~84KB) qua I2C vào RAM nội của cảm biến khi boot (tốn ~1.5s lúc boot máy).
  * Chiếm khoảng **~85KB Flash**, chỉ tốn **~3 - 5KB SRAM** lúc runtime, và **HOÀN TOÀN 0 MB PSRAM** (không đụng chạm tới Tensor Arena của AI).

---

### 5.4 Thuật toán Liveness 3 Vòng Kiểm Tra (Three-Stage 3D Filter)

Tạo module `firmware_esp32/ai_liveness_tof.h` và `ai_liveness_tof.cpp` chạy thuần túy bằng các phép toán số học cực nhanh (<0.05ms):

1. **Vòng 1 — Kiểm tra cự ly hợp lệ (Proximity Gate):**
   * Tính cự ly trung bình 4 điểm tâm mặt: $Z_{\text{center}} = (Z_{27} + Z_{28} + Z_{35} + Z_{36}) / 4$.
   * Nếu $Z_{\text{center}} < 250\text{mm}$ (quá sát) hoặc $Z_{\text{center}} > 850\text{mm}$ (quá xa / không có người) $\rightarrow$ Trả về `LIVENESS_NO_PERSON`.
2. **Vòng 2 — Độ lồi sống mũi (Nose Prominence Differential):**
   * Đỉnh mũi (Tâm 2×2 zones: 27, 28, 35, 36): $Z_{\text{nose}} = \min(Z_{27}, Z_{28}, Z_{35}, Z_{36})$.
   * Vùng gò má và trán bao quanh (8 zones lân cận: 18, 19, 20, 21, 26, 29, 34, 37): $Z_{\text{surround}} = \text{mean}(Z_{\text{lân cận}})$.
   * Độ chênh lệch: $\Delta Z = Z_{\text{surround}} - Z_{\text{nose}}$.
   * Ngưỡng quy chuẩn: Nếu $\Delta Z < 18\text{mm}$ $\rightarrow$ **Cảnh báo FAKE / MẶT PHẲNG 2D**.
3. **Vòng 3 — Khớp mặt phẳng tối thiểu (Planarity Residual Variance):**
   * Lấy 16 điểm trung tâm (vùng 4×4 từ hàng 2..5, cột 2..5) tính phương sai độ lệch so với độ sâu trung bình $\sigma^2$:
     $$\sigma^2 = \frac{1}{16} \sum_{i=1}^{16} (Z_i - \bar{Z})^2$$
   * Nếu $\sigma^2 < 12.0\text{mm}^2$ $\rightarrow$ Bề mặt quá phẳng $\rightarrow$ **Cảnh báo FAKE / ẢNH IN / MÀN HÌNH**.
   * Nếu $\sigma^2 \ge 12.0\text{mm}^2$ và $\Delta Z \ge 18\text{mm}$ $\rightarrow$ **LIVENESS PASS (Người thật 3D)**.

---

### 5.5 Tích hợp Pipeline Đa nhân & Cơ chế "Từ chối sớm" (Early Reject)

* **Core 1 (NetTask):** Đọc cảm biến ToF liên tục qua I2C mỗi khi hoàn tất giải mã 1 frame JPEG.
* **Cơ chế điều phối thông minh:**
  1. **Khi không có người (`LIVENESS_NO_PERSON`):** Core 0 ngủ đông `vTaskDelay(50ms)`, không chạy BlazeFace liên tục $\rightarrow$ Giữ chip mát (< 40°C), tiết kiệm điện.
  2. **Khi phát hiện tấn công giả mạo (`LIVENESS_SPOOF_2D`):**
     * Kích hoạt ngay: **LED Đỏ chớp nhanh + 2 tiếng Buzzer dài cảnh báo**.
     * In log Serial: `[SECURITY] GIAN LAN ANH IN / MAN HINH PHANG!`.
     * **HỦY BỎ LỆNH CHẠY RECOGNIZER:** Core 0 không tốn 1.3s suy luận vô ích, phản hồi từ chối ngay trong vòng < 5ms.
  3. **Khi đạt chuẩn người thật (`LIVENESS_LIVE_3D`):**
     * Core 0 tiếp tục chạy trích xuất vector đặc trưng Ghost-TinyFace và so khớp khuôn mặt như bình thường.
* **Lớp phòng thủ phụ (Software Passive & Active):**
  * Vẫn giữ thuật toán tính `variance` và `LBP` trên ảnh xám $64 \times 64$ sau bước `equalize_gray_u8` như lớp lọc kép thứ hai.
  * Active motion: Theo dõi độ dịch chuyển tâm mặt $\Delta(cx, cy) > 8\text{px}$ để đảm bảo người dùng có chuyển động tự nhiên.

---

### 5.6 Tiêu chuẩn đánh giá và nghiệm thu (Metrics & Validation)

* **Tỷ lệ từ chối giả mạo (APCER):** Đạt **$\le 1\%$** khi kiểm thử tấn công bằng:
  - Ảnh in màu khổ A4 giấy bóng chất lượng cao.
  - Ảnh in uốn cong mô phỏng vòm mặt.
  - Màn hình iPad Retina 11 inch và màn hình smartphone phát video khuôn mặt.
* **Tỷ lệ nhận diện đúng người thật (BPCER):** Đạt **$\le 2\%$** trong cự ly 35cm – 75cm.
* **Độ trễ kiểm tra Liveness:** Tổng thời gian đọc I2C và tính toán **$\le 2.5\text{ms}$**, không ảnh hưởng đến chu kỳ nhận diện chung.

---

## 6. 4.4 — Cứng hóa vận hành & enroll tại chỗ

- **Thời gian:** `millis()` `firmware_esp32.ino:22` mất khi reboot → `WiFi` → `configTime()` → `getLocalTime()` → lưu epoch vào SPIFFS mỗi phút; `log_attendance` ghi `YYYY-MM-DD HH:MM:SS` thay vì `millis()`. Fallback `DS3231` nếu offline.
- **Enroll tại chỗ:** Nút `B3F` `components.txt:19` (hoặc `Remote 1838T`) giữ 2s → bắt 20 ảnh qua TCP như `enroll_tool.py:17` (check sáng 65-200) → chạy lại `generate_embeddings` logic đã port (`equalize_gray_256`) ngay trên S3 → append JSON `/faces.json` trên SPIFFS → reboot load lại `FACE_DATABASE` trong PSRAM. Giữ **16 templates MAX-SIM** `face_database.h`, không trung bình 10 vector như bản nháp cũ.
- **OTA & bền:** HTTP `/update` (`Update.h` / `esp_https_ota`) để flash `model_data.h`/`face_database.h` không cần cáp. Leak test 24h: log `ESP.getFreeHeap()`/`getFreePsram()` mỗi phút qua Serial, heap không giảm >5%.

---

## 7. 4.5 — Kiểm thử thực địa & bàn giao

- **Matrix:** `≥10 người × 3 sáng (tối <80 lux / đèn tuýp / ngoài trời) × 3 khoảng cách (30/50/80cm)` + `Impostor_*` ≥20 ảnh. Báo FAR/FRR kèm CI 95%, không kết luận `FAR=0%` từ vài lần thử `README.md:281`.
- **Golden frame:** Lưu 1 JPEG 128 cố định `README.md:274` chạy qua `prepare_esp32_frame` `vision_utils.py:67` và `g_frame_buffer` S3, so `bbox/crop/embedding` trong sai số lượng tử hóa.
- **Camera trực tiếp (tuỳ chọn):** Khi bỏ laptop, đổi `image_decoder.cpp` từ `TJpg_Decoder` sang `esp_camera` OV2640 trên S3-EYE — giữ `ip_camera_streamer.py` chỉ cho debug.
- **Vỏ:** 3D print, cố định góc 1.2m, tản nhiệt, nguồn 5V/2A, hướng dẫn vận hành.

---

## 8. Tích hợp cảm biến từ `components.txt`

| Kit sẵn có | Dùng cho GĐ4 | Cách |
|---|---|---|
| `Keypad 4×4` / `RFID RC522` | 4.2 verification 1:1 | Nhập ID/rà thẻ trước khi so mặt |
| `MS-CDS05` ánh sáng | 4.1 gate | Không `Invoke` khi quá tối/sáng |
| `B3F` nút / `1838T` remote | 4.4 enroll/mode | Bấm để enroll, remote đổi chế độ |
| `SG90`/`28BYJ-48` | Không ưu tiên | Chỉ nếu làm gimbal theo mặt |

---

## 9. Lộ trình đề xuất (2-3 tuần, ưu tiên tốc độ+chính xác như yêu cầu)

**Tuần 1 — `4.1` + `4.2`:** profiling `[PERF]` → thử ESP-NN nhánh A → tính per-identity threshold + đo lại `SUCCESS` cách nhau ~1-2s.
**Tuần 2 — `4.3` passive + `4.4` NTP/SPIFFS JSON:** LBP/active + enroll 20 ảnh tại chỗ.
**Tuần 3 — `4.5`:** chạy matrix, vỏ, OTA.

---

## 10. Tiêu chí nghiệm thu GĐ4

- `rec ≤1s`, tổng ≤2s median 20 lần, `HB` không trễ.
- `TAR@FAR=1e-3` và per-identity FAR báo cáo, không chỉ `Accuracy`.
- Ảnh in bị reject ≥95% (passive), người thật không tăng FRR >5%.
- 24h heap/PSRAM ổn định, thời gian log đúng giờ, enroll tại chỗ không cần cáp.
