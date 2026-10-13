# Danh sách đề tài AI, AIoT & Edge AI (Khoa KTMT)

Đây là bản tổng hợp các đề tài đồ án liên quan đến **AI, AIoT (AI + IoT) và Edge AI (AI chạy trên thiết bị nhỏ)** của Khoa Kỹ thuật Máy tính. Mình gom lại thành **8 nhóm, 32 đề tài** cho dễ đọc.

Mỗi đề tài mình ghi ngắn gọn: mục tiêu, dùng gì (phần cứng + phần mềm), độ khó và thời gian. *(Project của mình — điểm danh khuôn mặt — chính là đề tài số 8.)*

---

## Nhóm 1: Edge AI & Thị giác máy tính trên kit nhúng (Raspberry Pi / Jetson / Kria)

**1. Nhận diện biển báo giao thông thời gian thực trên Jetson Nano**
- Mục tiêu: phát hiện và phân loại biển báo giao thông từ camera.
- Dùng: Jetson Nano hoặc Raspberry Pi + Google Coral, camera; model YOLOv8n / MobileNet-SSD, tối ưu bằng TensorRT.
- Độ khó: Trung bình, ~2.5-3 tháng. Hợp Đồ án 1 hoặc 2.

**2. So sánh SSD và YOLO trên máy tính nhúng**
- Mục tiêu: so sánh tốc độ và độ chính xác của 2 dòng model nhận diện vật thể.
- Dùng: Raspberry Pi 4 / Jetson Nano; TFLite, ONNX Runtime, TensorRT.
- Độ khó: Trung bình, ~2.5 tháng. Hợp Đồ án 1/2.

**3. Phân đoạn ảnh (tách từng điểm ảnh) trên thiết bị nhúng**
- Mục tiêu: tách chính xác từng điểm ảnh của vật thể/vùng (làn đường, vỉa hè).
- Dùng: Jetson Nano / Kria KV260; U-Net, Fast-SCNN, Mask R-CNN.
- Độ khó: Trung bình - Khó, ~3-3.5 tháng. Hợp Đồ án 2.

**4. Phát hiện đối tượng trong ảnh y khoa trên máy nhúng**
- Mục tiêu: hỗ trợ phát hiện tổn thương trong ảnh X-quang, nội soi, da.
- Dùng: Jetson Nano / Raspberry Pi; YOLOv8, EfficientDet, Grad-CAM.
- Độ khó: Khó, ~3.5 tháng. Hợp Đồ án 2.

**5. Mô hình Thị giác - Ngôn ngữ (VLM) trên thiết bị nhúng**
- Mục tiêu: trả lời câu hỏi về ảnh, mô tả cảnh cho người khiếm thị.
- Dùng: Jetson / Raspberry Pi 5; SmolVLM, Moondream2, MiniCPM-V.
- Độ khó: Khó, ~3.5 tháng. Hợp Đồ án 2.

**6. Phát hiện đối tượng đa phương thức (RGB + nhiệt hồng ngoại)**
- Mục tiêu: phát hiện người/vật cả trong bóng tối hoặc sương mù.
- Dùng: Jetson / Kria, camera RGB + camera nhiệt, YOLO-World, Grounding DINO.
- Độ khó: Khó, ~3.5 tháng. Hợp Đồ án 2.

**7. Gương thông minh (Smart Mirror)**
- Mục tiêu: gương hiển thị thông tin cá nhân theo người đứng trước.
- Dùng: Raspberry Pi 4/5, kính 2 chiều, webcam; MagicMirror² hoặc web tự viết.
- Độ khó: Dễ - Trung bình, ~1.5-2 tháng. Hợp Đồ án 1.

**8. Điểm danh khuôn mặt trong phòng học thông minh** ← *(project của mình)*
- Mục tiêu: điểm danh tự động, có chống gian lận.
- Dùng: Raspberry Pi / Jetson, camera, màn hình nhỏ, loa; RetinaFace, ArcFace/MobileFaceNet, FastAPI + web.
- Độ khó: Dễ - Trung bình, ~2 tháng. Hợp Đồ án 1/2.

---

## Nhóm 2: Tương tác Người - Máy (HCI) & nhận dạng cử chỉ / tư thế

**9. Nhận biết cử chỉ bàn tay qua các điểm mốc**
- Mục tiêu: nhận diện cử chỉ tay (nắm, xòe, like...) để điều khiển không chạm.
- Dùng: Raspberry Pi / laptop, webcam; MediaPipe Hands, MLP/SVM/LSTM.
- Độ khó: Trung bình, ~2.5 tháng. Hợp Đồ án 1/2.

**10. Nhận biết tư thế người & cảnh báo té ngã**
- Mục tiêu: nhận biết đứng/ngồi/nằm/cúi/té ngã, cảnh báo cho người cao tuổi.
- Dùng: Raspberry Pi / Jetson, camera, loa/Telegram; MediaPipe Pose, YOLOv8-pose, ST-GCN.
- Độ khó: Trung bình - Khó, ~3 tháng. Hợp Đồ án 2.

---

## Nhóm 3: Giao thông thông minh & hỗ trợ lái xe (ADAS/ITS)

**11. ADAS mini: cảnh báo lệch làn + giữ làn**
- Mục tiêu: phát hiện vạch làn, cảnh báo lệch làn, tự đánh lái giữ làn.
- Dùng: Raspberry Pi / Jetson, camera, servo; xử lý ảnh + PID.
- Độ khó: Trung bình - Khó, ~3 tháng. Hợp Đồ án 2.

**12. Kiểm soát hành trình thích ứng (ACC)**
- Mục tiêu: tự giữ tốc độ và giữ khoảng cách an toàn với xe trước.
- Dùng: xe mô hình, LiDAR/Radar, camera; YOLOv8 + ByteTrack, PID/Fuzzy.
- Độ khó: Trung bình - Khó, ~3 tháng. Hợp Đồ án 2.

**13. Phát hiện tài xế ngủ gật & mất tập trung**
- Mục tiêu: cảnh báo khi tài xế nhắm mắt lâu, ngáp, quay mặt đi.
- Dùng: Raspberry Pi / Jetson, camera IR, loa; MediaPipe Face Mesh, Dlib.
- Độ khó: Trung bình, ~2-2.5 tháng. Hợp Đồ án 1/2.

**14. Camera phạt nguội: xe sai làn & nhận diện biển số (ANPR)**
- Mục tiêu: phát hiện xe vi phạm và đọc biển số.
- Dùng: mini PC/Jetson, camera; YOLOv8 + DeepSORT, PaddleOCR/CRNN.
- Độ khó: Khó, ~3.5 tháng. Hợp Đồ án 2.

---

## Nhóm 4: AIoT công nghiệp, đô thị & an toàn lao động

**15. Giám sát xe ra vào trạm cân tự động**
- Mục tiêu: tự nhận biển số, đọc số cân, phát hiện gian lận cân.
- Dùng: 2-3 camera IP, máy tính trạm; YOLO + OCR, phân tích thể tích.
- Độ khó: Khó, ~3.5 tháng. Hợp Đồ án 2.

**16. Đánh giá hiệu suất & phát hiện hành vi bất thường của công nhân**
- Mục tiêu: biết công nhân có làm việc thật không, phát hiện ngồi chơi/ngủ gật.
- Dùng: camera IP + edge server/Jetson; ReID, YOLOv8-pose, SlowFast.
- Độ khó: Khó, ~3.5 tháng. Hợp Đồ án 2.

**17. Phát hiện gian lận tại quầy order chuỗi cafe**
- Mục tiêu: phát hiện nhân viên thu tiền nhưng không tạo bill.
- Dùng: camera nhìn từ trên xuống, đồng bộ phần mềm POS; hand tracking, event correlation.
- Độ khó: Khó, ~3.5 tháng. Hợp Đồ án 2.

**18. An toàn cẩu tháp công trình (Radar + Camera AI)**
- Mục tiêu: cảnh báo người/xe vào vùng nguy hiểm, phát hiện thiếu mũ bảo hộ.
- Dùng: Jetson, camera PTZ, radar 24/77GHz, IMU; YOLOv8 + radar.
- Độ khó: Khó, ~3.5 tháng. Hợp Đồ án 2.

**19. Kiểm soát chất lượng & môi trường cho bể chứa bùn**
- Mục tiêu: quan sát màu bùn, phát hiện vật lạ, theo dõi mức bùn/nước.
- Dùng: camera công nghiệp, cảm biến siêu âm/pH/DO, gateway ESP32; phân tích màu, CNN.
- Độ khó: Trung bình - Khó, ~3 tháng. Hợp Đồ án 2.

**20. Giám sát môi trường nhà máy bằng Thread/Matter + AI dự đoán cháy nổ**
- Mục tiêu: mạng cảm biến không dây thế hệ mới, cảnh báo cháy sớm.
- Dùng: ESP32-C6/nRF52840, cảm biến khí; Thread/Matter, YOLOv8-fire, LSTM.
- Độ khó: Khó, ~3.5 tháng. Hợp Đồ án 2.

---

## Nhóm 5: AIoT y tế & xử lý tín hiệu sinh học

**21. Tiên đoán bệnh bằng Deep Learning đa thông số sinh hiệu**
- Mục tiêu: thu nhiệt độ, SpO2, huyết áp, ECG, EMG để dự đoán nguy cơ bệnh.
- Dùng: ESP32/STM32 + cảm biến y tế; 1D-CNN + Bi-LSTM/TabNet, FastAPI, InfluxDB.
- Độ khó: Khó, ~3.5 tháng. Hợp Đồ án 2.

**22. Thiết bị đeo ECG/PPG kết hợp TinyML phát hiện loạn nhịp tim**
- Mục tiêu: thiết bị nhỏ, pin lâu, tự phát hiện nhịp tim bất thường.
- Dùng: ESP32-S3/Arduino Nano 33 BLE, cảm biến ECG/PPG; 1D-CNN lượng tử hóa TFLM.
- Độ khó: Trung bình - Khó, ~3 tháng. Hợp Đồ án 2.

**23. Phân loại sóng não EEG (động kinh/trầm cảm/điều khiển BCI)**
- Mục tiêu: đọc tín hiệu não, phân loại trạng thái, điều khiển robot/tay giả.
- Dùng: mũ điện cực EEG, kit xử lý; lọc tín hiệu, EEGNet/1D-CNN, SVM.
- Độ khó: Khó, ~3.5 tháng. Hợp Đồ án 2.

**24. Trạm khám bệnh từ xa (Telehealth mini) & phân tích ảnh que xét nghiệm**
- Mục tiêu: đo sinh hiệu gửi bác sĩ từ xa, phân tích que thử bằng camera.
- Dùng: Raspberry Pi + màn hình, cảm biến sinh hiệu, camera; xử lý màu, WebRTC.
- Độ khó: Trung bình, ~2.5-3 tháng. Hợp Đồ án 1/2.

---

## Nhóm 6: AIoT nông nghiệp, môi trường & robot

**25. Trạm quan trắc môi trường thông minh + AI dự báo**
- Mục tiêu: đo nhiệt độ, độ ẩm, bụi mịn... và dự báo thời tiết/ô nhiễm.
- Dùng: ESP32 + cảm biến, pin mặt trời, LoRa/4G; MQTT, FastAPI, InfluxDB, LSTM.
- Độ khó: Trung bình, ~2.5 tháng. Hợp Đồ án 1/2.

**26. Tưới cây tự động diện rộng dùng LoRa + AI thời tiết**
- Mục tiêu: nhiều nút cảm biến độ ẩm đất, tự tưới theo thời tiết.
- Dùng: ESP32 + module LoRa, van điện từ, gateway Raspberry Pi; Fuzzy Logic/Q-Learning.
- Độ khó: Trung bình - Khó, ~3 tháng. Hợp Đồ án 2.

**27. Drone nông nghiệp tự hành + xử lý ảnh cây trồng**
- Mục tiêu: bay tuần tra, chụp ảnh ghép bản đồ, phát hiện vùng sâu bệnh.
- Dùng: drone F450, Pixhawk, camera NDVI; OpenDroneMap, NDVI, U-Net/YOLOv8.
- Độ khó: Rất khó, ~3.5-4 tháng. Hợp Đồ án 2.

**28. Robot tự hành tránh vật cản dùng LiDAR + AI SLAM**
- Mục tiêu: tự khám phá phòng, dựng bản đồ 2D, tránh vật cản.
- Dùng: robot 2 bánh, LiDAR RPLIDAR, IMU, Raspberry Pi/Jetson; ROS 2, Cartographer, Nav2.
- Độ khó: Khó, ~3-3.5 tháng. Hợp Đồ án 2.

---

## Nhóm 7: Tăng tốc AI trên FPGA / SoC (Hardware AI)

**29. Thiết kế phần cứng tăng tốc mô hình ngôn ngữ (SLM/LLM) trên FPGA**
- Mục tiêu: đưa model ngôn ngữ nhỏ chạy trực tiếp trên FPGA, tăng tốc và tiết kiệm điện.
- Dùng: kit FPGA Xilinx (PYNQ-Z2, Kria KV260); Verilog/Vitis HLS, Vivado, lượng tử 1-4 bit.
- Độ khó: Rất khó, ~3.5-4 tháng. Hợp Đồ án 2.

**30. Thiết kế SoC / IP Core tăng tốc mạng nơ-ron (CNN, Autoencoder, Audio) trên FPGA**
- Mục tiêu: làm lõi phần cứng tăng tốc AI chuyên cho ảnh, tín hiệu y sinh hoặc âm thanh.
- Dùng: FPGA PYNQ-Z2/Basys 3/Kria; RTL, DSP48, AXI, driver C/C++.
- Độ khó: Rất khó, ~3.5 tháng. Hợp Đồ án 2.

---

## Nhóm 8: Nền tảng IoT, bán lẻ thông minh & quản trị dữ liệu

**31. Nền tảng IoT / IoT Agent + bảo trì phòng ngừa**
- Mục tiêu: quản lý nhiều thiết bị từ xa, tự phát hiện bất thường để bảo trì sớm.
- Dùng: node cảm biến rung/dòng/nhiệt, gateway Raspberry Pi; MQTT/TLS, FastAPI, InfluxDB, Isolation Forest.
- Độ khó: Trung bình - Khó, ~3 tháng. Hợp Đồ án 1/2.

**32. Máy bán hàng tự động thông minh (nhận diện hàng + khuôn mặt)**
- Mục tiêu: thanh toán QR/khuôn mặt, tự nhận diện sản phẩm khách lấy.
- Dùng: tủ mini, màn hình cảm ứng, camera, cơ cấu nhả hàng; YOLOv8-nano, nhận diện mặt, VietQR.
- Độ khó: Trung bình - Khó, ~3 tháng. Hợp Đồ án 2.

---

## Bảng so sánh 32 đề tài

| # | Tên đề tài | GVHD | Lĩnh vực | Độ khó | Thời gian | Hợp |
|---|---|---|---|:---:|:---:|:---:|
| 1 | Nhận diện biển báo (Jetson) | P. Đ. Duy | Thị giác / Edge AI | Trung bình | 2.5-3 th | 1 & 2 |
| 2 | So sánh SSD vs YOLO | P. Đ. Duy | Deep Learning | Trung bình | 2.5 th | 1 & 2 |
| 3 | Phân đoạn ảnh | P.Đ.Duy / N.T.Thiện | Thị giác / Edge AI | TB-Khó | 3-3.5 th | 2 |
| 4 | AI y khoa trên nhúng | N. T. Thiện | Medical AI | Khó | 3.5 th | 2 |
| 5 | VLM trên nhúng | N. T. Thiện | Edge VLM | Khó | 3.5 th | 2 |
| 6 | Phát hiện đa phương thức | N. T. Thiện | Multimodal | Khó | 3.5 th | 2 |
| 7 | Gương thông minh | P. Đ. Duy | Smart Device | Dễ-TB | 1.5-2 th | 1 |
| 8 | Điểm danh khuôn mặt | P.Đ.Duy / N.D.X.Bách | Face Recognition | Dễ-TB | 2 th | 1 |
| 9 | Cử chỉ bàn tay | Đỗ Trí Nhựt | HCI / MediaPipe | Trung bình | 2.5 th | 1 & 2 |
| 10 | Tư thế người & té ngã | Đỗ Trí Nhựt | Pose / Action | TB-Khó | 3 th | 2 |
| 11 | ADAS: LDW + LKA | P. Đ. Duy | ADAS | TB-Khó | 3 th | 2 |
| 12 | Kiểm soát hành trình ACC | P. Đ. Duy | ADAS | TB-Khó | 3 th | 2 |
| 13 | Tài xế ngủ gật | Đ.T.Nhựt / P.Đ.Duy | Thị giác / An toàn | Trung bình | 2-2.5 th | 1 & 2 |
| 14 | Xe sai làn & biển số | P. Đ. Duy | ITS / Traffic | Khó | 3.5 th | 2 |
| 15 | Trạm cân tự động | T. V. Cương | Industrial Vision | Khó | 3.5 th | 2 |
| 16 | Giám sát hành vi công nhân | T. V. Cương | Behavioral / ReID | Khó | 3.5 th | 2 |
| 17 | Gian lận quầy cafe | T. V. Cương | Hand-Object AI | Khó | 3.5 th | 2 |
| 18 | An toàn cẩu tháp | T. V. Cương | Construction AIoT | Khó | 3.5 th | 2 |
| 19 | Bể chứa bùn | T. V. Cương | Environmental AIoT | TB-Khó | 3 th | 2 |
| 20 | Môi trường nhà máy Thread/Matter | T. V. Cương | Next-Gen IoT | Khó | 3.5 th | 2 |
| 21 | Tiên đoán bệnh đa sinh hiệu | N. D. X. Bách | Medical AIoT | Khó | 3.5 th | 2 |
| 22 | Thiết bị đeo ECG/PPG | N.D.X.Bách / P.H.Luân | Wearable / TinyML | TB-Khó | 3 th | 2 |
| 23 | Sóng não EEG / BCI | N. D. X. Bách | Biosignal / BCI | Khó | 3.5 th | 2 |
| 24 | Telehealth mini | N. D. X. Bách | Telehealth | Trung bình | 2.5-3 th | 1 & 2 |
| 25 | Trạm quan trắc môi trường | N.D.X.Bách / P.Đ.Duy | Environmental IoT | Trung bình | 2.5 th | 1 & 2 |
| 26 | Tưới cây LoRa | P. Đ. Duy | Smart Agriculture | TB-Khó | 3 th | 2 |
| 27 | Drone nông nghiệp | N. D. X. Bách | UAV / Robotics | Rất khó | 3.5-4 th | 2 |
| 28 | Robot LiDAR + SLAM | P.Đ.Duy / T.T.Đức | Robotics / ROS 2 | Khó | 3-3.5 th | 2 |
| 29 | Tăng tốc SLM/LLM trên FPGA | TS. P. H. Luân | Hardware AI | Rất khó | 3.5-4 th | 2 |
| 30 | IP Core CNN trên FPGA | P.H.Luân / T.V.Cương | FPGA / SoC | Rất khó | 3.5 th | 2 |
| 31 | Nền tảng IoT / IoT Agent | N. D. X. Bách | Full-stack IoT | TB-Khó | 3 th | 1 & 2 |
| 32 | Máy bán hàng thông minh | N.D.X.Bách / P.Đ.Duy | Smart Retail | TB-Khó | 3 th | 2 |

---

## Gợi ý chọn đề tài theo hướng đi

1. **Muốn làm chắc, kịp tiến độ Đồ án 1:** #7 (Smart Mirror), #8 (Điểm danh khuôn mặt), #9 (Cử chỉ tay), #13 (Tài xế ngủ gật), #25 (Trạm quan trắc cơ bản).
2. **Muốn chuyên sâu AI & Thị giác máy tính:** #1, #3, #4, #5, #10, #14.
3. **Muốn làm IoT / Hệ thống nhúng công nghiệp:** #15, #18, #20, #26, #31.
4. **Muốn làm Y sinh & xử lý tín hiệu:** #21, #22, #23.
5. **Muốn làm Robotics & xe tự hành:** #11, #12, #27, #28.
6. **Muốn làm vi mạch & FPGA:** #29, #30 (portfolio rất tốt khi xin việc ngành vi mạch).
