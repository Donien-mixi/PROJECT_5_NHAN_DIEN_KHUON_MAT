Rất hoan nghênh tinh thần "vọc vạch" của bạn! Việc tự tay tinh chỉnh các ngưỡng (thresholds) sẽ giúp bạn hiểu rất sâu về cách hệ thống AI đưa ra quyết định. 

Để thay đổi tất cả các ngưỡng này, bạn chỉ cần mở duy nhất **1 file** là `host_laptop/main.py`. Bạn có thể mở file này bằng bất kỳ trình soạn thảo code nào (ví dụ: VSCode, Notepad++ hoặc chính giao diện IDE bạn đang dùng).

Hãy cuộn xuống **Dòng 25 đến dòng 28** trong file `host_laptop/main.py`. Bạn sẽ thấy đoạn code khởi tạo hệ thống như sau:

```python
    # 1. Khởi tạo các module (Phân chia logic rõ ràng)
    recognizer = FaceRecognizer(model_path=model_path, db_path=json_db_path, threshold=0.88, use_tflite=True)
    detector = UnifiedFaceDetector(target_size=(64, 64), conf_threshold=0.80)
    db = DatabaseManager(db_path=sqlite_db_path, cooldown_seconds=30)
    voter = TemporalVoter(required_votes=3)
```

Dưới đây là ý nghĩa và cách bạn có thể tinh chỉnh 3 con số quan trọng nhất:

### 1. Ngưỡng Nhận Diện Khuôn Mặt (Ai đây?)
* **Dòng 25:** Thay đổi giá trị `threshold=0.88`
* **Ý nghĩa:** Đây là "độ khó tính" của hệ thống khi so sánh mặt bạn với DataBase. 0.88 tương đương 88% độ giống nhau.
* **Cách tinh chỉnh:**
  * **Tăng lên (VD: `0.90` hoặc `0.92`):** Hệ thống sẽ cực kỳ bảo mật và khắt khe. Nó chỉ nhận diện ra bạn khi bạn nhìn thẳng, ánh sáng tốt. Tuyệt đối không bao giờ nhận nhầm người lạ hay đồ vật, nhưng bù lại đôi khi bạn hơi nghiêng mặt nó sẽ báo `UNKNOWN`.
  * **Giảm xuống (VD: `0.75` hoặc `0.80`):** Hệ thống sẽ rất nhạy. Đeo kính, nghiêng đầu, thiếu sáng nó vẫn nhận ra bạn. Nhưng rủi ro là nó có thể nhận nhầm người na ná bạn hoặc nhận nhầm cái bao tải.

### 2. Ngưỡng Dò Tìm Khuôn Mặt (Có mặt người không?)
* **Dòng 26:** Thay đổi giá trị `conf_threshold=0.80`
* **Ý nghĩa:** Đây là độ tự tin của Camera khi quét tìm một vật thể hình cái mặt. 
* **Cách tinh chỉnh:**
  * **Tăng lên (VD: `0.90` hoặc `0.95`):** Khung vuông sẽ rất kén chọn. Chỉ khi nào có một khuôn mặt cực kỳ rõ nét đập vào camera thì nó mới thèm vẽ khung. Bù lại, nó sẽ không bao giờ khoanh nhầm cái bao tải.
  * **Giảm xuống (VD: `0.60` hoặc `0.70`):** Camera cực nhạy, bạn đứng tít đằng xa nó cũng vẽ khung bắt được mặt bạn. Tuy nhiên, nếu nhà bạn có cái gối, cái bao tải hay vân gỗ hình tròn tròn, nó cũng sẽ vẽ khung lên vật đó (nhưng vì `threshold=0.88` ở trên đang hoạt động, nên nó sẽ báo vật đó là `UNKNOWN`).

### 3. Ngưỡng "Bảo chứng" Chống Nhiễu (Chắc chắn chưa?)
* **Dòng 28:** Thay đổi giá trị `required_votes=3`
* **Ý nghĩa:** Hệ thống sẽ không bao giờ vội vàng báo MATCH khi mới nhìn thấy bạn 1 khung hình (1 frame). Nó bắt buộc phải nhìn thấy khuôn mặt đó liên tục **3 lần liên tiếp** thì nó mới chốt kết quả và điểm danh.
* **Cách tinh chỉnh:**
  * **Tăng lên (VD: `5` hoặc `10`):** Cực kỳ an toàn chống nhận nhầm, nhưng bạn sẽ có cảm giác AI nhận diện hơi "chậm" vì phải đứng trước camera 0.5s - 1s nó mới chịu báo kết quả.
  * **Giảm xuống (VD: `1`):** Tốc độ phản hồi cực nhanh, lướt qua cái là nhận diện luôn. Nhược điểm là đôi khi kết quả trên màn hình bị chớp giật liên tục.

Bạn hãy tự do chỉnh sửa 3 con số trên, lưu file (`Ctrl + S`) và chạy lại lệnh `python host_laptop/main.py` để xem sự khác biệt. Tìm ra "điểm ngọt" (sweet spot) phù hợp nhất với ánh sáng phòng bạn chính là niềm vui lớn nhất khi làm AI đó!