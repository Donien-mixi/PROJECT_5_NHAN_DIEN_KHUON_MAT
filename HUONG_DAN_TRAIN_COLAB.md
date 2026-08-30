# HƯỚNG DẪN RETRAIN MÔ HÌNH TRÊN GOOGLE COLAB (QUY TRÌNH TRUYỀN THỐNG)

Theo như bạn nhớ, quy trình chuẩn mà chúng ta đã từng làm là chia làm 2 giai đoạn: **Huấn luyện nặng trên Colab** và **Đánh giá, lượng tử hóa nhẹ nhàng trên Laptop**. Việc này giúp bạn kiểm soát được chất lượng mô hình (`evaluate_model.py`) trước khi ép kiểu xuống TFLite.

Dưới đây là các bước mô phỏng lại chính xác quy trình quen thuộc của bạn:

---

## PHẦN 1: HUẤN LUYỆN TRÊN GOOGLE COLAB

**Bước 1: Chuẩn bị file ZIP trên máy tính**
Bạn cần nén (zip) **CẢ 2 THƯ MỤC** là `training_tinyml` và `data` thành một file có tên là `colab_training.zip`.
*(Lưu ý bắt buộc: Thư mục `data/registered_faces/` chứa khuôn mặt của bạn PHẢI có trong file zip này để AI học cách phân biệt ngay trên Colab).*

**Bước 2:** Truy cập [Google Colab](https://colab.research.google.com/) và tạo một Notebook mới.
**Bước 3:** Đảm bảo bạn đang dùng GPU (Vào `Runtime` > `Change runtime type` > Chọn `T4 GPU`).
**Bước 4:** Copy toàn bộ đoạn mã dưới đây dán vào 1 ô duy nhất trên Colab và nhấn nút **Run (Play)**.

Đoạn mã này sẽ làm 4 việc:
1. Yêu cầu bạn tải file `colab_training.zip` lên.
2. Tải tập dữ liệu LFW.
3. Chạy quá trình huấn luyện LFW (hơn 1 tiếng) **VÀ** tự động Fine-tune luôn trên dữ liệu cá nhân của bạn (để chống nhầm lẫn/overlap).
4. Tự động tải file `tinyface_backbone.keras` "hoàn hảo" về máy tính của bạn khi học xong.

```python
import os
from google.colab import files

print("🚀 BƯỚC 1: VUI LÒNG CHỌN FILE 'colab_training.zip' TỪ MÁY TÍNH CỦA BẠN ĐỂ TẢI LÊN...")
uploaded = files.upload()
zip_name = list(uploaded.keys())[0]

print("\n📦 BƯỚC 2: ĐANG GIẢI NÉN DỮ LIỆU...")
# Tạo thư mục gốc và giải nén toàn bộ vào đó (ĐÃ FIX LỖI TÊN FILE CÓ DẤU NGOẶC)
!mkdir -p colab_workspace
!unzip -q -o "{zip_name}" -d colab_workspace

print("\n🔍 ĐANG TÌM THƯ MỤC CHỨA MÃ NGUỒN...")
# Thuật toán quét sâu 100% chính xác
target_file = 'train_distillation.py'
found_path = None

for root, dirs, files_in_dir in os.walk('/content/colab_workspace'):
    if target_file in files_in_dir:
        found_path = root
        break

if not found_path:
    print(f"❌ LỖI: Không tìm thấy file '{target_file}' trong file zip!")
    print("Vui lòng kiểm tra lại xem bạn đã copy ĐÚNG thư mục 'training_tinyml' vào chưa.")
    print("Cấu trúc bên trong file zip của bạn hiện tại là:")
    !ls -R /content/colab_workspace
else:
    # Lùi lại 1 bậc để đứng ở thư mục gốc (chứa data và training_tinyml)
    project_root = os.path.dirname(found_path)
    os.chdir(project_root)
    print(f"[+] Đã chuyển đến thư mục làm việc: {project_root}")
    
    print("\n📥 BƯỚC 3: ĐANG TẢI LFW DATASET (Nếu chưa có)...")
    !pip install scikit-learn opencv-python-headless
    !python training_tinyml/download_lfw_dataset.py

    print("\n🧠 BƯỚC 4: ĐANG HUẤN LUYỆN LẠI MÔ HÌNH (Quá trình này sẽ mất hơn 1 tiếng)...")
    !python training_tinyml/train_distillation.py
    
    print("\n✅ BƯỚC 5: HUẤN LUYỆN XONG! ĐANG TẢI FILE '.keras' VỀ MÁY TÍNH CỦA BẠN...")
    if os.path.exists("training_tinyml/weights/tinyface_backbone.keras"):
        files.download("training_tinyml/weights/tinyface_backbone.keras")
    else:
        print("❌ Lỗi: Không tìm thấy file Keras. Quá trình huấn luyện có thể đã thất bại.")


```

Sau khi chạy xong, bạn sẽ nhận được file `tinyface_backbone.keras` tải về qua trình duyệt.

---

## PHẦN 2: ĐÁNH GIÁ VÀ LƯỢNG TỬ HÓA TRÊN LAPTOP CỦA BẠN

Sau khi tải được file `tinyface_backbone.keras` mới từ Colab về, bạn làm theo các bước sau trên máy tính của mình:

**Bước 1: Cập nhật trọng số mới**
- Chép file `tinyface_backbone.keras` vừa tải về đè lên file cũ nằm ở đường dẫn:
  `d:\PROJECT_5_DIEM_DANH_KHUON_MAT\training_tinyml\weights\tinyface_backbone.keras`

**Bước 2: Kích hoạt môi trường và Đánh giá mô hình (Evaluate)**
Mở Terminal/Command Prompt trong thư mục dự án và chạy:
```bash
# Đảm bảo đã kích hoạt môi trường ảo (ví dụ: .venv\Scripts\activate)
python training_tinyml/evaluate_model.py
```
*(Bước này để bạn xem biểu đồ, đánh giá xem AI mới học xoay nghiêng/lật ngang có tốt hơn bản cũ không).*

**Bước 3: Lượng tử hóa mô hình (Quantization)**
Nếu kết quả đánh giá (Evaluate) làm bạn hài lòng, hãy ép mô hình về định dạng siêu nhẹ cho ESP32-S3:
```bash
python training_tinyml/quantize_qat_int8.py
```
Lệnh này sẽ sinh ra file `tinyface_int8.tflite` (phiên bản trí tuệ nhân tạo mới nhất, thông minh hơn).

**Bước 4: Cập nhật lại cơ sở dữ liệu khuôn mặt và nạp Firmware**
Sau đó, tiếp tục chạy các lệnh sau để hoàn tất cập nhật vào ESP32:
```bash
conda activate projet_5
python training_tinyml/update_face_database.py
```
Mở phần mềm Arduino IDE (hoặc PlatformIO), cắm mạch ESP32-S3 vào và bấm **Upload** để nạp firmware mang não bộ AI mới nhất xuống.

---

## TỔNG HỢP: KHI NÀO CẦN CHẠY CÁC FILE PYTHON TRONG `training_tinyml/`?

Để bạn nắm rõ bức tranh toàn cảnh và không bị bối rối bởi nhiều file mã nguồn, dưới đây là vòng đời và thời điểm thực thi của 7 file Python quan trọng nhất trong hệ thống:

### 1. Nhóm chạy trên Google Colab (Quá trình Retrain)
*Mục tiêu: Đào tạo ra một "Bản năng nhận diện khuôn mặt" (Universal Feature Extractor) mới.*

* **`download_lfw_dataset.py`**: **Chạy ĐẦU TIÊN trên Colab.** File này có nhiệm vụ kết nối internet tải tập dữ liệu LFW (~13.000 khuôn mặt) về làm tài liệu học tập cho AI. Nếu không chạy file này, quá trình huấn luyện sẽ báo lỗi không có dữ liệu.
* **`train_distillation.py`**: **Chạy THỨ HAI trên Colab.** Đây là kịch bản huấn luyện chính. Nó sẽ dạy cho AI Ghost-TinyFace (học trò) cách bắt chước SFace (thầy giáo) trên tập ảnh LFW. Kết quả sinh ra file `tinyface_backbone.keras`.

### 2. Nhóm chạy trên Laptop Cục Bộ (Sau khi đã tải `.keras` về)
*Mục tiêu: Đánh giá mô hình mới, tối ưu hóa cho ESP32 và cập nhật dữ liệu của 3 người dùng.*

* **`evaluate_model.py`**: **Chạy ĐẦU TIÊN trên Laptop.** Sinh ra các biểu đồ đánh giá chất lượng của mô hình `.keras` mới. Bạn chạy file này để xem AI mới học xoay nghiêng/lật ngang có bị "tẩu hỏa nhập ma" không, độ chính xác có tốt hơn bản cũ không.
* **`quantize_qat_int8.py`**: **Chạy THỨ HAI trên Laptop.** Sau khi chạy Evaluate và thấy hài lòng, chạy file này để nén (lượng tử hóa) mô hình Keras khổng lồ xuống định dạng siêu nhẹ `.tflite` (số nguyên INT8) để nhét vừa vào bộ nhớ bé xíu của ESP32-S3.
* **`update_face_database.py`**: **Chạy THỨ BA trên Laptop.** Lúc này bạn đã có "não bộ" mới. Bạn phải chạy file này để AI mới nhìn lại hình ảnh của 3 người đăng ký (nằm trong thư mục `registered_faces/`) và tạo ra file `face_database.json` mới chứa các vector đặc trưng. Nếu không chạy, AI mới sẽ lấy nhầm vector của AI cũ và sẽ không nhận diện ra ai cả.

### 3. Nhóm file Thư viện (KHÔNG BAO GIỜ CHẠY TRỰC TIẾP)
*Mục tiêu: Đóng vai trò là công cụ/hàm hỗ trợ cho các file ở trên.*

* **`dataset_loader.py`**: Không cần chạy lệnh. Đây là file chứa các hàm tải ảnh và bộ Augmentation (như xoay ảnh 15 độ, lật ngang mà bạn vừa thêm). Các file như `train_distillation.py` hoặc `evaluate_model.py` sẽ tự động gọi file này vào (import) khi chúng nó chạy.
* **`generate_embeddings.py`**: Không cần chạy lệnh. Nhiệm vụ của nó là tính toán ra vector 128 chiều từ ảnh. File này sẽ được file `update_face_database.py` gọi tự động (import) để chạy ngầm bên dưới.

---

## PHẦN 3: HƯỚNG DẪN KHI MUỐN THÊM NGƯỜI MỚI (TẠI LAPTOP)

Nếu sau này bạn muốn **thêm người thứ 4, thứ 5** mà không muốn phải đưa lên Google Colab train lại (vì tốn hơn 1 tiếng đồng hồ), bạn có thể làm theo cách **chạy nhanh trên Laptop** như sau:

**Bước 1: Chụp ảnh người mới**
Tạo thư mục mới trong `data/registered_faces/` (ví dụ: `data/registered_faces/nguoi_thu_4`) và bỏ ảnh của người đó vào.

**Bước 2: Fine-tune cục bộ (Ép giãn khoảng cách để chống nhầm lẫn)**
Chạy lệnh sau trên Laptop của bạn (chỉ mất vài phút):
```bash
python training_tinyml/finetune_locally.py
```
*Script này sẽ tải file `.keras` hiện tại và dùng Thuật toán Phân loại (Classification / NormFace) siêu ổn định để học cách kéo giãn, phân biệt triệt để người cũ và người mới.*

**Bước 3: Lượng tử hóa lại mô hình**
Vì mô hình `.keras` vừa bị thay đổi ở Bước 2, bạn BẮT BUỘC phải ép kiểu lại xuống `.tflite`:
```bash
python training_tinyml/quantize_qat_int8.py
```

**Bước 4: Cập nhật cơ sở dữ liệu**
Dùng não bộ mới để quét lại toàn bộ ảnh và tạo file `face_database.json`:
```bash
python training_tinyml/update_face_database.py
```

Sau đó bạn chỉ cần nạp lại Firmware xuống ESP32 là hệ thống sẽ nhận diện người mới cực kỳ chính xác!
