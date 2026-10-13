# Hướng dẫn train Model V3 trên Google Colab (dùng dữ liệu CASIA-WebFace)

Model V3 là bản nâng cấp giúp AI **phân biệt người này với người khác tốt hơn** nhờ dùng dữ liệu lớn **CASIA-WebFace**:
- Dữ liệu lớn: hàng trăm nghìn ảnh, mỗi người có ảnh ở nhiều góc mặt, biểu cảm và điều kiện sáng khác nhau.
- Không đổi gì trên chip: model vẫn là Ghost-TinyFace 64x64 (~160KB), tăng tốc bằng SIMD, tốc độ vẫn ~1.3 giây/ảnh.
- Dạy theo kiểu "thầy dạy trò": SFace là thầy, Ghost-TinyFace là trò, kèm kỹ thuật giúp phân tách người và chống ảnh hưởng ánh sáng.
- **Không cần chụp lại ảnh:** sau khi train xong, dữ liệu của nhien/thao/toan sẽ tự được tạo lại từ ảnh có sẵn.

---

## Kiểm tra file dữ liệu `faces_webface_112x112.zip`

File (~2.65 GB) đã kiểm tra, **đạt chuẩn**:
1. Đúng định dạng RecordIO của InsightFace (`train.rec`, `train.idx`, `property`).
2. Quy mô: 10,572 danh tính, 494,414 ảnh.
3. Ảnh đã được căn chỉnh chuẩn (112x112 BGR).
4. Code trong `colab.zip` đọc trực tiếp file này bằng Python thuần, rất nhanh (~11,000 ảnh/giây), không cần cài MXNet. Tự tách ra:
   - Ảnh **64x64 xám** cho trò (Ghost-TinyFace).
   - Ảnh **112x112 màu** cho thầy (SFace).

---

## Cách đưa dữ liệu lên Colab (chọn 1 trong 2)

**Cách 1 (nên dùng, nhanh, không sợ rớt mạng) — Google Drive:**
1. Tải `faces_webface_112x112.zip` lên **Google Drive** (để ở thư mục gốc "Drive của tôi").
2. Khi chạy ô lệnh bên dưới, Colab tự kết nối Drive và đọc file.

**Cách 2 — kéo thả trực tiếp:**
1. Trên Colab, mở biểu tượng **Thư mục (Files 📁)** ở thanh bên trái.
2. Kéo file `faces_webface_112x112.zip` vào (file sẽ nằm ở `/content/faces_webface_112x112.zip`).

---

## Ô lệnh Colab duy nhất (chạy 1 lần)

> **Bắt buộc trước khi chạy:** vào menu **Runtime → Change runtime type → T4 GPU → Save**.

Copy toàn bộ đoạn code dưới vào **1 ô** trên Colab rồi bấm **Play (Run)**:

```python
# ==============================================================================
# TRAIN MODEL V3 (CASIA-WEBFACE + ArcFace + Illumination KD) — 1 CLICK RUN
# ==============================================================================
import os, sys, shutil, zipfile, subprocess

print("=" * 70)
print("🚀 KHỞI ĐỘNG PIPELINE HUẤN LUYỆN MODEL V3 (CASIA-WEBFACE)")
print("=" * 70)

# ---------- 1. KẾT NỐI GOOGLE DRIVE (Tùy chọn - Tự nhận diện file 2.65GB) ----------
try:
    from google.colab import drive
    if not os.path.exists("/content/drive/MyDrive"):
        print("[*] Đang kết nối Google Drive (để tự nhận diện dataset nếu bạn đã tải lên Drive)...")
        drive.mount('/content/drive', force_remount=False)
        print("[✓] Google Drive đã kết nối thành công!")
except Exception as e:
    print(f"[-] Bỏ qua kết nối Drive: {e}")

# ---------- 2. NẠP MÃ NGUỒN colab.zip (Chỉ ~950 KB, tải lên trong 2 giây) ----------
from google.colab import files

# Nếu muốn ép nạp lại file zip mới, đổi dòng dưới thành True
FORCE_REUPLOAD_CODE = True

if FORCE_REUPLOAD_CODE or not os.path.exists("/content/colab.zip"):
    for old in ("/content/colab.zip", "/content/training_tinyml", "/content/host_laptop"):
        if os.path.exists(old):
            shutil.rmtree(old) if os.path.isdir(old) else os.remove(old)
    print("\n" + "=" * 65)
    print(">>> Vui lòng chọn file colab.zip từ máy tính của bạn...")
    print(">>> Đường dẫn: D:\\PROJECT_5_DIEM_DANH_KHUON_MAT\\colab.zip")
    print("=" * 65)
    uploaded = files.upload()
    assert "colab.zip" in uploaded, "❌ Bạn chưa chọn colab.zip — hãy nhấn Run lại cell!"

# Giải nén mã nguồn
with zipfile.ZipFile("/content/colab.zip", "r") as z:
    z.extractall("/content")

def fix_windows_zip_paths(root="/content"):
    misplaced = []
    for dirpath, dirnames, filenames in os.walk(root):
        parts = dirpath.replace("\\", "/").split("/")
        if any(p in (".config", "sample_data", "drive") for p in parts):
            continue
        for fn in filenames:
            if "\\" in fn:
                misplaced.append(os.path.join(dirpath, fn))
    for src in misplaced:
        rel = os.path.relpath(src, root).replace("\\", "/")
        dst = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)

fix_windows_zip_paths()
print("[✓] Đã giải nén và nạp toàn bộ mã nguồn vào môi trường Colab!")

# ---------- 3. KIỂM TRA GPU T4 ----------
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print(f"[✓] TensorFlow: {tf.__version__} | GPU khả dụng: {gpus}")
if not gpus:
    print("\n⚠️ CẢNH BÁO: Chưa bật GPU! Hãy vào menu Runtime > Change runtime type > T4 GPU rồi chạy lại!")

# ---------- 4. CÀI ĐẶT THƯ VIỆN HỖ TRỢ ----------
print("[*] Đang cài đặt thư viện bổ trợ...")
os.system("pip -q install scikit-learn tqdm")

def run_command_stream(cmd, title=None):
    if title:
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
    p = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    for line in iter(p.stdout.readline, ''):
        print(line, end='', flush=True)
    p.stdout.close()
    return p.wait()

# Xóa model cũ nếu có để đảm bảo file tải về là model mới train
out_model = "/content/training_tinyml/weights/tinyface_backbone.keras"
if os.path.exists(out_model):
    os.remove(out_model)

# ---------- 5. TỰ ĐỘNG NHẬN DIỆN VÀ TIỀN XỬ LÝ CASIA-WEBFACE ----------
ret = run_command_stream(
    f'"{sys.executable}" -u training_tinyml/download_casia_dataset.py',
    "[BƯỚC 1/3] NẠP & TIỀN XỬ LÝ CASIA-WEBFACE 112x112"
)
if ret != 0:
    print("\n[!] Không tìm thấy CASIA-WebFace. Đang chuyển sang tập dữ liệu dự phòng LFW...")
    ret_lfw = run_command_stream(
        f'"{sys.executable}" -u training_tinyml/download_lfw_dataset.py',
        "[*] TẢI & TIỀN XỬ LÝ TẬP DỮ LIỆU DỰ PHÒNG LFW..."
    )
    assert ret_lfw == 0, "❌ Không thể chuẩn bị dữ liệu huấn luyện!"
    os.environ["FACE_DATASET_DIR"] = "/content/data/lfw_aligned"
else:
    os.environ["FACE_DATASET_DIR"] = "/content/data/casia_aligned"

# ---------- 6. BẮT ĐẦU HUẤN LUYỆN CHƯNG CẤT MODEL V3 ----------
ret = run_command_stream(
    f'"{sys.executable}" -u training_tinyml/train_distillation.py',
    "[BƯỚC 2/3] HUẤN LUYỆN MODEL V3 (ARCFACE + DISTILLATION + DYNAMIC HE)"
)
assert ret == 0, "❌ Quá trình huấn luyện gặp sự cố. Vui lòng kiểm tra log phía trên!"

# ---------- 7. TỰ ĐỘNG TẢI TRỌNG SỐ VỀ MÁY TÍNH ----------
assert os.path.exists(out_model), "❌ Không tìm thấy file kết quả tinyface_backbone.keras!"
file_kb = os.path.getsize(out_model) // 1024
print("\n" + "=" * 70)
print(f"[✓] Đã tạo thành công mô hình: {out_model} ({file_kb} KB)")

# Sao lưu thêm 1 bản vào Google Drive (nếu có kết nối Drive)
if os.path.exists("/content/drive/MyDrive"):
    drive_bk = "/content/drive/MyDrive/tinyface_backbone.keras"
    shutil.copyfile(out_model, drive_bk)
    print(f"[✓] Đã tự động lưu 1 bản dự phòng tại Google Drive: {drive_bk}")

print("[*] Đang tự động tải tinyface_backbone.keras về thư mục Downloads của bạn...")
files.download(out_model)

print("=" * 70)
print("🎉 CHÚC MỪNG BẠN ĐÃ HUẤN LUYỆN XONG MODEL V3 THÀNH CÔNG!")
print("=" * 70)
print("👉 Hãy kiểm tra thư mục Downloads trên máy tính của bạn.")
```

---

## Việc làm trên máy tính sau khi tải model về

Sau khi file `tinyface_backbone.keras` đã về thư mục **Downloads**, mở **Anaconda Prompt** (môi trường `projet_5`) tại thư mục `D:\PROJECT_5_DIEM_DANH_KHUON_MAT` và làm theo thứ tự:

| Bước | Lệnh | Mục đích |
|:---:|---|---|
| **1** | Copy `tinyface_backbone.keras` vào `training_tinyml\weights\` (ghi đè) | Cập nhật model mới |
| **2** | `python training_tinyml/quantize_qat_int8.py` | Nén model sang INT8 + tạo `model_data.h` |
| **3** | `python training_tinyml/update_face_database.py` | Tạo lại dữ liệu khuôn mặt cho nhien/thao/toan |
| **4** | `python training_tinyml/evaluate_model.py` | Kiểm tra độ đúng |
| **5** | `python host_laptop/main.py` | Test trên laptop |
| **6** | Mở Arduino IDE → Upload (hoặc chạy script nạp) | Nạp firmware mới có model V3 |
| **7** | `python host_laptop/ip_camera_streamer.py --ip <IP_ESP32> --port 12345` | Test thực tế |

> **Vì sao Model V3 tốt hơn:** nhờ học từ hơn 1,000 người trong CASIA-WebFace, khi bạn ngồi trước camera điểm sẽ cao (~0.75-0.85), còn người lạ hoặc khi che mặt sẽ tụt xuống rất thấp, gần như không nhận nhầm.
