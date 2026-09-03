# ☁️ HƯỚNG DẪN TRAIN MODEL V2 TRÊN GOOGLE COLAB (1 Ô LỆNH DUY NHẤT)

Model v2 giải quyết bài toán **nhầm lẫn giữa các người đã đăng ký** (thao↔toan) bằng:
- **ArcFace loss** (Deng et al., CVPR 2019) — ép margin góc giữa các danh tính LFW
- **Illumination-invariance KD** — teacher nhúng ảnh SẠCH, student nhận ảnh augment tối/sáng + HE
- Đồng bộ pipeline triển khai: HE (`equalize_gray_256`) có trong cả train và inference (Laptop + ESP32)

**Không cần chụp lại ảnh đăng ký** — sau khi train xong, DB sẽ được regenerate từ chính ảnh hiện có.

---

## CÁC BƯỚC

1. Mở [colab.research.google.com](https://colab.research.google.com) → **New Notebook**
2. **Runtime → Change runtime type → T4 GPU → Save** (bắt buộc — CPU sẽ chậm gấp 10 lần)
3. Copy **TOÀN BỘ ô lệnh bên dưới** vào 1 cell → nhấn **Play**
4. Khi được hỏi → chọn file `colab.zip` (nằm ở `D:\PROJECT_5_DIEM_DANH_KHUON_MAT\colab.zip`)
5. Chờ: tải LFW (~3-5 phút) → train (~40-90 phút, có early stopping) → model **tự tải về máy**
6. File về máy: `tinyface_backbone.keras` (khoảng 2MB, thư mục Downloads)

---

## Ô LỆNH COLAB — COPY TOÀN BỘ VÀO 1 CELL RỒI NHẤN PLAY

```python
# ==============================================================================
# TRAIN MODEL V2 (ArcFace + Illumination-Invariance KD) — CHẠY 1 LẦN NHẤN PLAY
# Yêu cầu: Runtime > Change runtime type > T4 GPU (đã chọn trước khi chạy)
# Kết quả: tự tải tinyface_backbone.keras về máy khi train xong
# ==============================================================================
import os, sys, shutil, zipfile

# ---------- 1. UPLOAD colab.zip (bỏ qua nếu đã upload trong phiên này) ----------
from google.colab import files
if os.path.exists("/content/colab.zip"):
    print("[OK] colab.zip đã có trong /content — bỏ qua upload (cell chạy lại).")
else:
    print("=" * 60)
    print(">>> Chọn file colab.zip (D:\\PROJECT_5_DIEM_DANH_KHUON_MAT\\colab.zip)")
    print("=" * 60)
    up = files.upload()
    assert "colab.zip" in up, "❌ Bạn chưa chọn colab.zip — chạy lại cell!"

# ---------- 2. GIẢI NÉ + CHUẨN HÓA ĐƯỜNG DẪN ----------
with zipfile.ZipFile("/content/colab.zip") as z:
    z.extractall("/content")

def fix_windows_zip_paths(root="/content"):
    """Zip tạo bằng PowerShell Compress-Archive trên Windows lưu entry bằng '\\'.
    Trên Linux chúng thành file có tên chứa '\\' thay vì cấu trúc thư mục.
    Hàm này chuyển về đúng chỗ. Nếu zip đã dùng '/' thì không làm gì cả."""
    misplaced = []
    for dirpath, dirnames, filenames in os.walk(root):
        parts = dirpath.replace("\\", "/").split("/")
        if any(p in (".config", "sample_data") for p in parts):
            continue  # bỏ qua thư mục hệ thống của Colab
        for fn in filenames:
            if "\\" in fn:
                misplaced.append(os.path.join(dirpath, fn))
    for src in misplaced:
        rel = os.path.relpath(src, root).replace("\\", "/")
        dst = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
    return len(misplaced)

fixed = fix_windows_zip_paths()
print(f"[OK] Giải nén xong (sửa {fixed} path kiểu Windows nếu có).")

# Xác nhận cấu trúc thư mục đúng trước khi chạy
for need in ("training_tinyml/train_distillation.py",
             "training_tinyml/download_lfw_dataset.py",
             "host_laptop/core/vision_utils.py",
             "host_laptop/detector/blazeface_esp32.py",
             "host_laptop/detector/face_detection_short_range_int8.tflite"):
    assert os.path.exists("/content/" + need), (
        f"❌ Thiếu {need} — zip sai cấu trúc. Dùng lại colab.zip mới nhất rồi chạy lại cell!")
print("[OK] Cấu trúc thư mục chuẩn.")

# ---------- 3. KIỂM TRA GPU ----------
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print(f"[OK] TensorFlow {tf.__version__} | GPU: {gpus}")
if not gpus:
    print("⚠️⚠️ KHÔNG CÓ GPU! Vào Runtime > Change runtime type > T4 GPU,")
    print("   rồi chạy lại cell này (Runtime > Restart and run all).")

# ---------- 4. CÀI DEPENDENCY ----------
print("[..] Cài scikit-learn (cho LFW loader)...")
os.system("pip -q install scikit-learn")

# ---------- 5. TẢI + TIỀN XỬ LÝ LFW (~200MB, lần đầu ~3-5 phút) ----------
print("=" * 60)
print("[5/6] TẢI LFW + CẮT MẶT BẰNG BLAZEFACE (bỏ qua nếu đã có)")
print("=" * 60)
r = os.system(f'"{sys.executable}" training_tinyml/download_lfw_dataset.py')
assert r == 0, "❌ Lỗi tải LFW — đọc log đỏ phía trên (nếu có) và gửi cho hỗ trợ!"

# ---------- 6. TRAIN: KD + ArcFace + ILLUMINATION-INVARIANCE ----------
print("=" * 60)
print("[6/6] TRAIN MODEL V2 (~40-90 phút trên T4, có early stopping)")
print("=" * 60)
r = os.system(f'"{sys.executable}" training_tinyml/train_distillation.py')
assert r == 0, "❌ Lỗi train — đọc log đỏ phía trên (nếu có) và gửi cho hỗ trợ!"

# ---------- 7. TỰ ĐỘNG TẢI MODEL VỀ MÁY ----------
out = "/content/training_tinyml/weights/tinyface_backbone.keras"
assert os.path.exists(out), "❌ Không thấy tinyface_backbone.keras!"
print(f"[OK] Model: {os.path.getsize(out)//1024} KB — đang tải về máy...")
files.download(out)

print("=" * 60)
print("✅ XONG! File tinyface_backbone.keras đã tải về máy (thư mục Downloads).")
print("=" * 60)
print("CÁC BƯỚC TIẾP THEO (trên máy, trong Anaconda Prompt projet_5):")
print("  1. Copy file vào:  training_tinyml\\weights\\tinyface_backbone.keras (ghi đè)")
print("  2. python training_tinyml/quantize_qat_int8.py")
print("  3. python training_tinyml/update_face_database.py")
print("  4. python training_tinyml/evaluate_model.py")
print("     → Mục 5 IDENTIFICATION phải >= 90% va khong con cap nham lan")
print("  5. python host_laptop/main.py  (test Laptop)")
print("  6. Flash lại ESP32 (Arduino IDE > Verify > Upload)")
print("  7. python host_laptop/ip_camera_streamer.py --ip <IP> --port 12345")
```

---

## SAU KHI MODEL VỀ MÁY — QUY TRÌNH LOCAL (trong `kich_hoat_moi_truong.md`)

| Bước | Lệnh | Kết quả mong đợi |
|---|---|---|
| 1 | Copy `tinyface_backbone.keras` → `training_tinyml\weights\` (ghi đè) | — |
| 2 | `python training_tinyml/quantize_qat_int8.py` | `tinyface_int8.tflite` mới + `model_data.h` |
| 3 | `python training_tinyml/update_face_database.py` | DB + `face_database.h` regenerate từ ảnh hiện có (HE) |
| 4 | `python training_tinyml/evaluate_model.py` | **Mục 5 IDENTIFICATION ≥ 90%, không còn cặp nhầm** |
| 5 | `python host_laptop/main.py` | 3 người nhận đúng tên |
| 6 | Arduino IDE → Verify → Upload | Firmware mới (có HE + model v2) |
| 7 | `python host_laptop/ip_camera_streamer.py --ip <IP>` | Serial in đúng tên từng người |

> **Nếu Identification vẫn < 90%:** gửi nguyên output `evaluate_model.py` cho tôi. Khi đó
> phương án kế tiếp là tăng kích thước model (Ghost channels ×1.5) hoặc bổ sung dataset —
> quyết định dựa trên số liệu, không đoán.

> **Lưu ý:** lần đầu chạy cell, Colab tải LFW ~200MB + SFace teacher 38MB (tự động).
> Nếu Colab ngắt kết nối giữa chừng: chạy lại cell — LFW đã cache, train bắt đầu lại từ đầu
> (colab.zip cũng đã có trong /content nên không phải upload lại). Cell có tự sửa lỗi zip
> tạo từ Windows (entry backslash) nên dùng phiên bản colab.zip mới nhất là chắc chắn nhất.
> Train dùng **early stopping** (patience 8 epochs) nên có thể kết thúc sớm hơn 50 epochs.
