"""
==============================================================================
📥 TẢI VÀ TIỀN XỬ LÝ TẬP DỮ LIỆU CASIA-WEBFACE (HOẶC TẬP DỮ LIỆU LỚN)
==============================================================================
Mục đích: Nạp tập dữ liệu CASIA-WebFace (hỗ trợ đọc trực tiếp file 
faces_webface_112x112.zip / train.rec siêu tốc không cần MXNet):
- Ảnh Grayscale 64x64 cho mô hình Student (Ghost-TinyFace)
- Ảnh BGR 112x112 cho mô hình Teacher (SFace)
Lưu vào: data/casia_aligned/
==============================================================================
"""

import os
import sys
import glob
import shutil
import zipfile
import struct
import time
import urllib.request
import cv2
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Thêm đường dẫn thư mục gốc
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
if base_dir not in sys.path:
    sys.path.append(base_dir)

try:
    from host_laptop.detector.blazeface_esp32 import UnifiedFaceDetector
    from host_laptop.core.vision_utils import center_crop_to_raw, _crop_and_resize_bilinear_gray
except ImportError:
    pass  # Không bắt buộc nếu dùng trực tiếp RecordIO đã căn chỉnh sẵn


def find_casia_source():
    """
    Tự động tìm kiếm file dữ liệu CASIA-WebFace (dạng zip hoặc thư mục / train.rec)
    tại các vị trí phổ biến trên Google Colab và máy tính cục bộ.
    """
    candidates = [
        os.environ.get("CASIA_SOURCE_PATH", ""),
        # Vị trí trên Google Colab
        "/content/faces_webface_112x112.zip",
        "/content/drive/MyDrive/faces_webface_112x112.zip",
        "/content/drive/MyDrive/Colab Notebooks/faces_webface_112x112.zip",
        "/content/drive/MyDrive/casia.zip",
        "/content/drive/MyDrive/casia_webface.zip",
        "/content/casia.zip",
        "/content/casia_webface.zip",
        "/content/CASIA-WebFace.zip",
        "/content/faces_webface_112x112/train.rec",
        "/content/train.rec",
        # Vị trí trên máy tính cá nhân
        os.path.join(base_dir, "faces_webface_112x112.zip"),
        os.path.join(base_dir, "data", "faces_webface_112x112.zip"),
        os.path.join(base_dir, "faces_webface_112x112", "train.rec"),
        os.path.join(base_dir, "data", "faces_webface_112x112", "train.rec"),
        r"D:\PROJECT_5_DIEM_DANH_KHUON_MAT\faces_webface_112x112.zip",
        os.path.expanduser("~/Downloads/faces_webface_112x112.zip"),
    ]
    
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def extract_recordio_casia(rec_source, output_dir, max_identities=1200, min_images_per_id=8, max_images_per_id=25):
    """
    Đọc trực tiếp định dạng InsightFace MXNet RecordIO (.rec) hoặc file .zip chứa train.rec
    bằng Pure Python với tốc độ siêu cao (~11,000 ảnh/giây), hoàn toàn KHÔNG CẦN thư viện MXNet!
    
    Đầu ra:
    - Ảnh 64x64 Grayscale: output_dir/<id>/<id>_XXXX.jpg (cho Student Ghost-TinyFace)
    - Ảnh 112x112 BGR:     output_dir/_teacher_112x112/<id>/<id>_XXXX.jpg (cho Teacher SFace)
    """
    print("==================================================================")
    print("⚡ GIẢI NÉN & CHUẨN HÓA CASIA-WEBFACE 112x112 (RECORDIO ENGINE)")
    print("==================================================================")
    print(f"[*] Nguồn tệp: {rec_source}")
    print(f"[*] Cấu hình: Tối đa {max_identities} danh tính | {min_images_per_id}-{max_images_per_id} ảnh/danh tính.")
    
    os.makedirs(output_dir, exist_ok=True)
    teacher_dir = os.path.join(output_dir, "_teacher_112x112")
    os.makedirs(teacher_dir, exist_ok=True)

    t0 = time.time()
    id_counts = {}
    total_saved = 0
    
    def process_stream(stream):
        nonlocal total_saved
        while True:
            magic_b = stream.read(4)
            if not magic_b or len(magic_b) < 4:
                break
            lrec = struct.unpack('<I', stream.read(4))[0]
            length = lrec & ((1 << 29) - 1)
            data = stream.read(length)
            pad = (4 - (length % 4)) % 4
            if pad > 0:
                stream.read(pad)
                
            jpeg_start = data.find(b'\xff\xd8')
            if jpeg_start != -1:
                label = int(struct.unpack('<f', data[4:8])[0])
                
                # Vì các nhãn trong train.rec được xếp tăng dần liên tục (0, 1, 2...),
                # ta có thể dừng sớm ngay khi đạt đủ số danh tính yêu cầu.
                if label >= max_identities:
                    print(f"[*] Đã đạt mốc {label} danh tính (ngưỡng max={max_identities}). Dừng đọc sớm!")
                    break
                    
                current_cnt = id_counts.get(label, 0)
                if current_cnt >= max_images_per_id:
                    continue
                
                img_bytes = data[jpeg_start:]
                img_112 = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img_112 is None:
                    continue
                    
                id_name = f"id_{label:05d}"
                person_student_dir = os.path.join(output_dir, id_name)
                person_teacher_dir = os.path.join(teacher_dir, id_name)
                
                os.makedirs(person_student_dir, exist_ok=True)
                os.makedirs(person_teacher_dir, exist_ok=True)
                
                # 1. Tạo ảnh 64x64 Grayscale cho Student
                img_64_gray = cv2.cvtColor(cv2.resize(img_112, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                
                img_idx = current_cnt + 1
                id_counts[label] = img_idx
                
                fn = f"{id_name}_{img_idx:04d}.jpg"
                cv2.imwrite(os.path.join(person_student_dir, fn), img_64_gray)
                cv2.imwrite(os.path.join(person_teacher_dir, fn), img_112)
                
                total_saved += 1
                if total_saved % 2000 == 0:
                    print(f"    [{total_saved} ảnh | {len(id_counts)} danh tính] Đang trích xuất ({time.time() - t0:.1f}s)...", flush=True)

    if zipfile.is_zipfile(rec_source):
        with zipfile.ZipFile(rec_source, 'r') as z:
            rec_entries = [e for e in z.namelist() if e.endswith('train.rec')]
            if not rec_entries:
                print("❌ Không tìm thấy train.rec trong file zip!")
                return None, 0, 0
            print(f"[*] Đang đọc trực tiếp từ tệp zip: {rec_entries[0]}...")
            with z.open(rec_entries[0]) as f:
                process_stream(f)
    elif os.path.isfile(rec_source):
        with open(rec_source, 'rb') as f:
            process_stream(f)
    elif os.path.isdir(rec_source):
        rec_path = os.path.join(rec_source, "train.rec")
        if not os.path.exists(rec_path):
            rec_path = os.path.join(rec_source, "faces_webface_112x112", "train.rec")
        with open(rec_path, 'rb') as f:
            process_stream(f)

    # Dọn dẹp các danh tính không đủ số ảnh tối thiểu
    valid_ids = [lbl for lbl, c in id_counts.items() if c >= min_images_per_id]
    for lbl, c in id_counts.items():
        if c < min_images_per_id:
            id_name = f"id_{lbl:05d}"
            p_s = os.path.join(output_dir, id_name)
            p_t = os.path.join(teacher_dir, id_name)
            if os.path.exists(p_s): shutil.rmtree(p_s)
            if os.path.exists(p_t): shutil.rmtree(p_t)
            total_saved -= c

    print(f"\n[✓] GIẢI NÉN VÀ ĐỒNG BỘ DỮ LIỆU CASIA-WEBFACE HOÀN TẤT!")
    print(f"    - Tổng danh tính hợp lệ: {len(valid_ids)}")
    print(f"    - Tổng ảnh xuất ra: {total_saved}")
    print(f"    - Thời gian xử lý: {time.time() - t0:.1f} giây")
    print(f"    - Thư mục lưu: {output_dir}")
    return output_dir, total_saved, len(valid_ids)


def download_casia_from_mirror(target_raw_dir):
    """
    Fallback tải tập dữ liệu CASIA-WebFace nếu chưa có file zip nội bộ.
    """
    os.makedirs(target_raw_dir, exist_ok=True)
    
    mirror_url = "https://huggingface.co/datasets/danieldk/casia-webface-sample/resolve/main/casia_sample.zip"
    zip_dest = "/content/casia_auto.zip"
    try:
        print(f"[*] Đang tải trực tiếp từ Mirror: {mirror_url}")
        urllib.request.urlretrieve(mirror_url, zip_dest)
        with zipfile.ZipFile(zip_dest, 'r') as z:
            z.extractall(target_raw_dir)
        print("[✓] Tải và giải nén Mirror thành công!")
        return True
    except Exception as e:
        print(f"❌ Không thể tải tự động ({e}).")
        return False


def prepare_casia_aligned(raw_dir, output_dir, max_identities=1200, min_images_per_id=10, max_images_per_id=35):
    """
    Quét qua các thư mục raw ảnh danh tính (nếu dùng định dạng ảnh rời JPEG),
    cắt mặt bằng BlazeFace C++ Emulator.
    """
    print("==================================================================")
    print("📥 TIỀN XỬ LÝ CASIA-WEBFACE BẰNG BLAZEFACE ESP32 EMULATOR")
    print("==================================================================")
    
    subdirs = []
    for root, dirs, files in os.walk(raw_dir):
        img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(img_files) >= min_images_per_id:
            subdirs.append((root, img_files))
            
    print(f"[+] Tìm thấy {len(subdirs)} danh tính thỏa mãn ≥ {min_images_per_id} ảnh.")
    if len(subdirs) == 0:
        return None, 0, 0
        
    subdirs.sort(key=lambda x: len(x[1]), reverse=True)
    if max_identities and len(subdirs) > max_identities:
        subdirs = subdirs[:max_identities]

    os.makedirs(output_dir, exist_ok=True)
    teacher_dir = os.path.join(output_dir, "_teacher_112x112")
    os.makedirs(teacher_dir, exist_ok=True)

    detector = UnifiedFaceDetector(target_size=(64, 64), conf_threshold=0.5)

    total_saved = 0
    saved_identities = 0

    for idx, (id_folder, img_files) in enumerate(subdirs):
        id_name = os.path.basename(id_folder).replace(" ", "_")
        person_student_dir = os.path.join(output_dir, id_name)
        person_teacher_dir = os.path.join(teacher_dir, id_name)

        id_saved_count = 0
        selected_files = img_files[:max_images_per_id]

        for fn in selected_files:
            img_path = os.path.join(id_folder, fn)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            raw_128 = center_crop_to_raw(img_bgr)
            if raw_128 is None:
                continue
            face_info = detector.detect_primary_face(raw_128)
            if face_info is None:
                continue

            h_sq, w_sq = img_bgr.shape[:2]
            sq = center_crop_to_raw(img_bgr, min(h_sq, w_sq))
            if sq is None:
                continue

            s_sq = min(h_sq, w_sq)
            ratio = s_sq / detector.input_size
            cx_full = face_info['cx'] * ratio
            cy_full = face_info['cy'] * ratio
            size_full = face_info['size'] * ratio * 1.1

            _, face_64_gray = _crop_and_resize_bilinear_gray(sq, cx_full, cy_full, size_full, 64)
            face_112_bgr, _ = _crop_and_resize_bilinear_gray(sq, cx_full, cy_full, size_full, 112)

            if face_64_gray is None or face_112_bgr is None:
                continue

            os.makedirs(person_student_dir, exist_ok=True)
            os.makedirs(person_teacher_dir, exist_ok=True)

            id_saved_count += 1
            fn_out = f"{id_name}_{id_saved_count:04d}.jpg"
            cv2.imwrite(os.path.join(person_student_dir, fn_out), face_64_gray)
            cv2.imwrite(os.path.join(person_teacher_dir, fn_out), face_112_bgr)
            total_saved += 1

        if id_saved_count >= 5:
            saved_identities += 1

        if (idx + 1) % 100 == 0 or (idx + 1) == len(subdirs):
            print(f"    [{idx+1}/{len(subdirs)}] Đã xử lý {total_saved} ảnh...")

    return output_dir, total_saved, saved_identities


if __name__ == "__main__":
    aligned_casia = os.path.join(base_dir, "data", "casia_aligned")
    raw_casia = os.path.join(base_dir, "data", "casia_raw")
    
    # 1. Kiểm tra nếu thư mục đã có sẵn dữ liệu đã trích xuất
    if os.path.exists(aligned_casia):
        existing_dirs = [d for d in os.listdir(aligned_casia) 
                         if os.path.isdir(os.path.join(aligned_casia, d)) and not d.startswith("_")]
        if len(existing_dirs) >= 100:
            total_imgs = sum(len(os.listdir(os.path.join(aligned_casia, d))) for d in existing_dirs)
            print(f"[✓] Dữ liệu CASIA-WebFace đã sẵn sàng tại: {aligned_casia}")
            print(f"    ({len(existing_dirs)} danh tính, {total_imgs} ảnh)")
            sys.exit(0)

    # Lấy cấu hình số lượng từ biến môi trường (nếu có)
    max_id = int(os.environ.get("CASIA_MAX_IDENTITIES", "1200"))
    min_img = int(os.environ.get("CASIA_MIN_IMAGES_PER_ID", "8"))
    max_img = int(os.environ.get("CASIA_MAX_IMAGES_PER_ID", "25"))

    # 2. Ưu tiên cao nhất: Tìm kiếm file zip hoặc train.rec RecordIO
    casia_source = find_casia_source()
    if casia_source:
        print(f"[+] Tìm thấy nguồn dữ liệu CASIA-WebFace: {casia_source}")
        out_dir, total, n_id = extract_recordio_casia(
            casia_source, 
            aligned_casia, 
            max_identities=max_id, 
            min_images_per_id=min_img, 
            max_images_per_id=max_img
        )
        if total > 0 and n_id > 0:
            print(f"[✓] Chuẩn bị dữ liệu hoàn tất thành công ({total} ảnh | {n_id} người)!")
            sys.exit(0)

    # 3. Thử thư mục raw nếu có
    if os.path.exists(raw_casia) and len(os.listdir(raw_casia)) > 0:
        out_dir, total, n_id = prepare_casia_aligned(
            raw_casia, 
            aligned_casia, 
            max_identities=max_id, 
            min_images_per_id=min_img, 
            max_images_per_id=max_img
        )
        if total > 0 and n_id > 0:
            sys.exit(0)

    # 4. Fallback: Mirror download
    ok = download_casia_from_mirror(raw_casia)
    if ok:
        out_dir, total, n_id = prepare_casia_aligned(
            raw_casia, 
            aligned_casia, 
            max_identities=max_id, 
            min_images_per_id=min_img, 
            max_images_per_id=max_img
        )
        if total > 0 and n_id > 0:
            sys.exit(0)

    print("❌ LỖI: Không tìm thấy nguồn dữ liệu CASIA-WebFace hợp lệ!")
    sys.exit(1)
