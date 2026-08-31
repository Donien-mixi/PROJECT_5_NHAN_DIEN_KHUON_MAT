"""
==============================================================================
📥 TẢI VÀ TIỀN XỬ LÝ TẬP DỮ LIỆU LFW (LABELED FACES IN THE WILD)
==============================================================================
Mục đích: Tải tập dữ liệu LFW (~13,233 ảnh, ~5,749 danh tính) qua sklearn,
tiền xử lý thành ảnh Grayscale 64x64 và lưu theo cấu trúc thư mục chuẩn.

Dữ liệu này dùng để huấn luyện Ghost-TinyFace thành Universal Feature Extractor
thông qua Knowledge Distillation từ SFace Teacher.
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import cv2
import numpy as np

# Thêm đường dẫn để import từ host_laptop
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from host_laptop.detector.blazeface_esp32 import UnifiedFaceDetector
    from host_laptop.core.vision_utils import center_square_crop
except ImportError:
    print("❌ LỖI: Không tìm thấy thư mục host_laptop!")
    print("   Hãy chắc chắn bạn đã nén cả thư mục 'host_laptop' và 'training_tinyml' lên Colab.")
    sys.exit(1)

def download_and_prepare_lfw(min_faces=5, output_dir=None):
    """
    Tải LFW dataset qua sklearn và lưu thành ảnh Grayscale 64x64.
    
    Tham số:
        min_faces: Chỉ lấy người có ≥ min_faces ảnh (mặc định: 5).
                   Giúp đảm bảo mỗi danh tính có đủ mẫu để Teacher SFace
                   tạo embedding đa dạng góc nhìn.
        output_dir: Thư mục đầu ra. Mặc định: data/lfw_aligned/
    
    Trả về:
        output_dir: Đường dẫn thư mục chứa ảnh đã xử lý.
        total_images: Tổng số ảnh đã xử lý.
        total_identities: Tổng số danh tính.
    """
    print("==================================================================")
    print("📥 TẢI TẬP DỮ LIỆU LFW (LABELED FACES IN THE WILD)")
    print("==================================================================")
    
    # Import sklearn (chỉ cần khi chạy script này)
    try:
        from sklearn.datasets import fetch_lfw_people
    except ImportError:
        print("❌ LỖI: Thư viện scikit-learn chưa được cài đặt.")
        print("   Chạy: pip install scikit-learn")
        return None, 0, 0
    
    # Xác định thư mục đầu ra
    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "data", "lfw_aligned")
    
    # Kiểm tra nếu đã tải rồi thì bỏ qua
    if os.path.exists(output_dir):
        existing_dirs = [d for d in os.listdir(output_dir) 
                         if os.path.isdir(os.path.join(output_dir, d))]
        if len(existing_dirs) > 50:
            total_imgs = sum(
                len([f for f in os.listdir(os.path.join(output_dir, d)) 
                     if f.endswith(('.jpg', '.png'))])
                for d in existing_dirs
            )
            print(f"[✓] Dataset LFW đã tồn tại tại: {output_dir}")
            print(f"    ({len(existing_dirs)} danh tính, {total_imgs} ảnh)")
            print(f"    Bỏ qua việc tải lại. Xóa thư mục nếu muốn tải lại.")
            return output_dir, total_imgs, len(existing_dirs)
    
    # Tải LFW qua sklearn (tự động download lần đầu, cache cho lần sau)
    print(f"[*] Đang tải LFW dataset (min_faces_per_person={min_faces})...")
    print(f"    (Lần đầu sẽ tải ~200MB từ Internet, sau đó sẽ dùng cache)")
    
    lfw = fetch_lfw_people(
        min_faces_per_person=min_faces,
        resize=1.0,  # Giữ nguyên kích thước gốc (250x250 hoặc tương tự)
        color=True,   # Tải ảnh màu RGB để SFace Teacher xử lý tốt nhất
        slice_=None   # Lấy toàn bộ vùng ảnh (không cắt xén)
    )
    
    images = lfw.images       # Shape: (N, H, W, 3) — RGB float [0, 1]
    targets = lfw.target      # Shape: (N,) — chỉ số danh tính
    target_names = lfw.target_names  # Mảng tên danh tính
    
    print(f"[+] Đã tải thành công:")
    print(f"    - Tổng ảnh: {len(images)}")
    print(f"    - Tổng danh tính: {len(target_names)}")
    print(f"    - Kích thước ảnh gốc: {images[0].shape}")
    
    # Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[*] Đang khởi tạo BlazeFace ESP32 C++ Emulator để cắt ảnh...")
    detector = UnifiedFaceDetector(target_size=(64, 64), conf_threshold=0.5)

    # Xử lý và lưu từng ảnh
    print(f"\n[*] Đang dùng BlazeFace để căn chỉnh và lưu ảnh Grayscale 64x64 (Student) & BGR 112x112 (Teacher)...")
    total_saved = 0
    identity_counts = {}
    
    # Đồng thời lưu ảnh BGR gốc (resize 112x112) để SFace Teacher dùng
    teacher_dir = os.path.join(output_dir, "_teacher_112x112")
    os.makedirs(teacher_dir, exist_ok=True)
    
    for i in range(len(images)):
        person_name = target_names[targets[i]]
        # Tạo tên thư mục an toàn
        safe_name = person_name.replace(" ", "_")
        person_dir = os.path.join(output_dir, safe_name)
        person_teacher_dir = os.path.join(teacher_dir, safe_name)
        
        # Chuyển ảnh từ float [0, 1] sang uint8 [0, 255]
        img_rgb = (images[i] * 255).astype(np.uint8)
        
        # Chuyển sang BGR cho OpenCV
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # CHẠY BLAZEFACE ĐỂ CẮT ẢNH CHUẨN ESP32
        face_info = detector.detect_primary_face(img_bgr)
        if face_info is None:
            continue # Bỏ qua ảnh nếu BlazeFace không nhận ra
            
        x, y, w_box, h_box = face_info['bbox']
        
        # Cắt 64x64 Grayscale cho Student
        face_64_bgr, face_64_gray = center_square_crop(img_bgr, x, y, w_box, h_box, target_size=(64, 64))
        # Cắt 112x112 BGR cho Teacher
        face_112_bgr, _ = center_square_crop(img_bgr, x, y, w_box, h_box, target_size=(112, 112))
        
        if face_64_gray is None or face_112_bgr is None:
            continue
            
        os.makedirs(person_dir, exist_ok=True)
        os.makedirs(person_teacher_dir, exist_ok=True)
        
        # Đếm số ảnh đã lưu cho mỗi người
        if safe_name not in identity_counts:
            identity_counts[safe_name] = 0
        identity_counts[safe_name] += 1
        
        # Lưu file ảnh Student
        img_filename = f"{safe_name}_{identity_counts[safe_name]:04d}.jpg"
        cv2.imwrite(os.path.join(person_dir, img_filename), face_64_gray)
        
        # Lưu file ảnh Teacher
        teacher_filename = f"teacher_{safe_name}_{identity_counts[safe_name]:04d}.jpg"
        cv2.imwrite(os.path.join(person_teacher_dir, teacher_filename), face_112_bgr)
        
        total_saved += 1
        
        # Hiển thị tiến trình
        if (i + 1) % 500 == 0 or (i + 1) == len(images):
            print(f"    [{i+1}/{len(images)}] Đã cắt và lưu bằng BlazeFace...")
    
    # Thống kê
    num_identities = len([d for d in os.listdir(output_dir) 
                          if os.path.isdir(os.path.join(output_dir, d)) and not d.startswith("_")])
    
    print(f"\n==================================================================")
    print(f"🎉 TẢI VÀ XỬ LÝ LFW THÀNH CÔNG!")
    print(f"==================================================================")
    print(f"📂 Thư mục lưu trữ: {output_dir}")
    print(f"👥 Số danh tính: {num_identities}")
    print(f"🖼️  Tổng ảnh 64x64 Grayscale: {total_saved}")
    print(f"🎓 Tổng ảnh 112x112 BGR (cho Teacher): {total_saved}")
    
    # Hiển thị top 10 danh tính có nhiều ảnh nhất
    sorted_counts = sorted(
        [(k, v) for k, v in identity_counts.items() if not k.startswith("teacher_")],
        key=lambda x: x[1], reverse=True
    )[:10]
    print(f"\n📊 Top 10 danh tính có nhiều ảnh nhất:")
    for name, count in sorted_counts:
        print(f"    - {name}: {count} ảnh")
    
    return output_dir, total_saved, num_identities


if __name__ == "__main__":
    download_and_prepare_lfw(min_faces=5)
