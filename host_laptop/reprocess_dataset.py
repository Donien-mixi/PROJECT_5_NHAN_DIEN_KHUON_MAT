import os
import cv2
import glob
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from detector.blazeface_esp32 import UnifiedFaceDetector

def main():
    print("==================================================")
    print(" BẮT ĐẦU XỬ LÝ LẠI DỮ LIỆU ĐỂ KHỚP VỚI ESP32 (BLAZEFACE)")
    print("==================================================")
    
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "registered_faces")
    if not os.path.exists(dataset_dir):
        print(f"[!] Không tìm thấy thư mục {dataset_dir}")
        return
        
    # Khởi tạo detector
    detector = UnifiedFaceDetector(target_size=(64, 64), conf_threshold=0.5)
    
    # Lấy tất cả ảnh
    image_paths = glob.glob(os.path.join(dataset_dir, "*", "*.jpg"))
    total_images = len(image_paths)
    
    if total_images == 0:
        print("[!] Không có ảnh nào để xử lý.")
        return
        
    print(f"[*] Tìm thấy {total_images} ảnh. Đang tiến hành Re-alignment...")
    
    success_count = 0
    fail_count = 0
    
    for idx, path in enumerate(image_paths):
        # Đọc ảnh gốc (đã bị crop bởi MediaPipe)
        img = cv2.imread(path)
        if img is None:
            continue
            
        # Tìm khuôn mặt lại bằng BlazeFace (mô phỏng ESP32)
        face_info = detector.detect_primary_face(img)
        
        if face_info is not None:
            # Cắt lại ảnh theo bounding box của BlazeFace
            face_bgr, _, _ = detector.align_and_crop(img, face_info)
            if face_bgr is not None:
                # Ghi đè ảnh cũ
                cv2.imwrite(path, face_bgr)
                success_count += 1
                print(f"[{idx+1}/{total_images}] Thành công: {os.path.basename(path)}")
            else:
                print(f"[{idx+1}/{total_images}] Lỗi cắt ảnh: {os.path.basename(path)}")
                fail_count += 1
        else:
            print(f"[{idx+1}/{total_images}] BlazeFace không tìm thấy mặt trong ảnh đã crop: {os.path.basename(path)}")
            fail_count += 1
            
    print("==================================================")
    print(f"[*] Hoàn thành! Thành công: {success_count} | Thất bại: {fail_count}")
    print("[*] Vui lòng chạy lại 'update_face_database.py' và 'evaluate_model.py'")
    print("==================================================")

if __name__ == "__main__":
    main()
