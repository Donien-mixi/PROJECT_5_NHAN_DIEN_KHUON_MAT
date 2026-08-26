import cv2
import os
import sys
import numpy as np

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from host_laptop.detector.yunet_detector import UnifiedFaceDetector

def enroll_user():
    print("==================================================================")
    print("📸 CÔNG CỤ THU THẬP KHUÔN MẶT ĐĂNG KÝ (FACE ENROLLMENT TOOL)")
    print("==================================================================")
    
    user_name = input("[?] Nhập Tên hoặc ID của người dùng (không dấu, viết liền): ").strip().replace(" ", "_")
    if not user_name:
        print("❌ Tên không hợp lệ. Thoát.")
        return

    # Khởi tạo thư mục lưu trữ
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, "data", "registered_faces", user_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"[*] Dữ liệu sẽ được lưu tại: {save_dir}")
    
    # Khởi tạo UnifiedFaceDetector (ưu tiên MediaPipe nếu có)
    detector = UnifiedFaceDetector(target_size=(64, 64), prefer_mediapipe=True, conf_threshold=0.75)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ LỖI: Không thể mở Webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    print("\n[+] HƯỚNG DẪN:")
    print("    - Nhấn phím 'C' để CHỤP VÀ LƯU khuôn mặt.")
    print("    - Nên quay mặt sang trái/phải/lên/xuống nhẹ để thu đa dạng góc độ.")
    print("    - Nhấn phím 'Q' hoặc 'ESC' để THOÁT.")
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        
        # Nhận diện chủ thể chính
        face = detector.detect_primary_face(frame)
        best_face_crop_bgr = None
        
        if face is not None:
            x, y, w, h = face['bbox']
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            all_lm = face.get('all_landmarks', [])
            if detector.mode == "MEDIAPIPE_MESH":
                for (lx, ly) in all_lm:
                    cv2.circle(display_frame, (lx, ly), 1, (0, 255, 255), -1)
            else:
                for (lx, ly) in face.get('landmarks_5', []):
                    cv2.circle(display_frame, (lx, ly), 4, (0, 215, 255), -1)
                    cv2.circle(display_frame, (lx, ly), 6, (0, 255, 0), 1)
                    
            crop_bgr, crop_gray, byte_data = detector.align_and_crop(frame, face)
            if crop_bgr is not None:
                best_face_crop_bgr = crop_bgr
                # Hiển thị ô ảnh đã canh chỉnh ở góc
                display_frame[10:74, 10:74] = crop_bgr
                cv2.rectangle(display_frame, (8, 8), (76, 76), (0, 255, 255), 2)
                cv2.putText(display_frame, "64x64 Aligned", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Hiển thị thanh Header thông tin người dùng và số ảnh đã chụp (2 dòng rõ ràng, không bị tràn viền)
        h, w = display_frame.shape[:2]
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (120, 10), (w - 10, 72), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)
        
        cv2.putText(display_frame, f"User: {user_name}", (130, 36), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Captured: {count} photos (Press 'C' to snap)", (130, 62), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2)
                    
        cv2.imshow("Enrollment Tool", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c') or key == ord('C'):
            if best_face_crop_bgr is not None:
                filename = os.path.join(save_dir, f"{user_name}_{count:03d}.jpg")
                cv2.imwrite(filename, best_face_crop_bgr)
                print(f"[+] Đã lưu: {filename}")
                count += 1
                
                # Hiệu ứng flash
                flash = np.ones_like(display_frame) * 255
                cv2.imshow("Enrollment Tool", flash)
                cv2.waitKey(50)
            else:
                print("⚠️ Không tìm thấy khuôn mặt rõ ràng để chụp!")
                
        elif key == ord('q') or key == ord('Q') or key == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[*] Đã thoát. Thu thập tổng cộng {count} ảnh cho {user_name}.")

if __name__ == "__main__":
    enroll_user()
