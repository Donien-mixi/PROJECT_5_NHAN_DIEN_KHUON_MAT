import cv2
import time
import os
import sys

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from host_laptop.detector.yunet_detector import UnifiedFaceDetector
from host_laptop.database.db_manager import DatabaseManager
from host_laptop.recognizer.face_recognizer import FaceRecognizer, TemporalVoter
from host_laptop.ui.hud_renderer import HUDRenderer

def main():
    print("==================================================================")
    print("🚀 KHỞI ĐỘNG HỆ THỐNG ĐIỂM DANH (LAPTOP-FIRST DEVELOPMENT)")
    print("==================================================================")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "training_tinyml", "weights", "tinyface_int8.tflite")
    json_db_path = os.path.join(base_dir, "data", "face_database.json")
    sqlite_db_path = os.path.join(base_dir, "data", "attendance.db")
    
    # 1. Khởi tạo các module (Phân chia logic rõ ràng)
    recognizer = FaceRecognizer(model_path=model_path, db_path=json_db_path, threshold=0.88, use_tflite=True)
    detector = UnifiedFaceDetector(target_size=(64, 64), conf_threshold=0.80)
    db = DatabaseManager(db_path=sqlite_db_path, cooldown_seconds=30)
    voter = TemporalVoter(required_votes=3)
    
    # 2. Khởi tạo Camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n[+] Đang mở luồng Camera... Nhấn 'Q' hoặc 'ESC' để thoát.")
    
    # Các biến thống kê
    fps = 0.0
    frame_count = 0
    fps_time = time.time()
    
    # Các biến trạng thái nhận diện
    last_crop_bgr = None
    recognition_start_time = None
    tracked_name = None
    tracked_sim = 0.0
    tracked_infer = 0.0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
                
            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            # Tính FPS
            now = time.time()
            dt = now - fps_time
            if dt >= 0.5:
                fps = frame_count / dt
                fps_time = now
                frame_count = 0
            
            # 3. Phát hiện khuôn mặt (Detector)
            face = detector.detect_primary_face(frame)
            
            # Biến phục vụ hiển thị
            infer = 0.0
            matched = False
            recognized_name = "Unknown"
            similarity = 0.0
            
            if face is not None:
                if recognition_start_time is None:
                    recognition_start_time = time.time()
                    sys.stdout.write("\n")
                    
                elapsed = time.time() - recognition_start_time
                crop_bgr, crop_gray, _ = detector.align_and_crop(frame, face)
                
                if crop_bgr is not None:
                    last_crop_bgr = crop_bgr
                    
                    if not tracked_name:
                        # 4. Trích xuất đặc trưng và so khớp (Recognizer)
                        matched_raw, name_raw, sim_raw, infer_ms_raw = recognizer.recognize(crop_gray)
                        
                        # Hiển thị log trạng thái đang quét
                        if not matched_raw or name_raw == "Unknown":
                            sys.stdout.write(f"\r👀 Đang so khớp: {sim_raw*100:.1f}%       ")
                            sys.stdout.flush()
                        
                        # 5. Bộ lọc nhiễu (Temporal Voter)
                        is_locked, final_name, final_sim, final_infer = voter.vote(name_raw, sim_raw, infer_ms_raw)
                        
                        if is_locked:
                            # Chốt kết quả
                            tracked_name = final_name
                            tracked_sim = final_sim
                            tracked_infer = final_infer
                            
                            sys.stdout.write(f"\r✅ Đã nhận diện: {tracked_name} ({tracked_sim*100:.1f}%) | {elapsed:.1f}s\n")
                            sys.stdout.flush()
                            
                            # 6. Ghi log CSDL (Database)
                            db.log_attendance(tracked_name, tracked_sim, tracked_infer)
                    
                    if tracked_name:
                        matched = True
                        recognized_name = tracked_name
                        similarity = tracked_sim
                        infer = tracked_infer
                    else:
                        matched = matched_raw
                        recognized_name = name_raw
                        similarity = sim_raw
                        infer = infer_ms_raw
                    
                    # Vẽ Bounding Box và thông tin khuôn mặt
                    HUDRenderer.draw_face_box(frame, face, matched, recognized_name, similarity, detector.mode)
            else:
                # Reset các trạng thái khi mất khuôn mặt
                recognition_start_time = None
                last_crop_bgr = None
                tracked_name = None
                tracked_sim = 0.0
                tracked_infer = 0.0
                voter.reset()
            
            # 7. Vẽ giao diện HUD (UI)
            HUDRenderer.draw_hud(frame, fps, detector.mode, infer, last_crop_bgr, matched, recognized_name, similarity, use_tflite=True)
            cv2.imshow("TinyML Face Recognition - Laptop First", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n[*] Đã đóng ứng dụng an toàn.")

if __name__ == "__main__":
    main()
