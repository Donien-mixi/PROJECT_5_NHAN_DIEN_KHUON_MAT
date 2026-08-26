import cv2
import time
import os
import sys
import threading
import queue
import numpy as np

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from host_laptop.detector.yunet_detector import UnifiedFaceDetector
from host_laptop.bridge.esp32_serial import ESP32Bridge
from host_laptop.database.db_manager import DatabaseManager

class ESP32WorkerThread(threading.Thread):
    """
    Luồng chạy ngầm độc lập (Background Worker) gửi nhận dữ liệu với ESP32-S3.
    Giúp Camera trên Laptop chạy mượt mà 30 - 60 FPS mà không bị nghẽn (Block) bởi Serial.
    """
    def __init__(self, bridge: ESP32Bridge):
        super().__init__(daemon=True)
        self.bridge = bridge
        self.image_queue = queue.Queue(maxsize=1) # Chỉ giữ khung hình mới nhất
        self.running = True
        self.last_rtt = 0.0
        self.last_infer_ms = 0.0
        self.last_matched = False
        self.last_name = "Unknown"
        self.last_similarity = 0.0
        self.is_processing = False

    def push_frame(self, image_bytes: bytes):
        """Đẩy ảnh mới nhất vào hàng đợi (bỏ qua ảnh cũ nếu ESP32 đang bận)."""
        if self.image_queue.full():
            try:
                self.image_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.image_queue.put_nowait(image_bytes)
        except queue.Full:
            pass

    def run(self):
        while self.running:
            try:
                # Chờ có ảnh mới trong queue
                img_bytes = self.image_queue.get(timeout=0.1)
                self.is_processing = True
                
                resp = self.bridge.send_inference(img_bytes)
                if resp.get("status") == "success":
                    self.last_rtt = resp.get("rtt_ms", 0.0)
                    self.last_infer_ms = resp.get("inference_ms", resp.get("simulated_infer_ms", 0.0))
                    self.last_matched = resp.get("matched", False)
                    self.last_name = resp.get("name", "Unknown")
                    self.last_similarity = resp.get("similarity", 0.0)
                elif resp.get("status") == "error":
                    print(f"\n[!] LỖI ESP32: {resp.get('message', 'Unknown error')}")
                    
                self.is_processing = False
                self.image_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.is_processing = False
                time.sleep(0.05)

    def stop(self):
        self.running = False

def draw_hud(frame, fps, detector_mode, esp_status, rtt_ms, infer_ms, last_crop_bgr=None, matched=False, name="Unknown", sim=0.0):
    """
    Vẽ thanh thông số HUD công nghệ cao và khung PiP 96x96 lên màn hình (tự động co giãn theo độ phân giải).
    """
    h, w = frame.shape[:2]
    
    # 1. Vẽ thanh Header trên cùng (nền tối mờ cao cấp)
    header_h = 42
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    font_scale = 0.48 if w <= 640 else 0.55
    y_text = 27
    
    # Cột 1: FPS Camera
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 2)
    
    # Cột 2: AI Detector Mode
    mode_text = "MESH(468p)" if detector_mode == "MEDIAPIPE_MESH" else "YUNET(5p)"
    col2_x = int(w * 0.20)
    cv2.putText(frame, f"AI: {mode_text}", (col2_x, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 200, 50), 2)
    
    # Cột 3: Trạng thái ESP32
    col3_x = int(w * 0.46)
    status_text = "ESP32: ON" if esp_status == "CONNECTED" else "ESP32: OFF"
    status_color = (0, 255, 0) if esp_status == "CONNECTED" else (0, 0, 255)
    cv2.putText(frame, status_text, (col3_x, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color, 2)
    
    # Cột 4 & 5: RTT & AI Latency
    if esp_status == "CONNECTED":
        col4_x = int(w * 0.68)
        col5_x = int(w * 0.85)
        cv2.putText(frame, f"RTT:{rtt_ms:.0f}ms", (col4_x, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 230, 255), 2)
        cv2.putText(frame, f"AI:{infer_ms:.0f}ms", (col5_x, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100, 255, 100), 2)

    # 3. Khung Picture-in-Picture (PiP) góc dưới phải
    if last_crop_bgr is not None:
        pip_w, pip_h = 100, 100
        margin = 12
        pip_x = w - pip_w - margin
        pip_y = h - pip_h - margin
        
        crop_resized = cv2.resize(last_crop_bgr, (pip_w, pip_h))
        frame[pip_y:pip_y+pip_h, pip_x:pip_x+pip_w] = crop_resized
        
        # Viền PiP (Xanh lá nếu nhận diện đúng, Đỏ nếu chưa khớp)
        pip_border_color = (0, 255, 0) if matched else (0, 165, 255)
        cv2.rectangle(frame, (pip_x, pip_y), (pip_x+pip_w, pip_y+pip_h), pip_border_color, 2)
        cv2.putText(frame, "64x64 ESP32", (pip_x, pip_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, pip_border_color, 1)

def main():
    print("==================================================================")
    print("🚀 KHỞI ĐỘNG HỆ THỐNG ĐIỂM DANH TINYML EDGE AI (ESP32-S3)")
    print("==================================================================")
    
    # 1. Khởi tạo bộ Face Detector cao cấp (Tự động ưu tiên MediaPipe Mesh 468p)
    detector = UnifiedFaceDetector(target_size=(64, 64))
    
    # 2. Khởi tạo kết nối Serial với ESP32-S3
    esp32_bridge = ESP32Bridge(port='COM6', baudrate=921600, timeout=1.0)
    esp_connected = esp32_bridge.connect()
    
    # 3. Khởi tạo Cơ sở dữ liệu SQLite
    db = DatabaseManager(db_path="data/attendance.db", cooldown_seconds=30)
    
    esp_worker = None
    if esp_connected:
        esp_worker = ESP32WorkerThread(esp32_bridge)
        esp_worker.start()
        print("[+] Đã khởi động Luồng giao tiếp nền ESP32 Worker!")
    else:
        print("[!] Không tìm thấy ESP32. Hệ thống sẽ chạy ở chế độ Standalone (Camera Only).")
        
    # 3. Mở Webcam Laptop
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n[+] Đang mở luồng Camera... Nhấn 'Q' hoặc 'ESC' để thoát.")
    
    fps = 0.0
    frame_count = 0
    fps_time = time.time()
    last_crop_bgr = None
    recognition_start_time = None
    
    # Biến trạng thái để KHÓA NHÃN (Lock) khi đã nhận diện thành công
    tracked_name = None
    tracked_sim = 0.0
    tracked_infer = 0.0
    
    # Biến hiển thị HUD (cần khởi tạo trước vòng lặp để tránh NameError)
    rtt = 0.0
    infer = 0.0
    matched = False
    recognized_name = "Unknown"
    similarity = 0.0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            # Tính FPS camera thực tế
            now = time.time()
            dt = now - fps_time
            if dt >= 0.5:
                fps = frame_count / dt
                fps_time = now
                frame_count = 0
            
            # 1. Phát hiện CHÍNH XÁC khuôn mặt chủ thể (Bỏ qua hoàn toàn nền nhiễu)
            face = detector.detect_primary_face(frame)
            
            if face is not None:
                if recognition_start_time is None:
                    recognition_start_time = time.time()
                    sys.stdout.write("\n") # Sang dòng mới cho block in Terminal
                    
                x, y, w, h = face['bbox']
                
                # Tính thời gian từ lúc phát hiện đến hiện tại
                elapsed = time.time() - recognition_start_time

                # Lấy thông số từ Worker hoặc từ biến Tracking
                if not tracked_name:
                    rtt = esp_worker.last_rtt if esp_worker else 0.0
                    infer = esp_worker.last_infer_ms if esp_worker else 0.0
                    matched = esp_worker.last_matched if esp_worker else False
                    recognized_name = esp_worker.last_name if esp_worker else "Unknown"
                    similarity = esp_worker.last_similarity if esp_worker else 0.0
                    
                    if matched:
                        tracked_name = recognized_name
                        tracked_sim = similarity
                        tracked_infer = infer
                        # In thông báo ĐÃ NHẬN DIỆN đúng 1 lần (sử dụng \n để chốt dòng)
                        sys.stdout.write(f"\r✅ Đã nhận diện: {tracked_name} ({tracked_sim*100:.1f}%) | {elapsed:.1f}s\n")
                        sys.stdout.flush()
                        
                        db.log_attendance(tracked_name, tracked_sim, tracked_infer)
                else:
                    # Trạng thái đã chốt (Lock)
                    matched = True
                    recognized_name = tracked_name
                    similarity = tracked_sim
                    infer = tracked_infer
                
                # In trạng thái đang xử lý ra Terminal (nếu chưa chốt)
                if not tracked_name:
                    if esp_worker and esp_worker.is_processing:
                        sys.stdout.write(f"\r⏳ ESP32 đang xử lý: {elapsed:.1f}s       ")
                        sys.stdout.flush()
                    else:
                        sys.stdout.write(f"\r👀 Đang lấy mẫu khuôn mặt: {elapsed:.1f}s       ")
                        sys.stdout.flush()
                
                # Màu sắc Bounding Box dựa trên kết quả nhận diện
                box_color = (0, 255, 0) if matched else (0, 165, 255)
                
                # Vẽ khung Bounding Box quanh chủ thể chính
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                
                # Vẽ các điểm mốc (Landmarks) màu vàng sáng (Golden Dots)
                all_lm = face.get('all_landmarks', [])
                if detector.mode == "MEDIAPIPE_MESH":
                    # Vẽ toàn bộ lưới điểm Face Mesh (màu vàng kim rực rỡ)
                    for (lx, ly) in all_lm:
                        cv2.circle(frame, (lx, ly), 1, (0, 255, 255), -1)
                else:
                    # Vẽ 5 điểm mốc chuẩn InsightFace với vòng tròn nổi bật
                    for (lx, ly) in face.get('landmarks_5', []):
                        cv2.circle(frame, (lx, ly), 4, (0, 215, 255), -1)
                        cv2.circle(frame, (lx, ly), 6, (0, 255, 0), 1)
                
                # 2. Căn chỉnh Affine Similarity Transformation chuẩn InsightFace 64x64
                crop_bgr, crop_gray, byte_data = detector.align_and_crop(frame, face)
                
                if crop_bgr is not None:
                    last_crop_bgr = crop_bgr
                    
                    # CHỈ đẩy sang ESP32 nếu CHƯA track được ai
                    if esp_worker is not None and not tracked_name:
                        esp_worker.push_frame(byte_data)
                        
                    # Hiển thị Tên người dùng và Độ tương đồng trực tiếp trên đỉnh khuôn mặt
                    if matched:
                        label_str = f"MATCH: {recognized_name} ({similarity*100:.1f}%) [{infer:.0f}ms]"
                    else:
                        label_str = f"UNKNOWN ({similarity*100:.1f}%) [{infer:.0f}ms]"
                        
                    # Vẽ khung nền chữ
                    (text_w, text_h), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(frame, (x, max(0, y - text_h - 10)), (x + text_w + 10, y), (18, 18, 18), -1)
                    cv2.putText(frame, label_str, (x + 5, max(15, y - 5)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
            else:
                recognition_start_time = None
                last_crop_bgr = None
                # Reset Tracking khi người rời khỏi khung hình
                tracked_name = None
                tracked_sim = 0.0
                tracked_infer = 0.0
                if esp_worker:
                    esp_worker.last_matched = False
                    esp_worker.last_name = "Unknown"
                    esp_worker.last_similarity = 0.0
            
            status_str = "CONNECTED" if esp_connected else "DISCONNECTED"
            
            # Vẽ thanh HUD công nghệ cao
            draw_hud(frame, fps, detector.mode, status_str, rtt, infer, last_crop_bgr, matched, recognized_name, similarity)
            
            cv2.imshow("TinyML Face Recognition - ESP32-S3 Edge AI", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                break
                
    finally:
        if esp_worker:
            esp_worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        esp32_bridge.close()
        print("\n[*] Đã đóng ứng dụng an toàn.")

if __name__ == "__main__":
    main()
