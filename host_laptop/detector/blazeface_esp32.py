import cv2
import numpy as np
import os
import sys
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# Thêm đường dẫn để import từ host_laptop
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from core.vision_utils import center_square_crop

class UnifiedFaceDetector:
    """
    Bộ phát hiện khuôn mặt BlazeFace đồng bộ 100% với mã C++ trên ESP32.
    Đảm bảo Bounding Box và các thuật toán nội suy hoàn toàn giống ESP32.
    """
    def __init__(self, target_size=(64, 64), prefer_mediapipe=False, conf_threshold=0.65):
        self.target_size = target_size
        self.conf_threshold = conf_threshold
        
        model_path = os.path.join(parent_dir, "face_detection_short_range.tflite")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy mô hình BlazeFace tại: {model_path}")
            
        print("[*] Đang khởi tạo BlazeFace (Đồng bộ ESP32-S3 C++)...")
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()
        
        self.anchors = self._generate_anchors()
        
        # Biến cho bộ lọc EMA (Exponential Moving Average) y hệt C++
        self.ema_cx = -1.0
        self.ema_cy = -1.0
        self.ema_w = -1.0
        self.ema_h = -1.0
        self.alpha = 0.35

    def _generate_anchors(self):
        anchors = []
        # 16x16 feature map
        for y in range(16):
            for x in range(16):
                cx = (x + 0.5) / 16.0
                cy = (y + 0.5) / 16.0
                anchors.append((cx, cy))
                anchors.append((cx, cy))
        # 8x8 feature map
        for y in range(8):
            for x in range(8):
                cx = (x + 0.5) / 8.0
                cy = (y + 0.5) / 8.0
                for i in range(6):
                    anchors.append((cx, cy))
        return anchors

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def detect_primary_face(self, frame):
        """
        Mô phỏng hàm detect_face() trong ai_face_detector.cpp của ESP32.
        """
        h, w = frame.shape[:2]
        # Bỏ qua nếu frame không phải 800x600 (hoặc thay đổi cho linh hoạt)
        # ESP32 crop từ frame gốc 240x240. Trên laptop ta resize ảnh về 240x240 trước 
        # để mô phỏng hoàn toàn 100% tỷ lệ khung hình và độ phân giải của ESP32.
        
        # Mô phỏng quá trình truyền từ Camera ESP32 (OV2640 thường capture ở 240x240 hoặc cắt vuông)
        # Để đảm bảo Center Square Crop y hệt ESP32, frame phải được cắt vuông trước
        size = min(h, w)
        y_ofs = (h - size) // 2
        x_ofs = (w - size) // 2
        square_frame = frame[y_ofs:y_ofs+size, x_ofs:x_ofs+size]
        
        # Resize về đúng độ phân giải mà thuật toán Bilinear Interpolation trên ESP32 làm việc
        esp32_frame_width = 240
        esp32_frame_height = 240
        frame_240 = cv2.resize(square_frame, (esp32_frame_width, esp32_frame_height))
        
        # Chuẩn bị input 128x128 cho BlazeFace
        input_128 = cv2.resize(frame_240, (128, 128))
        input_data = (input_128.astype(np.float32) - 127.5) / 128.0
        input_data = np.expand_dims(input_data, axis=0)
        
        self.interpreter.set_tensor(self.input_details['index'], input_data)
        self.interpreter.invoke()
        
        # BlazeFace trả về 2 tensors: Regressor và Classifier
        # Tự động hoán đổi nếu model xuất cls ở index 0
        out1 = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        out2 = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        
        if out1.shape[-1] == 1:
            cls_tensor = out1
            reg_tensor = out2
        else:
            cls_tensor = out2
            reg_tensor = out1
            
        max_score = 0.0
        best_anchor = -1
        
        for i in range(896):
            score = self._sigmoid(cls_tensor[i][0])
            if score > max_score:
                max_score = score
                best_anchor = i
                
        if max_score >= self.conf_threshold and best_anchor >= 0:
            best_dx = reg_tensor[best_anchor][0]
            best_dy = reg_tensor[best_anchor][1]
            best_w = reg_tensor[best_anchor][2]
            best_h = reg_tensor[best_anchor][3]
            
            raw_cx = best_dx / 128.0 + self.anchors[best_anchor][0]
            raw_cy = best_dy / 128.0 + self.anchors[best_anchor][1]
            raw_w = best_w / 128.0
            raw_h = best_h / 128.0
            
            # --- BỘ LỌC EMA ---
            if self.ema_cx < 0.0:
                self.ema_cx, self.ema_cy, self.ema_w, self.ema_h = raw_cx, raw_cy, raw_w, raw_h
            else:
                if abs(raw_cx - self.ema_cx) > 0.2 or abs(raw_cy - self.ema_cy) > 0.2:
                    self.ema_cx, self.ema_cy, self.ema_w, self.ema_h = raw_cx, raw_cy, raw_w, raw_h
                else:
                    self.ema_cx = self.alpha * raw_cx + (1.0 - self.alpha) * self.ema_cx
                    self.ema_cy = self.alpha * raw_cy + (1.0 - self.alpha) * self.ema_cy
                    self.ema_w = self.alpha * raw_w + (1.0 - self.alpha) * self.ema_w
                    self.ema_h = self.alpha * raw_h + (1.0 - self.alpha) * self.ema_h
                    
            cx, cy, bw, bh = self.ema_cx, self.ema_cy, self.ema_w, self.ema_h
            
            # Map tọa độ ngược về khung hình gốc (frame đầu vào)
            box_w = bw * size
            box_h = bh * size
            x_center = cx * size + x_ofs
            y_center = cy * size + y_ofs
            
            bx = max(0, int(x_center - box_w / 2))
            by = max(0, int(y_center - box_h / 2))
            bw_px = min(int(box_w), w - bx)
            bh_px = min(int(box_h), h - by)
            
            return {
                'bbox': [bx, by, bw_px, bh_px],
                'conf': max_score
            }
        else:
            # Reset EMA nếu mất dấu
            self.ema_cx = -1.0
            return None

    def align_and_crop(self, frame, face_info):
        """
        Cắt khuôn mặt vuông (Center Square Crop) y hệt ESP32.
        """
        if face_info is None:
            return None, None, None
            
        x, y, w_box, h_box = face_info['bbox']
        
        face_bgr, face_gray = center_square_crop(frame, x, y, w_box, h_box, self.target_size)
        
        if face_bgr is None:
            return None, None, None
            
        return face_bgr, face_gray, face_gray.tobytes()

FaceDetector = UnifiedFaceDetector
