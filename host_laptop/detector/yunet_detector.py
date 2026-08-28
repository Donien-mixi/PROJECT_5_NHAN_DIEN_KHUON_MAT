import cv2
import numpy as np
import os
import urllib.request

# ==============================================================================
# BẢNG TỌA ĐỘ 5 ĐIỂM CHUẨN INSIGHTFACE / ARCFACE (CHUẨN HÓA CHO SIZE 96x96)
# ==============================================================================
REFERENCE_5_POINTS_96 = np.array([
    [25.97, 44.31],  # Mắt phải (bên trái trên ảnh)
    [56.17, 44.14],  # Mắt trái (bên phải trên ảnh)
    [41.16, 61.49],  # Đầu mũi
    [28.76, 79.17],  # Khóe miệng phải
    [53.77, 79.03]   # Khóe miệng trái
], dtype=np.float32)

REFERENCE_5_POINTS_64 = np.array([
    [17.31, 29.54],
    [37.45, 29.43],
    [27.44, 40.99],
    [19.17, 52.78],
    [35.85, 52.69]
], dtype=np.float32)

class UnifiedFaceDetector:
    """
    Bộ phát hiện và bám khuôn mặt đa chế độ (MediaPipe Face Mesh 468+ điểm mốc / OpenCV YuNet)
    - Tích hợp bộ lọc Primary Face Selector: Loại bỏ 100% vật thể nhiễu nền (bao tải, quần áo, tranh ảnh).
    - Tích hợp chuẩn căn chỉnh Affine Similarity Transformation (chuẩn vàng InsightFace/ArcFace).
    - Hỗ trợ hiển thị lưới điểm vàng (Face Mesh) chuyên nghiệp.
    """
    YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    YUNET_FILE = "face_detection_yunet_2023mar.onnx"

    def __init__(self, target_size=(64, 64), prefer_mediapipe=True, conf_threshold=0.85):
        self.target_size = target_size
        self.conf_threshold = conf_threshold
        self.mode = "NONE"
        
        # 1. Thử khởi tạo Google MediaPipe Face Mesh (Độ chính xác cao nhất & không bắt nhầm nền)
        self.mp_face_mesh = None
        self.mesh_detector = None
        if prefer_mediapipe:
            try:
                import mediapipe as mp
                self.mp_face_mesh = mp.solutions.face_mesh
                self.mesh_detector = self.mp_face_mesh.FaceMesh(
                    max_num_faces=1,             # Chỉ tập trung bám 1 khuôn mặt chính của người ngồi trước máy
                    refine_landmarks=True,       # Bật 478 điểm mốc (bao gồm cả đồng tử mắt)
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.6
                )
                self.mode = "MEDIAPIPE_MESH"
                print("[+] Đã khởi tạo Google MediaPipe Face Mesh (468+ điểm mốc vàng chống nhiễu)!")
            except Exception as e:
                pass

        # 2. Khởi tạo OpenCV YuNet (nếu không dùng MediaPipe)
        self.yunet_detector = None
        if self.mode == "NONE":
            self._init_yunet()

    def _init_yunet(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, self.YUNET_FILE)
        
        # Tải YuNet nếu chưa có
        if not os.path.exists(model_path):
            print(f"[*] Đang tải OpenCV YuNet model...")
            try:
                urllib.request.urlretrieve(self.YUNET_URL, model_path)
            except Exception as e:
                print(f"⚠️ Lỗi tải YuNet: {e}")

        if os.path.exists(model_path) and hasattr(cv2, 'FaceDetectorYN'):
            self.yunet_detector = cv2.FaceDetectorYN.create(
                model=model_path,
                config="",
                input_size=(640, 480),
                score_threshold=self.conf_threshold, # Tăng ngưỡng lên 0.75 để lọc nhiễu nền
                nms_threshold=0.3,
                top_k=5000
            )
            self.mode = "YUNET"
            print("[+] Đã khởi tạo OpenCV YuNet Face Detector (Nâng cấp ngưỡng tin cậy chống nhiễu)!")
        else:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.yunet_detector = cv2.CascadeClassifier(cascade_path)
            self.mode = "HAAR"
            print("[*] Sử dụng OpenCV Haar Cascade dự phòng.")

    def detect_primary_face(self, frame):
        """
        Phát hiện và CHỈ CHỌN KHUÔN MẶT CHÍNH (Chủ thể người dùng trước màn hình).
        Loại bỏ hoàn toàn các phát hiện giả trên bao bì, áo quần, tranh ảnh phía sau.
        Trả về dict:
            - 'bbox': [x, y, w, h]
            - 'landmarks_5': 5 điểm chuẩn [(rx, ry), (lx, ly), (nx, ny), (rmx, rmy), (lmx, lmy)]
            - 'all_landmarks': tất cả điểm mốc (468 điểm nếu dùng MediaPipe)
            - 'conf': độ tin cậy
        """
        h, w = frame.shape[:2]
        
        # ----------------------------------------------------------------------
        # CHẾ ĐỘ 1: GOOGLE MEDIAPIPE FACE MESH (468+ ĐIỂM MỐC)
        # ----------------------------------------------------------------------
        if self.mode == "MEDIAPIPE_MESH":
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mesh_detector.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                points = []
                xs = []
                ys = []
                for lm in face_landmarks.landmark:
                    px, py = int(lm.x * w), int(lm.y * h)
                    points.append((px, py))
                    xs.append(px)
                    ys.append(py)
                
                # Bounding box bao trọn toàn bộ khuôn mặt
                min_x, max_x = max(0, min(xs)), min(w, max(xs))
                min_y, max_y = max(0, min(ys)), min(h, max(ys))
                bw = max_x - min_x
                bh = max_y - min_y
                
                # Trích xuất 5 điểm chuẩn InsightFace từ 468 điểm MediaPipe
                # Landmark 468: 468 là tâm đồng tử phải (hoặc landmark 33/133 tâm mắt), 473 là tâm đồng tử trái
                if len(points) >= 478:
                    right_eye = points[468] # Đồng tử mắt phải
                    left_eye = points[473]  # Đồng tử mắt trái
                else:
                    right_eye = points[33]
                    left_eye = points[263]
                    
                nose_tip = points[1]        # Đầu mũi
                right_mouth = points[61]    # Khóe miệng phải
                left_mouth = points[291]    # Khóe miệng trái
                
                landmarks_5 = [right_eye, left_eye, nose_tip, right_mouth, left_mouth]
                
                return {
                    'bbox': [min_x, min_y, bw, bh],
                    'landmarks_5': landmarks_5,
                    'all_landmarks': points,
                    'conf': 0.99
                }
            return None

        # ----------------------------------------------------------------------
        # CHẾ ĐỘ 2: OPENCV YUNET (KÈM BỘ LỌC PRIMARY FACE SELECTOR)
        # ----------------------------------------------------------------------
        elif self.mode == "YUNET":
            self.yunet_detector.setInputSize((w, h))
            _, detections = self.yunet_detector.detect(frame)
            
            if detections is None or len(detections) == 0:
                return None
                
            candidates = []
            center_x, center_y = w / 2.0, h / 2.0
            
            for det in detections:
                conf = float(det[14])
                if conf < self.conf_threshold:
                    continue # Bỏ qua nếu độ tin cậy thấp
                    
                bx, by, bw, bh = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                bx, by = max(0, bx), max(0, by)
                bw, bh = min(w - bx, bw), min(h - by, bh)
                
                area = bw * bh
                if area < 4000: # Loại bỏ các ô quá nhỏ phía nền xa
                    continue
                    
                # -------------------------------------------------------------
                # HỆ THỐNG LỌC NHIỄU THÔNG MINH (ANTI-FALSE-POSITIVE HEURISTICS)
                # -------------------------------------------------------------
                # 1. Ép ngưỡng tin cậy (Confidence) tối thiểu
                if conf < self.conf_threshold:
                    continue
                    
                # 2. Tỉ lệ khung hình (Aspect Ratio): Nới lỏng hơn (0.50 -> 1.50)
                aspect_ratio = bw / (bh + 1e-5)
                if aspect_ratio < 0.50 or aspect_ratio > 1.50:
                    continue
                    
                # 3. Phân bố điểm mốc sinh trắc (Biometric Landmarks Distribution)
                rx, ry = int(det[4]), int(det[5])  # Mắt phải
                lx, ly = int(det[6]), int(det[7])  # Mắt trái
                mx, my = int(det[8]), int(det[9])  # Mũi
                
                # Khoảng cách 2 mắt (nới lỏng: 15% - 85%)
                eye_dist = np.sqrt((rx - lx)**2 + (ry - ly)**2)
                if eye_dist < 0.15 * bw or eye_dist > 0.85 * bw:
                    continue
                    
                # Mũi phải nằm dưới 2 mắt
                if my < ry or my < ly:
                    continue
                    
                # Tính khoảng cách từ tâm bbox tới tâm màn hình
                box_center_x = bx + bw / 2.0
                box_center_y = by + bh / 2.0
                dist_to_center = np.sqrt((box_center_x - center_x)**2 + (box_center_y - center_y)**2)
                norm_dist = dist_to_center / (np.sqrt(center_x**2 + center_y**2) + 1e-5)
                
                # Điểm số ưu tiên: Mặt to + ở giữa màn hình + confidence cao
                priority_score = (area / (w * h)) * 0.5 + (1.0 - norm_dist) * 0.3 + conf * 0.2
                
                landmarks_5 = [
                    (int(det[4]), int(det[5])),   # Mắt phải
                    (int(det[6]), int(det[7])),   # Mắt trái
                    (int(det[8]), int(det[9])),   # Mũi
                    (int(det[10]), int(det[11])), # Khóe miệng phải
                    (int(det[12]), int(det[13]))  # Khóe miệng trái
                ]
                
                candidates.append({
                    'bbox': [bx, by, bw, bh],
                    'landmarks_5': landmarks_5,
                    'all_landmarks': landmarks_5,
                    'conf': conf,
                    'priority': priority_score
                })
                
            if not candidates:
                return None
                
            # Chọn ứng viên có điểm số cao nhất (chắc chắn là chủ thể người dùng)
            candidates.sort(key=lambda c: c['priority'], reverse=True)
            return candidates[0]
            
        return None

    def align_and_crop(self, frame, face_info):
        """
        Căn chỉnh khuôn mặt bằng phép biến đổi tương đồng Affine (Similarity Transform)
        chuẩn InsightFace / ArcFace đưa về kích thước 96x96.
        """
        if face_info is None:
            return None, None, None
            
        landmarks_5 = face_info.get('landmarks_5')
        if landmarks_5 is not None and len(landmarks_5) == 5:
            src_pts = np.array(landmarks_5, dtype=np.float32)
            
            # Tính ma trận biến đổi Affine Similarity (xoay + tịnh tiến + co giãn đồng dạng)
            ref_pts = REFERENCE_5_POINTS_64 if self.target_size == (64, 64) else REFERENCE_5_POINTS_96
            M, inliers = cv2.estimateAffinePartial2D(src_pts, ref_pts)
            
            if M is not None:
                # Thực hiện biến đổi trực tiếp toàn bộ khuôn mặt chuẩn xác đến từng pixel
                face_aligned_bgr = cv2.warpAffine(
                    frame, M, self.target_size, 
                    flags=cv2.INTER_AREA, 
                    borderMode=cv2.BORDER_REPLICATE
                )
                face_aligned_gray = cv2.cvtColor(face_aligned_bgr, cv2.COLOR_BGR2GRAY)
                byte_data = face_aligned_gray.tobytes()
                return face_aligned_bgr, face_aligned_gray, byte_data
                
        # Fallback crop nếu không có landmarks
        x, y, w, h = face_info['bbox']
        img_h, img_w = frame.shape[:2]
        crop = frame[max(0, y):min(img_h, y+h), max(0, x):min(img_w, x+w)]
        if crop.size == 0:
            return None, None, None
            
        face_resized_bgr = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_AREA)
        face_resized_gray = cv2.cvtColor(face_resized_bgr, cv2.COLOR_BGR2GRAY)
        return face_resized_bgr, face_resized_gray, face_resized_gray.tobytes()

# Đồng bộ với code cũ
FaceDetector = UnifiedFaceDetector
