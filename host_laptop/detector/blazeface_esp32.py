"""
Unified BlazeFace Emulator 128x128 — đồng bộ 100% với firmware_esp32/ai_face_detector.cpp
Chuẩn: README.md:56,57,242 + mo_ta_project.md:33,82-91 + CAC_FILE_CHINH_NHIEU.md:2.4
- Input: JPEG 128x128 RGB (từ ip_camera_streamer, không resize trung gian)
- Detector: BlazeFace 128, conf 0.80; tự đọc dtype/quantization của model
- Crop: Bilinear thủ công 128→64 qua host_laptop/core/vision_utils.py (không dùng resize OpenCV)
"""
import cv2
import numpy as np
import os
import sys

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# Import Bilinear đồng bộ
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # host_laptop
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
try:
    from host_laptop.core.vision_utils import _crop_and_resize_bilinear_gray, rgb565_roundtrip, equalize_gray_256
except ImportError:
    from core.vision_utils import _crop_and_resize_bilinear_gray, rgb565_roundtrip, equalize_gray_256


class UnifiedFaceDetector:
    def __init__(self, target_size=(64, 64), conf_threshold=0.80):
        self.target_size = target_size
        self.conf_threshold = conf_threshold
        self.input_size = 128

        # Tìm model BlazeFace 128 (ưu tiên bản full INT8 để đồng bộ firmware).
        candidates = [
            os.path.join(parent_dir, "detector", "face_detection_short_range_int8.tflite"),
            os.path.join(parent_dir, "face_detection_short_range.tflite"),
            os.path.join(parent_dir, "detector", "face_detection_short_range.tflite"),
        ]
        model_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy model BlazeFace 128 tại: {model_path}")
        if "int8" in os.path.basename(model_path):
            self.mode = "BLAZEFACE_INT8"
        else:
            self.mode = "BLAZEFACE_F32"

        print("[*] Đang khởi tạo BlazeFace Emulator 128 (đồng bộ ESP32)...")
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()
        # BlazeFace short-range có 2 output: scores + boxes
        self.anchors = self._generate_anchors()
        self.ema_cx = -1.0
        self.ema_cy = -1.0
        self.ema_w = -1.0
        self.ema_h = -1.0
        self.alpha = 0.35

    def _generate_anchors(self):
        anchors = []
        # 16x16
        for y in range(16):
            for x in range(16):
                cx = (x + 0.5) / 16.0
                cy = (y + 0.5) / 16.0
                anchors.append((cx, cy))
                anchors.append((cx, cy))
        # 8x8 x6
        for y in range(8):
            for x in range(8):
                cx = (x + 0.5) / 8.0
                cy = (y + 0.5) / 8.0
                for _ in range(6):
                    anchors.append((cx, cy))
        return anchors  # 896 anchors

    @staticmethod
    def _sigmoid(x):
        """Sigmoid ổn định, tránh RuntimeWarning khi tensor có logit lớn."""
        x = np.clip(x, -80.0, 80.0)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _dequantize(tensor, details):
        """Đưa output TFLite về float, bất kể output nằm ở index nào."""
        if np.issubdtype(details['dtype'], np.integer):
            scale, zero_point = details.get('quantization', (0.0, 0))
            if scale:
                return (tensor.astype(np.float32) - zero_point) * scale
        return tensor.astype(np.float32, copy=False)

    def _decode_boxes(self, raw_scores, raw_boxes):
        """Decode anchor có score cao nhất theo BlazeFace short-range.

        Regressor của model chứa tọa độ theo đơn vị pixel (x, y, w, h),
        không phải logarithm. Công thức này tương ứng với C++: ``raw / 128
        + anchor``. Cách cũ dùng ``exp`` khiến một logit bình thường trở thành
        box khổng lồ, dẫn tới chỉ số crop âm.
        """
        scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)
        boxes = np.asarray(raw_boxes, dtype=np.float32)
        if scores.size != len(self.anchors) or boxes.shape[0] != len(self.anchors) or boxes.shape[1] < 4:
            return None

        score_values = self._sigmoid(scores)
        best_idx = int(np.argmax(score_values))
        score = float(score_values[best_idx])
        if score < self.conf_threshold:
            return None

        dx, dy, dw, dh = boxes[best_idx, :4]
        cx_anchor, cy_anchor = self.anchors[best_idx]
        cx = float(dx + cx_anchor * self.input_size)
        cy = float(dy + cy_anchor * self.input_size)
        width = float(dw)
        height = float(dh)
        if not np.isfinite([cx, cy, width, height]).all() or width <= 0.0 or height <= 0.0:
            return None

        # Một box hoàn toàn ở ngoài frame không thể dùng để enroll/recognize.
        if cx + width / 2.0 <= 0.0 or cy + height / 2.0 <= 0.0:
            return None
        if cx - width / 2.0 >= self.input_size or cy - height / 2.0 >= self.input_size:
            return None
        return cx, cy, width, height, score

    def detect_primary_face(self, frame):
        """
        Nhận frame 128x128 BGR (từ ip_camera_streamer, đã là 128)
        Trả về face dict hoặc None
        """
        h, w = frame.shape[:2]
        # Đồng nhất ESP32: frame phải đã là 128x128 từ ip_camera_streamer (README.md:240)
        # Không tự resize ở đây để tránh lệch Domain Shift — nếu sai kích thước, báo lỗi và bỏ qua
        if w != self.input_size or h != self.input_size:
            print(f"[WARN] Frame {w}x{h} != {self.input_size}x{self.input_size} — bỏ qua để giữ đồng nhất 128 (chỉ streamer mới được resize)")
            return None
        # Mô phỏng RGB565 của ESP32 (mo_ta_project.md:42-48): g_frame_buffer là RGB565,
        # detector đọc pixel qua RGB565 decode. Bước này có thể chạy nhiều lần (idempotent).
        frame_128 = rgb565_roundtrip(frame)

        # Preprocess chuẩn BlazeFace: RGB và normalize [-1, 1]. Nếu model
        # quantized thì lượng tử hóa chính tensor đã normalize, không phải RGB thô.
        rgb = cv2.cvtColor(frame_128, cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) - 127.5) / 128.0
        input_dtype = self.input_details['dtype']
        if np.issubdtype(input_dtype, np.integer):
            input_scale, input_zero = self.input_details.get('quantization', (0.0, 0))
            if not input_scale:
                return None
            limits = np.iinfo(input_dtype)
            inp = np.clip(
                np.round(normalized / input_scale) + input_zero,
                limits.min,
                limits.max,
            ).astype(input_dtype)
        else:
            inp = normalized.astype(input_dtype)
        inp = np.expand_dims(inp, axis=0)

        self.interpreter.set_tensor(self.input_details['index'], inp)
        self.interpreter.invoke()
        # Lấy outputs
        out0_details, out1_details = self.output_details[:2]
        out0 = self.interpreter.get_tensor(out0_details['index'])
        out1 = self.interpreter.get_tensor(out1_details['index'])
        # Regressor/classificator có thể ở thứ tự khác nhau tùy file TFLite.
        if out0.shape[-1] == 1:
            scores = self._dequantize(out0[0], out0_details)
            boxes = self._dequantize(out1[0], out1_details)
        else:
            scores = self._dequantize(out1[0], out1_details)
            boxes = self._dequantize(out0[0], out0_details)

        candidate = self._decode_boxes(scores, boxes)
        if candidate is None:
            # Reset EMA khi mất mặt
            self.ema_cx = self.ema_cy = self.ema_w = self.ema_h = -1.0
            return None

        cx, cy, w_box, h_box, score = candidate
        # EMA smoothing y hệt C++
        if self.ema_cx < 0:
            self.ema_cx, self.ema_cy, self.ema_w, self.ema_h = cx, cy, w_box, h_box
        else:
            self.ema_cx = self.alpha * cx + (1 - self.alpha) * self.ema_cx
            self.ema_cy = self.alpha * cy + (1 - self.alpha) * self.ema_cy
            self.ema_w = self.alpha * w_box + (1 - self.alpha) * self.ema_w
            self.ema_h = self.alpha * h_box + (1 - self.alpha) * self.ema_h

        cx, cy = self.ema_cx, self.ema_cy
        size = max(self.ema_w, self.ema_h)
        left, top = max(0.0, cx - size / 2.0), max(0.0, cy - size / 2.0)
        right = min(float(self.input_size), cx + size / 2.0)
        bottom = min(float(self.input_size), cy + size / 2.0)
        if right <= left or bottom <= top:
            self.ema_cx = self.ema_cy = self.ema_w = self.ema_h = -1.0
            return None
        x1, y1 = int(left), int(top)
        x2, y2 = int(right), int(bottom)
        if x2 <= x1 or y2 <= y1:
            self.ema_cx = self.ema_cy = self.ema_w = self.ema_h = -1.0
            return None
        w_clamp, h_clamp = x2 - x1, y2 - y1

        # Trả về face dict tương thích main.py / hud_renderer
        return {
            'bbox': (x1, y1, w_clamp, h_clamp),
            'cx': cx,
            'cy': cy,
            'size': size,
            'score': float(score),
            'landmarks_5': [],  # BlazeFace không có 5 landmarks sẵn, để rỗng
        }

    def align_and_crop(self, frame, face, equalize=True):
        """
        Dùng Bilinear thủ công đồng bộ C++ để crop 64x64.
        equalize=True (mặc định, inference live): áp Histogram Equalization khử nhạy
        ánh sáng — ĐỒNG BỘ với preprocess_face() trên ESP32 và generate_embeddings.
        enroll_tool truyền equalize=False để lưu ảnh RAW (HE chạy lúc suy luận/DB-gen).
        """
        if face is None:
            return None, None, None
        # Đảm bảo frame là 128 — không resize để giữ đồng nhất (chỉ streamer resize)
        h, w = frame.shape[:2]
        if w != self.input_size or h != self.input_size:
            return None, None, None
        # Ground truth pixel y hệt g_frame_buffer (RGB565) như detect_primary_face
        frame_128 = rgb565_roundtrip(frame)
        cx, cy, size = face['cx'], face['cy'], face['size']
        if not np.isfinite([cx, cy, size]).all() or size <= 0.0:
            return None, None, None
        # Mở rộng box 10% để lấy đủ mặt (y hệt C++ scale)
        box_size = size * 1.1
        face_bgr, face_gray = _crop_and_resize_bilinear_gray(frame_128, cx, cy, box_size, 64)
        if face_gray is not None and equalize:
            face_gray = equalize_gray_256(face_gray)
        return face_bgr, face_gray, None
