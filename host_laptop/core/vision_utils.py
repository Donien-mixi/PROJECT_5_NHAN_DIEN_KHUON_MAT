"""
Bilinear thủ công đồng bộ 100% với firmware_esp32/ai_face_detector.cpp:preprocess_face
Chuẩn: mo_ta_project.md:82-91, README.md:57,242,246 + KE_HOACH_PULL_VA_CHINH_SUA.md:247-299
Cấm dùng resize OpenCV cho crop mặt — phải vòng lặp pixel để khử Domain Shift.

Module này cũng chứa hợp đồng pixel ESP32 (RGB565 round-trip) và hàm tạo frame
chuẩn từ webcam, dùng chung cho enroll_tool.py / main.py / ip_camera_streamer.py.
"""
import numpy as np
import cv2

# Hợp đồng chung (mo_ta_project.md:42-44, README.md:246)
RAW_SIZE = 128                 # RAW 128x128 — webcam được resize đúng 1 lần tại bước này
JPEG_QUALITY_TO_ESP32 = 80     # ip_camera_streamer.py encode JPEG quality 80


def rgb565_roundtrip(img_bgr):
    """
    Mô phỏng vòng đổi màu RGB565 trên ESP32 — biểu diễn pixel y hệt g_frame_buffer.
    - TJpgDec.setSwapBytes(false) (little-endian) khi giải mã RGB888 -> RGB565:
        c = (r>>3)<<11 | (g>>2)<<5 | (b>>3)
    - ai_face_detector.cpp:227-232 giải mã ngược:
        r = ((c>>11)&0x1F)<<3 ; g = ((c>>5)&0x3F)<<2 ; b = (c&0x1F)<<3
    Kết quả là một số nguyên phép chỉ giữ 5/6/5 bit cao, tạo lệch bit như ESP32 thật.
    """
    if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr must be a non-empty BGR image with three channels")

    h, w = img_bgr.shape[:2]
    img = img_bgr.astype(np.int32)
    r5 = (img[:, :, 2] >> 3) & 0x1F           # R: 5 bit
    g6 = (img[:, :, 1] >> 2) & 0x3F           # G: 6 bit
    b5 = (img[:, :, 0] >> 3) & 0x1F           # B: 5 bit
    c = (r5 << 11) | (g6 << 5) | b5

    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:, :, 2] = ((c >> 11) & 0x1F) << 3    # R
    out[:, :, 1] = ((c >> 5) & 0x3F) << 2     # G
    out[:, :, 0] = (c & 0x1F) << 3            # B
    return out


def center_crop_to_raw(frame_bgr, raw_size=RAW_SIZE):
    """Center-crop vuông rồi resize 1 lần duy nhất về raw_size — giống ip_camera_streamer.py."""
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        return None
    height, width = frame_bgr.shape[:2]
    side = min(height, width)
    if side <= 0:
        return None
    top = (height - side) // 2
    left = (width - side) // 2
    square = frame_bgr[top:top + side, left:left + side]
    return cv2.resize(square, (raw_size, raw_size), interpolation=cv2.INTER_AREA)


def jpeg_roundtrip(img_bgr, quality=JPEG_QUALITY_TO_ESP32):
    """Encode/decode JPEG quality 80 — mô phỏng đúng ảnh ESP32 nhận qua TCP."""
    if img_bgr is None:
        return None
    ok, encoded = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def prepare_esp32_frame(frame_bgr):
    """
    Frame BGR 128x128 giống hệt pixel ESP32 sau decode TCP:
    center crop vuông -> resize RAW 128 (1 lần) -> JPEG encode/decode q80 -> RGB565 round-trip
    Không mirror (Không flip) — ESP32 cũng nhận ảnh gốc không lật.
    Chuẩn: mo_ta_project.md:29-31,42-48, README.md:246
    """
    raw = center_crop_to_raw(frame_bgr)
    if raw is None:
        return None
    decoded = jpeg_roundtrip(raw)
    if decoded is None:
        return None
    try:
        return rgb565_roundtrip(decoded)
    except ValueError:
        return None


def equalize_gray_256(img_gray):
    """
    Histogram Equalization (LUT số nguyên) trên ảnh gray uint8 — khử nhạy ánh sáng
    trước khi trích xuất embedding (tham khảo: Shan et al., AMFG 2003; Xie & Lam, 2006).

    Công thức chuẩn (đồng bộ bit-exact với equalize_gray_u8 trong ai_face_detector.cpp):
        cdf[v]   = tổng hist[0..v]
        cdf_min  = cdf tại giá trị nonzero đầu tiên
        lut[v]   = (cdf[v] - cdf_min) * 255 // (N - cdf_min)   [phép chia nguyên floor]
        ảnh đơn điệu (N == cdf_min) → giữ nguyên (identity)
    """
    arr = np.asarray(img_gray, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError("img_gray must be a 2-D grayscale image")
    hist = np.bincount(arr.ravel(), minlength=256)
    cdf = np.cumsum(hist)
    nz = np.nonzero(hist)[0]
    if nz.size == 0:
        return np.asarray(img_gray, dtype=np.uint8).copy()
    cdf_min = int(cdf[nz[0]])
    den = int(arr.size) - cdf_min
    if den <= 0:
        # Ảnh đơn điệu (mọi pixel cùng giá trị) → identity
        return np.asarray(img_gray, dtype=np.uint8).copy()
    lut = ((cdf - cdf_min) * 255) // den          # số nguyên floor — trùng C++
    lut = np.clip(lut, 0, 255).astype(np.int64)
    return lut[arr].astype(np.uint8)


def _crop_and_resize_bilinear_gray(img_bgr, cx, cy, box_size, target_size=64):
    """
    Cắt vùng vuông (cx,cy,box_size) từ ảnh 128x128 BGR và thu về target_size x target_size
    bằng Bilinear Interpolation thủ công y hệt C++.
    Trả về (face_bgr, face_gray) — face_gray đã là 64x64 uint8.
    """
    if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr must be a non-empty BGR image with three channels")
    if not np.isfinite([cx, cy, box_size]).all() or box_size <= 0 or target_size <= 0:
        return None, None

    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return None, None
    half_box = box_size / 2.0
    x1_box = cx - half_box
    y1_box = cy - half_box
    scale = box_size / target_size

    face_bgr = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    face_gray = np.zeros((target_size, target_size), dtype=np.uint8)

    for ty in range(target_size):
        for tx in range(target_size):
            # Tọa độ nguồn với half-pixel correction (đồng bộ C++)
            sx = x1_box + (tx + 0.5) * scale - 0.5
            sy = y1_box + (ty + 0.5) * scale - 0.5

            # Clamp tọa độ thực trước khi tính 4 điểm lân cận.  Tính x1/y1
            # từ x0/y0 chưa clamp sẽ tạo chỉ số âm khi detector trả box lệch.
            sx = max(0.0, min(sx, w - 1.0))
            sy = max(0.0, min(sy, h - 1.0))
            x0 = int(np.floor(sx))
            y0 = int(np.floor(sy))
            x1 = min(x0 + 1, w - 1)
            y1 = min(y0 + 1, h - 1)

            wx = sx - x0
            wy = sy - y0

            # Bilinear 4 điểm cho từng kênh BGR
            for c in range(3):
                top = (1 - wx) * float(img_bgr[y0, x0, c]) + wx * float(img_bgr[y0, x1, c])
                bot = (1 - wx) * float(img_bgr[y1, x0, c]) + wx * float(img_bgr[y1, x1, c])
                face_bgr[ty, tx, c] = int(np.clip((1 - wy) * top + wy * bot, 0, 255))

            # Grayscale 0.299R+0.587G+0.114B
            def _gray_at(y_, x_):
                b, g, r = float(img_bgr[y_, x_, 0]), float(img_bgr[y_, x_, 1]), float(img_bgr[y_, x_, 2])
                return 0.114 * b + 0.587 * g + 0.299 * r

            g00 = _gray_at(y0, x0)
            g01 = _gray_at(y0, x1)
            g10 = _gray_at(y1, x0)
            g11 = _gray_at(y1, x1)
            top_g = (1 - wx) * g00 + wx * g01
            bot_g = (1 - wx) * g10 + wx * g11
            face_gray[ty, tx] = int(np.clip((1 - wy) * top_g + wy * bot_g, 0, 255))

    return face_bgr, face_gray


def center_square_crop(img_bgr, cx, cy, box_size, target_size=64):
    """Alias để tương thích code cũ — gọi Bilinear đồng bộ"""
    return _crop_and_resize_bilinear_gray(img_bgr, cx, cy, box_size, target_size)
