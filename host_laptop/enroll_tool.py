"""Enroll 70/30 đồng bộ streamer/ESP32: RAW 128 → RGB565 → Bilinear gray 64.

Nguồn webcam được center-crop vuông và resize một lần về 128x128, encode/decode
JPEG quality 80 rồi mô phỏng RGB565 như g_frame_buffer — giống hệt input mà ESP32
nhận qua TCP. Chỉ ảnh grayscale 64x64 lossless được lưu.
Chuẩn: mo_ta_project.md:31,42-48, README.md:246
"""
import cv2
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from host_laptop.detector.blazeface_esp32 import UnifiedFaceDetector
from host_laptop.core.vision_utils import prepare_esp32_frame

# Cổng chất lượng ảnh (bài học: ảnh chụp tối mean=42 gây nhầm lẫn thao↔toan).
# Ảnh tối/quá sáng → embedding kém tin cậy → từ chối ngay lúc chụp.
BRIGHT_MIN = 65    # dưới ngưỡng này → TỪ CHỐI lưu (quá tối)
BRIGHT_WARN = 85   # cảnh báo nhưng vẫn cho phép lưu
BRIGHT_MAX = 200   # trên ngưỡng này → TỪ CHỐI lưu (cháy sáng)


def prepare_stream_equivalent_frame(frame_bgr):
    """Trả về frame BGR 128x128 theo đúng bước nguồn của streamer + RGB565 ESP32.

    Không mirror frame inference: streamer/ESP32 cũng dùng chiều ảnh gốc.
    ``cv2.resize`` chỉ xuất hiện một lần tại bước tạo RAW 128x128.
    """
    out = prepare_esp32_frame(frame_bgr)
    if out is None:
        return None
    return out


def _existing_image_paths(save_dir):
    return sorted(
        os.path.join(save_dir, name)
        for name in os.listdir(save_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def _has_legacy_non_gray_images(paths):
    """Ảnh cũ BGR/JPEG không được trộn với gray PNG chuẩn mới."""
    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 2 or image.shape != (64, 64):
            return True
    return False


def enroll_user():
    print("==================================================================")
    print("📸 ENROLL TOOL — 70/30 POSE, BILINEAR 128→64 ĐỒNG BỘ")
    print("==================================================================")
    user_name = input("[?] Nhập Tên/ID (không dấu, viết liền): ").strip().replace(" ", "_")
    if not user_name:
        print("❌ Tên không hợp lệ.")
        return
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, "data", "registered_faces", user_name)
    os.makedirs(save_dir, exist_ok=True)
    existing_paths = _existing_image_paths(save_dir)
    if _has_legacy_non_gray_images(existing_paths):
        print("❌ Phát hiện ảnh cũ không phải grayscale 64x64 trong thư mục này.")
        print("   Không trộn ảnh cũ với chuẩn mới. Hãy sao lưu/xóa ảnh cũ rồi enroll lại.")
        return
    print(f"[*] Lưu tại: {save_dir}")
    print("[*] Tỷ lệ vàng 70/30: 14 ảnh thẳng + 6 ảnh nghiêng 10-15°")

    detector = UnifiedFaceDetector(target_size=(64, 64), conf_threshold=0.80)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n[+] Nhấn 'C' để CHỤP, 'Q' để THOÁT. Quay nhẹ trái/phải/lên/xuống để đủ pose.")
    count = len(existing_paths)
    print(f"[*] Đã có {count} ảnh, sẽ tiếp tục từ {count:03d}")

    MAX_IMAGES = 20  # keep 80% = 16 templates = MAX_TEMPLATES trong firmware

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_128 = prepare_stream_equivalent_frame(frame)
        if frame_128 is None:
            continue
        display = frame_128.copy()
        face = detector.detect_primary_face(frame_128)
        best_bgr = None
        best_gray = None
        if face is not None:
            x, y, w, h = face['bbox']
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for (lx, ly) in face.get('landmarks_5', []):
                cv2.circle(display, (lx, ly), 3, (0, 215, 255), -1)
            bgr, gray, _ = detector.align_and_crop(frame_128, face, equalize=False)
            if bgr is not None and gray is not None:
                best_bgr = bgr
                best_gray = gray
                display[10:74, 10:74] = bgr
                cv2.rectangle(display, (8, 8), (76, 76), (0, 255, 255), 2)
                cv2.putText(display, "64x64 Bilinear", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        h, w = display.shape[:2]
        overlay = display.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 72), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)
        # Hiển thị độ sáng khuôn mặt để chỉnh ánh sáng TRƯỚC khi chụp
        bright_now = float(best_gray.mean()) if best_gray is not None else 0.0
        if bright_now < BRIGHT_MIN:
            b_color = (0, 0, 255)     # đỏ: quá tối
        elif bright_now > BRIGHT_MAX:
            b_color = (0, 0, 255)     # đỏ: cháy sáng
        elif bright_now < BRIGHT_WARN:
            b_color = (0, 165, 255)   # cam: hơi tối
        else:
            b_color = (0, 255, 0)     # xanh: tốt
        cv2.putText(display, f"User: {user_name}  {count}/20", (15, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        cv2.putText(display, f"Do sang: {bright_now:.0f} (dat {BRIGHT_MIN}-{BRIGHT_MAX})", (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, b_color, 1)
        cv2.putText(display, f"Press 'C' snap  'Q' quit  70/30", (15, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1)
        if count >= MAX_IMAGES:
            cv2.putText(display, "DU 20 ANH - NHAN 'Q' DE THOAT", (15, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        cv2.imshow("Enroll 128->64 Bilinear", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') or key == ord('C'):
            if count >= MAX_IMAGES:
                # Chặn cứng: >20 ảnh → keep_count > MAX_TEMPLATES làm hỏng face_database.h
                print("⚠️ Đã đủ 20 ảnh/người (16 templates). Nhấn 'Q' thoát; muốn chụp lại thì xóa bớt ảnh cũ.")
                continue
            if best_gray is None:
                print("⚠️ Không thấy mặt rõ (conf<0.80)")
                continue
            # CỔNG CHẤT LƯỢNG: từ chối ảnh tối/cháy sáng — embedding kém gây nhầm người
            bright = float(best_gray.mean())
            if bright < BRIGHT_MIN:
                print(f"⚠️ TỪ CHỐI: ảnh QUÁ TỐI (độ sáng {bright:.0f} < {BRIGHT_MIN}) — thêm đèn/chỉnh ánh sáng rồi chụp lại")
                continue
            if bright > BRIGHT_MAX:
                print(f"⚠️ TỪ CHỐI: ảnh CHÁY SÁNG (độ sáng {bright:.0f} > {BRIGHT_MAX}) — giảm ánh sáng rồi chụp lại")
                continue
            if bright < BRIGHT_WARN:
                print(f"⚠️ Cảnh báo: hơi tối ({bright:.0f}) — vẫn lưu, nên tăng ánh sáng cho các ảnh sau")
            # PNG grayscale không nén mất dữ liệu; generate_embeddings.py hỗ trợ .png.
            path = os.path.join(save_dir, f"{user_name}_{count:03d}.png")
            cv2.imwrite(path, best_gray)
            print(f"[+] Saved {path} (độ sáng {bright:.0f})")
            count += 1
            flash = np.ones_like(display) * 255
            cv2.imshow("Enroll 128->64 Bilinear", flash)
            cv2.waitKey(80)
        elif key == ord('q') or key == ord('Q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[*] Thu thập {count} ảnh cho {user_name}. Chạy: python training_tinyml/update_face_database.py")

if __name__ == "__main__":
    enroll_user()
