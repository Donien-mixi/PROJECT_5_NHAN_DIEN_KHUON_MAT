"""
Dumb IP Camera — Gửi JPEG 128x128 RGB qua TCP 12345 tới ESP32 (HEADLESS, không cửa sổ)
Chuẩn: README.md:3,161,178,247 + mo_ta_project.md:30
- Chỉ resize 1 lần duy nhất ở streamer: center crop → 128x128 (không detect/crop mặt ở Laptop)
- KHÔNG mở cửa sổ OpenCV: Laptop chỉ là camera; kết quả điểm danh báo bằng
  Serial Monitor + 2 LED + Buzzer trên ESP32 (README.md:3,247)
- Dừng bằng Ctrl+C
"""
import cv2
import socket
import argparse
import time
import struct
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from host_laptop.core.vision_utils import center_crop_to_raw

def stream_camera(ip, port, resolution=(128, 128), quality=80):
    print(f"🚀 Streaming JPEG {resolution[0]}x{resolution[1]} quality={quality} qua TCP {port} (headless)")
    print("Kết quả điểm danh xem trên Serial Monitor ESP32 (115200) + 2 LED + Buzzer. Nhấn Ctrl+C để thoát.")

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không mở được webcam")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Khử độ trễ đệm 4-5 frame nội bộ của OpenCV DirectShow

    sock = None
    frames = 0
    try:
        while True:
            # Kết nối / tự kết nối lại khi ESP32 restart hoặc mất kết nối
            if sock is None:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Tắt Nagle algorithm, gửi gói ngay
                print(f"Đang kết nối tới ESP32 TCP {ip}:{port}...")
                try:
                    sock.connect((ip, port))
                    print("✅ TCP connected!")
                except Exception as e:
                    print(f"❌ Chưa kết nối được ({e}) — thử lại sau 3s...")
                    sock.close()
                    sock = None
                    time.sleep(3)
                    continue

            ret, frame = cap.read()
            if not ret:
                break
            # Bỏ ~15 frame đầu (~0.5-1s) để webcam auto-exposure ổn định,
            # tránh gửi frame quá sáng/quá tối làm điểm số thấp ở lượt nhận diện đầu
            frames += 1
            if frames <= 15:
                continue
            # Center crop vuông rồi resize 128x128 — duy nhất 1 lần (dùng chung vision_utils)
            resized = center_crop_to_raw(frame, resolution[0])
            if resized is None:
                continue

            ok, buf = cv2.imencode('.jpg', resized, encode_param)
            if ok:
                data = buf.tobytes()
                try:
                    sock.sendall(struct.pack('<I', len(data)))
                    sock.sendall(data)
                    if frames % 15 == 0:  # in 1 lần/giây thay vì spam mỗi frame
                        print(f"→ đã gửi {frames} frame JPEG 128x128 TCP")
                except Exception as e:
                    # ESP32 restart/đứt kết nối — đóng socket và tự kết nối lại,
                    # KHÔNG thoát chương trình (tự hồi phục)
                    print(f"\n⚠️ Mất kết nối ({e}) — sẽ tự kết nối lại...")
                    try:
                        sock.close()
                    except Exception:
                        pass
                    sock = None
                    time.sleep(1)
            time.sleep(1/15.0)
    except KeyboardInterrupt:
        print("\nĐã dừng.")
    finally:
        cap.release()
        if sock is not None:
            sock.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dumb IP Camera 128 TCP 12345")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="IP ESP32")
    parser.add_argument("--port", type=int, default=12345, help="Port TCP (mặc định 12345)")
    args = parser.parse_args()
    stream_camera(args.ip, args.port)
