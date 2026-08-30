import cv2
import socket
import argparse
import time
import struct
import sys
import os

# Thêm đường dẫn để import được module detector
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detector.yunet_detector import FaceDetector

def stream_camera(ip, port, resolution=(240, 240), quality=50):
    # Cấu hình Socket TCP
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Đang kết nối tới ESP32 tại {ip}:{port}...")
    try:
        sock.connect((ip, port))
        print("✅ Kết nối TCP thành công!")
    except Exception as e:
        print(f"❌ Không thể kết nối tới ESP32: {e}")
        print("Vui lòng đảm bảo ESP32 đã khởi động và in ra 'Listening on TCP port...'")
        return
        
    print(f"🚀 Bắt đầu phát luồng video tới ESP32 tại {ip}:{port}")
    print(f"Độ phân giải gửi: 64x64 (Đã qua xử lý Affine Alignment)")
    print("Nhấn Ctrl+C để thoát...")
    
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Lỗi: Không thể mở camera trên Laptop")
        return
        
    # Laptop bây giờ chỉ đóng vai trò là Camera IP (Dumb Camera)
    # Không thực hiện nhận diện hay cắt ảnh nữa, gửi thẳng 240x240 xuống ESP32
    print(f"Đang chuẩn bị gửi nguyên khung hình thô {resolution[0]}x{resolution[1]} xuống ESP32...")
        
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize khung hình về 240x240 (vuông) để tiết kiệm băng thông TCP
            # Bạn có thể cắt ở giữa (Center crop) nếu muốn không bị méo tỉ lệ
            h, w = frame.shape[:2]
            size = min(h, w)
            y_start = (h - size) // 2
            x_start = (w - size) // 2
            cropped_frame = frame[y_start:y_start+size, x_start:x_start+size]
            resized_frame = cv2.resize(cropped_frame, resolution)
            
            # Hiển thị trên Laptop để xem
            cv2.imshow("Raw Frame (Streaming to ESP32)", resized_frame)
            cv2.waitKey(1)
            
            # Nén ảnh thành JPEG
            ret, buffer = cv2.imencode('.jpg', resized_frame, encode_param)
            
            if ret:
                data = buffer.tobytes()
                # Gửi bằng TCP: Gửi 4 bytes độ dài trước, sau đó gửi data
                try:
                    sock.sendall(struct.pack('<I', len(data)))
                    sock.sendall(data)
                    print(f"Đang gửi {len(data)} bytes (JPEG {resolution[0]}x{resolution[1]}) qua TCP...", end='\r')
                except Exception as e:
                    print(f"\n❌ Lỗi gửi mạng (Có thể ESP32 ngắt kết nối): {e}")
                    break
                    
            # Khống chế FPS (Truyền 10-15 FPS là đủ mượt cho ESP32)
            time.sleep(1/15.0)
            
    except KeyboardInterrupt:
        print("\nĐã dừng luồng Camera IP.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        sock.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Laptop IP Camera Streamer cho ESP32")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="Địa chỉ IP tĩnh của board ESP32")
    parser.add_argument("--port", type=int, default=12345, help="Port UDP lắng nghe trên ESP32")
    args = parser.parse_args()
    
    stream_camera(args.ip, args.port)
