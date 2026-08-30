import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def update_db():
    print("==================================================================")
    print("🔄 CẬP NHẬT LẠI CƠ SỞ DỮ LIỆU KHUÔN MẶT")
    print("==================================================================")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    import subprocess
    
    # 1. Cập nhật Face Embeddings (Từ ảnh đã chụp)
    gen_script = os.path.join(current_dir, "generate_embeddings.py")
    print("\n[Bước 1] Đang trích xuất lại đặc trưng khuôn mặt...")
    subprocess.run([sys.executable, gen_script], check=True)
    
    # 2. Bỏ qua bước kiểm tra model_data.h vì script không tồn tại
    print("\n✅ Cập nhật CSDL hoàn tất! Hãy mở Arduino IDE nạp lại Firmware ESP32 nếu bạn vừa thêm ảnh người mới.")

if __name__ == "__main__":
    update_db()
