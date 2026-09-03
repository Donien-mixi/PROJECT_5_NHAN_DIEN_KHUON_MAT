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
    
    # 2. Regenerate ai_config.h đồng bộ PSRAM (768KB+350KB+32KB)
    try:
        export_script = os.path.join(current_dir, "export_config.py")
        if os.path.exists(export_script):
            print("\n[Bước 2] Đang regenerate ai_config.h...")
            subprocess.run([sys.executable, export_script], check=False)
    except Exception as e:
        print(f"[!] Bỏ qua export_config: {e}")

    print("\n✅ Cập nhật CSDL hoàn tất! Hãy mở Arduino IDE nạp lại Firmware ESP32 nếu bạn vừa thêm ảnh người mới.")

if __name__ == "__main__":
    update_db()
