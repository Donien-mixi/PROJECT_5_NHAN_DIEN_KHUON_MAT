import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_tinyml.train_arcface_distill import train_model
from training_tinyml.quantize_qat_int8 import quantize_to_int8
from training_tinyml.generate_embeddings import generate_database
from training_tinyml.export_to_c_header import export_tflite_to_c_header
from training_tinyml.evaluate_model import evaluate

def main():
    print("==================================================================")
    print("🚀 BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN VÀ ĐÓNG GÓI TINYML CHO ESP32-S3")
    print("==================================================================")

    # Bước 1: Huấn luyện mô hình TinyFaceNet với ArcFace Loss
    print("\n>>> BƯỚC 1/5: HUẤN LUYỆN MÔ HÌNH ARCFACE...")
    train_model(epochs=25, batch_size=16)

    # Bước 2: Lượng tử hóa Full INT8 sang TFLite
    print("\n>>> BƯỚC 2/5: LƯỢNG TỬ HÓA FULL INT8 CHO ESP32-S3...")
    quantize_to_int8()

    # Bước 3: Trích xuất Vector đặc trưng 128D cho Database
    print("\n>>> BƯỚC 3/5: TRÍCH XUẤT FACE DATABASE 128D...")
    generate_database()

    # Bước 4: Xuất mô hình thành mảng C++ Header
    print("\n>>> BƯỚC 4/5: XUẤT C++ HEADER (model_data.h)...")
    export_tflite_to_c_header()

    # Bước 5: Đánh giá định lượng độ chính xác
    print("\n>>> BƯỚC 5/5: ĐÁNH GIÁ ĐỘ CHÍNH XÁC MÔ HÌNH...")
    evaluate()

    print("\n==================================================================")
    print("🎉 HOÀN THÀNH 100% QUY TRÌNH GIAI ĐOẠN 3!")
    print("📦 Mô hình AI và CSDL C++ đã sẵn sàng để nạp vào ESP32-S3 (Giai đoạn 4 & 5).")
    print("==================================================================")

if __name__ == "__main__":
    main()
