import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import glob
import cv2
import numpy as np
import tensorflow as tf

def quantize_to_int8():
    print("==================================================================")
    print("⚡ LƯỢNG TỬ HÓA FULL INT8 CHO ESP32-S3 (TFLITE CONVERTER)")
    print("==================================================================")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "weights", "tinyface_backbone.keras")
    
    if not os.path.exists(model_path):
        print(f"❌ LỖI: Không tìm thấy model tại {model_path}. Vui lòng chạy train_arcface_distill.py trước!")
        return

    # 1. Nạp mô hình Keras gốc (Float32)
    print(f"[*] Đang nạp mô hình Float32 từ: {model_path}")
    from models.ghost_tinyface import build_tinyface_ghost
    model = build_tinyface_ghost()
    model.load_weights(model_path)

    # 2. Xây dựng Representative Dataset để hiệu chuẩn dải giá trị Quantization
    data_dir = os.path.join(os.path.dirname(current_dir), "data", "registered_faces")
    img_files = glob.glob(os.path.join(data_dir, "*", "*.jpg")) + glob.glob(os.path.join(data_dir, "*", "*.png"))
    
    print(f"[*] Tìm thấy {len(img_files)} ảnh làm mẫu hiệu chuẩn (Representative Dataset)...")

    def representative_dataset_gen():
        for path in img_files:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
            norm_img = (img.astype(np.float32) - 127.5) / 128.0
            norm_img = np.expand_dims(norm_img, axis=(0, -1)) # Shape: (1, 64, 64, 1)
            yield [norm_img]

    # 3. Cấu hình Lượng tử hóa Full INT8
    # QUAN TRỌNG: Sử dụng concrete function với batch_size=1 cố định
    # để TFLite Converter không sinh ra ops SHAPE/PACK/STRIDED_SLICE
    # (các ops này không tương thích tốt với TFLite Micro trên ESP32-S3)
    run_model = tf.function(lambda x: model(x, training=False))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([1, 64, 64, 1], tf.float32)
    )
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    
    # Ép buộc 100% các phép tính chạy bằng INT8 cho tập lệnh phần cực Vector ESP-NN
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("[*] Đang tiến hành lượng tử hóa và nén mô hình sang INT8...")
    tflite_quant_model = converter.convert()

    # 4. Lưu file .tflite
    tflite_path = os.path.join(current_dir, "weights", "tinyface_int8.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_quant_model)

    size_kb = len(tflite_quant_model) / 1024.0
    print("\n==================================================================")
    print("🎉 LƯỢNG TỬ HÓA INT8 THÀNH CÔNG!")
    print("==================================================================")
    print(f"📦 Đường dẫn file: {tflite_path}")
    print(f"💾 Kích thước mô hình INT8: {size_kb:.2f} KB (Rất nhỏ gọn, nạp mượt vào 16MB Flash của ESP32-S3)!")
    
    return tflite_path

if __name__ == "__main__":
    quantize_to_int8()
