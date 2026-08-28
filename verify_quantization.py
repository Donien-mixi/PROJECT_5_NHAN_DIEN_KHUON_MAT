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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from training_tinyml.models.ghost_tinyface import build_tinyface_ghost

def verify_quantization():
    print("==================================================================")
    print("🔍 XÁC THỰC LƯỢNG TỬ HÓA: Keras Float32 vs TFLite INT8")
    print("==================================================================")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    keras_model_path = os.path.join(base_dir, "training_tinyml", "weights", "tinyface_backbone.keras")
    tflite_model_path = os.path.join(base_dir, "training_tinyml", "weights", "tinyface_int8.tflite")
    data_dir = os.path.join(base_dir, "data", "registered_faces")

    if not os.path.exists(keras_model_path) or not os.path.exists(tflite_model_path):
        print("❌ LỖI: Không tìm thấy model Keras hoặc TFLite.")
        return

    # 1. Nạp mô hình Keras
    print("[*] Đang nạp mô hình Keras Float32...")
    keras_model = build_tinyface_ghost()
    keras_model.load_weights(keras_model_path)

    # 2. Nạp mô hình TFLite
    print("[*] Đang nạp mô hình TFLite INT8...")
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 3. Lấy danh sách ảnh
    img_files = glob.glob(os.path.join(data_dir, "*", "*.jpg")) + glob.glob(os.path.join(data_dir, "*", "*.png"))
    if len(img_files) == 0:
        print("❌ Không tìm thấy ảnh nào trong data/registered_faces/")
        return

    print(f"[*] Đang so sánh trên {len(img_files)} ảnh...")
    
    similarities = []
    
    for path in img_files:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        norm_img = (img.astype(np.float32) - 127.5) / 128.0
        norm_img = np.expand_dims(norm_img, axis=(0, -1))
        
        # Keras Inference
        keras_emb = keras_model(norm_img, training=False)[0].numpy()
        keras_emb = keras_emb / (np.linalg.norm(keras_emb) + 1e-7)
        
        # TFLite Inference
        scale, zero_point = input_details[0]['quantization']
        if scale > 0:
            input_data = np.clip(np.round(norm_img / scale) + zero_point, -128, 127).astype(np.int8)
        else:
            input_data = norm_img.astype(np.float32)
            
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        out_scale, out_zero_point = output_details[0]['quantization']
        if out_scale > 0:
            tflite_emb = (output_data.astype(np.float32) - out_zero_point) * out_scale
        else:
            tflite_emb = output_data.astype(np.float32)
            
        tflite_emb = tflite_emb[0]
        tflite_emb = tflite_emb / (np.linalg.norm(tflite_emb) + 1e-7)
        
        # Calculate Cosine Similarity
        sim = np.dot(keras_emb, tflite_emb)
        similarities.append(sim)

    # Thống kê
    similarities = np.array(similarities)
    avg_sim = np.mean(similarities)
    min_sim = np.min(similarities)
    
    print("\n[+] KẾT QUẢ SO SÁNH:")
    print(f"  - Độ tương đồng trung bình (Average Cosine Sim): {avg_sim*100:.2f}%")
    print(f"  - Độ tương đồng thấp nhất (Worst-case Sim): {min_sim*100:.2f}%")
    
    if avg_sim >= 0.95:
        print("\n🎉 LƯỢNG TỬ HÓA RẤT TỐT! Mô hình INT8 gần như giữ nguyên độ chính xác của Float32.")
    elif avg_sim >= 0.90:
        print("\n⚠️ CẢNH BÁO NHẸ: Lượng tử hóa có gây suy giảm chút ít, nhưng vẫn ở mức chấp nhận được.")
    else:
        print("\n❌ LỖI LƯỢNG TỬ HÓA: Độ lệch quá lớn! Cần xem lại Representative Dataset.")

if __name__ == "__main__":
    verify_quantization()
