import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import glob
import json
import cv2
import numpy as np
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

def evaluate():
    print("==================================================================")
    print("📊 ĐÁNH GIÁ ĐỊNH LƯỢNG MÔ HÌNH TINYFACENET (BENCHMARK & CONFUSION)")
    print("==================================================================")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "weights", "tinyface_int8.tflite")
    db_path = os.path.join(os.path.dirname(current_dir), "data", "face_database.json")

    if not os.path.exists(model_path) or not os.path.exists(db_path):
        print("❌ LỖI: Vui lòng chạy extract_tflite.py và generate_embeddings.py trước!")
        return

    # Nạp mô hình TFLite INT8
    print(f"[*] Đang nạp mô hình INT8 TFLite: {model_path}")
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']
    with open(db_path, "r", encoding="utf-8") as f:
        database = json.load(f)

    data_dir = os.path.join(os.path.dirname(current_dir), "data", "registered_faces")

    user_names = list(database.keys())
    user_test_embeddings = {}

    # Trích xuất embeddings cho tất cả ảnh test của từng người
    for user_name in user_names:
        user_dir = os.path.join(data_dir, user_name)
        img_paths = glob.glob(os.path.join(user_dir, "*.jpg")) + glob.glob(os.path.join(user_dir, "*.png"))
        
        embs = []
        for p in img_paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
            norm_img = (img.astype(np.float32) - 127.5) / 128.0
            norm_img = np.expand_dims(norm_img, axis=(0, -1))

            # Lượng tử hóa Input thành INT8
            if input_scale > 0:
                input_data = (norm_img / input_scale) + input_zero_point
                input_data = np.clip(input_data, -128, 127).astype(np.int8)
            else:
                input_data = norm_img.astype(np.float32)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # Giải lượng tử hóa Output về Float32
            if output_scale > 0:
                emb = (output_data.astype(np.float32) - output_zero_point) * output_scale
            else:
                emb = output_data.astype(np.float32)

            emb = emb[0]
            emb = emb / (np.linalg.norm(emb) + 1e-7)
            embs.append(emb)
        user_test_embeddings[user_name] = embs

    min_mean_intra = 1.0

    print("\n--- 1. KIỂM THỬ ĐỘ TƯƠNG ĐỒNG NỘI BỘ (INTRA-CLASS SIMILARITY) ---")
    for user_name in user_names:
        ref_emb = np.array(database[user_name]["embedding"], dtype=np.float32)
        embs = user_test_embeddings[user_name]
        
        if embs:
            sims = [np.dot(e, ref_emb) for e in embs]
            mean_s = np.mean(sims)
            min_mean_intra = min(min_mean_intra, mean_s)
            p5_s = np.percentile(sims, 5) # Use 5th percentile to ignore extreme outliers
            max_s = np.max(sims)
            
            print(f"👤 Người dùng: '{user_name}' ({len(sims)} ảnh test)")
            print(f"   🔹 Độ tương đồng trung bình (Mean):          {mean_s:.4f} ({mean_s*100:.1f}%)")
            print(f"   🔹 Độ tương đồng thấp nhất (P5 - Bỏ nhiễu): {p5_s:.4f} ({p5_s*100:.1f}%)")
            print(f"   🔹 Ảnh mờ/tệ nhất (Min tuyệt đối):          {np.min(sims):.4f} ({np.min(sims)*100:.1f}%)")

    max_mean_inter = 0.0

    print("\n--- 2. KIỂM THỬ ĐỘ TƯƠNG ĐỒNG CHÉO (INTER-CLASS CROSS-SIMILARITY) ---")
    if len(user_names) > 1:
        for i in range(len(user_names)):
            for j in range(len(user_names)):
                if i != j:
                    u_src = user_names[i]
                    u_dst = user_names[j]
                    ref_dst = np.array(database[u_dst]["embedding"], dtype=np.float32)
                    embs_src = user_test_embeddings[u_src]
                    
                    if embs_src:
                        cross_sims = [np.dot(e, ref_dst) for e in embs_src]
                        mean_cross = np.mean(cross_sims)
                        max_mean_inter = max(max_mean_inter, mean_cross)
                        p95_cross = np.percentile(cross_sims, 95) # Use 95th percentile
                        max_cross = np.max(cross_sims)
                        print(f"🔀 Ảnh của '{u_src}' so khớp với CSDL của '{u_dst}':")
                        print(f"   🔸 Tương đồng chéo trung bình:           {mean_cross:.4f} ({mean_cross*100:.1f}%)")
                        print(f"   🔸 Tương đồng chéo cao nhất (P95):       {p95_cross:.4f} ({p95_cross*100:.1f}%)")
                        print(f"   🔸 Ảnh nhầm lẫn cao nhất (Max tuyệt đối): {max_cross:.4f} ({max_cross*100:.1f}%)")
    else:
        print("ℹ️ Chỉ có 1 người dùng thực tế trong CSDL.")

    # 3. Tính toán khoảng cách phân cách dựa trên MEAN (Ổn định hơn Percentile đối với Data nén hẹp)
    print("\n--- 3. ĐÁNH GIÁ ĐỘ PHÂN CÁCH (SEPARATION MARGIN) & NGƯỠNG TỐI ƯU ---")
    margin = min_mean_intra - max_mean_inter
    recommended_thresh = (min_mean_intra + max_mean_inter) / 2.0
    
    # Không giới hạn cận trên ở 0.70 nữa, vì mô hình KD có thể có dải similarity từ 0.85 - 0.99
    if recommended_thresh < 0.55:
        recommended_thresh = 0.60

    print(f"📐 Khoảng cách trung bình giữa các lớp (Mean Margin): {margin:.4f} ({margin*100:.1f}%)")
    if margin > 0.04:
        print("✅ ĐÁNH GIÁ: XUẤT SẮC! Khoảng cách trung bình an toàn. Threshold đề xuất sẽ hoạt động tốt.")
    elif margin > 0.02:
        print("⚠️ ĐÁNH GIÁ: TỐT, nhưng 2 người dùng có nét khá giống nhau đối với AI.")
    else:
        print("❌ ĐÁNH GIÁ: Quá giống nhau! Hãy thử chụp lại ảnh ở các góc sáng sủa hơn.")

    print(f"\n🎯 Ngưỡng Cosine Threshold tối ưu đề xuất cho ESP32-S3: {recommended_thresh:.2f} ({recommended_thresh*100:.0f}%)")
    print(f"   - Nếu Similarity >= {recommended_thresh:.2f}: XÁC NHẬN ĐÚNG NGƯỜI (MATCH)")
    print(f"   - Nếu Similarity < {recommended_thresh:.2f}:  NGƯỜI LẠ / KHÔNG KHỚP (UNKNOWN)")
    print("==================================================================")

if __name__ == "__main__":
    evaluate()
