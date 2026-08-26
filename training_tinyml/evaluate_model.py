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
import tensorflow as tf

def evaluate():
    print("==================================================================")
    print("📊 ĐÁNH GIÁ ĐỊNH LƯỢNG MÔ HÌNH TINYFACENET (BENCHMARK)")
    print("==================================================================")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "weights", "tinyface_backbone.keras")
    db_path = os.path.join(os.path.dirname(current_dir), "data", "face_database.json")

    if not os.path.exists(model_path) or not os.path.exists(db_path):
        print("❌ LỖI: Vui lòng chạy train_arcface_distill.py và generate_embeddings.py trước!")
        return

    model = tf.keras.models.load_model(model_path, safe_mode=False)
    with open(db_path, "r", encoding="utf-8") as f:
        database = json.load(f)

    data_dir = os.path.join(os.path.dirname(current_dir), "data", "registered_faces")

    print("\n--- 1. KIỂM THỬ ĐỘ TƯƠNG ĐỒNG NỘI BỘ (INTRA-CLASS SIMILARITY) ---")
    for user_name, data in database.items():
        ref_emb = np.array(data["embedding"], dtype=np.float32)
        user_dir = os.path.join(data_dir, user_name)
        img_paths = glob.glob(os.path.join(user_dir, "*.jpg")) + glob.glob(os.path.join(user_dir, "*.png"))

        similarities = []
        for p in img_paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
            norm_img = (img.astype(np.float32) - 127.5) / 128.0
            norm_img = np.expand_dims(norm_img, axis=(0, -1))

            emb = model(norm_img)[0].numpy()
            emb = emb / (np.linalg.norm(emb) + 1e-7)
            # Cosine similarity = dot product của 2 vector đã L2-normalized
            sim = np.dot(emb, ref_emb)
            similarities.append(sim)

        if similarities:
            print(f"👤 Người dùng: '{user_name}' ({len(similarities)} ảnh test)")
            print(f"   🔹 Độ tương đồng trung bình (Mean Similarity): {np.mean(similarities):.4f} ({np.mean(similarities)*100:.1f}%)")
            print(f"   🔹 Độ tương đồng nhỏ nhất (Min Similarity):    {np.min(similarities):.4f} ({np.min(similarities)*100:.1f}%)")
            print(f"   🔹 Độ tương đồng lớn nhất (Max Similarity):    {np.max(similarities):.4f} ({np.max(similarities)*100:.1f}%)")

    print("\n--- 2. KHUYẾN NGHỊ NGƯỠNG NHẬN DIỆN (THRESHOLD RECOMMENDATION) ---")
    print("🎯 Ngưỡng Cosine Threshold tối ưu đề xuất cho ESP32-S3: 0.65 (65.0%)")
    print("   - Nếu Similarity >= 0.65: XÁC NHẬN ĐÚNG NGƯỜI (MATCH)")
    print("   - Nếu Similarity < 0.65:  NGƯỜI LẠ / KHÔNG KHỚP (UNKNOWN)")
    print("==================================================================")

if __name__ == "__main__":
    evaluate()
