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
    print("📊 ĐÁNH GIÁ ĐỊNH LƯỢNG MÔ HÌNH TINYFACENET (BENCHMARK & CONFUSION)")
    print("==================================================================")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "weights", "tinyface_backbone.keras")
    db_path = os.path.join(os.path.dirname(current_dir), "data", "face_database.json")

    if not os.path.exists(model_path) or not os.path.exists(db_path):
        print("❌ LỖI: Vui lòng chạy train_arcface_distill.py và generate_embeddings.py trước!")
        return

    from models.ghost_tinyface import build_tinyface_ghost
    model = build_tinyface_ghost()
    model.load_weights(model_path)
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

            emb = model(norm_img)[0].numpy()
            emb = emb / (np.linalg.norm(emb) + 1e-7)
            embs.append(emb)
        user_test_embeddings[user_name] = embs

    print("\n--- 1. KIỂM THỬ ĐỘ TƯƠNG ĐỒNG NỘI BỘ (INTRA-CLASS SIMILARITY) ---")
    min_intra_all = 1.0
    for user_name in user_names:
        ref_emb = np.array(database[user_name]["embedding"], dtype=np.float32)
        embs = user_test_embeddings[user_name]
        
        if embs:
            sims = [np.dot(e, ref_emb) for e in embs]
            mean_s = np.mean(sims)
            min_s = np.min(sims)
            max_s = np.max(sims)
            min_intra_all = min(min_intra_all, min_s)
            
            print(f"👤 Người dùng: '{user_name}' ({len(sims)} ảnh test)")
            print(f"   🔹 Độ tương đồng trung bình (Mean Similarity): {mean_s:.4f} ({mean_s*100:.1f}%)")
            print(f"   🔹 Độ tương đồng nhỏ nhất (Min Similarity):    {min_s:.4f} ({min_s*100:.1f}%)")
            print(f"   🔹 Độ tương đồng lớn nhất (Max Similarity):    {max_s:.4f} ({max_s*100:.1f}%)")

    print("\n--- 2. KIỂM THỬ ĐỘ TƯƠNG ĐỒNG CHÉO (INTER-CLASS CROSS-SIMILARITY) ---")
    max_inter_all = 0.0
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
                        max_cross = np.max(cross_sims)
                        max_inter_all = max(max_inter_all, max_cross)
                        print(f"🔀 Ảnh của '{u_src}' so khớp với CSDL của '{u_dst}':")
                        print(f"   🔸 Tương đồng chéo trung bình: {mean_cross:.4f} ({mean_cross*100:.1f}%)")
                        print(f"   🔸 Tương đồng chéo cao nhất:   {max_cross:.4f} ({max_cross*100:.1f}%)")
    else:
        print("ℹ️ Chỉ có 1 người dùng thực tế trong CSDL.")

    # 3. Tính toán khoảng cách phân cách (Separation Margin)
    print("\n--- 3. ĐÁNH GIÁ ĐỘ PHÂN CÁCH (SEPARATION MARGIN) & NGƯỠNG TỐI ƯU ---")
    margin = min_intra_all - max_inter_all
    recommended_thresh = (min_intra_all + max_inter_all) / 2.0
    if recommended_thresh < 0.55:
        recommended_thresh = 0.60
    elif recommended_thresh > 0.75:
        recommended_thresh = 0.70

    print(f"📐 Khoảng cách phân cách an toàn (Separation Margin): {margin:.4f} ({margin*100:.1f}%)")
    if margin > 0.15:
        print("✅ ĐÁNH GIÁ: XUẤT SẮC! Hai người dùng được phân tách rõ rệt trong không gian cầu 128D.")
    elif margin > 0.0:
        print("⚠️ ĐÁNH GIÁ: TỐT, nhưng nên bổ sung thêm góc chụp để tăng biên độ phân cách.")
    else:
        print("❌ ĐÁNH GIÁ: Có sự chồng lấn giữa các lớp, cần huấn luyện lại!")

    print(f"\n🎯 Ngưỡng Cosine Threshold tối ưu đề xuất cho ESP32-S3: {recommended_thresh:.2f} ({recommended_thresh*100:.0f}%)")
    print(f"   - Nếu Similarity >= {recommended_thresh:.2f}: XÁC NHẬN ĐÚNG NGƯỜI (MATCH)")
    print(f"   - Nếu Similarity < {recommended_thresh:.2f}:  NGƯỜI LẠ / KHÔNG KHỚP (UNKNOWN)")
    print("==================================================================")

if __name__ == "__main__":
    evaluate()
