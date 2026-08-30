import os
import json
import cv2
import numpy as np
import tensorflow as tf
from models.ghost_tinyface import build_tinyface_ghost
import glob

def find_outliers():
    print("=========================================================")
    print("🕵️ ĐANG TÌM KIẾM CÁC BỨC ẢNH LỖI (OUTLIERS) TRONG DATASET")
    print("=========================================================")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "weights", "tinyface_backbone.keras")
    db_path = os.path.join(os.path.dirname(current_dir), "data", "face_database.json")

    model = build_tinyface_ghost()
    model.load_weights(model_path)
    
    with open(db_path, "r", encoding="utf-8") as f:
        database = json.load(f)

    data_dir = os.path.join(os.path.dirname(current_dir), "data", "registered_faces")
    
    for user_name in database.keys():
        ref_emb = np.array(database[user_name]["embedding"], dtype=np.float32)
        user_dir = os.path.join(data_dir, user_name)
        img_paths = glob.glob(os.path.join(user_dir, "*.jpg")) + glob.glob(os.path.join(user_dir, "*.png"))
        
        for p in img_paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
            norm = (img.astype(np.float32) - 127.5) / 128.0
            emb = model(np.expand_dims(norm, axis=(0,-1)))[0].numpy()
            emb = emb / (np.linalg.norm(emb) + 1e-7)
            
            sim = np.dot(emb, ref_emb)
            # Nếu bức ảnh nào giống chính chủ dưới 70%, đó chắc chắn là ảnh rác/lỗi
            if sim < 0.70:
                print(f"❌ ẢNH BỊ LỖI (Độ tương đồng chỉ {sim*100:.1f}%): {p}")

if __name__ == "__main__":
    find_outliers()
