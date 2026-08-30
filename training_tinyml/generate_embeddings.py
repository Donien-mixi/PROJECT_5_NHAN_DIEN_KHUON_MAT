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

def generate_database():
    print("==================================================================")
    print("🧠 TRÍCH XUẤT DATABASE VECTOR ĐẶC TRƯNG 128D (EMBEDDINGS)")
    print("==================================================================")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "weights", "tinyface_backbone.keras")
    
    if not os.path.exists(model_path):
        print(f"❌ LỖI: Không tìm thấy mô hình tại {model_path}. Hãy chạy train_arcface_distill.py trước!")
        return

    # 1. Nạp mô hình
    print(f"[*] Đang nạp mô hình: {model_path}")
    from models.ghost_tinyface import build_tinyface_ghost
    model = build_tinyface_ghost()
    model.load_weights(model_path)

    # 2. Quét thư mục người dùng thực tế trong data/registered_faces
    base_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(base_dir, "data", "registered_faces")
    
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith("Impostor_")]
    subdirs.sort()

    database = {}
    cpp_db_entries = []

    for idx, user_name in enumerate(subdirs):
        user_dir = os.path.join(data_dir, user_name)
        img_paths = glob.glob(os.path.join(user_dir, "*.jpg")) + glob.glob(os.path.join(user_dir, "*.png"))
        
        if len(img_paths) == 0:
            continue
            
        print(f"\n[*] Đang xử lý danh tính '{user_name}' ({len(img_paths)} ảnh)...")
        user_embeddings = []

        for p in img_paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
            norm_img = (img.astype(np.float32) - 127.5) / 128.0
            norm_img = np.expand_dims(norm_img, axis=(0, -1)) # Shape: (1, 64, 64, 1)

            # Trích xuất vector 128D và chuẩn hóa L2
            emb = model(norm_img)[0].numpy() # shape: (128,)
            norm_e = emb / (np.linalg.norm(emb) + 1e-7)
            user_embeddings.append(norm_e)

        if not user_embeddings:
            continue

        user_embeddings = np.array(user_embeddings)

        # 3. Thuật toán Trimmed Centroid (Lọc bỏ 20% ảnh ngoại lai/mờ/lệch)
        initial_mean = np.mean(user_embeddings, axis=0)
        initial_mean /= (np.linalg.norm(initial_mean) + 1e-7)

        # Tính độ tương đồng của từng ảnh với tâm sơ bộ
        sims = np.dot(user_embeddings, initial_mean)
        
        # Giữ lại 80% ảnh có độ tương đồng cao nhất (loại bỏ 20% nhiễu)
        keep_count = max(5, int(len(user_embeddings) * 0.80))
        top_indices = np.argsort(sims)[-keep_count:]
        clean_embeddings = user_embeddings[top_indices]

        # Tính Vector đại diện chuẩn mực từ các ảnh nét nhất
        mean_emb = np.mean(clean_embeddings, axis=0)
        norm_mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-7)

        database[user_name] = {
            "id": idx + 1,
            "name": user_name,
            "embedding": norm_mean_emb.tolist(),
            "samples_count": len(user_embeddings),
            "clean_samples_used": keep_count
        }

        # Tạo chuỗi C++ cho firmware ESP32
        cpp_emb_str = ", ".join([f"{v:.6f}f" for v in norm_mean_emb])
        cpp_db_entries.append(f'  {{"{user_name}", {idx + 1}, {{{cpp_emb_str}}}}}')

        print(f"    ✅ Đã tạo vector đại diện 128 chiều sạch cho '{user_name}' (dùng {keep_count}/{len(user_embeddings)} ảnh chất lượng nhất).")

    # 4. Lưu ra file JSON
    json_path = os.path.join(base_dir, "data", "face_database.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(database, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Đã lưu CSDL JSON tại: {json_path}")

    # 5. Xuất ra file Header C++ cho ESP32-S3 Firmware
    cpp_header_path = os.path.join(base_dir, "firmware_esp32", "face_database.h")
    
    # Dành cho môi trường Colab (Không có thư mục firmware_esp32)
    colab_export_path = os.path.join(current_dir, "colab_exports", "face_database.h")
    os.makedirs(os.path.dirname(colab_export_path), exist_ok=True)
    
    cpp_content = (
        "#pragma once\n#include <Arduino.h>\n\n"
        "struct RegisteredFace {\n"
        "    const char* name;\n"
        "    int id;\n"
        "    float embedding[128];\n"
        "};\n\n"
        f"const int NUM_REGISTERED_FACES = {len(cpp_db_entries)};\n\n"
        "const RegisteredFace FACE_DATABASE[NUM_REGISTERED_FACES] = {\n"
        + ",\n".join(cpp_db_entries)
        + "\n};\n"
    )

    with open(colab_export_path, "w", encoding="utf-8") as f:
        f.write(cpp_content)
        
    print(f"[+] Đã tạo thành công file CSDL C++ cho Colab tại:\n    👉 {colab_export_path}")
        
    # Copy sang firmware_esp32 nếu đang chạy ở máy tính local
    if os.path.exists(os.path.dirname(cpp_header_path)):
        with open(cpp_header_path, "w", encoding="utf-8") as f:
            f.write(cpp_content)
        print(f"[+] Đã copy mã nguồn C++ CSDL nạp cho ESP32 tại:\n    👉 {cpp_header_path}")

if __name__ == "__main__":
    generate_database()
