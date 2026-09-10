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

# Cho phép chạy trực tiếp: python training_tinyml/generate_embeddings.py
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from host_laptop.core.vision_utils import equalize_gray_256

# Đồng bộ ESP32: số template mỗi người trong struct C++ (lấp 0.0f nếu ít hơn)
MAX_TEMPLATES = 16
FACE_EMBEDDING_DIM = 128

# 4.2 Per-identity threshold (Verheyen ARES 2023 — identity-level thresholds):
# người dễ nhầm (cross-sim cao với người khác) có ngưỡng riêng cao hơn ngưỡng global,
# giúp FAR ổn định khi DB dày lên mà không cần train lại. Floor = global, cap = 0.80.
GLOBAL_THRESHOLD = 0.70
PER_ID_MARGIN = 0.02   # cross_max + margin
PER_ID_CAP = 0.75      # Ngưỡng trần tối ưu (tránh quá gắt, cho phép góc mặt dao động tự nhiên 0.75-0.85)


def compute_per_identity_thresholds(database):
    """Tính ngưỡng riêng từng người: max cosine giữa template của người này và
    template của MỌI người khác (cross-user MAX), + margin. Trả về dict name->thr."""
    thresholds = {}
    names = list(database.keys())
    for u in names:
        cross_max = 0.0
        u_templates = [np.asarray(t, dtype=np.float32) for t in database[u]["templates"]]
        for v in names:
            if v == u:
                continue
            for t_v in database[v]["templates"]:
                tv = np.asarray(t_v, dtype=np.float32)
                for t_u in u_templates:
                    s = float(np.dot(t_u, tv))
                    if s > cross_max:
                        cross_max = s
        thr = min(PER_ID_CAP, max(GLOBAL_THRESHOLD, cross_max + PER_ID_MARGIN))
        thresholds[u] = {"cross_max": cross_max, "threshold": thr}
    return thresholds

def generate_database():
    print("==================================================================")
    print("🧠 TRÍCH XUẤT DATABASE VECTOR ĐẶC TRƯNG 128D (EMBEDDINGS)")
    print("==================================================================")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "weights", "tinyface_int8.tflite")
    
    if not os.path.exists(model_path):
        print(f"❌ LỖI: Không tìm thấy mô hình tại {model_path}. Hãy chạy python extract_tflite.py trước!")
        return

    # 1. Nạp mô hình TFLite INT8
    print(f"[*] Đang nạp mô hình INT8 TFLite: {model_path}")
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

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
            # HE khử nhạy ánh sáng — ĐỒNG BỘ preprocess_face() ESP32 + align_and_crop live
            img = equalize_gray_256(img)
            norm_img = (img.astype(np.float32) - 127.5) / 128.0
            norm_img = np.expand_dims(norm_img, axis=(0, -1)) # Shape: (1, 64, 64, 1)

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
                
            emb = emb[0] # shape: (128,)
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

        # Lưu đa templates (mỗi ảnh sạch một vector) để matching lấy MAX-SIM —
        # tăng độ khớp khi góc/ánh sáng live khác lúc enroll mà không cần chụp thêm.
        templates = [e / (np.linalg.norm(e) + 1e-7) for e in clean_embeddings]
        # GIỚI HẠN đúng MAX_TEMPLATES — nếu chụp >20 ảnh (keep > 16) mà không cắt thì
        # header C++ sinh ra sẽ thừa initializer cho float embeddings[16][128] → compile error.
        if len(templates) > MAX_TEMPLATES:
            templates = templates[:MAX_TEMPLATES]

        database[user_name] = {
            "id": idx + 1,
            "name": user_name,
            "embedding": norm_mean_emb.tolist(),
            "templates": [t.tolist() for t in templates],
            "samples_count": len(user_embeddings),
            "clean_samples_used": keep_count
        }

        print(f"    ✅ Đã tạo vector đại diện 128-D + {len(templates)}/{MAX_TEMPLATES} templates cho '{user_name}' (dùng {keep_count}/{len(user_embeddings)} ảnh chất lượng nhất).")

    # 3.5 [4.2] Per-identity threshold — tính SAU khi tất cả người đã có templates
    id_thresholds = compute_per_identity_thresholds(database)
    for u, info in id_thresholds.items():
        database[u]["threshold"] = round(info["threshold"], 4)
        print(f"    🔒 Ngưỡng riêng '{u}': {info['threshold']:.3f} "
              f"(cross_max={info['cross_max']:.3f} + margin {PER_ID_MARGIN})")

    # 3.6 Tạo chuỗi C++ cho firmware ESP32 (đa templates + ngưỡng riêng — MAX-SIM đồng bộ Laptop)
    # Mỗi người giữ tối đa MAX_TEMPLATES vector; padding 0.0f nếu ít hơn.
    cpp_db_entries = []
    for idx, user_name in enumerate(subdirs):
        if user_name not in database:
            continue
        templates = database[user_name]["templates"]
        templ_list = []
        for t in templates:
            templ_list.append(", ".join(f"{v:.6f}f" for v in t))
        # Chèn padding cho đủ MAX_TEMPLATES (để struct tĩnh C++)
        pad = "0.0f"
        while len(templ_list) < MAX_TEMPLATES:
            templ_list.append(", ".join([pad] * 128))
        templ_cpp = ", ".join("{" + t + "}" for t in templ_list)
        cpp_db_entries.append(
            f'  {{"{user_name}", {database[user_name]["id"]}, {database[user_name]["threshold"]:.4f}f, {len(templates)}, {{{templ_cpp}}}}}'
        )

    # 4. Lưu ra file JSON
    json_path = os.path.join(base_dir, "data", "face_database.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(database, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Đã lưu CSDL JSON tại: {json_path}")

    # 5. Xuất ra file Header C++ cho ESP32-S3 Firmware (multi-template)
    cpp_header_path = os.path.join(base_dir, "firmware_esp32", "face_database.h")
    
    # Dành cho môi trường Colab (Không có thư mục firmware_esp32)
    colab_export_path = os.path.join(current_dir, "colab_exports", "face_database.h")
    os.makedirs(os.path.dirname(colab_export_path), exist_ok=True)
    
    cpp_content = (
        "#pragma once\n#include <Arduino.h>\n\n"
        f"#define MAX_TEMPLATES {MAX_TEMPLATES}\n"
        "struct RegisteredFace {\n"
        "    const char* name;\n"
        "    int id;\n"
        "    float threshold;  // [4.2] per-identity threshold (global 0.60, cap 0.80)\n"
        "    int num_templates;\n"
        f"    float embeddings[MAX_TEMPLATES][{FACE_EMBEDDING_DIM}];\n"
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
