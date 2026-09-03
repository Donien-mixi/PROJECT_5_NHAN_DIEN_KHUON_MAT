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

# Cho phép chạy trực tiếp: python training_tinyml/evaluate_model.py
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from host_laptop.core.vision_utils import equalize_gray_256


def embed_image(interpreter, input_details, output_details, input_scale, input_zero_point,
                output_scale, output_zero_point, path):
    """Trích xuất embedding L2-normalized 128-D từ một ảnh grayscale (64x64 nếu khác sẽ resize).
    Áp dụng HE khử nhạy ánh sáng — đồng bộ generate_embeddings + ESP32 preprocess_face."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    img = equalize_gray_256(img)
    norm_img = (img.astype(np.float32) - 127.5) / 128.0
    norm_img = np.expand_dims(norm_img, axis=(0, -1))
    if input_scale > 0:
        input_data = (norm_img / input_scale) + input_zero_point
        input_data = np.clip(input_data, -128, 127).astype(np.int8)
    else:
        input_data = norm_img.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if output_scale > 0:
        emb = (output_data.astype(np.float32) - output_zero_point) * output_scale
    else:
        emb = output_data.astype(np.float32)
    emb = emb[0]
    return emb / (np.linalg.norm(emb) + 1e-7)


def trimmed_centroid(embs, keep_ratio=0.8):
    """Vector đại diện: giữ 80% embedding tương đồng với tâm sơ bộ nhất — đồng bộ generate_embeddings.py."""
    if not embs:
        return None
    arr = np.array(embs, dtype=np.float32)
    mean = arr.mean(axis=0)
    mean = mean / (np.linalg.norm(mean) + 1e-7)
    sims = np.dot(arr, mean)
    keep_count = max(5, int(len(arr) * keep_ratio))
    keep_count = min(keep_count, len(arr))
    top = np.argsort(sims)[-keep_count:]
    centroid = arr[top].mean(axis=0)
    return centroid / (np.linalg.norm(centroid) + 1e-7)


def evaluate():
    print("==================================================================")
    print("📊 ĐÁNH GIÁ ĐỊNH LƯỢNG MÔ HÌNH TINYFACENET (BENCHMARK & CONFUSION)")
    print("==================================================================")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "weights", "tinyface_int8.tflite")
    db_path = os.path.join(os.path.dirname(current_dir), "data", "face_database.json")

    if not os.path.exists(model_path):
        print("❌ LỖI: Vui lòng chạy quantize_qat_int8.py trước để tạo tinyface_int8.tflite!")
        return

    # Nạp mô hình TFLite INT8
    print(f"[*] Đang nạp mô hình INT8 TFLite: {model_path}")
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    data_dir = os.path.join(os.path.dirname(current_dir), "data", "registered_faces")

    database = {}
    if os.path.exists(db_path):
        with open(db_path, "r", encoding="utf-8") as f:
            database = json.load(f)

    user_names = [d for d in os.listdir(data_dir)
                  if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith("Impostor_")]
    user_names.sort()

    if not user_names:
        print("❌ LỖI: Không có thư mục người dùng trong data/registered_faces! Hãy enroll trước.")
        return

    # Trích xuất embeddings cho tất cả ảnh của từng người (ảnh 64x64 PNG/JPG)
    print(f"[*] Đang trích xuất embedding cho {len(user_names)} người dùng (gallery + probe)...")
    user_imgs = {}    # user -> list path
    user_embs = {}    # user -> list embedding
    impostor_embs = []  # ảnh thư mục Impostor_* (nếu có) + ảnh người khác dùng làm impostor

    for user_name in user_names:
        user_dir = os.path.join(data_dir, user_name)
        img_paths = sorted(glob.glob(os.path.join(user_dir, "*.jpg")) + glob.glob(os.path.join(user_dir, "*.png")))
        embs = []
        for p in img_paths:
            e = embed_image(interpreter, input_details, output_details,
                            input_scale, input_zero_point, output_scale, output_zero_point, p)
            if e is not None:
                embs.append(e)
        user_imgs[user_name] = img_paths
        user_embs[user_name] = embs
        print(f"   - {user_name}: {len(embs)} ảnh")

    # Impostor: ảnh trong các thư mục Impostor_* (kho người lạ độc lập)
    for imp_dirname in sorted(os.listdir(data_dir)):
        if not imp_dirname.startswith("Impostor_"):
            continue
        imp_dir = os.path.join(data_dir, imp_dirname)
        for p in sorted(glob.glob(os.path.join(imp_dir, "*.jpg")) + glob.glob(os.path.join(imp_dir, "*.png"))):
            e = embed_image(interpreter, input_details, output_details,
                            input_scale, input_zero_point, output_scale, output_zero_point, p)
            if e is not None:
                impostor_embs.append(e)
    print(f"   - Impostor folder: {len(impostor_embs)} ảnh "
          f"(đặt vào data/registered_faces/Impostor_XX/ để làm người lạ độc lập)")

    def split_gallery_probe(embs, keep_ratio=0.8):
        """Chia gallery (80%) / probe (20%) theo độ tương đồng với tâm sơ bộ (không dùng ảnh trùng).
        Trả về (centroid, probe, gallery_list) — gallery_list là templates cho mục 5."""
        if len(embs) < 2:
            return None, None, None
        arr = np.array(embs, dtype=np.float32)
        mean = arr.mean(axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-7)
        sims = np.dot(arr, mean)
        sort_idx = np.argsort(sims)  # tương đồng tăng dần
        gallery_count = max(1, int(len(arr) * keep_ratio))
        gallery_embs = arr[sort_idx[-gallery_count:]]
        probe_embs = arr[sort_idx[:-gallery_count]]
        if len(probe_embs) == 0:
            probe_embs = arr[sort_idx[-1:]]  # fallback nếu gallery lấy hết (users 1 ảnh)
            gallery_embs = arr[:len(arr)-1]
        centroid = trimmed_centroid(list(gallery_embs))
        return centroid, list(probe_embs), list(gallery_embs)

    # 1. Intra-class similarity (với centroid tổng của từng người — same phép DB)
    print("\n--- 1. KIỂM THỬ ĐỘ TƯƠNG ĐỒNG NỘI BỘ (INTRA-CLASS SIMILARITY vs DB CENTROID) ---")
    min_mean_intra = 1.0
    for user_name in user_names:
        ref_emb = None
        if user_name in database:
            ref_emb = np.array(database[user_name]["embedding"], dtype=np.float32)
        elif user_embs[user_name]:
            ref_emb = trimmed_centroid(user_embs[user_name])
        embs = user_embs[user_name]
        if embs and ref_emb is not None:
            sims = [float(np.dot(e, ref_emb)) for e in embs]
            mean_s = np.mean(sims)
            min_mean_intra = min(min_mean_intra, mean_s)
            print(f"👤 Người dùng: '{user_name}' ({len(sims)} ảnh)")
            print(f"   🔹 Mean: {mean_s:.4f} | P5: {np.percentile(sims, 5):.4f} | Min: {np.min(sims):.4f}")

    # 2. Inter-class cross-similarity
    print("\n--- 2. KIỂM THỬ ĐỘ TƯƠNG ĐỒNG CHÉO (INTER-CLASS CROSS-SIMILARITY) ---")
    max_mean_inter = 0.0
    for u_src in user_names:
        refs = {}
        for u_dst in user_names:
            if u_dst == u_src:
                continue
            if u_dst in database:
                refs[u_dst] = np.array(database[u_dst]["embedding"], dtype=np.float32)
            elif user_embs[u_dst]:
                refs[u_dst] = trimmed_centroid(user_embs[u_dst])
        for u_dst, ref_dst in refs.items():
            for e in user_embs[u_src]:
                sim = float(np.dot(e, ref_dst))
                max_mean_inter = max(max_mean_inter, sim)
        cross_all = [float(np.dot(e, ref)) for ref in refs.values() for e in user_embs[u_src]]
        if cross_all:
            print(f"🔀 '{u_src}' vs khác người: mean={np.mean(cross_all):.4f} p95={np.percentile(cross_all, 95):.4f} max={np.max(cross_all):.4f}")

    # 3. Separation margin
    print("\n--- 3. ĐÁNH GIÁ ĐỘ PHÂN CÁCH (SEPARATION MARGIN) ---")
    margin = min_mean_intra - max_mean_inter
    print(f"📐 Margin = {margin:.4f} ({margin*100:.1f}%)")
    if margin > 0.04:
        print("✅ ĐÁNH GIÁ: XUẤT SẮC! Khoảng cách trung bình an toàn.")
    elif margin > 0.02:
        print("⚠️ ĐÁNH GIÁ: TỐT, nhưng 2 người dùng có nét khá giống nhau đối với AI.")
    else:
        print("❌ ĐÁNH GIÁ: Quá giống nhau! Hãy thử chụp lại ảnh ở các góc sáng sủa hơn.")

    # 4. GALLERY/PROBE HOLD-OUT + TAR/FAR (KE_HOACH 2.2.5, README 84-110)
    print("\n--- 4. TAR/FAR VỚI GALLERY/PROBE/IMPOSTOR ĐỘC LẬP ---")
    print("   (mỗi người: 80% ảnh làm gallery, 20% còn lại làm probe; người lạ = ảnh khác người + Impostor_*)")

    thresholds = np.arange(0.60, 0.96, 0.01)
    probe_scores = []    # (label=1, score)
    impostor_scores = []
    holdout = {}         # user -> {"templates": [...], "probe": [...]} cho mục 5 (identification)

    for user_name in user_names:
        if len(user_embs[user_name]) < 2:
            print(f"   ⚠️ Bỏ qua {user_name}: chỉ có {len(user_embs[user_name])} ảnh (cần ≥2 để tách gallery/probe)")
            continue
        centroid, probe, gallery_list = split_gallery_probe(user_embs[user_name])
        if centroid is None:
            continue
        holdout[user_name] = {"templates": list(gallery_list), "probe": list(probe)}
        for e in probe:
            probe_scores.append(float(np.dot(e, centroid)))
        # Impostor cho user này: ảnh của mọi người dùng KHÁC + Impostor_*
        for other in user_names:
            if other == user_name:
                continue
            for e in user_embs[other]:
                impostor_scores.append(float(np.dot(e, centroid)))
        for e in impostor_embs:
            impostor_scores.append(float(np.dot(e, centroid)))

    if not probe_scores:
        print("❌ Không đủ dữ liệu để tạo probe. Mỗi người cần ≥2 ảnh; tách lại nếu cần.")
        return

    probe_scores = np.array(probe_scores)
    impostor_scores = np.array(impostor_scores) if impostor_scores else np.array([])

    best = None
    for t in thresholds:
        tar = float(np.mean(probe_scores >= t)) if len(probe_scores) else float("nan")
        far = float(np.mean(impostor_scores >= t)) if len(impostor_scores) else float("nan")
        delta = (tar if not np.isnan(tar) else 0.0) - (far if not np.isnan(far) else 0.0)
        if best is None or delta > best[3]:
            best = (float(t), tar, far, delta)
    t_best, tar_best, far_best, _ = best

    OP_THRESHOLD = 0.60  # ngưỡng đang dùng trong hệ thống (main.py + FACE_THRESHOLD trên ESP32)
    tar_op = float(np.mean(probe_scores >= OP_THRESHOLD)) if len(probe_scores) else float("nan")
    far_op = float(np.mean(impostor_scores >= OP_THRESHOLD)) if len(impostor_scores) else float("nan")
    tar88 = float(np.mean(probe_scores >= 0.88)) if len(probe_scores) else float("nan")
    far88 = float(np.mean(impostor_scores >= 0.88)) if len(impostor_scores) else float("nan")

    print(f"   Số mẫu probe: {len(probe_scores)} | Số mẫu impostor: {len(impostor_scores)}")
    print(f"   Đề xuất tối ưu (max TAR-FAR): threshold={t_best:.2f} | TAR={tar_best:.4f} | FAR={far_best:.4f}")
    print(f"   Tại 0.60 (đang dùng):            TAR={tar_op:.4f} | FAR={far_op:.4f}")
    print(f"   Tại 0.88 (cũ):                   TAR={tar88:.4f} | FAR={far88:.4f}")
    print(f"   Ghi chú: FAR ở mục này đo bằng CENTROID (đơn giản hơn hệ thống thật);")
    print(f"   xem mục 5 — Identification Accuracy dùng đúng logic MAX-SIM của hệ thống.")

    # 5. IDENTIFICATION ACCURACY (top-1) — ĐÚNG LOGIC HỆ THỐNG
    #    Mỗi probe được so với TẤT CẢ templates của MỌI người (MAX-SIM), chọn người
    #    có điểm cao nhất (argmax) — giống face_recognizer.py (Laptop) và
    #    identify_face() trên ESP32. Đây là chỉ số thật của hệ thống đa người dùng.
    print("\n--- 5. IDENTIFICATION ACCURACY (argmax MAX-SIM — đồng bộ Laptop ↔ ESP32) ---")
    id_results = []  # (true_user, pred_user, best_score)
    for true_user, ho in holdout.items():
        for e in ho["probe"]:
            pred_user, best_s = "Unknown", -1.0
            for u, tlist in holdout.items():
                for t in tlist["templates"]:
                    s = float(np.dot(e, np.asarray(t, dtype=np.float32)))
                    if s > best_s:
                        best_s, pred_user = s, u
            id_results.append((true_user, pred_user, best_s))

    total_id = len(id_results)
    correct_id = sum(1 for t, p, _ in id_results if t == p)
    print(f"   Top-1 đúng người: {correct_id}/{total_id} = {correct_id/total_id*100:.1f}%" if total_id else "   Không có probe")
    wrong = [(t, p, s) for t, p, s in id_results if t != p]
    if wrong:
        print("   ⚠️ Các cặp NHẦM LẪN (người thật → bị nhận thành):")
        for t, p, s in wrong:
            print(f"      {t} → {p} (score {s:.3f})")
    # Tại ngưỡng hệ thống: probe được CHẤP NHẬN đúng người (argmax đúng + score ≥ ngưỡng)
    op_ok = sum(1 for t, p, s in id_results if t == p and s >= OP_THRESHOLD)
    print(f"   Tại ngưỡng {OP_THRESHOLD:.2f}: nhận đúng + đủ ngưỡng = {op_ok}/{total_id} = {op_ok/total_id*100:.1f}%" if total_id else "")
    low = [(t, s) for t, p, s in id_results if t == p and s < OP_THRESHOLD]
    if low:
        print(f"   ⚠️ {len(low)} probe ĐÚNG NGƯỜI nhưng ĐIỂM < {OP_THRESHOLD:.2f} (bị bỏ qua):")
        for t, s in low:
            print(f"      {t} (score {s:.3f})")
    print(f"   Ghi chú: FAR với NGƯỜI LẠI THẬT cần thêm ảnh vào data/registered_faces/Impostor_XX/")

    # 7. [4.2] PER-IDENTITY THRESHOLD + TAR@FAR (đồng bộ generate_embeddings.py)
    #    - Ngưỡng riêng từng người: max cross-sim (gallery của người này vs người khác) + margin
    #    - Ngưỡng an toàn FAR=0 trên tập impostor hiện có: max(impostor_scores)
    #    - OPIS-lite: FRR từng người tại ngưỡng global — đo "threshold inconsistency"
    print("\n--- 7. PER-IDENTITY THRESHOLD + TAR@FAR (mục 4.2 README) ---")
    PER_ID_MARGIN = 0.02
    PER_ID_CAP = 0.80
    if len(impostor_scores):
        far0_thresh = float(np.max(impostor_scores)) + 0.01
        print(f"   Ngưỡng an toàn FAR=0 trên {len(impostor_scores)} mẫu impostor: {far0_thresh:.3f}")
    for u, ho in holdout.items():
        u_templates = [np.asarray(t, dtype=np.float32) for t in ho["templates"]]
        cross_max = 0.0
        for v, ho_v in holdout.items():
            if v == u:
                continue
            for t_v in ho_v["templates"]:
                tv = np.asarray(t_v, dtype=np.float32)
                for t_u in u_templates:
                    s = float(np.dot(t_u, tv))
                    if s > cross_max:
                        cross_max = s
        thr = min(PER_ID_CAP, max(OP_THRESHOLD, cross_max + PER_ID_MARGIN))
        frr_u = float(np.mean([np.dot(e, np.asarray(t, dtype=np.float32)) < OP_THRESHOLD
                               for e in ho["probe"] for t in u_templates])) if ho["probe"] else float("nan")
        db_thr = database.get(u, {}).get("threshold") if database else None
        db_str = f" | DB: {db_thr}" if db_thr is not None else ""
        print(f"   👤 {u}: cross_max={cross_max:.3f} → ngưỡng đề xuất {thr:.3f} | FRR@{OP_THRESHOLD}={frr_u*100:.1f}%{db_str}")
    print("   (Ngưỡng đã được ghi vào face_database.json/.h bởi generate_embeddings.py —")
    print("    matching hiệu lực = max(0.60, ngưỡng riêng) trên CẢ Laptop và ESP32.)")

    print("\n--- 6. KẾT LUẬN THRESHOLD ---")
    print("   Chọn threshold trên tập dev (tối ưu TAR-FAR + Identification), sau đó KHÓA và")
    print("   chạy lại trên tập test độc lập. Nếu có cặp nhầm lẫn ở mục 5, xem lại ảnh 2 người")
    print("   đó (góc/ánh sáng) hoặc nâng ngưỡng trên mức max cross-similarity của cặp đó.")
    print("==================================================================")


if __name__ == "__main__":
    evaluate()
