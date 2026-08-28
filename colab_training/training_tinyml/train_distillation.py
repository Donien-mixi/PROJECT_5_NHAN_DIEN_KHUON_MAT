"""
==============================================================================
🎓 HUẤN LUYỆN UNIVERSAL FEATURE EXTRACTOR
   KNOWLEDGE DISTILLATION TỪ SFACE TEACHER TRÊN TẬP DỮ LIỆU LFW
==============================================================================
Chiến lược:
  - Teacher: SFace (OpenCV, pretrained trên 10,000+ danh tính)
  - Student: Ghost-TinyFace (64x64 INT8 cho ESP32-S3)
  - Dữ liệu: LFW (~5,000+ ảnh, ~1,600+ danh tính) — KHÔNG dùng ảnh của người dùng.
  - Loss: Cosine Distance + MSE + Hard Negative Mining (Triplet-style)
  - Mục tiêu: Student học cách trích xuất đặc trưng khuôn mặt TỔNG QUÁT,
    không phụ thuộc vào bối cảnh hay điều kiện ánh sáng cụ thể.

Sau khi train xong:
  - Đóng băng mô hình Student (Universal Extractor).
  - Chỉ dùng ảnh 3 người dùng để tạo Database Vector (Matching), KHÔNG train lại.
==============================================================================
"""

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
from tensorflow.keras import optimizers

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_tinyml.models.ghost_tinyface import build_tinyface_ghost
from training_tinyml.dataset_loader import FaceDatasetLoader


# ==============================================================================
# AUGMENTATION MODULE (TÁI SỬ DỤNG TỪ dataset_loader.py)
# ==============================================================================
def augment_image(image_np):
    """
    Data Augmentation chuyên sâu cho khuôn mặt (tái sử dụng logic từ FaceDatasetLoader).
    Input: ảnh grayscale uint8 (H, W)
    Output: ảnh grayscale uint8 (H, W) đã augment
    """
    img = image_np.copy()
    h, w = img.shape

    # 1. Xoay và Scale/Tịnh tiến nhẹ
    angle = np.random.uniform(-10.0, 10.0)
    scale = np.random.uniform(0.92, 1.08)
    tx = np.random.uniform(-3.0, 3.0)
    ty = np.random.uniform(-3.0, 3.0)
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # 2. Lật ngang ngẫu nhiên (quan trọng cho face recognition)
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)

    # 3. Biến đổi Gamma (mô phỏng ánh sáng)
    if np.random.rand() > 0.35:
        gamma = np.random.uniform(0.60, 1.50)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        img = cv2.LUT(img, table)

    # 4. Thay đổi Contrast & Brightness
    alpha = np.random.uniform(0.70, 1.30)
    beta = np.random.uniform(-30, 30)
    img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

    # 5. Làm mờ nhẹ ngẫu nhiên (Motion Blur / Defocus)
    if np.random.rand() > 0.55:
        ksize = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), np.random.uniform(0.3, 1.2))

    # 6. Random Cutout (che 1-2 ô nhỏ giả lập kính/bóng râm/tóc)
    if np.random.rand() > 0.55:
        for _ in range(np.random.randint(1, 3)):
            rw = np.random.randint(4, 14)
            rh = np.random.randint(4, 14)
            rx = np.random.randint(1, max(2, w - rw - 1))
            ry = np.random.randint(1, max(2, h - rh - 1))
            fill_val = np.random.randint(20, 200)
            img[ry:ry+rh, rx:rx+rw] = fill_val

    # 7. Nhiễu Gaussian hạt camera
    if np.random.rand() > 0.45:
        noise = np.random.normal(0, np.random.uniform(2, 8), img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img


# ==============================================================================
# HARD NEGATIVE MINING LOSS
# ==============================================================================
def compute_hard_negative_loss(student_embeddings, identity_labels, margin=0.4):
    """
    Hard Negative Mining Loss (Triplet-style):
    Trong mỗi batch, tìm cặp ảnh khác danh tính nhưng có embedding gần nhau nhất.
    Phạt nặng nếu khoảng cách < margin để ép mô hình đẩy các danh tính ra xa.
    
    Công thức: L_hn = mean(max(0, margin - ||emb_i - emb_j||₂))
    
    Tham số:
        student_embeddings: (batch, 128) — đã L2-normalize
        identity_labels: (batch,) — nhãn danh tính
        margin: Khoảng cách tối thiểu yêu cầu giữa 2 danh tính khác nhau
    """
    # Chuẩn hóa L2
    emb_norm = tf.nn.l2_normalize(student_embeddings, axis=-1)
    
    # Ma trận khoảng cách Euclidean: ||a - b||₂² = ||a||² + ||b||² - 2*a·b
    # Vì đã L2-normalize: ||a||² = ||b||² = 1, nên: dist² = 2 - 2*cosine
    dot_product = tf.matmul(emb_norm, emb_norm, transpose_b=True)  # (batch, batch)
    distances = tf.sqrt(tf.maximum(2.0 - 2.0 * dot_product, 1e-8))  # (batch, batch)
    
    # Tạo mask: True nếu 2 ảnh thuộc KHÁC danh tính
    labels_equal = tf.equal(tf.expand_dims(identity_labels, 0),
                            tf.expand_dims(identity_labels, 1))
    diff_identity_mask = tf.logical_not(labels_equal)  # (batch, batch)
    diff_identity_mask = tf.cast(diff_identity_mask, tf.float32)
    
    # Chỉ tính loss cho các cặp khác danh tính
    # Thay các cặp cùng danh tính bằng giá trị lớn (không ảnh hưởng đến min)
    masked_distances = distances * diff_identity_mask + (1.0 - diff_identity_mask) * 999.0
    
    # Tìm Hard Negative: cặp khác danh tính GẦN NHẤT cho mỗi anchor
    hardest_negative_dist = tf.reduce_min(masked_distances, axis=1)  # (batch,)
    
    # Hinge Loss: phạt nếu khoảng cách < margin
    hn_loss = tf.maximum(margin - hardest_negative_dist, 0.0)
    
    # Chỉ tính trung bình trên các anchor có hard negative hợp lệ (có ít nhất 1 cặp khác danh tính)
    has_negative = tf.reduce_any(tf.cast(diff_identity_mask, tf.bool), axis=1)
    has_negative = tf.cast(has_negative, tf.float32)
    
    valid_count = tf.maximum(tf.reduce_sum(has_negative), 1.0)
    loss = tf.reduce_sum(hn_loss * has_negative) / valid_count
    
    return loss


# ==============================================================================
# HÀM HUẤN LUYỆN CHÍNH
# ==============================================================================
def train_universal_distillation(epochs=50, batch_size=32, learning_rate=5e-4, 
                                  augment_repeat=8, warmup_epochs=5):
    """
    Huấn luyện Universal Feature Extractor qua Knowledge Distillation.
    
    Pipeline:
    1. Tải toàn bộ ảnh LFW đã tiền xử lý (64x64 Grayscale + 112x112 BGR).
    2. SFace Teacher trích xuất embedding 128-D cho mỗi ảnh (trên bản 112x112).
    3. Ghost-TinyFace Student học bắt chước Teacher trên bản 64x64 Grayscale.
    4. Loss = α*Cosine + β*MSE + γ*HardNegative
    5. Lưu mô hình Student đã thông minh → Sẵn sàng lượng tử hóa INT8.
    
    Tham số:
        epochs: Số vòng huấn luyện (mặc định: 50)
        batch_size: Kích thước batch (mặc định: 32, tăng lên nếu RAM cho phép)
        learning_rate: Tốc độ học tối đa (mặc định: 5e-4)
        augment_repeat: Số lần nhân bản mỗi ảnh với augmentation (mặc định: 8)
        warmup_epochs: Số epoch khởi động (learning rate tăng dần từ 0)
    """
    print("==================================================================")
    print("🎓 HUẤN LUYỆN UNIVERSAL FEATURE EXTRACTOR")
    print("   (KNOWLEDGE DISTILLATION TỪ SFACE TRÊN TẬP DỮ LIỆU LFW)")
    print("==================================================================")
    print(f"   Epochs: {epochs} | Batch Size: {batch_size} | LR: {learning_rate}")
    print(f"   Augment Repeat: {augment_repeat}x | Warmup: {warmup_epochs} epochs")
    print("==================================================================")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    
    # =========================================================================
    # BƯỚC 1: Khởi tạo SFace Teacher Model
    # =========================================================================
    teacher_model_path = os.path.join(current_dir, "weights", "face_recognition_sface_2021dec.onnx")
    
    if not os.path.exists(teacher_model_path):
        print(f"[*] Đang tải mô hình SFace Teacher...")
        import urllib.request
        os.makedirs(os.path.dirname(teacher_model_path), exist_ok=True)
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
        urllib.request.urlretrieve(url, teacher_model_path)
    
    teacher = cv2.FaceRecognizerSF.create(teacher_model_path, "")
    print("[+] Đã khởi tạo SFace Teacher Model thành công!")

    # =========================================================================
    # BƯỚC 2: Nạp toàn bộ ảnh LFW đã tiền xử lý
    # =========================================================================
    lfw_dir = os.path.join(base_dir, "data", "lfw_aligned")
    teacher_112_dir = os.path.join(lfw_dir, "_teacher_112x112")
    
    if not os.path.exists(lfw_dir) or not os.path.exists(teacher_112_dir):
        print("❌ LỖI: Chưa tải LFW dataset. Chạy download_lfw_dataset.py trước!")
        return
    
    # Liệt kê tất cả danh tính (bỏ qua thư mục _teacher)
    identity_dirs = [
        d for d in os.listdir(lfw_dir)
        if os.path.isdir(os.path.join(lfw_dir, d)) and not d.startswith("_")
    ]
    identity_dirs.sort()
    
    identity_to_idx = {name: idx for idx, name in enumerate(identity_dirs)}
    
    print(f"[+] Tìm thấy {len(identity_dirs)} danh tính LFW để huấn luyện.")

    # =========================================================================
    # BƯỚC 3: Trích xuất Teacher Embeddings + Chuẩn bị cặp (Student Input, Teacher Target)
    # =========================================================================
    print(f"\n[*] Đang trích xuất đặc trưng Teacher và chuẩn bị dữ liệu huấn luyện...")
    print(f"    (Mỗi ảnh gốc x {augment_repeat} biến thể augmentation)")
    
    all_student_inputs = []   # Ảnh 64x64 Grayscale chuẩn hóa [-1, 1]
    all_teacher_targets = []  # Vector embedding 128-D từ SFace Teacher
    all_identity_labels = []  # Nhãn danh tính (cho Hard Negative Mining)
    
    processed_count = 0
    skipped_count = 0
    
    for identity_name in identity_dirs:
        identity_idx = identity_to_idx[identity_name]
        
        # Lấy đường dẫn ảnh 64x64 Grayscale (cho Student)
        student_img_dir = os.path.join(lfw_dir, identity_name)
        student_img_paths = sorted(
            glob.glob(os.path.join(student_img_dir, "*.jpg")) +
            glob.glob(os.path.join(student_img_dir, "*.png"))
        )
        
        # Lấy đường dẫn ảnh 112x112 BGR (cho Teacher)
        teacher_img_dir = os.path.join(teacher_112_dir, identity_name)
        teacher_img_paths = sorted(
            glob.glob(os.path.join(teacher_img_dir, "*.jpg")) +
            glob.glob(os.path.join(teacher_img_dir, "*.png"))
        )
        
        if len(student_img_paths) == 0 or len(teacher_img_paths) == 0:
            continue
        
        # Ghép cặp theo thứ tự (student_path[i] <-> teacher_path[i])
        num_pairs = min(len(student_img_paths), len(teacher_img_paths))
        
        for i in range(num_pairs):
            # Đọc ảnh Student (64x64 Grayscale)
            student_gray = cv2.imread(student_img_paths[i], cv2.IMREAD_GRAYSCALE)
            if student_gray is None:
                skipped_count += 1
                continue
            student_gray = cv2.resize(student_gray, (64, 64), interpolation=cv2.INTER_AREA)
            
            # Đọc ảnh Teacher (112x112 BGR)
            teacher_bgr = cv2.imread(teacher_img_paths[i])
            if teacher_bgr is None:
                skipped_count += 1
                continue
            teacher_bgr = cv2.resize(teacher_bgr, (112, 112), interpolation=cv2.INTER_AREA)
            
            # Nhân bản với augmentation
            for aug_idx in range(augment_repeat):
                if aug_idx == 0:
                    # Bản gốc (không augment)
                    aug_gray = student_gray.copy()
                    aug_teacher_bgr = teacher_bgr.copy()
                else:
                    # Augment ảnh Student
                    aug_gray = augment_image(student_gray)
                    # Tạo bản BGR 112x112 tương ứng cho Teacher
                    # (Augment riêng ảnh grayscale rồi scale lên BGR cho Teacher)
                    aug_teacher_bgr = cv2.cvtColor(
                        cv2.resize(aug_gray, (112, 112), interpolation=cv2.INTER_LINEAR),
                        cv2.COLOR_GRAY2BGR
                    )
                
                # Teacher trích xuất embedding 128-D
                teacher_feat = teacher.feature(aug_teacher_bgr)[0]
                teacher_feat = teacher_feat / (np.linalg.norm(teacher_feat) + 1e-7)
                
                # Chuẩn hóa ảnh Student về [-1.0, 1.0]
                norm_64 = (aug_gray.astype(np.float32) - 127.5) / 128.0
                norm_64 = np.expand_dims(norm_64, axis=-1)  # (64, 64, 1)
                
                all_student_inputs.append(norm_64)
                all_teacher_targets.append(teacher_feat)
                all_identity_labels.append(identity_idx)
        
        processed_count += num_pairs
        
        if processed_count % 200 == 0:
            print(f"    [{processed_count} ảnh gốc xử lý] "
                  f"({len(all_student_inputs)} mẫu tổng cộng)...")
    
    # Xáo trộn dữ liệu
    print(f"\n[*] Đang xáo trộn {len(all_student_inputs)} mẫu huấn luyện...")
    
    indices = np.random.permutation(len(all_student_inputs))
    X_train = np.array(all_student_inputs, dtype=np.float32)[indices]
    Y_teacher = np.array(all_teacher_targets, dtype=np.float32)[indices]
    Z_labels = np.array(all_identity_labels, dtype=np.int32)[indices]
    
    # Giải phóng bộ nhớ
    del all_student_inputs, all_teacher_targets, all_identity_labels
    
    print(f"\n[+] Tổng số mẫu huấn luyện: {len(X_train)} cặp (Student ↔ Teacher)")
    print(f"    ({processed_count} ảnh gốc x {augment_repeat} augmentations)")
    print(f"    Bỏ qua: {skipped_count} ảnh lỗi")
    
    # =========================================================================
    # BƯỚC 4: Xây dựng mạng Student Ghost-TinyFace
    # =========================================================================
    student = build_tinyface_ghost(input_shape=(64, 64, 1), embedding_dim=128)
    student.summary()
    
    # =========================================================================
    # BƯỚC 5: Cấu hình Optimizer với Warmup + Cosine Annealing
    # =========================================================================
    steps_per_epoch = len(X_train) // batch_size
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    
    # Learning Rate Schedule: Linear Warmup → Cosine Decay
    # Warmup: LR tăng tuyến tính từ 0 → learning_rate trong warmup_steps đầu
    # Sau đó: Cosine Decay từ learning_rate → learning_rate * 0.01
    class WarmupCosineDecay(optimizers.schedules.LearningRateSchedule):
        def __init__(self, max_lr, warmup_steps, total_steps):
            super().__init__()
            self.max_lr = max_lr
            self.warmup_steps = warmup_steps
            self.total_steps = total_steps
            
        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            warmup = tf.cast(self.warmup_steps, tf.float32)
            total = tf.cast(self.total_steps, tf.float32)
            
            # Giai đoạn Warmup: LR tăng tuyến tính
            warmup_lr = self.max_lr * (step / tf.maximum(warmup, 1.0))
            
            # Giai đoạn Cosine Decay
            decay_step = (step - warmup) / tf.maximum(total - warmup, 1.0)
            decay_step = tf.minimum(decay_step, 1.0)
            cosine_lr = self.max_lr * 0.5 * (1.0 + tf.cos(np.pi * decay_step))
            cosine_lr = tf.maximum(cosine_lr, self.max_lr * 0.01)
            
            return tf.where(step < warmup, warmup_lr, cosine_lr)
        
        def get_config(self):
            return {
                "max_lr": self.max_lr,
                "warmup_steps": self.warmup_steps,
                "total_steps": self.total_steps
            }
    
    lr_schedule = WarmupCosineDecay(
        max_lr=learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )
    optimizer = optimizers.Adam(learning_rate=lr_schedule)
    
    print(f"\n[+] Cấu hình Optimizer:")
    print(f"    Steps/Epoch: {steps_per_epoch}")
    print(f"    Total Steps: {total_steps}")
    print(f"    Warmup Steps: {warmup_steps} ({warmup_epochs} epochs)")
    
    # =========================================================================
    # BƯỚC 6: Vòng lặp huấn luyện tùy chỉnh (Custom Training Loop)
    # =========================================================================
    # Hệ số Loss:
    #   α = 1.0 (Cosine Distance - quan trọng nhất, giữ hướng vector)
    #   β = 0.5 (MSE - giữ biên độ)
    #   γ = 0.3 (Hard Negative Mining - đẩy xa cặp khác người)
    ALPHA = 1.0   # Cosine Loss weight
    BETA = 0.5    # MSE Loss weight
    GAMMA = 0.3   # Hard Negative Mining Loss weight
    HN_MARGIN = 0.5  # Margin cho Hard Negative (khoảng cách tối thiểu)
    
    print(f"\n[*] Hệ số Loss: α={ALPHA} (Cosine) + β={BETA} (MSE) + γ={GAMMA} (HN, margin={HN_MARGIN})")
    print(f"\n{'='*70}")
    print(f"   BẮT ĐẦU HUẤN LUYỆN UNIVERSAL DISTILLATION ({epochs} Epochs)")
    print(f"{'='*70}")
    
    # Tạo tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_teacher, Z_labels))
    dataset = dataset.shuffle(buffer_size=min(len(X_train), 10000))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 8  # Early stopping nếu loss không giảm sau 8 epochs
    
    @tf.function
    def train_step(batch_x, batch_y_teacher, batch_labels):
        with tf.GradientTape() as tape:
            # Forward pass Student
            student_emb = student(batch_x, training=True)
            
            # 1. Cosine Distance Loss
            t_norm = tf.nn.l2_normalize(batch_y_teacher, axis=-1)
            s_norm = tf.nn.l2_normalize(student_emb, axis=-1)
            cosine_sim = tf.reduce_sum(t_norm * s_norm, axis=-1)
            cosine_loss = tf.reduce_mean(1.0 - cosine_sim)
            
            # 2. MSE Loss (trên vector đã chuẩn hóa L2)
            mse_loss = tf.reduce_mean(tf.square(t_norm - s_norm))
            
            # 3. Hard Negative Mining Loss
            hn_loss = compute_hard_negative_loss(
                student_emb, batch_labels, margin=HN_MARGIN
            )
            
            # Tổng hợp Loss
            total_loss = ALPHA * cosine_loss + BETA * mse_loss + GAMMA * hn_loss
        
        # Backward pass
        grads = tape.gradient(total_loss, student.trainable_variables)
        
        # Gradient Clipping (chống gradient explosion)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        
        optimizer.apply_gradients(zip(grads, student.trainable_variables))
        
        return cosine_loss, mse_loss, hn_loss, total_loss

    for epoch in range(epochs):
        epoch_cosine_loss = []
        epoch_mse_loss = []
        epoch_hn_loss = []
        epoch_total_loss = []
        
        for batch_idx, (batch_x, batch_y_teacher, batch_labels) in enumerate(dataset):
            c_loss, m_loss, h_loss, t_loss = train_step(batch_x, batch_y_teacher, batch_labels)
            
            epoch_cosine_loss.append(c_loss)
            epoch_mse_loss.append(m_loss)
            epoch_hn_loss.append(h_loss)
            epoch_total_loss.append(t_loss)
        
        # Thống kê epoch
        avg_total = np.mean(epoch_total_loss)
        avg_cosine = np.mean(epoch_cosine_loss)
        avg_mse = np.mean(epoch_mse_loss)
        avg_hn = np.mean(epoch_hn_loss)
        
        current_lr = optimizer.learning_rate
        if hasattr(current_lr, '__call__'):
            current_lr = current_lr(optimizer.iterations).numpy()
        elif hasattr(current_lr, 'numpy'):
            current_lr = current_lr.numpy()
        
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Loss: {avg_total:.4f} "
              f"(Cos: {avg_cosine:.4f}, MSE: {avg_mse:.4f}, HN: {avg_hn:.4f}) | "
              f"LR: {current_lr:.6f}")
        
        # Early Stopping Check
        if avg_total < best_loss:
            best_loss = avg_total
            patience_counter = 0
            # Lưu checkpoint tốt nhất
            best_weights = student.get_weights()
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"\n[!] Early Stopping: Loss không giảm sau {patience_limit} epochs liên tiếp.")
                print(f"    Khôi phục trọng số tốt nhất (Best Loss: {best_loss:.4f})")
                student.set_weights(best_weights)
                break
    
    # =========================================================================
    # BƯỚC 7: Lưu mô hình Universal Backbone
    # =========================================================================
    weights_dir = os.path.join(current_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    backbone_path = os.path.join(weights_dir, "tinyface_backbone.keras")
    
    # Tạo mô hình sạch (uncompiled) để inference
    inference_student = build_tinyface_ghost(input_shape=(64, 64, 1), embedding_dim=128)
    inference_student.set_weights(student.get_weights())
    inference_student.save(backbone_path)
    
    print(f"\n{'='*70}")
    print(f"🎉 HUẤN LUYỆN UNIVERSAL FEATURE EXTRACTOR THÀNH CÔNG!")
    print(f"{'='*70}")
    print(f"📦 Mô hình đã lưu tại: {backbone_path}")
    print(f"📊 Best Loss đạt được: {best_loss:.4f}")
    print(f"\n💡 Mô hình này đã được dạy trên hàng nghìn khuôn mặt đa dạng (LFW).")
    print(f"   Nó giờ đây có khả năng nhận diện khuôn mặt TỔNG QUÁT,")
    print(f"   không phụ thuộc vào bối cảnh, ánh sáng hay vị trí chụp.")
    print(f"\n🔜 Tiếp theo: Chạy lượng tử hóa INT8 và tạo Database Vector cho 3 người dùng.")


if __name__ == "__main__":
    train_universal_distillation(
        epochs=50,
        batch_size=32,
        learning_rate=5e-4,
        augment_repeat=8,
        warmup_epochs=5
    )
