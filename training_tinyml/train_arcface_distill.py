import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses, metrics, callbacks

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_tinyml.models.ghost_tinyface import build_tinyface_ghost
from training_tinyml.losses.arcface import ArcFaceHead
from training_tinyml.dataset_loader import FaceDatasetLoader

def train_model(epochs=30, batch_size=16, learning_rate=1e-3):
    print("==================================================================")
    print("🚀 BẮT ĐẦU HUẤN LUYỆN TINYFACENET-GHOST VỚI ARCFACE LOSS")
    print("==================================================================")

    # 1. Nạp tập dữ liệu huấn luyện
    loader = FaceDatasetLoader(target_size=(64, 64), batch_size=batch_size)
    dataset, num_classes = loader.get_tf_dataset(augment=True, repeat=15)
    
    print(f"\n[+] Số lớp danh tính (Classes): {num_classes}")
    print(f"[+] Danh sách nhãn: {loader.classes}")

    # 2. Xây dựng mạng Backbone TinyFaceNet-Ghost
    backbone = build_tinyface_ghost(input_shape=(64, 64, 1), embedding_dim=128)
    
    # 3. Ghép nối tầng ArcFace Head cho quá trình huấn luyện
    image_input = layers.Input(shape=(64, 64, 1), name="train_image_input")
    label_input = layers.Input(shape=(), dtype=tf.int32, name="train_label_input")
    
    embeddings = backbone(image_input)
    arcface_logits = ArcFaceHead(
        num_classes=num_classes, 
        embedding_dim=128, 
        margin=0.5, 
        scale=32.0, 
        name="arcface_head"
    )([embeddings, label_input])
    
    train_model = Model(inputs=[image_input, label_input], outputs=arcface_logits, name="Train_ArcFace_Model")
    
    # 4. Cấu hình Optimizer và Learning Rate Scheduler
    lr_schedule = optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=epochs * 10,
        alpha=0.01
    )
    optimizer = optimizers.Adam(learning_rate=lr_schedule)
    
    loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
    acc_metric = metrics.SparseCategoricalAccuracy()

    train_model.compile(optimizer=optimizer, loss=loss_fn, metrics=[acc_metric])
    
    print("\n--- CẤU TRÚC MÔ HÌNH HUẤN LUYỆN ---")
    backbone.summary()

    # 5. Huấn luyện mô hình
    print(f"\n[*] Bắt đầu huấn luyện qua {epochs} Epochs...")
    history = train_model.fit(
        dataset,
        epochs=epochs,
        verbose=1
    )

    # 6. Lưu trữ trọng số
    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    backbone_path = os.path.join(weights_dir, "tinyface_backbone.keras")
    backbone.save(backbone_path)
    print(f"\n[+] ĐÃ LƯU THÀNH CÔNG MÔ HÌNH BACKBONE (CHO INFERENCE & LƯỢNG TỬ HÓA):")
    print(f"    👉 {backbone_path}")

    return backbone, loader.classes

if __name__ == "__main__":
    train_model(epochs=60, batch_size=16)
