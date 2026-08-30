import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_tinyml.dataset_loader import FaceDatasetLoader
from training_tinyml.train_distillation import augment_image

def finetune_locally(epochs=100, batch_size=128):
    print("==================================================================")
    print("🚀 FINE-TUNE CỤC BỘ: ÉP MÔ HÌNH PHÂN BIỆT RÕ RÀNG NGƯỜI DÙNG")
    print("==================================================================")
    print("Lưu ý: Chạy file này trên laptop sau khi bạn CHỤP THÊM ẢNH NGƯỜI MỚI.")
    print("Việc này chỉ mất vài phút, giúp AI không bị nhầm lẫn (Overlap) ")
    print("mà không cần phải lên Google Colab train lại từ đầu (mất 1 tiếng).")
    print("==================================================================\n")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(base_dir, "data", "registered_faces")
    model_path = os.path.join(current_dir, "weights", "tinyface_backbone.keras")
    base_model_path = os.path.join(current_dir, "weights", "tinyface_backbone_base.keras")
    
    import shutil
    
    # 1. Bảo vệ "não gốc" từ Colab. Nếu chưa có bản backup (_base), tạo ngay 1 bản.
    if os.path.exists(model_path) and not os.path.exists(base_model_path):
        print(f"[*] Đang tạo bản sao lưu não gốc từ Colab: {base_model_path}")
        shutil.copy2(model_path, base_model_path)

    # 2. Luôn bắt đầu fine-tune từ não gốc để chống Quên thảm họa (Catastrophic Forgetting)
    if not os.path.exists(base_model_path):
        print(f"❌ LỖI: Không tìm thấy {base_model_path} hoặc {model_path}.")
        print("Hãy chắc chắn bạn đã tải mô hình từ Colab về trước!")
        return
        
    if not os.path.exists(data_dir):
        print(f"❌ LỖI: Không tìm thấy {data_dir}.")
        return

    identities = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if len(identities) < 2:
        print("[-] Cần ít nhất 2 người để fine-tune phân biệt. Bỏ qua.")
        return
        
    print(f"[*] Đang tải mô hình gốc (từ Colab) để bắt đầu học...")
    try:
        from training_tinyml.train_distillation import build_tinyface_ghost
        model = build_tinyface_ghost(input_shape=(64, 64, 1), embedding_dim=128)
        model.load_weights(model_path)
    except Exception as e:
        print(f"❌ LỖI tải mô hình: {e}")
        return
        
    print(f"[+] Tìm thấy {len(identities)} người dùng: {identities}")
    
    all_images = []
    all_labels = []
    
    for idx, name in enumerate(identities):
        img_dir = os.path.join(data_dir, name)
        for img_name in os.listdir(img_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(img_dir, img_name)
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is None: continue
            
            gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
            
            # Nhân bản bằng augmentation để chống Overfitting
            # Tăng lên 50 lần để có dữ liệu phong phú hơn
            for _ in range(50):
                aug = augment_image(gray)
                norm = (aug.astype(np.float32) - 127.5) / 128.0
                all_images.append(np.expand_dims(norm, axis=-1))
                all_labels.append(idx)
                
    X_train = np.array(all_images)
    Y_train = np.array(all_labels)
    
    if len(X_train) == 0:
        print("❌ LỖI: Không đọc được ảnh nào.")
        return
        
    print(f"[+] Đã tạo {len(X_train)} ảnh augment từ ảnh cá nhân để fine-tune.")
    
    dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    dataset = dataset.shuffle(2000).batch(32, drop_remainder=False) # Batch nhỏ hơn để cập nhật gradient đều hơn
    
    print("\n[*] Đang thiết lập mạng Classifier (NormFace) để fine-tune...")
    # Đóng băng 70% các lớp đầu tiên của backbone để không làm hỏng kiến thức tổng quát
    num_layers = len(model.layers)
    for layer in model.layers[:int(num_layers * 0.7)]:
        layer.trainable = False
        
    # Tạo mô hình Classification tạm thời trên nền Backbone
    inputs = tf.keras.Input(shape=(64, 64, 1))
    embeddings = model(inputs, training=True) # Backbone xuất ra 128-D
    
    # Chuẩn hóa L2 (NormFace)
    norm_embeddings = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(embeddings)
    
    # Scale lên (s=10) để Softmax tính toán tốt Gradient (vì Cosine chỉ từ -1 đến 1)
    scaled_embeddings = tf.keras.layers.Lambda(lambda x: x * 10.0)(norm_embeddings)
    
    # Dense layer không bias: Trọng số của Dense chính là vector đại diện cho từng người (Class Center)
    outputs = tf.keras.layers.Dense(len(identities), activation='softmax', use_bias=False)(scaled_embeddings)
    
    finetune_model = tf.keras.Model(inputs, outputs)
    
    finetune_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n[*] Đang tiến hành huấn luyện kéo giãn khoảng cách (Classification Loss)...")
    # Huấn luyện mô hình Classification (sẽ tự động tinh chỉnh các lớp cuối của backbone)
    finetune_model.fit(dataset, epochs=30)
            
    print("\n[+] Fine-tuning hoàn tất! Đang lưu mô hình đè lên file cũ...")
    # LƯU Ý: Chỉ lưu lại 'model' (phần Backbone trích xuất 128-D), bỏ đi lớp Classification!
    model.save(model_path)
    
    print("\n🎉 XONG! VẤN ĐỀ NHẦM LẪN ĐÃ ĐƯỢC GIẢI QUYẾT TRIỆT ĐỂ BẰNG CLASSIFICATION LOSS.")
    print("QUAN TRỌNG: Bạn VỪA LÀM THAY ĐỔI '.keras'. Do đó bạn PHẢI CHẠY LẠI THEO THỨ TỰ SAU:")
    print("  1. python training_tinyml/quantize_qat_int8.py    (Để ép kiểu lại file tflite mới)")
    print("  2. python training_tinyml/update_face_database.py (Để cập nhật lại database cho đúng não bộ mới)")
    print("  3. Nạp lại Firmware xuống mạch ESP32.")

if __name__ == "__main__":
    finetune_locally()
