import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import cv2
import glob
import numpy as np
import tensorflow as tf

class FaceDatasetLoader:
    """
    Module nạp và tăng cường dữ liệu (Data Augmentation) cho bài toán Face Recognition.
    - Quét toàn bộ danh tính trong data/registered_faces/
    - Tự động sinh dữ liệu đối kháng (Negative classes) nếu chỉ có 1 người dùng.
    - Áp dụng Data Augmentation đa dạng để mô hình học tốt trên vi điều khiển thực tế.
    """
    def __init__(self, data_dir=None, target_size=(64, 64), batch_size=16):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.data_dir = os.path.join(base_dir, "data", "registered_faces")
        else:
            self.data_dir = data_dir
            
        self.target_size = target_size
        self.batch_size = batch_size
        self.classes = []
        self.class_to_idx = {}
        self.image_paths = []
        self.labels = []
        
        self._load_dataset_paths()

    def _load_dataset_paths(self):
        """Quét thư mục và gán nhãn cho từng danh tính."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)

        subdirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        subdirs.sort()
        
        # Nếu chỉ có 1 người dùng, tự động tạo 2 danh tính đối kháng (Negative identities)
        # để hàm mất mát ArcFace có thể tính toán góc phân cách giữa các người khác nhau
        if len(subdirs) < 2:
            print("[*] Phát hiện chỉ có 1 người dùng. Đang tự động tạo 2 lớp dữ liệu đối kháng (Synthetic Impostors)...")
            neg_dir_1 = os.path.join(self.data_dir, "Impostor_Sample_A")
            neg_dir_2 = os.path.join(self.data_dir, "Impostor_Sample_B")
            os.makedirs(neg_dir_1, exist_ok=True)
            os.makedirs(neg_dir_2, exist_ok=True)
            
            # Tạo các ảnh đối kháng giả lập (với các cấu trúc tần số khác nhau)
            for i in range(15):
                # Impostor A: Texture dạng lưới / noise có cấu trúc
                pattern_a = np.zeros((64, 64, 3), dtype=np.uint8)
                pattern_a[::4, :] = 180
                pattern_a[:, ::4] = 120
                noise_a = np.random.randint(50, 150, (64, 64, 3), dtype=np.uint8)
                cv2.imwrite(os.path.join(neg_dir_1, f"sample_a_{i:02d}.jpg"), cv2.addWeighted(pattern_a, 0.4, noise_a, 0.6, 0))
                
                # Impostor B: Texture dạng gradient tròn
                y, x = np.ogrid[:64, :64]
                dist_from_center = np.sqrt((x - 48)**2 + (y - 48)**2)
                pattern_b = np.clip(255 - dist_from_center * 3 + np.random.normal(0, 10, (64, 64)), 0, 255).astype(np.uint8)
                pattern_b = cv2.cvtColor(pattern_b, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(os.path.join(neg_dir_2, f"sample_b_{i:02d}.jpg"), pattern_b)
                
            subdirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
            subdirs.sort()

        self.classes = subdirs
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        for cls_name in self.classes:
            cls_dir = os.path.join(self.data_dir, cls_name)
            img_files = glob.glob(os.path.join(cls_dir, "*.jpg")) + glob.glob(os.path.join(cls_dir, "*.png"))
            for img_path in img_files:
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])

        print(f"[+] Tìm thấy {len(self.classes)} danh tính với tổng cộng {len(self.image_paths)} ảnh trong cơ sở dữ liệu.")
        for cls_name, idx in self.class_to_idx.items():
            count = sum(1 for label in self.labels if label == idx)
            print(f"    - Class {idx}: '{cls_name}' ({count} ảnh)")

    def augment_image(self, image_np):
        """
        Data Augmentation thời gian thực:
        - Lật gương ngang ngẫu nhiên
        - Thay đổi độ sáng ngẫu nhiên (+-15%)
        - Thay đổi độ tương phản ngẫu nhiên (+-15%)
        - Thêm nhiễu Gaussian nhẹ (mô phỏng camera thực tế)
        """
        img = image_np.copy()
        
        # 1. Lật gương ngang 50%
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)

        # 2. Thay đổi độ sáng & tương phản
        alpha = np.random.uniform(0.85, 1.15) # Contrast
        beta = np.random.uniform(-15, 15)     # Brightness
        img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        # 3. Thêm nhiễu nhẹ 30%
        if np.random.rand() > 0.7:
            noise = np.random.normal(0, 5, img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)

        return img

    def get_tf_dataset(self, augment=True, repeat=10):
        """
        Chuyển đổi thành tf.data.Dataset hiệu năng cao.
        """
        if len(self.image_paths) == 0:
            raise ValueError(f"❌ Không tìm thấy ảnh nào trong {self.data_dir} để huấn luyện!")

        all_images = []
        all_labels = []

        # Nạp ảnh vào bộ nhớ RAM
        for path, label in zip(self.image_paths, self.labels):
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                continue
            # Chuyển sang Grayscale và resize
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.resize(img_gray, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Nhân bản dữ liệu với Data Augmentation
            for _ in range(repeat if augment else 1):
                if augment:
                    aug_img = self.augment_image(img_gray)
                else:
                    aug_img = img_gray
                    
                # Chuẩn hóa về dải [-1.0, 1.0] chuẩn cho Neural Networks
                norm_img = (aug_img.astype(np.float32) - 127.5) / 128.0
                norm_img = np.expand_dims(norm_img, axis=-1) # shape: (64, 64, 1)
                
                all_images.append(norm_img)
                all_labels.append(label)

        all_images = np.array(all_images, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int32)
        
        print(f"[+] Tạo Dataset thành công: {len(all_images)} mẫu huấn luyện (sau khi Augmentation).")
        
        # Tạo tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices(((all_images, all_labels), all_labels))
        dataset = dataset.shuffle(buffer_size=1024).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset, len(self.classes)


if __name__ == "__main__":
    loader = FaceDatasetLoader()
    ds, num_classes = loader.get_tf_dataset(augment=True, repeat=5)
    for (imgs, labels), target in ds.take(1):
        print(f"Batch Image Shape: {imgs.shape}")
        print(f"Batch Label Shape: {labels.shape}")
