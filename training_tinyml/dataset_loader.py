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
    Module nạp và tăng cường dữ liệu (Data Augmentation) mạnh mẽ cho TinyML Face Recognition.
    - Quét toàn bộ danh tính trong data/registered_faces/
    - Luôn tạo hệ thống 8 lớp đối kháng đa tần số (Multi-Anchor Impostor Classes)
      để đảm bảo không gian cầu 128D của ArcFace học phân tách hình học khuôn mặt thực thụ,
      chống học vẹt tuyệt đối dù chỉ có 1 hay 2 người dùng.
    - Áp dụng Data Augmentation đa dạng: Xoay, Zoom, Cutout, Contrast, Brightness, Noise.
    """
    def __init__(self, data_dir=None, target_size=(64, 64), batch_size=16):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(base_dir, "data", "registered_faces")
        else:
            self.data_dir = data_dir
            
        self.target_size = target_size
        self.batch_size = batch_size
        self.classes = []
        self.class_to_idx = {}
        self.image_paths = []
        self.labels = []
        self.synthetic_images = []
        self.synthetic_labels = []
        
        self._load_dataset_paths()

    def _generate_synthetic_impostors(self, start_idx=2, count=8, samples_per_class=30):
        """
        Sinh 8 lớp đối kháng nhân tạo (Synthetic Impostors) với các mẫu vân hình học & tần số khác nhau.
        Điều này ép ArcFace loss tạo các biên phân cách góc cực rộng cho các người dùng thực tế.
        """
        synth_imgs = []
        synth_lbls = []
        
        for k in range(count):
            cls_idx = start_idx + k
            cls_name = f"Synthetic_Anchor_{k+1}"
            self.classes.append(cls_name)
            self.class_to_idx[cls_name] = cls_idx
            
            for s in range(samples_per_class):
                h, w = self.target_size
                img = np.zeros((h, w), dtype=np.uint8)
                
                if k == 0:
                    # Anchor 1: Lưới kẻ ngang dọc
                    freq = np.random.randint(3, 8)
                    img[::freq, :] = np.random.randint(160, 240)
                    img[:, ::freq] = np.random.randint(100, 180)
                    noise = np.random.randint(20, 80, (h, w), dtype=np.uint8)
                    img = cv2.add(img, noise)
                elif k == 1:
                    # Anchor 2: Gradient tròn từ tâm (Radial)
                    cx, cy = np.random.randint(24, 40), np.random.randint(24, 40)
                    y, x = np.ogrid[:h, :w]
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    img = np.clip(255 - dist * np.random.uniform(3.5, 6.0), 0, 255).astype(np.uint8)
                elif k == 2:
                    # Anchor 3: Sọc chéo (Diagonal stripes)
                    angle = np.random.uniform(0.3, 1.2)
                    y, x = np.mgrid[:h, :w]
                    pattern = np.sin((x * np.cos(angle) + y * np.sin(angle)) * 0.4) * 127 + 128
                    img = pattern.astype(np.uint8)
                elif k == 3:
                    # Anchor 4: Đốm Gaussian đa tâm (Blobs)
                    for _ in range(np.random.randint(3, 7)):
                        bx, by = np.random.randint(10, 54), np.random.randint(10, 54)
                        rad = np.random.randint(6, 16)
                        cv2.circle(img, (bx, by), rad, int(np.random.randint(120, 255)), -1)
                    img = cv2.GaussianBlur(img, (9, 9), 3.0)
                elif k == 4:
                    # Anchor 5: Bàn cờ (Checkerboard)
                    bs = np.random.randint(4, 10)
                    for r in range(0, h, bs):
                        for c in range(0, w, bs):
                            if ((r // bs) + (c // bs)) % 2 == 0:
                                img[r:r+bs, c:c+bs] = np.random.randint(150, 240)
                elif k == 5:
                    # Anchor 6: Khối hình học ngẫu nhiên
                    for _ in range(np.random.randint(2, 5)):
                        pt1 = (np.random.randint(5, 45), np.random.randint(5, 45))
                        pt2 = (np.random.randint(20, 60), np.random.randint(20, 60))
                        cv2.rectangle(img, pt1, pt2, int(np.random.randint(80, 220)), -1)
                elif k == 6:
                    # Anchor 7: Gradient tuyến tính xoay góc
                    grad = np.linspace(20, 235, w, dtype=np.uint8)
                    img = np.tile(grad, (h, 1))
                    M = cv2.getRotationMatrix2D((w//2, h//2), np.random.randint(0, 360), 1.0)
                    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                else:
                    # Anchor 8: Nhiễu đốm hạt có tương quan không gian (Speckle)
                    low_res = np.random.randint(0, 255, (h//4, w//4), dtype=np.uint8)
                    img = cv2.resize(low_res, (h, w), interpolation=cv2.INTER_CUBIC)
                
                # Chuẩn hóa về [-1.0, 1.0]
                norm_img = (img.astype(np.float32) - 127.5) / 128.0
                norm_img = np.expand_dims(norm_img, axis=-1)
                synth_imgs.append(norm_img)
                synth_lbls.append(cls_idx)
                
        return synth_imgs, synth_lbls

    def _load_dataset_paths(self):
        """Quét thư mục và gán nhãn cho từng người dùng thực tế."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)

        # Lọc chỉ lấy thư mục người dùng thực tế (không phải Impostor hay Synthetic)
        subdirs = [
            d for d in os.listdir(self.data_dir) 
            if os.path.isdir(os.path.join(self.data_dir, d)) and not d.startswith("Impostor_") and not d.startswith("Synthetic_")
        ]
        subdirs.sort()
        
        self.classes = subdirs.copy()
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        for cls_name in subdirs:
            cls_dir = os.path.join(self.data_dir, cls_name)
            img_files = glob.glob(os.path.join(cls_dir, "*.jpg")) + glob.glob(os.path.join(cls_dir, "*.png"))
            for img_path in img_files:
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])

        print(f"[+] Tìm thấy {len(subdirs)} người dùng thực tế với tổng cộng {len(self.image_paths)} ảnh:")
        for cls_name, idx in self.class_to_idx.items():
            count = sum(1 for label in self.labels if label == idx)
            print(f"    - Class {idx}: '{cls_name}' ({count} ảnh)")

        # Luôn tạo thêm 8 lớp đối kháng đa dạng tần số
        num_real_classes = len(subdirs)
        print(f"[*] Đang khởi tạo 8 lớp đối kháng nhân tạo (Synthetic Anchors) để rèn luyện không gian hình học khuôn mặt...")
        self.synthetic_images, self.synthetic_labels = self._generate_synthetic_impostors(
            start_idx=num_real_classes, 
            count=8, 
            samples_per_class=35
        )
        print(f"[+] Tổng số lớp sau khi bổ sung đối kháng: {len(self.classes)} lớp.")

    def augment_image(self, image_np):
        """
        Data Augmentation chuyên sâu cho khuôn mặt đã căn chỉnh (Aligned Faces):
        - Xoay ngẫu nhiên (-8 độ đến +8 độ)
        - Phóng to/Thu nhỏ (Scale 0.94 - 1.06), tịnh tiến nhẹ (+-2 pixel)
        - Biến đổi cường độ chiếu sáng đa dạng (Gamma, Contrast, Brightness)
        - Làm mờ ngẫu nhiên (Gaussian Blur) mô phỏng camera rung lắc
        - Cutout ngẫu nhiên (che 1 ô nhỏ giả lập bóng râm/tóc)
        - Nhiễu hạt cảm biến Gaussian
        """
        img = image_np.copy()
        h, w = img.shape
        
        # 0. Lật ngang ngẫu nhiên (Horizontal Flip)
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)

        # 1. Xoay và Scale/Tịnh tiến nhẹ (Affine Transform giữ nguyên tâm mắt)
        angle = np.random.uniform(-15.0, 15.0)
        scale = np.random.uniform(0.94, 1.06)
        tx = np.random.uniform(-2.0, 2.0)
        ty = np.random.uniform(-2.0, 2.0)
        
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # 2. Biến đổi Gamma (Mô phỏng ánh sáng ngày vs phòng tối)
        if np.random.rand() > 0.4:
            gamma = np.random.uniform(0.70, 1.40)
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            img = cv2.LUT(img, table)

        # 3. Thay đổi độ sáng & tương phản
        alpha = np.random.uniform(0.75, 1.25) # Contrast
        beta = np.random.uniform(-25, 25)      # Brightness
        img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        # 4. Làm mờ nhẹ ngẫu nhiên (Motion Blur / Defocus)
        if np.random.rand() > 0.6:
            img = cv2.GaussianBlur(img, (3, 3), np.random.uniform(0.4, 1.0))

        # 5. Random Cutout (Che 1 ô nhỏ giả lập kính/bóng râm)
        if np.random.rand() > 0.6:
            rw = np.random.randint(5, 12)
            rh = np.random.randint(5, 12)
            rx = np.random.randint(2, w - rw - 2)
            ry = np.random.randint(2, h - rh - 2)
            fill_val = np.random.randint(30, 180)
            img[ry:ry+rh, rx:rx+rw] = fill_val

        # 6. Thêm nhiễu Gaussian hạt camera
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, np.random.uniform(2, 6), img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)

        return img

    def get_tf_dataset(self, augment=True, repeat=12):
        """
        Chuyển đổi thành tf.data.Dataset hiệu năng cao.
        """
        if len(self.image_paths) == 0:
            raise ValueError(f"❌ Không tìm thấy ảnh nào trong {self.data_dir} để huấn luyện!")

        all_images = []
        all_labels = []

        # Nạp ảnh của người dùng thật vào bộ nhớ RAM
        for path, label in zip(self.image_paths, self.labels):
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                continue
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.resize(img_gray, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Nhân bản dữ liệu với Data Augmentation
            for _ in range(repeat if augment else 1):
                if augment:
                    aug_img = self.augment_image(img_gray)
                else:
                    aug_img = img_gray
                    
                # Chuẩn hóa về dải [-1.0, 1.0]
                norm_img = (aug_img.astype(np.float32) - 127.5) / 128.0
                norm_img = np.expand_dims(norm_img, axis=-1)
                
                all_images.append(norm_img)
                all_labels.append(label)

        # Bổ sung các mẫu Synthetic Impostors
        all_images.extend(self.synthetic_images)
        all_labels.extend(self.synthetic_labels)

        all_images = np.array(all_images, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int32)
        
        print(f"[+] Tạo Dataset thành công: {len(all_images)} mẫu huấn luyện ({len(self.classes)} lớp).")
        
        # Tạo tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices(((all_images, all_labels), all_labels))
        dataset = dataset.shuffle(buffer_size=2048).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset, len(self.classes)


if __name__ == "__main__":
    loader = FaceDatasetLoader()
    ds, num_classes = loader.get_tf_dataset(augment=True, repeat=5)
    for (imgs, labels), target in ds.take(1):
        print(f"Batch Image Shape: {imgs.shape}")
        print(f"Batch Label Shape: {labels.shape}")
