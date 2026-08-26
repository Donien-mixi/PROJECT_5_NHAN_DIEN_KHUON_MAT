import math
import tensorflow as tf
from tensorflow.keras import layers

class ArcFaceHead(layers.Layer):
    """
    ArcFace (Additive Angular Margin Loss) Head:
    Ép khoảng cách góc giữa các ảnh của cùng một người co cụm sát lại gần nhau,
    và đẩy khoảng cách giữa các người khác nhau ra xa tối đa trong không gian cầu 128D.
    
    Công thức: L = -log( e^{s * cos(theta_yi + m)} / (e^{s * cos(theta_yi + m)} + sum_{j!=yi} e^{s * cos(theta_j)}) )
    """
    def __init__(self, num_classes, embedding_dim=128, margin=0.5, scale=32.0, **kwargs):
        super(ArcFaceHead, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.scale = scale
        
        # Các hằng số tính toán lượng giác
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def build(self, input_shape):
        # Ma trận trọng số trọng tâm các lớp W (normalized)
        self.W = self.add_weight(
            name='arcface_weights',
            shape=(self.embedding_dim, self.num_classes),
            initializer='glorot_uniform',
            trainable=True
        )
        super(ArcFaceHead, self).build(input_shape)

    def call(self, inputs, training=None):
        # inputs gồm 2 thành phần: [embeddings (batch, 128), labels (batch,)]
        embeddings, labels = inputs
        
        # 1. Chuẩn hóa L2 vector embeddings và vector trọng số W
        embeddings = tf.math.l2_normalize(embeddings, axis=-1)
        norm_W = tf.math.l2_normalize(self.W, axis=0)
        
        # 2. Tính Cosine Similarity giữa embedding và trọng tâm các lớp: cos(theta) = e . W
        cosine = tf.matmul(embeddings, norm_W) # shape: (batch, num_classes)
        
        if not training:
            # Khi Inference chỉ cần nhân hệ số scale
            return cosine * self.scale
            
        # 3. Tính toán sin(theta) = sqrt(1 - cos^2(theta))
        sine = tf.math.sqrt(tf.clip_by_value(1.0 - tf.math.square(cosine), 1e-7, 1.0))
        
        # 4. Tính cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Xử lý vùng góc vượt quá pi
        phi = tf.where(cosine > self.th, phi, cosine - self.mm)
        
        # 5. Thay thế logits tại đúng nhãn lớp mục tiêu y_i bằng phi = cos(theta_yi + m)
        one_hot = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # 6. Phóng đại logits bằng hệ số bán kính s (Scale)
        output = output * self.scale
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'embedding_dim': self.embedding_dim,
            'margin': self.margin,
            'scale': self.scale
        })
        return config


class CosineDistillationLoss(tf.keras.losses.Loss):
    """
    Hàm mất mát chưng cất tri thức (Knowledge Distillation) dựa trên khoảng cách Cosine.
    Ép vector embedding của mạng học sinh (TinyFaceNet) phải có góc tương đồng với mạng giáo viên (Teacher Model).
    """
    def __init__(self, name='cosine_distill_loss', **kwargs):
        super(CosineDistillationLoss, self).__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        # y_true: embedding từ Teacher Model (batch, 128)
        # y_pred: embedding từ Student Model (batch, 128)
        # Chuẩn hóa L2 cả 2 vector
        t_norm = tf.math.l2_normalize(y_true, axis=-1)
        s_norm = tf.math.l2_normalize(y_pred, axis=-1)
        
        # Cosine Loss = 1.0 - cos(t, s)
        cos_sim = tf.reduce_sum(t_norm * s_norm, axis=-1)
        loss = 1.0 - cos_sim
        return tf.reduce_mean(loss)
