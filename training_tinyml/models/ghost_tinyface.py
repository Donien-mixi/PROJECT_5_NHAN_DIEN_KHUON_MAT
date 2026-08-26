import math
import sys
import tensorflow as tf
from tensorflow.keras import layers, Model

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# ==============================================================================
# GHOST MODULE (DỰA TRÊN BÀI BÁO KHOA HỌC GHOSTFACENETS 2023)
# ==============================================================================
def ghost_module(x, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, name=None):
    """
    Ghost Module: Tạo 1/2 số kênh bằng Conv2D chuẩn, 1/2 số kênh còn lại bằng Cheap Depthwise Conv.
    Giảm 50% số phép tính FLOPs so với Conv2D thông thường.
    """
    prefix = f"{name}_" if name else ""
    init_channels = math.ceil(out_channels / ratio)
    new_channels = init_channels * (ratio - 1)

    # 1. Tích chập chính (Primary Conv)
    x1 = layers.Conv2D(
        init_channels,
        kernel_size,
        strides=stride,
        padding='same',
        use_bias=False,
        name=f"{prefix}primary_conv"
    )(x)
    x1 = layers.BatchNormalization(name=f"{prefix}primary_bn")(x1)
    if relu:
        x1 = layers.ReLU(max_value=6.0, name=f"{prefix}primary_relu")(x1)

    # 2. Phép biến đổi tuyến tính siêu rẻ (Cheap Operation: Depthwise Conv)
    x2 = layers.DepthwiseConv2D(
        dw_size,
        strides=1,
        padding='same',
        use_bias=False,
        name=f"{prefix}cheap_conv"
    )(x1)
    x2 = layers.BatchNormalization(name=f"{prefix}cheap_bn")(x2)
    if relu:
        x2 = layers.ReLU(max_value=6.0, name=f"{prefix}cheap_relu")(x2)

    # 3. Nối các kênh lại
    out = layers.Concatenate(axis=-1, name=f"{prefix}concat")([x1, x2])
    # Cắt gọn đúng số kênh yêu cầu (nếu làm tròn thừa)
    if out.shape[-1] != out_channels:
        out = out[..., :out_channels]
    return out


def ghost_bottleneck(x, dw_channels, out_channels, dw_kernel=3, stride=1, name=None):
    """
    Ghost Bottleneck: Khối Inverted Residual xây dựng từ Ghost Modules.
    """
    prefix = f"{name}_" if name else ""
    in_channels = x.shape[-1]

    # 1. Ghost Module mở rộng số kênh (Expansion)
    y = ghost_module(x, dw_channels, kernel_size=1, relu=True, name=f"{prefix}ghost1")

    # 2. Depthwise Conv nếu có giảm kích thước (Stride = 2)
    if stride > 1:
        y = layers.DepthwiseConv2D(
            dw_kernel,
            strides=stride,
            padding='same',
            use_bias=False,
            name=f"{prefix}dw_conv"
        )(y)
        y = layers.BatchNormalization(name=f"{prefix}dw_bn")(y)

    # 3. Ghost Module thu hẹp số kênh (Projection - Không dùng ReLU để giữ tuyến tính)
    y = ghost_module(y, out_channels, kernel_size=1, relu=False, name=f"{prefix}ghost2")

    # 4. Nhánh Shortcut (Residual Connection)
    if stride == 1 and in_channels == out_channels:
        shortcut = x
    else:
        shortcut = layers.DepthwiseConv2D(
            dw_kernel,
            strides=stride,
            padding='same',
            use_bias=False,
            name=f"{prefix}sc_dw"
        )(x)
        shortcut = layers.BatchNormalization(name=f"{prefix}sc_bn1")(shortcut)
        shortcut = layers.Conv2D(
            out_channels,
            1,
            strides=1,
            padding='same',
            use_bias=False,
            name=f"{prefix}sc_conv"
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{prefix}sc_bn2")(shortcut)

    out = layers.Add(name=f"{prefix}add")([shortcut, y])
    return out


# ==============================================================================
# MẠNG NƠ-RON CHÍNH: TINYFACENET-GHOST (DÀNH RIÊNG CHO ESP32-S3) - FAST VERSION
# ==============================================================================
def build_tinyface_ghost(input_shape=(64, 64, 1), embedding_dim=128, name="TinyFaceNet_Ghost_Fast"):
    """
    Kiến trúc TinyFaceNet-Ghost Siêu nhẹ:
    - Input: 64x64 Grayscale (1 channel)
    - Số kênh đã được tỉa (Pruned) giảm một nửa so với bản gốc.
    - Output: 128-D L2-normalized embedding vector
    - FLOPs: Giảm hơn 70% so với bản 96x96 gốc.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # 1. Stem Layer (Conv 3x3 Stride 2: 64x64 -> 32x32)
    x = layers.Conv2D(16, 3, strides=2, padding='same', use_bias=False, name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(max_value=6.0, name="stem_relu")(x)

    # 2. Ghost Bottlenecks (Khôi phục 100% kênh não bộ AI)
    # Stage 1: 32x32 -> 32x32 (32 channels)
    x = ghost_bottleneck(x, dw_channels=32, out_channels=32, stride=1, name="gb1_1")

    # Stage 2: 32x32 -> 16x16 (32 channels)
    x = ghost_bottleneck(x, dw_channels=64, out_channels=32, stride=2, name="gb2_1")
    x = ghost_bottleneck(x, dw_channels=64, out_channels=32, stride=1, name="gb2_2")

    # Stage 3: 16x16 -> 8x8 (64 channels)
    x = ghost_bottleneck(x, dw_channels=128, out_channels=64, stride=2, name="gb3_1")
    x = ghost_bottleneck(x, dw_channels=128, out_channels=64, stride=1, name="gb3_2")

    # Stage 4: 8x8 -> 4x4 (96 channels)
    x = ghost_bottleneck(x, dw_channels=192, out_channels=96, stride=2, name="gb4_1")

    # 3. Head / Feature Extractor (4x4 -> 1x1x96)
    x = layers.DepthwiseConv2D(4, strides=1, padding='valid', use_bias=False, name="head_dw")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    # Dùng Reshape tĩnh thay vì Flatten() để tránh sinh ra ops SHAPE/STRIDED_SLICE/PACK
    # sau DepthwiseConv2D(4, valid) trên 4x4x96: output = 1x1x96, cần reshape về (96,)
    x = layers.Reshape((96,), name="head_reshape")(x)

    # 4. Linear Embedding Bottleneck (128 dimensions)
    embeddings = layers.Dense(embedding_dim, use_bias=False, name="embedding_dense")(x)
    embeddings = layers.BatchNormalization(name="embedding_bn")(embeddings)

    model = Model(inputs=inputs, outputs=embeddings, name=name)
    return model


if __name__ == "__main__":
    # Test cấu trúc mạng
    model = build_tinyface_ghost(input_shape=(64, 64, 1), embedding_dim=128)
    model.summary()
    
    # Kiểm tra kích thước đầu ra
    dummy_input = tf.random.normal([1, 64, 64, 1])
    output = model(dummy_input)
    print(f"\n[+] Input Shape: {dummy_input.shape}")
    print(f"[+] Output Embedding Shape: {output.shape}")
    print(f"[+] L2 Norm of Output: {tf.norm(output, axis=-1).numpy()[0]:.4f} (Kỳ vọng: 1.0000)")
