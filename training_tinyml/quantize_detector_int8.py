"""
PTQ BlazeFace short-range -> full INT8 (giữ nguyên topo 2-output: regressors + classificators).

Cấu trúc chuẩn BlazeFace short-range (128x128, fdlite/mediapipe), model gốc là float16-hybrid.
Script đọc trọng số từ file .tflite float32, dựng lại Keras model, validate với interpreter gốc
(chỉ số đi khi maxdiff < 1e-3), rồi PTQ INT8 với representative dataset từ ảnh khuôn mặt thật.

Đầu ra: training_tinyml/weights/face_detection_short_range_int8.tflite
Sau đó chạy: python host_laptop/convert_tflite_to_c.py --model <tflite> --out firmware_esp32/detector_model_data.h --symbol g_detector_model
"""
import os
import sys
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "host_laptop", "detector", "face_detection_short_range.tflite")
DST = os.path.join(ROOT, "training_tinyml", "weights", "face_detection_short_range_int8.tflite")


def build_model(interp, td, tname, tshape, ops):
    """Dựng lại Keras BlazeFace short-range từ graph TFLite, nạp trọng số float16 -> float32."""
    name2idx = {d['name']: i for i, d in enumerate(td)}

    def _orig_idx(idx):
        nm = tname[idx]
        if nm.endswith("_dequantize"):
            base = nm.replace("_dequantize", "")
            if base in name2idx:
                return name2idx[base]
        return idx

    def const(idx):
        return np.asarray(interp.get_tensor(_orig_idx(idx)), dtype=np.float32)

    tensors = {}
    inputs = keras.Input(shape=(128, 128, 3), name='input')
    tensors[0] = inputs

    def stride_from(x, oidx):
        out_shape = tshape[oidx]
        if len(x.shape) == 4 and len(out_shape) == 4:
            return max(1, x.shape[1] // out_shape[1])
        return 1

    def conv2d_op(op, name):
        x = tensors[op['inputs'][0]]
        k = const(op['inputs'][1]); bias = const(op['inputs'][2])
        out = k.shape[0]; kh, kw = k.shape[1], k.shape[2]
        st = stride_from(x, op['outputs'][0])
        layer = L.Conv2D(out, kernel_size=(kh, kw), strides=(st, st), padding='same', use_bias=True, name=name)
        y = layer(x)
        layer.set_weights([k.transpose(1, 2, 3, 0), bias])
        return y

    def dw_op(op, name):
        x = tensors[op['inputs'][0]]
        k = const(op['inputs'][1]); bias = const(op['inputs'][2])
        kh, kw = k.shape[1], k.shape[2]
        st = stride_from(x, op['outputs'][0])
        layer = L.DepthwiseConv2D(kernel_size=(kh, kw), strides=(st, st), padding='same', use_bias=True,
                                  depth_multiplier=1, name=name)
        y = layer(x)
        layer.set_weights([k[0][:, :, :, None], bias])
        return y

    def add_op(op, name):
        return L.Add(name=name)([tensors[op['inputs'][0]], tensors[op['inputs'][1]]])

    def relu_op(op, name):
        return L.ReLU(name=name)(tensors[op['inputs'][0]])

    def pad_op(op, name):
        x = tensors[op['inputs'][0]]
        p = const(op['inputs'][1]).astype(np.int32)
        pad_c = (int(p[-1][0]), int(p[-1][1])) if p.shape[0] in (2, 4) else (0, 0)
        return L.Lambda(lambda t, pc=pad_c: tf.pad(t, [[0, 0], [0, 0], [0, 0], list(pc)]), name=name)(x)

    def maxpool_op(op, name):
        return L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name=name)(tensors[op['inputs'][0]])

    def reshape_op(op, name):
        return L.Reshape(tshape[op['outputs'][0]][1:], name=name)(tensors[op['inputs'][0]])

    def concat_op(op, name):
        return L.Concatenate(axis=1, name=name)([tensors[i] for i in op['inputs']])

    handlers = {
        'CONV_2D': conv2d_op, 'DEPTHWISE_CONV_2D': dw_op, 'ADD': add_op, 'RELU': relu_op,
        'PAD': pad_op, 'MAX_POOL_2D': maxpool_op, 'RESHAPE': reshape_op, 'CONCATENATION': concat_op,
    }

    for op in ops:
        oname = op['op_name']
        if oname in ('DELEGATE', 'DEQUANTIZE'):
            continue
        h = handlers.get(oname)
        if h is None:
            print("  [warn] skip op", oname, op['index'])
            continue
        y = h(op, f"{oname.lower()}_{op['index']}")
        for oidx in op['outputs']:
            tensors[oidx] = y

    # tensors[174]=classificators (896,1), tensors[175]=regressors (896,16)
    model = keras.Model(inputs=inputs, outputs=[tensors[174], tensors[175]], name='blazeface')
    return model


def validate(model, interp, n=3):
    ins = interp.get_input_details()[0]
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(n):
        img = rng.uniform(0, 255, size=(1, 128, 128, 3)).astype(np.float32)
        norm = (img - 127.5) / 128.0
        interp.set_tensor(ins['index'], norm)
        interp.invoke()
        # TFLite output order: [0]=regressors(896,16), [1]=classificators(896,1)
        tfl_reg, tfl_cos = interp.get_tensor(interp.get_output_details()[0]['index']), \
                           interp.get_tensor(interp.get_output_details()[1]['index'])
        keras_reg, keras_cos = model(norm, training=False)[1].numpy(), model(norm, training=False)[0].numpy()
        d = max(np.abs(keras_cos - tfl_cos).max(), np.abs(keras_reg - tfl_reg).max())
        worst = max(worst, float(d))
    print(f"  [validate] maxdiff = {worst:.6f}  ({'PASS' if worst < 1e-3 else 'FAIL'})")
    if worst >= 1e-3:
        raise SystemExit("❌ Dựng lại model không khớp — dừng PTQ.")
    return worst


def rep_dataset(size=128, samples=64):
    """Representative dataset: ảnh khuôn mặt thật (registered_faces) + pha nhiễu để hiệu chuẩn."""
    reg = os.path.join(ROOT, "data", "registered_faces")
    files = []
    for p in glob.glob(os.path.join(reg, "*", "*.jpg")) + glob.glob(os.path.join(reg, "*", "*.png")):
        files.append(p)
    rng = np.random.default_rng(3)
    if not files:
        print("  [rep] Không có ảnh thật, dùng ảnh ngẫu nhiên.")
        files = []
    idx = 0
    for _ in range(samples):
        if files:
            p = files[idx % len(files)]
            idx += 1
            img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
            img = tf.image.resize(tf.image.resize(img, (size, size)), (size, size)).numpy()
            img = np.asarray(img, dtype=np.float32)
        else:
            img = rng.uniform(0, 255, size=(size, size, 3)).astype(np.float32)
        norm = (img - 127.5) / 128.0
        yield [norm[None, :, :, :]]


def main():
    print("=== QUANTIZE DETECTOR: BlazeFace short-range FLOAT16 -> FULL INT8 ===")
    if not os.path.exists(SRC):
        raise SystemExit(f"Không tìm thấy {SRC}")

    interp = tf.lite.Interpreter(model_path=SRC)
    interp.allocate_tensors()
    td = interp.get_tensor_details()
    tname = {i: d['name'] for i, d in enumerate(td)}
    tshape = {i: d['shape'].tolist() for i, d in enumerate(td)}
    ops = interp._get_ops_details()

    model = build_model(interp, td, tname, tshape, ops)
    print("  [build] params =", model.count_params())
    validate(model, interp)

    # PTQ full INT8
    run_model = tf.function(lambda x: model(x, training=False))
    concrete = run_model.get_concrete_function(tf.TensorSpec([1, 128, 128, 3], tf.float32))
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.experimental_new_quantizer = True

    print("  [ptq] Converting to full INT8 ...")
    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with open(DST, "wb") as f:
        f.write(tflite_model)
    print(f"  [ptq] OK → {DST} ({len(tflite_model)/1024:.1f} KB)")

    # Audit kết quả
    q = tf.lite.Interpreter(model_path=DST)
    q.allocate_tensors()
    print("  [audit] input :", [(d['shape'].tolist(), d['dtype'].__name__, d['quantization']) for d in q.get_input_details()])
    print("  [audit] output:", [(d['name'], d['shape'].tolist(), d['dtype'].__name__, d['quantization']) for d in q.get_output_details()])
    print("  [audit] ops   :", sorted({o['op_name'] for o in q._get_ops_details()}))
    return DST


if __name__ == "__main__":
    main()
