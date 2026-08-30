import tensorflow as tf

def analyze(path):
    print('Model:', path)
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    print('Inputs:')
    for i in interpreter.get_input_details():
        print(f"  name: {i['name']}, shape: {i['shape']}, type: {i['dtype']}, quant: {i['quantization']}")
    print('Outputs:')
    for o in interpreter.get_output_details():
        print(f"  name: {o['name']}, shape: {o['shape']}, type: {o['dtype']}, quant: {o['quantization']}")

analyze('host_laptop/detector.tflite')
analyze(r'host_laptop/030_BlazeFace/04_full_integer_quantization/face_detection_front_128x128_full_integer_quant.tflite')
