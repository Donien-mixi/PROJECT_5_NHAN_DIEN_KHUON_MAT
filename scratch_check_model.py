import tensorflow as tf

model_path = 'host_laptop/030_BlazeFace/face_detection_front_128_integer_quant.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

print('Input details:', interpreter.get_input_details()[0]['dtype'])
print('Output details:', interpreter.get_output_details()[0]['dtype'])
