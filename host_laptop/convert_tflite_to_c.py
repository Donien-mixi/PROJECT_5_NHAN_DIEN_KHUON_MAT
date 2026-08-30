import os

TFLITE_PATH = "detector.tflite"
HEADER_PATH = "../firmware_esp32/detector_model_data.h"

def convert():
    print(f"Đang chuyển đổi sang {HEADER_PATH}...")
    with open(TFLITE_PATH, "rb") as f:
        tflite_content = f.read()

    hex_array = [f"0x{byte:02x}" for byte in tflite_content]
    
    with open(HEADER_PATH, "w") as f:
        f.write("#ifndef DETECTOR_MODEL_DATA_H\n")
        f.write("#define DETECTOR_MODEL_DATA_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"const unsigned int g_detector_model_len = {len(hex_array)};\n")
        f.write("const unsigned char g_detector_model[] = {\n")
        
        for i in range(0, len(hex_array), 12):
            line = ", ".join(hex_array[i:i+12])
            if i + 12 < len(hex_array):
                line += ","
            f.write(f"    {line}\n")
            
        f.write("};\n\n")
        f.write("#endif // DETECTOR_MODEL_DATA_H\n")
        
    print(f"🎉 Đã tạo thành công {HEADER_PATH}!")

if __name__ == "__main__":
    convert()
