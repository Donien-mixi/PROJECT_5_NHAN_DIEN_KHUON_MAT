import os
import urllib.request
import binascii

# URL tải mô hình MediaPipe BlazeFace siêu nhẹ (128x128)
MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite"
TFLITE_PATH = "detector.tflite"
HEADER_PATH = "../firmware_esp32/detector_model_data.h"

def download_and_convert():
    print(f"Đang tải mô hình từ: {MODEL_URL}")
    try:
        # Thêm User-Agent để tránh bị block
        req = urllib.request.Request(MODEL_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(TFLITE_PATH, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"✅ Đã tải thành công {TFLITE_PATH}")
    except Exception as e:
        print(f"❌ Lỗi tải mô hình: {e}")
        return

    # Chuyển đổi thành C array
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
        
        # Ghi từng dòng, mỗi dòng 12 bytes
        for i in range(0, len(hex_array), 12):
            line = ", ".join(hex_array[i:i+12])
            if i + 12 < len(hex_array):
                line += ","
            f.write(f"    {line}\n")
            
        f.write("};\n\n")
        f.write("#endif // DETECTOR_MODEL_DATA_H\n")
        
    print(f"🎉 Đã tạo thành công {HEADER_PATH}!")

if __name__ == "__main__":
    download_and_convert()
