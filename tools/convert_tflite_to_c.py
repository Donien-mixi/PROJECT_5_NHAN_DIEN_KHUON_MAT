import os
import argparse

def convert_tflite_to_c(tflite_path, c_file_path, array_name="g_model"):
    """
    Chuyển đổi file .tflite thành mảng C (unsigned char array) để nhúng vào ESP32.
    """
    if not os.path.exists(tflite_path):
        print(f"Error: Lỗi không tìm thấy file {tflite_path}")
        return False
        
    with open(tflite_path, 'rb') as f:
        tflite_content = f.read()
        
    hex_array = [f"0x{b:02x}" for b in tflite_content]
    
    with open(c_file_path, 'w', encoding='utf-8') as f:
        f.write(f"// Được tự động sinh ra từ file {os.path.basename(tflite_path)}\n")
        f.write(f"// Kích thước: {len(tflite_content)} bytes\n\n")
        f.write(f"#ifndef {array_name.upper()}_H\n")
        f.write(f"#define {array_name.upper()}_H\n\n")
        f.write(f"extern const unsigned char {array_name}[];\n")
        f.write(f"extern const int {array_name}_len;\n\n")
        f.write(f"const unsigned char {array_name}[] = {{\n")
        
        # Format the array (12 bytes per line)
        for i in range(0, len(hex_array), 12):
            f.write("    " + ", ".join(hex_array[i:i+12]))
            if i + 12 < len(hex_array):
                f.write(",\n")
            else:
                f.write("\n")
                
        f.write("};\n\n")
        f.write(f"const int {array_name}_len = {len(tflite_content)};\n\n")
        f.write(f"#endif // {array_name.upper()}_H\n")
        
    print(f"Thành công! Đã tạo file {c_file_path} (Kích thước: {len(tflite_content)} bytes)")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TFLite to C array")
    parser.add_argument("--input", type=str, default="../training_tinyml/weights/tinyface_int8.tflite", help="Đường dẫn file .tflite đầu vào")
    parser.add_argument("--output", type=str, default="../firmware_esp32/model_data.h", help="Đường dẫn file .h đầu ra")
    args = parser.parse_args()
    
    # Resolve absolute paths
    input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.input))
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.output))
    
    # Đảm bảo thư mục đầu ra tồn tại
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    convert_tflite_to_c(input_path, output_path)
