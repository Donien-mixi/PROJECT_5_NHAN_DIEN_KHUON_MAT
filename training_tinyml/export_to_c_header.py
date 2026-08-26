import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def export_tflite_to_c_header():
    print("==================================================================")
    print("💻 XUẤT MÔ HÌNH INT8 THÀNH MẢNG C++ HEADER CHO ESP32-S3")
    print("==================================================================")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    tflite_path = os.path.join(current_dir, "weights", "tinyface_int8.tflite")
    
    if not os.path.exists(tflite_path):
        print(f"❌ LỖI: Không tìm thấy file {tflite_path}. Hãy chạy quantize_qat_int8.py trước!")
        return

    with open(tflite_path, "rb") as f:
        tflite_bytes = f.read()

    base_dir = os.path.dirname(current_dir)
    header_path = os.path.join(base_dir, "firmware_esp32", "src", "model_data.h")
    sketch_header_path = os.path.join(base_dir, "firmware_esp32", "src", "tinyml_recognizer", "model_data.h")
    os.makedirs(os.path.dirname(header_path), exist_ok=True)
    os.makedirs(os.path.dirname(sketch_header_path), exist_ok=True)

    print(f"[*] Đang chuyển đổi {len(tflite_bytes)} bytes sang mảng Byte C++...")

    hex_array = []
    for i, b in enumerate(tflite_bytes):
        hex_array.append(f"0x{b:02x}")

    # Nhóm 12 byte trên 1 dòng cho gọn gàng
    lines = []
    for i in range(0, len(hex_array), 12):
        lines.append("  " + ", ".join(hex_array[i:i+12]))

    body = ",\n".join(lines)
    c_content = f"""#pragma once
#include <Arduino.h>

// ==============================================================================
// MÔ HÌNH AI TINYFACENET-GHOST (LƯỢNG TỬ HÓA FULL INT8)
// TỔNG KÍCH THƯỚC: {len(tflite_bytes)} BYTES (~{len(tflite_bytes)/1024:.2f} KB)
// TỐI ƯU HÓA 100% CHO ESP32-S3 TFLITE MICRO & ESP-NN VECTOR ACCELERATOR
// ==============================================================================

alignas(16) const unsigned char g_model_data[] = {{
{body}
}};

const unsigned int g_model_data_len = {len(tflite_bytes)};
"""

    with open(header_path, "w", encoding="utf-8") as f:
        f.write(c_content)
    with open(sketch_header_path, "w", encoding="utf-8") as f:
        f.write(c_content)

    print(f"🎉 XUẤT THÀNH CÔNG FILE C++ HEADER TẠI:\n    👉 {sketch_header_path}")

if __name__ == "__main__":
    export_tflite_to_c_header()
