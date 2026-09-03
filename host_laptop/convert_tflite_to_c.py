"""
Script nhỏ để đổi file `.tflite` INT8 thành `.h` (mảng C array) cho firmware ESP32.

Cách dùng:
    python host_laptop/convert_tflite_to_c.py \
        --model training_tinyml/weights/tinyface_int8.tflite \
        --out firmware_esp32/model_data.h \
        --symbol g_model_recognizer

Ví dụ cho detector:
    python host_laptop/convert_tflite_to_c.py \
        --model host_laptop/detector/face_detection_short_range.tflite \
        --out firmware_esp32/detector_model_data.h \
        --symbol g_detector_model
"""
import argparse
import os


def tflite_to_c_header(tflite_path, symbol="g_model", header_guard=None):
    """Đọc file .tflite -> nội dung header C với extern + len (chuẩn firmware_esp32)."""
    with open(tflite_path, "rb") as f:
        data = f.read()

    if header_guard is None:
        header_guard = f"{symbol}_H".upper()

    hex_lines = []
    for i in range(0, len(data), 12):
        hex_lines.append("    " + ", ".join(f"0x{b:02x}" for b in data[i:i + 12]) + ",")

    header = (
        f"// Được tự động sinh ra từ file {os.path.basename(tflite_path)}\n"
        f"// Kích thước: {len(data)} bytes — bởi {os.path.basename(__file__)}\n\n"
        f"#ifndef {header_guard}\n"
        f"#define {header_guard}\n\n"
        f"extern const unsigned char {symbol}[];\n"
        f"extern const int {symbol}_len;\n\n"
        f"const unsigned char {symbol}[] = {{\n"
        + "\n".join(hex_lines)
        + "\n};\n\n"
        f"const int {symbol}_len = {len(data)};\n\n"
        f"#endif // {header_guard}\n"
    )
    return header


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .tflite -> .h cho ESP32")
    parser.add_argument("--model", required=True, help="Đường dẫn file .tflite")
    parser.add_argument("--out", required=True, help="Đường dẫn file .h đầu ra")
    parser.add_argument("--symbol", default="g_model", help="Tên symbol mảng C (mặc định: g_model)")
    parser.add_argument("--guard", default=None, help="Header guard tùy chỉnh (mặc định: <SYMBOL>_H)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"Không tìm thấy model: {args.model}")

    content = tflite_to_c_header(args.model, args.symbol, args.guard)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] Created {args.out} from {args.model}")
