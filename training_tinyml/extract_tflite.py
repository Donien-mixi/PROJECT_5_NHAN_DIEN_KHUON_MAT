import os
import re

def extract_tflite():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    h_path = os.path.join(current_dir, "..", "firmware_esp32", "model_data.h")
    tflite_path = os.path.join(current_dir, "weights", "tinyface_int8_from_h.tflite")
    
    if not os.path.exists(h_path):
        print(f"Cannot find {h_path}")
        return
        
    with open(h_path, "r") as f:
        content = f.read()
        
    # Extract the array content between { and }
    match = re.search(r'\{([^}]+)\}', content)
    if not match:
        print("Could not find the array in model_data.h")
        return
        
    array_content = match.group(1)
    
    # Extract all hex values
    hex_values = re.findall(r'0x[0-9a-fA-F]{1,2}', array_content)
    
    # Convert to bytes
    byte_array = bytearray(int(x, 16) for x in hex_values)
    
    # Write to .tflite
    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(byte_array)
        
    print(f"Extracted {len(byte_array)} bytes to {tflite_path}")

if __name__ == "__main__":
    extract_tflite()
