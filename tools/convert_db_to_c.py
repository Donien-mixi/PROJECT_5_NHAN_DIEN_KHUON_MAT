import os
import json
import argparse

def convert_db_to_c(json_path, c_file_path):
    """
    Chuyển đổi CSDL dạng JSON thành C++ header (array).
    """
    if not os.path.exists(json_path):
        print(f"Error: Không tìm thấy file {json_path}")
        return False
        
    with open(json_path, 'r', encoding='utf-8') as f:
        db = json.load(f)
        
    num_users = len(db)
    vector_len = 128
    
    with open(c_file_path, 'w', encoding='utf-8') as f:
        f.write(f"// Được tự động sinh ra từ file {os.path.basename(json_path)}\n")
        f.write(f"// Số người đăng ký: {num_users}\n\n")
        f.write("#ifndef FACE_DATABASE_H\n")
        f.write("#define FACE_DATABASE_H\n\n")
        
        f.write(f"#define NUM_REGISTERED_USERS {num_users}\n")
        f.write(f"#define EMBEDDING_SIZE {vector_len}\n\n")
        
        f.write("typedef struct {\n")
        f.write("    const char* name;\n")
        f.write("    const float embedding[EMBEDDING_SIZE];\n")
        f.write("} RegisteredUser;\n\n")
        
        f.write(f"const RegisteredUser face_database[NUM_REGISTERED_USERS] = {{\n")
        
        for name, embedding in db.items():
            if len(embedding) != vector_len:
                print(f"Warning: Vector của {name} có độ dài {len(embedding)} (mong đợi {vector_len})")
            
            f.write(f"    {{\n")
            f.write(f"        \"{name}\",\n")
            f.write(f"        {{\n")
            
            # Format the embedding array (8 floats per line)
            for i in range(0, len(embedding), 8):
                chunk = embedding[i:i+8]
                f.write("            " + ", ".join([f"{val:.6f}f" for val in chunk]))
                if i + 8 < len(embedding):
                    f.write(",\n")
                else:
                    f.write("\n")
                    
            f.write(f"        }}\n")
            f.write(f"    }},\n")
            
        f.write("};\n\n")
        f.write("#endif // FACE_DATABASE_H\n")
        
    print(f"Thành công! Đã tạo file {c_file_path} cho {num_users} người dùng.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON Face DB to C header")
    parser.add_argument("--input", type=str, default="../data/face_database.json", help="Đường dẫn file .json đầu vào")
    parser.add_argument("--output", type=str, default="../firmware_esp32/src/tinyml_recognizer/face_database.h", help="Đường dẫn file .h đầu ra")
    args = parser.parse_args()
    
    # Resolve absolute paths
    input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.input))
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.output))
    
    # Đảm bảo thư mục đầu ra tồn tại
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    convert_db_to_c(input_path, output_path)
