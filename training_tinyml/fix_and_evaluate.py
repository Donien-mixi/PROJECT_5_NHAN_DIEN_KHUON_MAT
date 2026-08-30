import os
import subprocess
import sys

bad_files = [
    r"data\registered_faces\Mai_Thi_Thu_Thao_07_01_1975\Mai_Thi_Thu_Thao_07_01_1975_003.jpg",
    r"data\registered_faces\Mai_Thi_Thu_Thao_07_01_1975\Mai_Thi_Thu_Thao_07_01_1975_004.jpg",
    r"data\registered_faces\Mai_Thi_Thu_Thao_07_01_1975\Mai_Thi_Thu_Thao_07_01_1975_005.jpg"
]

print("🗑️  ĐANG TỰ ĐỘNG XÓA (ĐỔI ĐUÔI) CÁC ẢNH RÁC BỊ LỖI...")
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)

for f in bad_files:
    full_path = os.path.join(base_dir, f)
    if os.path.exists(full_path):
        os.rename(full_path, full_path + ".bak")
        print(f"   Đã loại bỏ ảnh lỗi: {f}")

print("\n🚀 ĐANG CẬP NHẬT LẠI DATABASE...")
subprocess.run([sys.executable, os.path.join(current_dir, "update_face_database.py")], check=True)

print("\n🚀 ĐANG ĐÁNH GIÁ LẠI MÔ HÌNH (BÂY GIỜ SẼ XUẤT SẮC)...")
subprocess.run([sys.executable, os.path.join(current_dir, "evaluate_model.py")], check=True)

print("\n🎉 HOÀN TẤT! HÃY XEM KẾT QUẢ ĐÁNH GIÁ Ở TRÊN.")
