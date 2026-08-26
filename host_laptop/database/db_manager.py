import sqlite3
import os
import time
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path="data/attendance.db", cooldown_seconds=30):
        """
        Quản lý CSDL SQLite điểm danh.
        :param db_path: Đường dẫn lưu file CSDL.
        :param cooldown_seconds: Khoảng thời gian (giây) giữa 2 lần điểm danh liên tiếp của cùng một người để chống spam.
        """
        self.db_path = db_path
        self.cooldown_seconds = cooldown_seconds
        
        # Lưu trữ thời điểm điểm danh cuối cùng của mỗi người {tên: timestamp}
        self.last_seen = {}
        
        # Đảm bảo thư mục tồn tại
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def init_db(self):
        """Khởi tạo bảng nếu chưa có."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tạo bảng lưu lịch sử nhận diện
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                similarity REAL,
                inference_ms REAL
            )
        ''')
        
        conn.commit()
        conn.close()

    def log_attendance(self, name: str, similarity: float, inference_ms: float) -> bool:
        """
        Ghi nhận lịch sử điểm danh.
        Tránh ghi spam nếu người dùng vừa được ghi nhận cách đây vài giây.
        
        :return: True nếu đã ghi vào DB thành công, False nếu bị chặn bởi cooldown.
        """
        current_time = time.time()
        
        # Kiểm tra Cooldown
        if name in self.last_seen:
            elapsed = current_time - self.last_seen[name]
            if elapsed < self.cooldown_seconds:
                return False # Bỏ qua vì khoảng cách điểm danh quá ngắn
                
        # Ghi vào DB
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO attendance_logs (name, similarity, inference_ms) VALUES (?, ?, ?)",
                (name, similarity, inference_ms)
            )
            
            conn.commit()
            conn.close()
            
            # Cập nhật thời điểm điểm danh
            self.last_seen[name] = current_time
            
            # In ra màn hình Terminal với màu sắc để dễ nhìn
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[+] \033[92mĐIỂM DANH THÀNH CÔNG\033[0m: {name} | {now_str} | Sim: {similarity*100:.1f}%")
            return True
            
        except Exception as e:
            print(f"[-] Lỗi ghi CSDL điểm danh: {e}")
            return False
