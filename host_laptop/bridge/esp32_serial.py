import serial
import time
import struct
import json

class ESP32Bridge:
    """
    Module quản lý kết nối và truyền nhận dữ liệu với ESP32-S3 qua cổng Serial tốc độ cao.
    """
    HEADER = b'\xFF\xAA'
    CMD_PING = 0x00
    CMD_INFO = 0x01
    CMD_INFERENCE = 0x02
    CMD_IMAGE = 0x02  # Alias for inference
    
    def __init__(self, port='COM6', baudrate=921600, timeout=1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.is_connected = False

    def connect(self) -> bool:
        """Khởi tạo kết nối với ESP32-S3."""
        try:
            print(f"[*] Đang kết nối tới ESP32-S3 trên cổng {self.port} ({self.baudrate} bps)...")
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.1,
                write_timeout=self.timeout,
                rtscts=False,
                dsrdtr=False
            )
            # Kích hoạt DTR/RTS cho ESP32-S3
            self.ser.dtr = True
            self.ser.rts = True
            
            print("[*] Đang chờ ESP32 khởi động và đọc log (2 giây)...")
            time.sleep(2.0)
            
            # Đọc toàn bộ boot log của ESP32 để debug lỗi AllocateTensors!
            while self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    print(f"[ESP32 BOOT LOG] {line}")
            
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            self.ser.timeout = self.timeout
            
            self.is_connected = True
            print(f"[+] Kết nối ESP32-S3 thành công trên cổng {self.port}!")
            return True
        except serial.SerialException as e:
            print(f"❌ Không thể kết nối cổng {self.port}: {e}")
            self.is_connected = False
            return False

    @staticmethod
    def calculate_checksum(data: bytes) -> int:
        """Tính toán XOR checksum."""
        checksum = 0
        for byte in data:
            checksum ^= byte
        return checksum

    def _create_packet(self, command_id: int, payload: bytes) -> bytes:
        """Đóng gói tin nhị phân."""
        cmd_byte = struct.pack('<B', command_id)
        length_bytes = struct.pack('>I', len(payload)) # Big-endian 4 bytes
        checksum_byte = struct.pack('<B', self.calculate_checksum(payload))
        return self.HEADER + cmd_byte + length_bytes + payload + checksum_byte

    def send_inference(self, image_bytes: bytes) -> dict:
        """
        Gửi mảng byte ảnh 96x96 (9216 bytes) xuống ESP32 để suy luận TinyML và nhận kết quả nhận diện.
        """
        if not self.is_connected or self.ser is None:
            return {"status": "error", "message": "Chưa kết nối ESP32"}
            
        packet = self._create_packet(self.CMD_INFERENCE, image_bytes)
        
        try:
            t_start = time.perf_counter()
            self.ser.write(packet)
            self.ser.flush()
            
            resp_line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            t_end = time.perf_counter()
            rtt_ms = (t_end - t_start) * 1000
            
            if not resp_line:
                return {"status": "timeout", "rtt_ms": round(rtt_ms, 2)}
                
            try:
                res_json = json.loads(resp_line)
                res_json["rtt_ms"] = round(rtt_ms, 2)
                return res_json
            except json.JSONDecodeError:
                return {"status": "raw", "data": resp_line, "rtt_ms": round(rtt_ms, 2)}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def send_image(self, image_bytes: bytes) -> dict:
        """Alias cho send_inference."""
        return self.send_inference(image_bytes)

    def ping(self) -> dict:
        """Gửi lệnh Ping kiểm tra kết nối."""
        if not self.is_connected or self.ser is None:
            return {"status": "error", "message": "Chưa kết nối ESP32"}
            
        packet = self._create_packet(self.CMD_PING, b"PING")
        try:
            t0 = time.perf_counter()
            self.ser.write(packet)
            self.ser.flush()
            resp = self.ser.readline().decode('utf-8', errors='ignore').strip()
            t1 = time.perf_counter()
            return {"status": "success", "response": resp, "rtt_ms": round((t1 - t0) * 1000, 2)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def close(self):
        """Đóng kết nối."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.is_connected = False
            print("[*] Đã đóng kết nối Serial với ESP32-S3.")

# ==============================================================================
# HÀM TEST ĐỘC LẬP
# ==============================================================================
if __name__ == "__main__":
    bridge = ESP32Bridge(port='COM6')
    if bridge.connect():
        print("[*] Đang test Ping...")
        print(bridge.ping())
        
        print("[*] Đang test gửi ảnh giả lập 96x96 (9216 bytes)...")
        dummy_img = bytes([i % 256 for i in range(96 * 96)])
        print(bridge.send_image(dummy_img))
        
        bridge.close()
