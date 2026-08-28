import os
import time
import numpy as np
import json
import tensorflow as tf

class FaceRecognizer:
    """
    Chịu trách nhiệm nạp mô hình (Keras Float32 hoặc TFLite INT8),
    nạp cơ sở dữ liệu JSON và thực hiện trích xuất đặc trưng (embedding)
    cũng như so khớp (Matching) bằng Cosine Similarity.
    """
    def __init__(self, model_path, db_path, threshold=0.75, use_tflite=False):
        self.threshold = threshold
        self.use_tflite = use_tflite
        self.model = None
        self.interpreter = None
        self.database = {}
        
        if self.use_tflite:
            self._load_tflite_model(model_path)
        else:
            self._load_keras_model(model_path)
            
        self._load_database(db_path)

    def _load_tflite_model(self, model_path):
        if not os.path.exists(model_path):
            print(f"❌ LỖI: Không tìm thấy {model_path}.")
            return
        print(f"[*] Đang nạp mô hình TFLite INT8 từ {model_path}...")
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("[+] Đã nạp mô hình TFLite thành công.")

    def _load_keras_model(self, model_path):
        if not os.path.exists(model_path):
            print(f"❌ LỖI: Không tìm thấy {model_path}.")
            print("Vui lòng đợi Colab train xong, tải về và chép vào thư mục training_tinyml/weights/.")
            return
            
        print("[*] Đang nạp mô hình Universal Keras Ghost-TinyFace...")
        # Import local để tránh circular dependency và tăng tốc độ nạp thư viện
        from training_tinyml.models.ghost_tinyface import build_tinyface_ghost
        self.model = build_tinyface_ghost()
        self.model.load_weights(model_path)
        print("[+] Đã nạp mô hình Keras Float32 thành công.")

    def _load_database(self, db_path):
        if os.path.exists(db_path):
            with open(db_path, "r", encoding="utf-8") as f:
                self.database = json.load(f)
                print(f"[*] Đã nạp CSDL JSON với {len(self.database)} người dùng.")
        else:
            print("[!] CẢNH BÁO: Chưa có CSDL (face_database.json). Vẫn chạy được nhưng không so khớp được ai.")

    def recognize(self, crop_gray):
        """
        Nhận vào ảnh khuôn mặt đã được align và crop (grayscale 64x64).
        Trả về: (matched(bool), best_name(str), best_sim(float), infer_time_ms(float))
        """
        if not self.use_tflite and self.model is None:
            return False, "Unknown", 0.0, 0.0
        if self.use_tflite and self.interpreter is None:
            return False, "Unknown", 0.0, 0.0
            
        # Tiền xử lý giống ESP32 (Chuẩn hóa về [-1, 1])
        norm_img = (crop_gray.astype(np.float32) - 127.5) / 128.0
        norm_img = np.expand_dims(norm_img, axis=(0, -1)) # Shape: (1, 64, 64, 1)
        
        t_start = time.time()
        
        if self.use_tflite:
            scale, zero_point = self.input_details[0]['quantization']
            if scale > 0:
                input_data = np.clip(np.round(norm_img / scale) + zero_point, -128, 127).astype(np.int8)
            else:
                input_data = norm_img.astype(np.float32)
                
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            out_scale, out_zero_point = self.output_details[0]['quantization']
            if out_scale > 0:
                emb = (output_data.astype(np.float32) - out_zero_point) * out_scale
            else:
                emb = output_data.astype(np.float32)
            emb = emb[0]
        else:
            emb = self.model(norm_img, training=False)[0].numpy()
            
        infer_ms = (time.time() - t_start) * 1000.0
        
        # L2 Normalize
        emb = emb / (np.linalg.norm(emb) + 1e-7)
        
        best_name = "Unknown"
        best_sim = -1.0
        
        # So khớp
        for uname, udata in self.database.items():
            ref_emb = np.array(udata["embedding"], dtype=np.float32)
            sim = np.dot(emb, ref_emb)
            if sim > best_sim:
                best_sim = sim
                best_name = uname
                
        matched = (best_sim >= self.threshold)
        if not matched:
            best_name = "Unknown"
            
        return matched, best_name, float(best_sim), infer_ms

class TemporalVoter:
    """
    Bộ đếm phiếu chống nhiễu (Temporal Voting).
    Yêu cầu n frame liên tiếp phải nhận diện ra cùng 1 người mới "chốt" kết quả.
    """
    def __init__(self, required_votes=5):
        self.required_votes = required_votes
        self.vote_name = None
        self.vote_count = 0
        
    def vote(self, recognized_name, similarity, infer_ms):
        """
        Trả về (is_locked, tracked_name, tracked_sim, tracked_infer)
        Nếu chưa đủ phiếu, trả về (False, None, 0, 0)
        """
        if recognized_name != "Unknown":
            if recognized_name == self.vote_name:
                self.vote_count += 1
            else:
                self.vote_name = recognized_name
                self.vote_count = 1

            if self.vote_count >= self.required_votes:
                # Đủ phiếu
                final_name = self.vote_name
                self.reset()
                return True, final_name, similarity, infer_ms
        else:
            self.reset()
            
        return False, None, 0.0, 0.0
        
    def reset(self):
        self.vote_name = None
        self.vote_count = 0
