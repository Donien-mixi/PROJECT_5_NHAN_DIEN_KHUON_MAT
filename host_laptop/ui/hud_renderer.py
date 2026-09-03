import cv2

class HUDRenderer:
    """
    Chịu trách nhiệm vẽ các thành phần UI (HUD, bounding box, text) lên frame.
    """
    @staticmethod
    def draw_hud(frame, fps, detector_mode, infer_ms, last_crop_bgr=None, matched=False, name="Unknown", sim=0.0, use_tflite=False):
        h, w = frame.shape[:2]
        
        # Vẽ thanh header
        header_h = 42
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_h), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        font_scale = 0.48 if w <= 640 else 0.55
        y_text = 27
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 2)
        
        mode_text = "BLAZEFACE_INT8 128" if detector_mode == "BLAZEFACE_INT8" else "BLAZEFACE_INT8 128"
        col2_x = int(w * 0.20)
        cv2.putText(frame, f"AI: {mode_text}", (col2_x, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 200, 50), 2)
        
        col3_x = int(w * 0.46)
        inference_text = "LAPTOP INFERENCE: TFLITE INT8" if use_tflite else "LAPTOP INFERENCE: KERAS"
        cv2.putText(frame, inference_text, (col3_x, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
        
        col4_x = int(w * 0.78)
        cv2.putText(frame, f"Infer:{infer_ms:.0f}ms", (col4_x, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100, 255, 100), 2)

        # Vẽ Picture-in-Picture (PiP)
        if last_crop_bgr is not None:
            pip_w, pip_h = 100, 100
            margin = 12
            pip_x = w - pip_w - margin
            pip_y = h - pip_h - margin
            
            crop_resized = cv2.resize(last_crop_bgr, (pip_w, pip_h))
            frame[pip_y:pip_y+pip_h, pip_x:pip_x+pip_w] = crop_resized
            
            pip_border_color = (0, 255, 0) if matched else (0, 165, 255)
            cv2.rectangle(frame, (pip_x, pip_y), (pip_x+pip_w, pip_y+pip_h), pip_border_color, 2)
            cv2.putText(frame, "64x64 Input", (pip_x, pip_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, pip_border_color, 1)

    @staticmethod
    def draw_face_box(frame, face, matched, recognized_name, similarity, detector_mode):
        x, y, w, h = face['bbox']
        box_color = (0, 255, 0) if matched else (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        
        # Vẽ điểm mốc BlazeFace 5 điểm (đồng bộ ESP32, không dùng Mesh 468)
        for (lx, ly) in face.get('landmarks_5', []):
            cv2.circle(frame, (lx, ly), 4, (0, 215, 255), -1)
            cv2.circle(frame, (lx, ly), 6, (0, 255, 0), 1)
                
        # Vẽ tên
        label_str = f"MATCH: {recognized_name} ({similarity*100:.1f}%)" if matched else f"UNKNOWN ({similarity*100:.1f}%)"
        (text_w, text_h), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x, max(0, y - text_h - 10)), (x + text_w + 10, y), (18, 18, 18), -1)
        cv2.putText(frame, label_str, (x + 5, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
