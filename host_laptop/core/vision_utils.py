import cv2

def center_square_crop(frame, bbox_x, bbox_y, bbox_w, bbox_h, target_size=(64, 64)):
    """
    Chuẩn hóa thuật toán cắt ảnh (Center Square Crop) dùng chung cho cả Python và C++ (ESP32).
    Đảm bảo ảnh lúc Enroll và lúc Detect hoàn toàn giống hệt nhau về mặt toán học.
    """
    img_h, img_w = frame.shape[:2]
    
    # 1. Tìm cạnh lớn nhất để tạo hình vuông
    size = max(bbox_w, bbox_h)
    
    # 2. Căn giữa hình vuông so với bbox ban đầu
    x_adj = bbox_x - (size - bbox_w) // 2
    y_adj = bbox_y - (size - bbox_h) // 2
    
    # 3. Chống tràn viền ảnh
    box_x = max(0, x_adj)
    box_y = max(0, y_adj)
    box_w = min(size, img_w - box_x)
    box_h = min(size, img_h - box_y)
    
    # 4. Cắt ảnh
    crop = frame[box_y:box_y+box_h, box_x:box_x+box_w]
    if crop.size == 0:
        return None, None
        
    # 5. Resize về target size (64x64)
    face_resized_bgr = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
    face_resized_gray = cv2.cvtColor(face_resized_bgr, cv2.COLOR_BGR2GRAY)
    
    return face_resized_bgr, face_resized_gray
