#ifndef WEB_SERVER_H
#define WEB_SERVER_H

#include <stdbool.h>

// Khởi tạo Wi-Fi và Web Server HTTP nền (tự động bỏ qua nếu không có mạng)
void setup_web_server();

// Cập nhật thông tin nhận diện AI từ Core 0 sang Web Server để hiển thị lên Web Dashboard
void update_ai_status(const char* name, float similarity, bool is_matched);

#endif // WEB_SERVER_H
