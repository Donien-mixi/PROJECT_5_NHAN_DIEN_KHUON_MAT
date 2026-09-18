#include "web_server.h"
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include "esp_http_server.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "camera_driver.h"

static const char* TAG = "WEB_SERVER";

// Cấu hình Wi-Fi (Tự động kết nối, nếu không có mạng thì vẫn chạy AI bình thường)
static const char *wifi_ssid = "Phong 2";
static const char *wifi_password = "Kimhoang";

static httpd_handle_t s_httpd = NULL;

static char s_last_name[32] = "Dang cho mat...";
static float s_last_sim = 0.0f;
static bool s_last_matched = false;

static int s_vflip = 0;
static int s_hmirror = 0;

static uint8_t* s_stream_buf = nullptr;
static const size_t STREAM_BUF_SIZE = 65536; // 64KB PSRAM

void update_ai_status(const char* name, float similarity, bool is_matched) {
    if (name) {
        strncpy(s_last_name, name, sizeof(s_last_name) - 1);
        s_last_name[sizeof(s_last_name) - 1] = '\0';
    }
    s_last_sim = similarity;
    s_last_matched = is_matched;
}

// -------------------------------------------------------------------------
// HTTP Handlers
// -------------------------------------------------------------------------

static esp_err_t index_handler(httpd_req_t *req) {
    const char index_html[] = R"rawliteral(
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ESP32-S3 AI Face Attendance (OV5640)</title>
    <style>
        :root {
            --bg: #0f172a;
            --card: #1e293b;
            --text: #f8fafc;
            --primary: #38bdf8;
            --success: #22c55e;
            --danger: #ef4444;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .container {
            max-width: 600px;
            width: 100%;
        }
        h1 {
            font-size: 1.3rem;
            color: var(--primary);
            text-align: center;
            margin-bottom: 16px;
        }
        .card {
            background: var(--card);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
            text-align: center;
        }
        img.stream {
            width: 100%;
            max-width: 320px;
            height: auto;
            border-radius: 8px;
            border: 2px solid var(--primary);
            background: #000;
        }
        .status-badge {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 1rem;
            font-weight: bold;
            margin-top: 10px;
        }
        .matched { background: rgba(34, 197, 94, 0.2); color: var(--success); border: 1px solid var(--success); }
        .unknown { background: rgba(239, 68, 68, 0.2); color: var(--danger); border: 1px solid var(--danger); }
        .btn-group {
            display: flex;
            gap: 8px;
            justify-content: center;
            margin-top: 12px;
            flex-wrap: wrap;
        }
        button {
            background: #334155;
            color: white;
            border: none;
            padding: 8px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85rem;
            transition: 0.2s;
        }
        button:hover { background: var(--primary); color: #000; }
        .btn-danger { background: #7f1d1d; }
        .btn-danger:hover { background: var(--danger); color: white; }
        pre.log-box {
            background: #020617;
            padding: 12px;
            border-radius: 8px;
            text-align: left;
            max-height: 180px;
            overflow-y: auto;
            font-size: 0.8rem;
            color: #94a3b8;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 ESP32-S3 TinyML Face Attendance (OV5640) - ESP-IDF 5.3</h1>
        
        <div class="card">
            <img id="cameraFeed" class="stream" src="/capture" alt="Live Camera Preview" />
            <div id="badge" class="status-badge unknown">Đang chờ quét...</div>
            <div style="margin-top: 8px; font-size: 0.85rem; color: #94a3b8;">
                Độ tin cậy: <span id="conf" style="color:var(--text);font-weight:bold;">0%</span> | 
                PSRAM: <span id="psram">--</span>
            </div>
            
            <div class="btn-group">
                <button onclick="toggleFlip('v')">🔄 Lật dọc (V-Flip)</button>
                <button onclick="toggleFlip('h')">↔️ Lật ngang (Mirror)</button>
                <button onclick="fetchAttendance()">📄 Xem lịch sử</button>
                <button class="btn-danger" onclick="clearAttendance()">🗑️ Xóa log</button>
            </div>
        </div>

        <div class="card">
            <h3 style="margin-top:0;font-size:1rem;color:var(--primary);text-align:left;">📋 Lịch sử Điểm danh (SPIFFS)</h3>
            <pre class="log-box" id="logContent">Đang tải lịch sử...</pre>
        </div>
    </div>

    <script>
        function updateStatus() {
            fetch('/status')
                .then(r => r.json())
                .then(d => {
                    const badge = document.getElementById('badge');
                    if (d.name && d.name !== 'Unknown' && d.matched) {
                        badge.className = 'status-badge matched';
                        badge.innerText = '✅ ' + d.name;
                    } else if (d.name === 'Unknown') {
                        badge.className = 'status-badge unknown';
                        badge.innerText = '⚠️ Người lạ (Unknown)';
                    } else {
                        badge.className = 'status-badge unknown';
                        badge.innerText = d.name;
                    }
                    document.getElementById('conf').innerText = (d.sim * 100).toFixed(1) + '%';
                    document.getElementById('psram').innerText = (d.psram / (1024*1024)).toFixed(1) + 'MB';
                })
                .catch(() => {});
        }
        setInterval(updateStatus, 1000);

        function toggleFlip(type) {
            fetch('/flip?type=' + type);
        }

        function fetchAttendance() {
            fetch('/attendance')
                .then(r => r.text())
                .then(t => {
                    document.getElementById('logContent').innerText = t || 'Chưa có bản ghi điểm danh nào.';
                });
        }

        function clearAttendance() {
            if (confirm('Bạn có chắc muốn xóa toàn bộ lịch sử điểm danh trên ESP32?')) {
                fetch('/attendance/clear').then(() => fetchAttendance());
            }
        }

        const streamImg = document.getElementById('cameraFeed');
        let isFetchingFrame = false;
        function refreshLiveFrame() {
            if (isFetchingFrame) return;
            isFetchingFrame = true;
            const nextImg = new Image();
            nextImg.onload = function() {
                if (streamImg) streamImg.src = nextImg.src;
                isFetchingFrame = false;
                setTimeout(refreshLiveFrame, 80);
            };
            nextImg.onerror = function() {
                isFetchingFrame = false;
                setTimeout(refreshLiveFrame, 300);
            };
            nextImg.src = '/capture?t=' + Date.now();
        }
        refreshLiveFrame();
        fetchAttendance();
    </script>
</body>
</html>
)rawliteral";
    httpd_resp_set_type(req, "text/html; charset=utf-8");
    return httpd_resp_send(req, index_html, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t status_handler(httpd_req_t *req) {
    char json[256];
    snprintf(json, sizeof(json),
             "{\"name\":\"%s\",\"sim\":%.3f,\"matched\":%s,\"heap\":%u,\"psram\":%u,\"uptime\":%llu}",
             s_last_name, s_last_sim, s_last_matched ? "true" : "false",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
             (unsigned long long)(esp_timer_get_time() / 1000000));
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, json, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t capture_handler(httpd_req_t *req) {
    size_t stream_len = 0;
    if (s_stream_buf && get_latest_jpeg_frame(s_stream_buf, 65536, &stream_len) && stream_len > 0) {
        httpd_resp_set_type(req, "image/jpeg");
        httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=capture.jpg");
        httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
        return httpd_resp_send(req, (const char *)s_stream_buf, stream_len);
    }
    return httpd_resp_send_404(req);
}

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static esp_err_t stream_handler(httpd_req_t *req) {
    esp_err_t res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if (res != ESP_OK) return res;
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    size_t stream_len = 0;
    char part_buf[64];

    while (true) {
        if (s_stream_buf && get_latest_jpeg_frame(s_stream_buf, 65536, &stream_len) && stream_len > 0) {
            res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
            if (res != ESP_OK) break;

            size_t hlen = snprintf(part_buf, sizeof(part_buf), _STREAM_PART, stream_len);
            res = httpd_resp_send_chunk(req, part_buf, hlen);
            if (res != ESP_OK) break;

            res = httpd_resp_send_chunk(req, (const char *)s_stream_buf, stream_len);
            if (res != ESP_OK) break;
        }
        vTaskDelay(pdMS_TO_TICKS(60));
    }
    return res;
}

static esp_err_t attendance_handler(httpd_req_t *req) {
    FILE* f = fopen("/spiffs/attendance.csv", "r");
    if (!f) {
        return httpd_resp_send(req, "Chua co du lieu diem danh.", HTTPD_RESP_USE_STRLEN);
    }
    httpd_resp_set_type(req, "text/plain; charset=utf-8");
    char chunk[256];
    size_t bytes;
    while ((bytes = fread(chunk, 1, sizeof(chunk), f)) > 0) {
        httpd_resp_send_chunk(req, chunk, bytes);
    }
    fclose(f);
    return httpd_resp_send_chunk(req, NULL, 0);
}

static esp_err_t attendance_clear_handler(httpd_req_t *req) {
    unlink("/spiffs/attendance.csv");
    FILE* f = fopen("/spiffs/attendance.csv", "w");
    if (f) fclose(f);
    httpd_resp_set_type(req, "text/plain");
    return httpd_resp_send(req, "OK", HTTPD_RESP_USE_STRLEN);
}

static esp_err_t flip_handler(httpd_req_t *req) {
    char param[32];
    if (httpd_req_get_url_query_str(req, param, sizeof(param)) == ESP_OK) {
        char type[8];
        if (httpd_query_key_value(param, "type", type, sizeof(type)) == ESP_OK) {
            if (strcmp(type, "v") == 0) {
                s_vflip = !s_vflip;
            } else if (strcmp(type, "h") == 0) {
                s_hmirror = !s_hmirror;
            }
            set_camera_orientation(s_vflip, s_hmirror);
        }
    }
    httpd_resp_set_type(req, "text/plain");
    return httpd_resp_send(req, "OK", HTTPD_RESP_USE_STRLEN);
}

void setup_web_server() {
    ESP_LOGI(TAG, "Đang kết nối Wi-Fi tới: %s ...", wifi_ssid);

    // Khởi tạo Wi-Fi Station Mode trên ESP-IDF
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_t* sta_netif = esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);

    wifi_config_t wifi_config;
    memset(&wifi_config, 0, sizeof(wifi_config));
    strncpy((char*)wifi_config.sta.ssid, wifi_ssid, sizeof(wifi_config.sta.ssid));
    strncpy((char*)wifi_config.sta.password, wifi_password, sizeof(wifi_config.sta.password));

    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    esp_wifi_start();
    esp_wifi_connect();

    // Chờ kết nối Wi-Fi với timeout 4s
    int retry = 0;
    esp_netif_ip_info_t ip_info;
    bool connected = false;
    while (retry < 16) {
        vTaskDelay(pdMS_TO_TICKS(250));
        if (esp_netif_get_ip_info(sta_netif, &ip_info) == ESP_OK && ip_info.ip.addr != 0) {
            connected = true;
            break;
        }
        retry++;
    }

    if (!connected) {
        ESP_LOGW(TAG, "⚠️ Không thể kết nối Wi-Fi! Hệ thống chạy 100%% OFFLINE độc lập.");
        return;
    }

    ESP_LOGI(TAG, "✅ Đã kết nối Wi-Fi thành công!");
    ESP_LOGI(TAG, "👉 Địa chỉ IP Web Dashboard: http://" IPSTR, IP2STR(&ip_info.ip));

    if (!s_stream_buf) {
        s_stream_buf = (uint8_t*)heap_caps_malloc(STREAM_BUF_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!s_stream_buf) {
            ESP_LOGE(TAG, "❌ LỖI: Không thể cấp phát s_stream_buf trên PSRAM!");
            return;
        }
    }

    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;
    config.ctrl_port = 32768;
    config.max_open_sockets = 4;
    config.stack_size = 12288;

    httpd_uri_t uri_index = { .uri = "/", .method = HTTP_GET, .handler = index_handler, .user_ctx = NULL };
    httpd_uri_t uri_capture = { .uri = "/capture", .method = HTTP_GET, .handler = capture_handler, .user_ctx = NULL };
    httpd_uri_t uri_status = { .uri = "/status", .method = HTTP_GET, .handler = status_handler, .user_ctx = NULL };
    httpd_uri_t uri_stream = { .uri = "/stream", .method = HTTP_GET, .handler = stream_handler, .user_ctx = NULL };
    httpd_uri_t uri_attendance = { .uri = "/attendance", .method = HTTP_GET, .handler = attendance_handler, .user_ctx = NULL };
    httpd_uri_t uri_att_clear = { .uri = "/attendance/clear", .method = HTTP_GET, .handler = attendance_clear_handler, .user_ctx = NULL };
    httpd_uri_t uri_flip = { .uri = "/flip", .method = HTTP_GET, .handler = flip_handler, .user_ctx = NULL };

    if (httpd_start(&s_httpd, &config) == ESP_OK) {
        httpd_register_uri_handler(s_httpd, &uri_index);
        httpd_register_uri_handler(s_httpd, &uri_capture);
        httpd_register_uri_handler(s_httpd, &uri_status);
        httpd_register_uri_handler(s_httpd, &uri_stream);
        httpd_register_uri_handler(s_httpd, &uri_attendance);
        httpd_register_uri_handler(s_httpd, &uri_att_clear);
        httpd_register_uri_handler(s_httpd, &uri_flip);
        ESP_LOGI(TAG, "🌐 [Web Server] HTTP Server đã khởi động trên port 80!");
    } else {
        ESP_LOGE(TAG, "❌ [Web Server] Khởi động thất bại!");
    }
}
