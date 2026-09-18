#ifndef CAMERA_DRIVER_H
#define CAMERA_DRIVER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "esp_camera.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

extern volatile bool is_new_frame_available;
extern SemaphoreHandle_t image_mutex;

bool setup_camera();
bool capture_frame_to_buffer();
bool get_latest_jpeg_frame(uint8_t* dest, size_t max_len, size_t* out_len);
void set_camera_orientation(int vflip, int hmirror);

#endif // CAMERA_DRIVER_H
