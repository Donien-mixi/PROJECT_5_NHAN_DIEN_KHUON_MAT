#ifndef IMAGE_DECODER_H
#define IMAGE_DECODER_H

#include <TJpg_Decoder.h>
#include <Arduino.h>

void setup_image_decoder();
void decode_jpeg_frame(const uint8_t* jpeg_data, uint32_t length);

#endif // IMAGE_DECODER_H
