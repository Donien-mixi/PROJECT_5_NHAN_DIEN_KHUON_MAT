import os
import re

lib_dir = r"D:\Arduino\libraries\TensorFlowLite_ESP32\src\tensorflow\lite\micro\kernels"
common_file = os.path.join(lib_dir, "dequantize_common.cpp")
deq_file = os.path.join(lib_dir, "dequantize.cpp")

# 1. Patch dequantize_common.cpp
if os.path.exists(common_file):
    with open(common_file, "r") as f:
        code = f.read()
    
    if "kTfLiteFloat16" not in code:
        code = code.replace("input->type == kTfLiteUInt8);", "input->type == kTfLiteUInt8 ||\n                              input->type == kTfLiteFloat16);")
        with open(common_file, "w") as f:
            f.write(code)
        print("Patched dequantize_common.cpp")
    else:
        print("dequantize_common.cpp already patched.")
else:
    print(f"Not found: {common_file}")

# 2. Patch dequantize.cpp
if os.path.exists(deq_file):
    with open(deq_file, "r") as f:
        code = f.read()
    
    fp16_func = """
#include <string.h>
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    if (exp == 0) {
        if (frac == 0) return sign ? -0.0f : 0.0f;
        while ((frac & 0x400) == 0) { frac <<= 1; exp--; }
        frac &= 0x3FF; exp++;
    } else if (exp == 0x1F) {
        return sign ? -1.0f : 1.0f; // Simplified INF/NAN
    }
    uint32_t f32 = (sign << 31) | ((exp - 15 + 127) << 23) | (frac << 13);
    float f; memcpy(&f, &f32, sizeof(f));
    return f;
}
"""

    if "kTfLiteFloat16" not in code:
        # Add fp16_to_fp32 function before DequantizeEval
        code = code.replace("namespace tflite {", "namespace tflite {\n" + fp16_func)
        
        # Add case kTfLiteFloat16:
        case_float16 = """
      case kTfLiteFloat16: {
        int num_elements = 1;
        for (int i = 0; i < input->dims->size; ++i) num_elements *= input->dims->data[i];
        const uint16_t* input_data = tflite::micro::GetTensorData<uint16_t>(input);
        float* output_data = tflite::micro::GetTensorData<float>(output);
        for (int i = 0; i < num_elements; ++i) {
            output_data[i] = fp16_to_fp32(input_data[i]);
        }
        break;
      }
      case kTfLiteUInt8:"""
        code = code.replace("case kTfLiteUInt8:", case_float16)
        
        with open(deq_file, "w") as f:
            f.write(code)
        print("Patched dequantize.cpp")
    else:
        print("dequantize.cpp already patched.")
else:
    print(f"Not found: {deq_file}")
