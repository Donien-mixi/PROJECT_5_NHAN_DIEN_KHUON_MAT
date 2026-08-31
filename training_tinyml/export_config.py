import os
import datetime

def export_ai_config():
    print("[*] Generating ai_config.h for ESP32...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    firmware_dir = os.path.join(project_root, "firmware_esp32")
    config_path = os.path.join(firmware_dir, "ai_config.h")
    
    # In a real dynamic scenario, you would parse the .tflite file using tflite runtime
    # to determine exact arena size. Here we define safe limits.
    detector_arena_size = 1500 * 1024  # 1500 KB for BlazeFace float32
    recognizer_arena_size = 768 * 1024 # 768 KB ( increased from 240KB ) for new GhostFaceNet int8
    
    version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    h_content = f"""// AUTO-GENERATED CONFIGURATION FILE - DO NOT EDIT MANUALLY
// Generated on: {version}

#ifndef AI_CONFIG_H
#define AI_CONFIG_H

// --- ESP32 MEMORY ALLOCATION FOR TENSORFLOW LITE ---
// These values are automatically generated based on the models in the AI Builder.

// 1. Face Detector (BlazeFace)
#define DETECTOR_ARENA_SIZE ({detector_arena_size}) // {detector_arena_size / 1024:.0f} KB

// 2. Face Recognizer (GhostFaceNet)
#define RECOGNIZER_ARENA_SIZE ({recognizer_arena_size}) // {recognizer_arena_size / 1024:.0f} KB

// 3. Face Image Input Size
#define FACE_TARGET_SIZE 64

#endif // AI_CONFIG_H
"""
    
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(h_content)
        
    print(f"[+] Successfully exported configuration to: {config_path}")
    print("    ESP32 Firmware will automatically allocate the correct RAM sizes on next build.")

if __name__ == "__main__":
    export_ai_config()
