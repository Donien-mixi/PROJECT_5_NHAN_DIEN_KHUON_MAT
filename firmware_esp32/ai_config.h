// AUTO-GENERATED CONFIGURATION FILE - DO NOT EDIT MANUALLY
// Generated on: 20260831_101428

#ifndef AI_CONFIG_H
#define AI_CONFIG_H

// --- ESP32 MEMORY ALLOCATION FOR TENSORFLOW LITE ---
// These values are automatically generated based on the models in the AI Builder.

// 1. Face Detector (BlazeFace)
#define DETECTOR_ARENA_SIZE (1536000) // 1500 KB

// 2. Face Recognizer (GhostFaceNet)
#define RECOGNIZER_ARENA_SIZE (786432) // 768 KB

// 3. Face Image Input Size
#define FACE_TARGET_SIZE 64

#endif // AI_CONFIG_H
