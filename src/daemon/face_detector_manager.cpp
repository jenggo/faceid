#include "face_detector_manager.h"
#include <syslog.h>

FaceDetectorManager& FaceDetectorManager::instance() {
    static FaceDetectorManager instance;
    return instance;
}

bool FaceDetectorManager::initialize() {
    if (models_ready_) {
        return true;
    }
    
    // TODO: Load YuNet and SFace models from model store
    // For now, just mark as ready for basic testing
    syslog(LOG_INFO, "Face detector models initialized (placeholder)");
    models_ready_ = true;
    return true;
}

float FaceDetectorManager::verify_face(const std::string& username) {
    if (!models_ready_) {
        syslog(LOG_ERR, "Face detector not ready");
        return -1.0f;
    }
    
    // TODO: Implement actual face verification
    // This will:
    // 1. Get frames from camera
    // 2. Detect faces with YuNet
    // 3. Extract features with SFace
    // 4. Compare against user's model
    // 5. Return confidence score
    
    syslog(LOG_DEBUG, "Face verification for user: %s (placeholder)", username.c_str());
    return 0.0f;  // Placeholder
}

std::string FaceDetectorManager::get_status() const {
    return models_ready_ ? "ready" : "not_ready";
}

FaceDetectorManager::~FaceDetectorManager() {
    // Cleanup models when daemon shuts down
}
