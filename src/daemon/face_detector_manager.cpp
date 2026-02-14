#include "face_detector_manager.h"
#include "config.h"
#include <syslog.h>

FaceDetectorManager::FaceDetectorManager() : models_ready_(false) {
    detector_ = std::make_unique<faceid::FaceDetector>();
}

FaceDetectorManager& FaceDetectorManager::instance() {
    static FaceDetectorManager instance;
    return instance;
}

bool FaceDetectorManager::initialize() {
    if (models_ready_) {
        return true;  // Already initialized
    }
    
    syslog(LOG_ERR, "FaceDetectorManager: !!!INITIALIZING!!! calling detector_->loadModels()");
    
    // Load embedded YuNet and SFace models
    bool load_result = detector_->loadModels();
    syslog(LOG_ERR, "FaceDetectorManager: !!!LOADMODELS_RETURNED!!! %s", load_result ? "true" : "false");
    
    if (!load_result) {
        syslog(LOG_ERR, "FaceDetectorManager: Failed to load models");
        return false;
    }
    
    models_ready_ = true;
    syslog(LOG_INFO, "FaceDetectorManager: YuNet and SFace loaded successfully");
    return true;
}

std::vector<faceid::Rect> FaceDetectorManager::detect_faces(const faceid::Image& frame) {
    if (!models_ready_) {
        syslog(LOG_WARNING, "FaceDetectorManager: Not initialized");
        return {};
    }
    
    std::lock_guard<std::mutex> lock(detector_mutex_);
    
    // Create non-owning view into frame
    faceid::ImageView view(
        const_cast<uint8_t*>(frame.data()),
        frame.width(),
        frame.height(),
        frame.channels()
    );
    
    // Detect faces using YuNet
    std::vector<faceid::Rect> faces = detector_->detectFaces(view);
    
    syslog(LOG_DEBUG, "FaceDetectorManager: Detected %zu faces", faces.size());
    return faces;
}

float FaceDetectorManager::verify_face(
    const faceid::Image& frame,
    const std::vector<faceid::Rect>& faces,
    const std::vector<faceid::FaceEncoding>& stored_models) {
    
    if (!models_ready_ || faces.empty() || stored_models.empty()) {
        return 0.0f;
    }
    
    std::lock_guard<std::mutex> lock(detector_mutex_);
    
    // Create non-owning view
    faceid::ImageView view(
        const_cast<uint8_t*>(frame.data()),
        frame.width(),
        frame.height(),
        frame.channels()
    );
    
    // Extract features from detected face(s) using SFace
    std::vector<faceid::FaceEncoding> encodings = 
        detector_->encodeFaces(view, faces);
    
    if (encodings.empty()) {
        syslog(LOG_WARNING, "FaceDetectorManager: Failed to extract features");
        return 0.0f;
    }
    
    // Compare with all stored models, find best match
    double best_match = 0.0;
    for (const auto& stored : stored_models) {
        double similarity = detector_->compareFaces(encodings[0], stored);
        if (similarity > best_match) {
            best_match = similarity;
        }
    }
    
    syslog(LOG_INFO, "FaceDetectorManager: Best match confidence: %.3f", best_match);
    
    return static_cast<float>(best_match);
}

faceid::FaceEncoding FaceDetectorManager::encode_face(
    const faceid::Image& frame,
    const faceid::Rect& face_box) {
    
    syslog(LOG_ERR, "FaceDetectorManager: !!!ENCODE_FACE_CALLED!!! models_ready_=%d", models_ready_);
    
    if (!models_ready_) {
        syslog(LOG_WARNING, "FaceDetectorManager: Not initialized for encoding (models_ready_=false)");
        return {};  // Return empty vector on error
    }
    
    syslog(LOG_DEBUG, "FaceDetectorManager: encode_face called with frame %dx%d, face_box (%d,%d,%d,%d)",
           frame.width(), frame.height(), face_box.x, face_box.y, face_box.width, face_box.height);
    
    std::lock_guard<std::mutex> lock(detector_mutex_);
    
    // Create non-owning view into frame
    faceid::ImageView view(
        const_cast<uint8_t*>(frame.data()),
        frame.width(),
        frame.height(),
        frame.channels()
    );
    
    // Get quality threshold from daemon YAML config
    double min_quality = Config::instance().recognition.min_face_quality;
    syslog(LOG_INFO, "FaceDetectorManager: Using min_face_quality=%.3f from YAML config", min_quality);
    
    // Extract features from the single face using SFace with quality threshold
    std::vector<faceid::FaceEncoding> encodings = 
        detector_->encodeFaces(view, {face_box}, min_quality);
    
    syslog(LOG_DEBUG, "FaceDetectorManager: encodeFaces returned %zu encodings", encodings.size());
    
    if (encodings.empty()) {
        syslog(LOG_WARNING, "FaceDetectorManager: Failed to extract face encoding (encodeFaces returned empty)");
        return {};
    }
    
    const auto& encoding = encodings[0];
    syslog(LOG_INFO, "FaceDetectorManager: Extracted face encoding (%zu dimensions)", encoding.size());
    
    return encoding;
}

std::string FaceDetectorManager::get_status() const {
    return models_ready_ ? "ready" : "not_ready";
}

FaceDetectorManager::~FaceDetectorManager() {
    // Unique_ptr automatically cleans up detector
}
