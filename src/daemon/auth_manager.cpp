#include "auth_manager.h"
#include "config.h"
#include "face_detector_manager.h"
#include "camera_manager.h"
#include "model_store.h"
#include "fingerprint_manager.h"
#include <syslog.h>
#include <cstring>
#include <chrono>
#include <cerrno>

AuthManager::AuthManager() {
    syslog(LOG_INFO, "AuthManager initialized");
}

AuthManager::~AuthManager() {
    verify_stop();  // Clean up any running threads
    syslog(LOG_INFO, "AuthManager destroyed");
}

bool AuthManager::should_use_face(const std::string& context) const {
    // Face is always used by default (unless explicitly disabled)
    const Config& config = Config::instance();
    return config.authentication.face;
}

bool AuthManager::should_use_fingerprint(const std::string& context) const {
    const Config& config = Config::instance();
    
    // First check if fingerprint auth is enabled globally
    if (!config.authentication.fingerprint) {
        return false;
    }
    
    // Then check context-specific settings
    if (context == "lockscreen") {
        return config.authentication.lockscreen.fingerprint;
    } else if (context == "sudo") {
        return config.authentication.sudo.fingerprint;
    }
    
    // Default: use fingerprint for other contexts
    return true;
}

bool AuthManager::verify_start(const std::string& username, const std::string& context) {
    AuthStatus expected = AuthStatus::IDLE;
    
    // Atomic compare-and-swap: only proceed if IDLE
    if (!status_.compare_exchange_strong(expected, AuthStatus::VERIFYING)) {
        syslog(LOG_WARNING, "VerifyStart rejected: already verifying");
        return false;
    }
    
    current_username_ = username;
    current_context_ = context;
    cancel_verification_ = false;
    
    syslog(LOG_INFO, "VerifyStart: username=%s context=%s", 
           username.c_str(), context.c_str());
    
    // Determine which biometric methods to use for this context
    bool use_face = should_use_face(context);
    bool use_fingerprint = should_use_fingerprint(context);
    
    syslog(LOG_DEBUG, "VerifyStart: use_face=%d use_fingerprint=%d", 
           use_face, use_fingerprint);
    
    // Launch verification threads
    // Note: These threads will set status_ to COMPLETED or FAILED
    // when they finish. First one to succeed will signal success.
    
    if (use_face) {
        // Face verification runs immediately
        verification_threads_.push_back(
            std::thread(&AuthManager::run_face_verification, this, nullptr, username)
        );
    }
    
    if (use_fingerprint) {
        // Fingerprint verification gets a 500ms delay (face head start)
        verification_threads_.push_back(
            std::thread(&AuthManager::run_fingerprint_verification, this, nullptr, username)
        );
    }
    
    return true;
}

void AuthManager::verify_stop() {
    cancel_verification_ = true;
    
    // Wait for all verification threads to finish
    for (auto& thread : verification_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    verification_threads_.clear();
    
    if (status_.exchange(AuthStatus::IDLE) == AuthStatus::VERIFYING) {
        syslog(LOG_INFO, "VerifyStop: authentication cancelled by user");
    }
}

void AuthManager::run_face_verification(sd_bus* bus, const std::string& username) {
    syslog(LOG_DEBUG, "Face verification thread started for %s", username.c_str());
    
    // Step 1: Initialize face detector if needed
    FaceDetectorManager& face_mgr = FaceDetectorManager::instance();
    if (!face_mgr.is_ready()) {
        syslog(LOG_DEBUG, "Face detector not ready, initializing...");
        if (!face_mgr.initialize()) {
            syslog(LOG_ERR, "Failed to initialize face detector");
            emit_verify_status(bus, "face_error", true);
            status_ = AuthStatus::FAILED;
            return;
        }
    }
    
    // Step 2: Initialize camera if needed
    CameraManager& camera_mgr = CameraManager::instance();
    if (!camera_mgr.is_ready()) {
        syslog(LOG_DEBUG, "Camera not ready, initializing...");
        if (!camera_mgr.initialize()) {
            syslog(LOG_ERR, "Failed to initialize camera");
            emit_verify_status(bus, "camera_error", true);
            status_ = AuthStatus::FAILED;
            return;
        }
    }
    
    // Step 3: Capture frame
    faceid::Image frame = camera_mgr.capture_frame();
    if (frame.empty()) {
        syslog(LOG_ERR, "Failed to capture frame from camera");
        emit_verify_status(bus, "camera_error", true);
        status_ = AuthStatus::FAILED;
        return;
    }
    
    if (cancel_verification_) {
        syslog(LOG_DEBUG, "Face verification cancelled after frame capture");
        return;
    }
    
    // Step 4: Detect faces
    std::vector<faceid::Rect> faces = face_mgr.detect_faces(frame);
    if (faces.empty()) {
        syslog(LOG_INFO, "No faces detected for %s", username.c_str());
        emit_verify_status(bus, "no_face", true);
        status_ = AuthStatus::FAILED;
        return;
    }
    
    if (cancel_verification_) {
        syslog(LOG_DEBUG, "Face verification cancelled after detection");
        return;
    }
    
    // Step 5: Load user models
    ModelStore& model_store = ModelStore::instance();
    std::vector<faceid::FaceEncoding> stored_models = model_store.get_user_model(username);
    if (stored_models.empty()) {
        syslog(LOG_INFO, "No face models enrolled for %s", username.c_str());
        emit_verify_status(bus, "no_model", true);
        status_ = AuthStatus::FAILED;
        return;
    }
    
    // Step 6: Verify face
    float confidence = face_mgr.verify_face(frame, faces, stored_models);
    
    if (cancel_verification_) {
        syslog(LOG_DEBUG, "Face verification cancelled");
        return;
    }
    
    // Step 7: Check confidence threshold
    float threshold = 0.40f;  // Default threshold (can be made configurable later)
    
    if (confidence >= threshold) {
        syslog(LOG_INFO, "Face verification successful for %s (confidence: %.3f >= %.3f)", 
               username.c_str(), confidence, threshold);
        emit_verify_status(bus, "success", true);
        status_ = AuthStatus::COMPLETED;
    } else {
        syslog(LOG_WARNING, "Face verification failed for %s (confidence: %.3f < %.3f)", 
               username.c_str(), confidence, threshold);
        emit_verify_status(bus, "face_failure", true);
        status_ = AuthStatus::FAILED;
    }
}

void AuthManager::run_fingerprint_verification(sd_bus* bus, const std::string& username) {
    // Give face a 500ms head start
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    if (cancel_verification_ || status_ == AuthStatus::COMPLETED) {
        syslog(LOG_DEBUG, "Fingerprint verification cancelled or already succeeded");
        return;
    }
    
    syslog(LOG_DEBUG, "Fingerprint verification thread started for %s", username.c_str());
    
    faceid::daemon::FingerprintManager& fp_mgr = faceid::daemon::FingerprintManager::instance();
    
    if (!fp_mgr.is_available()) {
        syslog(LOG_DEBUG, "Fingerprint reader not available");
        return;
    }
    
    // Run fingerprint verification with cancel flag
    std::atomic<bool> cancel_flag{false};
    float confidence = fp_mgr.verify_fingerprint(username, cancel_flag);
    
    if (cancel_verification_) {
        cancel_flag.store(true);
        syslog(LOG_DEBUG, "Fingerprint verification cancelled");
        return;
    }
    
    if (confidence > 0.7f && status_ == AuthStatus::VERIFYING) {
        syslog(LOG_INFO, "Fingerprint verification successful (confidence: %.2f)", confidence);
        emit_verify_status(bus, "success", true);
        status_ = AuthStatus::COMPLETED;
    } else if (status_ == AuthStatus::VERIFYING) {
        syslog(LOG_WARNING, "Fingerprint verification failed (confidence: %.2f)", confidence);
        emit_verify_status(bus, "fingerprint_failure", true);
        status_ = AuthStatus::FAILED;
    }
}

void AuthManager::emit_verify_status(sd_bus* bus, const std::string& result, bool done) {
    if (!bus) {
        syslog(LOG_DEBUG, "No bus available for signal emission");
        return;
    }
    
    sd_bus_error error = SD_BUS_ERROR_NULL;
    int ret = sd_bus_emit_signal(bus,
        "/org/freedesktop/FaceID/Device",
        "org.freedesktop.FaceID.Device",
        "VerifyStatus",
        "sb", result.c_str(), done);
    
    if (ret < 0) {
        syslog(LOG_ERR, "Failed to emit VerifyStatus signal: %s", strerror(-ret));
    }
    
    sd_bus_error_free(&error);
}

int AuthManager::register_methods(sd_bus* bus) {
    // TODO: Register D-Bus method handlers using sd_bus_add_object_vtable
    // This will be connected to the D-Bus interface
    syslog(LOG_INFO, "AuthManager: D-Bus methods registered (placeholder)");
    return 0;
}
