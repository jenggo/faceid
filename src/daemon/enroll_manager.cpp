#include "enroll_manager.h"
#include "model_store.h"
#include "face_detector_manager.h"
#include <syslog.h>
#include <cerrno>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <pwd.h>

EnrollmentManager::EnrollmentManager() {
    syslog(LOG_INFO, "EnrollmentManager initialized");
}

EnrollmentManager::~EnrollmentManager() {
    enroll_stop();  // Clean up any running threads
}

bool EnrollmentManager::enroll_start(const std::string& username) {
    EnrollStatus expected = EnrollStatus::IDLE;
    
    // Atomic compare-and-swap: only proceed if IDLE
    if (!status_.compare_exchange_strong(expected, EnrollStatus::ENROLLING)) {
        syslog(LOG_WARNING, "EnrollStart rejected: already enrolling");
        return false;
    }
    
    enrolling_username_ = username;
    cancel_enrollment_ = false;
    
    syslog(LOG_INFO, "EnrollStart: username=%s", username.c_str());
    
    // Launch enrollment thread
    try {
        enrollment_thread_ = std::make_unique<std::thread>(
            &EnrollmentManager::run_enrollment, this, nullptr, username
        );
        return true;
    } catch (const std::exception& e) {
        syslog(LOG_ERR, "Failed to start enrollment thread: %s", e.what());
        status_ = EnrollStatus::FAILED;
        return false;
    }
}

void EnrollmentManager::enroll_stop() {
    cancel_enrollment_ = true;
    
    if (enrollment_thread_ && enrollment_thread_->joinable()) {
        enrollment_thread_->join();
    }
    
    if (status_.exchange(EnrollStatus::IDLE) == EnrollStatus::ENROLLING) {
        syslog(LOG_INFO, "EnrollStop: enrollment cancelled by user");
    }
}

void EnrollmentManager::run_enrollment(sd_bus* bus, const std::string& username) {
    syslog(LOG_DEBUG, "Enrollment thread started for %s", username.c_str());
    
    try {
        // Phase 1: Capture training frames (progress 0-60%)
        int progress = 0;
        emit_enroll_status(bus, "capturing", progress, false);
        
        syslog(LOG_INFO, "Enrollment: capturing training frames...");
        capture_training_frames(progress);
        
        if (cancel_enrollment_) {
            syslog(LOG_INFO, "Enrollment cancelled during capture");
            emit_enroll_status(bus, "cancelled", progress, true);
            status_ = EnrollStatus::FAILED;
            return;
        }
        
        progress = 60;
        emit_enroll_status(bus, "training", progress, false);
        
        // Phase 2: Train model (progress 60-90%)
        syslog(LOG_INFO, "Enrollment: training model...");
        if (!train_model(username, progress)) {
            syslog(LOG_ERR, "Failed to train model");
            emit_enroll_status(bus, "train_error", 0, true);
            status_ = EnrollStatus::FAILED;
            return;
        }
        
        if (cancel_enrollment_) {
            syslog(LOG_INFO, "Enrollment cancelled during training");
            emit_enroll_status(bus, "cancelled", progress, true);
            status_ = EnrollStatus::FAILED;
            return;
        }
        
        progress = 90;
        emit_enroll_status(bus, "saving", progress, false);
        
        // Phase 3: Save model (progress 90-100%)
        syslog(LOG_INFO, "Enrollment: saving model...");
        if (!save_model(username)) {
            syslog(LOG_ERR, "Failed to save model");
            emit_enroll_status(bus, "save_error", 90, true);
            status_ = EnrollStatus::FAILED;
            return;
        }
        
        // Reload model store to pick up new model
        ModelStore::instance().load_models();
        
        progress = 100;
        syslog(LOG_INFO, "Enrollment completed successfully for %s", username.c_str());
        emit_enroll_status(bus, "success", progress, true);
        status_ = EnrollStatus::COMPLETED;
        
    } catch (const std::exception& e) {
        syslog(LOG_ERR, "Enrollment exception: %s", e.what());
        emit_enroll_status(bus, "error", 0, true);
        status_ = EnrollStatus::FAILED;
    }
}

void EnrollmentManager::capture_training_frames(int& progress) {
    // TODO: Implement actual frame capture
    // This should:
    // 1. Capture frames from camera
    // 2. Detect face in each frame
    // 3. Request user to rotate head for different angles
    // 4. Capture ~10-20 frames
    // 5. Update progress (0 → 60)
    
    // For now, simulate with delay
    for (int i = 0; i < 6; i++) {
        if (cancel_enrollment_.load()) break;
        
        progress = (i * 10);  // 0, 10, 20, 30, 40, 50
        syslog(LOG_DEBUG, "Enrollment: captured %d frames", i + 1);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    progress = 60;
}

bool EnrollmentManager::train_model(const std::string& username, int& progress) {
    // TODO: Implement actual model training
    // This should use the captured frames to train an SFace model
    
    // For now, simulate with delay
    for (int i = 0; i < 3; i++) {
        if (cancel_enrollment_) return false;
        
        progress = 60 + (i * 10);  // 60, 70, 80
        syslog(LOG_DEBUG, "Enrollment: training progress %d%%", progress);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
    
    progress = 90;
    return true;
}

bool EnrollmentManager::save_model(const std::string& username) {
    std::string model_path = get_model_path(username);
    
    // TODO: Write actual model file
    // This should save the trained SFace model to disk
    
    // Create a placeholder file for now
    FILE* f = fopen(model_path.c_str(), "w");
    if (!f) {
        syslog(LOG_ERR, "Failed to create model file: %s", strerror(errno));
        return false;
    }
    
    // Write placeholder model header
    fprintf(f, "FaceID Model v2.0\nUsername: %s\n", username.c_str());
    fclose(f);
    
    syslog(LOG_INFO, "Model saved: %s", model_path.c_str());
    return true;
}

std::string EnrollmentManager::get_model_path(const std::string& username) {
    const char* home = getenv("HOME");
    if (!home) {
        home = "/tmp";
    }
    
    std::string model_dir = std::string(home) + "/.local/share/faceid/models";
    std::string model_path = model_dir + "/" + username + ".faceid";
    
    return model_path;
}

void EnrollmentManager::emit_enroll_status(sd_bus* bus, const std::string& result, 
                                          int progress, bool done) {
    if (!bus) {
        syslog(LOG_DEBUG, "No bus available for signal emission");
        return;
    }
    
    sd_bus_error error = SD_BUS_ERROR_NULL;
    int ret = sd_bus_emit_signal(bus,
        "/org/freedesktop/FaceID/Device",
        "org.freedesktop.FaceID.Device",
        "EnrollStatus",
        "sib", result.c_str(), progress, done);
    
    if (ret < 0) {
        syslog(LOG_ERR, "Failed to emit EnrollStatus signal: %s", strerror(-ret));
    }
    
    sd_bus_error_free(&error);
}
