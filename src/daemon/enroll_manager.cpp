#include "enroll_manager.h"
#include "fingerprint_manager.h"
#include "model_store.h"
#include "face_detector_manager.h"
#include "camera_manager.h"
#include "../models/binary_model.h"
#include <syslog.h>
#include <cerrno>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <pwd.h>
#include <ctime>
#include <sys/stat.h>

EnrollmentManager::EnrollmentManager() {
    syslog(LOG_INFO, "EnrollmentManager initialized");
}

EnrollmentManager::~EnrollmentManager() {
    enroll_stop();  // Clean up any running threads
}

bool EnrollmentManager::enroll_start(sd_bus* bus, const std::string& username) {
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
            &EnrollmentManager::run_enrollment, this, bus, username
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
         int progress = 0;
         emit_enroll_status(bus, username, "in_progress", progress, "Initializing enrollment...");
         syslog(LOG_INFO, "Enrollment: Starting enrollment for %s", username.c_str());
         
         // Initialize managers (reuse from migrate-ml-models)
         FaceDetectorManager& detector = FaceDetectorManager::instance();
         if (!detector.is_ready()) {
             if (!detector.initialize()) {
                 syslog(LOG_ERR, "Enrollment: Failed to initialize FaceDetectorManager");
                 emit_enroll_status(bus, username, "error", 0, "Failed to initialize face detector");
                 status_ = EnrollStatus::FAILED;
                 return;
             }
         }
         
         CameraManager& camera = CameraManager::instance();
         if (!camera.is_ready()) {
             if (!camera.initialize()) {
                 syslog(LOG_ERR, "Enrollment: Failed to initialize CameraManager");
                 emit_enroll_status(bus, username, "error", 0, "Failed to access camera");
                 status_ = EnrollStatus::FAILED;
                 return;
             }
         }
         
         progress = 10;
         emit_enroll_status(bus, username, "in_progress", progress, "Managers initialized - starting capture");
         
         if (cancel_enrollment_) {
             emit_enroll_status(bus, username, "cancelled", progress, "Enrollment cancelled by user");
             status_ = EnrollStatus::FAILED;
             return;
         }
        
        // === MULTI-POSE CAPTURE (5 Poses) ===
        // Task 2.1 & 2.2 & 2.3: Multi-pose iteration with prompts and delays
        
        std::vector<std::string> pose_names = {"center", "left", "right", "up", "down"};
        std::vector<std::string> pose_prompts = {
            "Face detected - hold still",
            "Turn your head slightly left",
            "Turn your head slightly right",
            "Tilt your head slightly up",
            "Tilt your head slightly down"
        };
        
        std::vector<faceid::FaceEncoding> pose_encodings;
        
        for (size_t pose_idx = 0; pose_idx < pose_names.size(); ++pose_idx) {
            const auto& pose_name = pose_names[pose_idx];
            const auto& prompt = pose_prompts[pose_idx];
            
            // Task 2.3: Wait for user adjustment (except first pose)
            if (pose_idx > 0) {
                syslog(LOG_DEBUG, "Enrollment: Waiting for pose adjustment - %s", prompt.c_str());
                std::this_thread::sleep_for(std::chrono::milliseconds(1500));
            }
            
            // Task 3.1 & 3.2: Retry logic for face detection
            const int MAX_RETRIES = 3;
            const int POSE_TIMEOUT_SECONDS = 30;
            const int GLOBAL_TIMEOUT_SECONDS = 180;
            
            auto pose_start_time = std::chrono::steady_clock::now();
            auto global_start_time = std::chrono::steady_clock::now();
            int retry_count = 0;
            faceid::FaceEncoding encoding;
            bool pose_success = false;
            
            while (retry_count < MAX_RETRIES && !pose_success) {
                // Task 3.2: Check per-pose timeout (30 seconds)
                auto elapsed = std::chrono::steady_clock::now() - pose_start_time;
                if (elapsed > std::chrono::seconds(POSE_TIMEOUT_SECONDS)) {
                     syslog(LOG_WARNING, "Enrollment: Pose %s timed out after %d seconds", 
                            pose_name.c_str(), POSE_TIMEOUT_SECONDS);
                     emit_enroll_status(bus, username, "timeout", progress, 
                                       "Pose capture timed out - please try again");
                     status_ = EnrollStatus::FAILED;
                     return;
                 }
                 
                 // Task 3.3: Check global timeout (3 minutes)
                 auto global_elapsed = std::chrono::steady_clock::now() - global_start_time;
                 if (global_elapsed > std::chrono::seconds(GLOBAL_TIMEOUT_SECONDS)) {
                     syslog(LOG_WARNING, "Enrollment: Global timeout after %d seconds", GLOBAL_TIMEOUT_SECONDS);
                     emit_enroll_status(bus, username, "timeout", progress, 
                                       "Enrollment took too long - please try again");
                     status_ = EnrollStatus::FAILED;
                     return;
                 }
                 
                 // Check for cancellation
                 if (cancel_enrollment_) {
                     emit_enroll_status(bus, username, "cancelled", progress, "Enrollment cancelled");
                     status_ = EnrollStatus::FAILED;
                     return;
                 }
                
                // Task 2.2: Emit pose prompt
                if (retry_count == 0) {
                    syslog(LOG_INFO, "Enrollment: Capturing pose %zu/5 - %s", 
                           pose_idx + 1, pose_name.c_str());
                } else {
                    syslog(LOG_INFO, "Enrollment: Retry %d/3 for pose %s", 
                           retry_count, pose_name.c_str());
                }
                
                // Task 2.1: Capture frame
                faceid::Image frame = camera.capture_frame();
                
                 if (frame.empty()) {
                     syslog(LOG_ERR, "Enrollment: Failed to capture frame for pose %s", 
                            pose_name.c_str());
                     emit_enroll_status(bus, username, "error", progress, 
                                       "Failed to capture frame from camera");
                     status_ = EnrollStatus::FAILED;
                     return;
                 }
                
                syslog(LOG_DEBUG, "Enrollment: Frame captured for pose %s (%dx%d)", 
                       pose_name.c_str(), frame.width(), frame.height());
                
                // Task 2.1: Detect faces
                std::vector<faceid::Rect> faces = detector.detect_faces(frame);
                syslog(LOG_DEBUG, "Enrollment: Detected %zu faces in pose %s", 
                       faces.size(), pose_name.c_str());
                
                // Task 3.1: Validate exactly 1 face with retry logic
                if (faces.empty()) {
                    syslog(LOG_WARNING, "Enrollment: No face detected in pose %s (retry %d)", 
                           pose_name.c_str(), retry_count + 1);
                    retry_count++;
                    
                    if (retry_count >= MAX_RETRIES) {
                        emit_enroll_status(bus, username, "no_face", progress, "No face detected - please look at camera");
                        status_ = EnrollStatus::FAILED;
                        return;
                    }
                    continue;  // Retry this pose
                }
                
                if (faces.size() > 1) {
                    syslog(LOG_WARNING, "Enrollment: Multiple faces detected in pose %s (%zu, retry %d)", 
                           pose_name.c_str(), faces.size(), retry_count + 1);
                    retry_count++;
                    
                    if (retry_count >= MAX_RETRIES) {
                        emit_enroll_status(bus, username, "multiple_faces", progress, "Multiple faces detected - ensure only one person is visible");
                        status_ = EnrollStatus::FAILED;
                        return;
                    }
                    continue;  // Retry this pose
                }
                
                // Task 3.4: Validate face bounding box size
                const auto& face_box = faces[0];
                const int MIN_FACE_SIZE = 64;
                
                if (face_box.width < MIN_FACE_SIZE || face_box.height < MIN_FACE_SIZE) {
                    syslog(LOG_WARNING, "Enrollment: Face too small (%dx%d) in pose %s (retry %d)", 
                           face_box.width, face_box.height, pose_name.c_str(), retry_count + 1);
                    retry_count++;
                    
                    if (retry_count >= MAX_RETRIES) {
                        emit_enroll_status(bus, username, "quality_error", progress, "Face too small - move closer to camera");
                        status_ = EnrollStatus::FAILED;
                        return;
                    }
                    continue;  // Retry this pose
                }
                
                // Task 2.1: Extract encoding
                encoding = detector.encode_face(frame, faces[0]);
                
                if (encoding.empty()) {
                    syslog(LOG_ERR, "Enrollment: Failed to extract encoding for pose %s", 
                           pose_name.c_str());
                    emit_enroll_status(bus, username, "encoding_error", progress, "Failed to extract face encoding");
                    status_ = EnrollStatus::FAILED;
                    return;
                }
                
                syslog(LOG_INFO, "Enrollment: Extracted encoding for pose %s (%zu dims)", 
                       pose_name.c_str(), encoding.size());
                
                pose_success = true;
            }
            
            if (!pose_success) {
                syslog(LOG_ERR, "Enrollment: Failed to capture pose %s after %d retries", 
                       pose_name.c_str(), MAX_RETRIES);
                emit_enroll_status(bus, username, "no_face", progress, "No face detected - please look at camera");
                status_ = EnrollStatus::FAILED;
                return;
            }
            
            // Task 2.1: Store encoding
            pose_encodings.push_back(encoding);
            
            // Task 2.1: Update progress (0->20->40->60->80)
            progress = static_cast<int>((pose_idx + 1) * 20);
            emit_enroll_status(bus, username, "in_progress", progress, "Capturing poses...");
        }
        
        // === MODEL CREATION (Task 1.3 / Task 2.4 - Multi-pose version) ===
        syslog(LOG_INFO, "Enrollment: Creating face model with %zu encodings...", pose_encodings.size());
        
        faceid::BinaryFaceModel model;
        model.version = 2;
        model.username = username;
        model.valid = true;
        model.timestamp = static_cast<uint32_t>(std::time(nullptr));
        model.face_ids = {
            username + "_pose_0",
            username + "_pose_1",
            username + "_pose_2",
            username + "_pose_3",
            username + "_pose_4"
        };
        
        // Task 2.4: Store all 5 encodings in model (as multi-sample format)
        model.sample_encodings.push_back(pose_encodings);
        
        // Task 2.4: Quality scores for each pose (all 1.0 for MVP)
        model.quality_scores.push_back(
            std::vector<float>(pose_encodings.size(), 1.0f)
        );
        
        syslog(LOG_INFO, "Enrollment: Model created with %zu total encodings", 
               model.getTotalEncodingCount());
        
         progress = 75;
         emit_enroll_status(bus, username, "in_progress", progress, "Creating model...");
         
         if (cancel_enrollment_) {
             emit_enroll_status(bus, username, "cancelled", progress, "Enrollment cancelled");
             status_ = EnrollStatus::FAILED;
             return;
         }
        
        // Task 1.3: Save model to disk
        syslog(LOG_INFO, "Enrollment: Saving model to disk...");
        std::string model_path = get_model_path(username);
        
        if (!faceid::BinaryModelLoader::saveUserModel(model_path, model)) {
            syslog(LOG_ERR, "Enrollment: Failed to save model to %s", model_path.c_str());
            emit_enroll_status(bus, username, "storage_error", 75, "Failed to save model file");
            status_ = EnrollStatus::FAILED;
            return;
        }
        
        syslog(LOG_INFO, "Enrollment: Model saved to %s", model_path.c_str());
        
        // Task 1.3: Validate model file and set permissions
        struct stat st;
        if (stat(model_path.c_str(), &st) != 0) {
            syslog(LOG_ERR, "Enrollment: Model file not found after save: %s", model_path.c_str());
            emit_enroll_status(bus, username, "storage_error", 75, "Failed to save model file");
            status_ = EnrollStatus::FAILED;
            return;
        }
        
        if (chmod(model_path.c_str(), 0600) != 0) {
            syslog(LOG_WARNING, "Enrollment: Failed to set model file permissions: %s", 
                   strerror(errno));
        }
        
        progress = 90;
        emit_enroll_status(bus, username, "in_progress", progress, "Capturing poses...");
        
        // Task 1.3: Validate model after saving
        faceid::BinaryFaceModel loaded_model;
        if (!faceid::BinaryModelLoader::loadUserModel(model_path, loaded_model)) {
            syslog(LOG_ERR, "Enrollment: Failed to validate model after saving");
            emit_enroll_status(bus, username, "validation_error", 90, "Model validation failed");
            status_ = EnrollStatus::FAILED;
            return;
        }
        
        if (loaded_model.getTotalEncodingCount() != model.getTotalEncodingCount()) {
            syslog(LOG_ERR, "Enrollment: Model validation failed - encoding count mismatch");
            emit_enroll_status(bus, username, "validation_error", 90, "Model validation failed");
            status_ = EnrollStatus::FAILED;
            return;
        }
        
        syslog(LOG_INFO, "Enrollment: Model validated - %zu encodings recovered", 
               loaded_model.getTotalEncodingCount());
        
        // Task 1.3: Invalidate ModelStore cache
        ModelStore::instance().invalidate_cache(username);
        syslog(LOG_DEBUG, "Enrollment: ModelStore cache invalidated for %s", username.c_str());
        
        progress = 100;
        syslog(LOG_INFO, "Enrollment completed successfully for %s (5 poses)", username.c_str());
        emit_enroll_status(bus, username, "success", progress, "Enrollment complete - ready for authentication");
        status_ = EnrollStatus::COMPLETED;
        
    } catch (const std::exception& e) {
        syslog(LOG_ERR, "Enrollment exception: %s", e.what());
        emit_enroll_status(bus, username, "error", 0, "Enrollment failed with exception");
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

void EnrollmentManager::emit_enroll_status(sd_bus* bus, const std::string& username,
                                           const std::string& status, int progress, 
                                           const std::string& message) {
    if (!bus) {
        syslog(LOG_DEBUG, "No bus available for signal emission");
        return;
    }
    
    int ret = sd_bus_emit_signal(bus,
        "/org/freedesktop/FaceID/Device",
        "org.freedesktop.FaceID.Device",
        "EnrollStatus",
        "siss", status.c_str(), progress, username.c_str(), message.c_str());
    
    if (ret < 0) {
        syslog(LOG_ERR, "Failed to emit EnrollStatus signal: %s", strerror(-ret));
    }
}

void EnrollmentManager::enroll_fingerprint(sd_bus* bus, const std::string& username) {
    using FPMgr = faceid::daemon::FingerprintManager;
    FPMgr& fp_mgr = FPMgr::instance();
    
    // Check if fingerprint reader available
    if (!fp_mgr.is_available()) {
        syslog(LOG_WARNING, "Fingerprint enrollment requested but device unavailable");
        return;
    }
    
    syslog(LOG_INFO, "Starting fingerprint enrollment for user: %s", username.c_str());
    
    // Emit initial progress
    int ret = sd_bus_emit_signal(bus,
        "/org/freedesktop/FaceID/Device",
        "org.freedesktop.FaceID.Device",
        "FingerprintEnrollStatus",
        "siss", "in_progress", 0, username.c_str(), 
        "Place finger on scanner to begin enrollment");
    
    if (ret < 0) {
        syslog(LOG_ERR, "Failed to emit FingerprintEnrollStatus signal: %s", strerror(-ret));
    }
    
    // Get FingerprintAuth reference
    faceid::FingerprintAuth& fp_auth = fp_mgr.get_fingerprint_auth();
    std::atomic<bool> cancel_flag{false};
    
    // Define progress callback to emit D-Bus signals
    auto progress_callback = [bus, &username](const std::string& status, int progress) {
        // Emit progress signal
        int ret = sd_bus_emit_signal(bus,
            "/org/freedesktop/FaceID/Device",
            "org.freedesktop.FaceID.Device",
            "FingerprintEnrollStatus",
            "siss", "in_progress", progress, username.c_str(), 
            status.c_str());
        
        if (ret < 0) {
            syslog(LOG_ERR, "Failed to emit enrollment progress signal: %s", strerror(-ret));
        }
    };
    
    // Use proper fprintd enrollment API instead of authenticate()
    bool success = fp_auth.enroll(username, 120, cancel_flag, progress_callback);
    
    if (cancel_flag.load()) {
        syslog(LOG_INFO, "Fingerprint enrollment cancelled for user: %s", username.c_str());
        ret = sd_bus_emit_signal(bus,
            "/org/freedesktop/FaceID/Device",
            "org.freedesktop.FaceID.Device",
            "FingerprintEnrollStatus",
            "siss", "cancelled", 100, username.c_str(), "Enrollment cancelled by user");
        return;
    }
    
    if (success) {
        syslog(LOG_INFO, "Fingerprint enrollment succeeded for user: %s", username.c_str());
        ret = sd_bus_emit_signal(bus,
            "/org/freedesktop/FaceID/Device",
            "org.freedesktop.FaceID.Device",
            "FingerprintEnrollStatus",
            "siss", "success", 100, username.c_str(), 
            "Fingerprint enrolled successfully - 5 samples collected");
    } else {
        std::string error = fp_auth.getLastError();
        syslog(LOG_WARNING, "Fingerprint enrollment failed for user %s: %s",
               username.c_str(), error.c_str());
        ret = sd_bus_emit_signal(bus,
            "/org/freedesktop/FaceID/Device",
            "org.freedesktop.FaceID.Device",
            "FingerprintEnrollStatus",
            "siss", "error", 100, username.c_str(), error.c_str());
    }
}
