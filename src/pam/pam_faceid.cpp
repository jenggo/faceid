#include <security/pam_appl.h>
#include <security/pam_modules.h>
#include <syslog.h>
#include <fstream>
#include <json/json.h>
#include <thread>
#include <atomic>
#include <future>
#include "../config.h"
#include "../logger.h"
#include "../fingerprint_auth.h"
#include "../lid_detector.h"
#include "../display_detector.h"

// Suppress OpenCV warnings from external headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include "../camera.h"
#include "../face_detector.h"
#pragma GCC diagnostic pop

#include "config_paths.h"

using namespace faceid;

static bool authenticate_user(const char* username) {
    auto auth_start = std::chrono::steady_clock::now();
    
    // Initialize logger
    Logger& logger = Logger::getInstance();
    logger.setLogFile("/var/log/faceid.log");
    
    openlog("pam_faceid", LOG_PID, LOG_AUTH);
    
    // Load configuration
    Config& config = Config::getInstance();
    std::string config_path = std::string(CONFIG_DIR) + "/faceid.conf";
    if (!config.load(config_path)) {
        syslog(LOG_ERR, "Failed to load configuration");
        logger.auditAuthFailure(username, "biometric", "config_load_failed");
        closelog();
        return false;
    }
    
    // Check if lid detection is enabled
    bool check_lid = config.getBool("authentication", "check_lid_state").value_or(true);
    
    if (check_lid) {
        LidDetector lid_detector;
        LidState lid_state = lid_detector.getLidState();
        
        if (lid_state == LidState::CLOSED) {
            logger.info(std::string("Lid is CLOSED, skipping biometric authentication for user ") + username);
            logger.auditAuthFailure(username, "biometric", "lid_closed");
            syslog(LOG_INFO, "Lid closed, skipping biometric auth for user %s", username);
            closelog();
            return false;
        } else if (lid_state == LidState::OPEN) {
            logger.debug(std::string("Lid is OPEN, proceeding with biometric authentication (method: ") + 
                        lid_detector.getDetectionMethod() + ")");
            syslog(LOG_DEBUG, "Lid open, proceeding with biometric auth");
        } else {
            // Unknown state - proceed with caution but log it
            logger.warning(std::string("Could not determine lid state (") + 
                          lid_detector.getLastError() + "), proceeding with biometric auth");
            syslog(LOG_WARNING, "Unknown lid state, proceeding with biometric auth");
        }
    }
    
    // Check display state (screen on/off, locked/unlocked)
    bool check_display = config.getBool("authentication", "check_display_state").value_or(true);
    
    if (check_display) {
        DisplayDetector display_detector;
        DisplayState display_state = display_detector.getDisplayState();
        
        // If we're on lock screen, add delay before checking display state again
        // This gives the screen time to turn off after pressing lock button (Meta+L)
        if (display_detector.isLockScreenGreeter() || display_detector.isScreenLocked()) {
            int delay_ms = config.getInt("authentication", "lock_screen_delay_ms").value_or(1000);
            if (delay_ms > 0) {
                logger.debug(std::string("Lock screen detected, waiting ") + std::to_string(delay_ms) + 
                           "ms before checking display state");
                std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
                
                // Recheck display state after delay
                display_state = display_detector.getDisplayState();
            }
        }
        
        if (display_state == DisplayState::OFF) {
            logger.info(std::string("Display is OFF (") + display_detector.getDetectionMethod() + 
                       "), skipping biometric authentication for user " + username);
            logger.auditAuthFailure(username, "biometric", "display_off");
            syslog(LOG_INFO, "Display off, skipping biometric auth for user %s", username);
            closelog();
            return false;
        } else if (display_state == DisplayState::ON) {
            logger.debug(std::string("Display is ON (") + display_detector.getDetectionMethod() + 
                        "), proceeding with biometric authentication");
            syslog(LOG_DEBUG, "Display on, proceeding with biometric auth");
        } else {
            // Unknown state - proceed with caution but log it
            logger.warning(std::string("Could not determine display state (") + 
                          display_detector.getLastError() + "), proceeding with biometric auth");
            syslog(LOG_WARNING, "Unknown display state, proceeding with biometric auth");
        }
    }
    
    logger.auditAuthAttempt(username, "face+fingerprint");
    
    // Get user model path
    std::string model_path = std::string(MODELS_DIR) + "/" + username + ".json";
    std::ifstream model_file(model_path);
    bool face_enrolled = model_file.is_open();
    model_file.close();
    
    // Check if user has face enrolled
    if (!face_enrolled) {
        logger.info(std::string("No face model found for user ") + username);
    }
    
    // Initialize fingerprint auth
    FingerprintAuth fingerprint;
    bool fingerprint_available = fingerprint.initialize() && fingerprint.isAvailable();
    
    if (!fingerprint_available) {
        logger.info("Fingerprint authentication not available");
    }
    
    // If neither method is available, fail early
    if (!face_enrolled && !fingerprint_available) {
        logger.auditAuthFailure(username, "face+fingerprint", "no_auth_methods_available");
        closelog();
        return false;
    }
    
    int timeout = config.getInt("recognition", "timeout").value_or(5);
    std::atomic<bool> auth_success(false);
    std::atomic<bool> cancel_flag(false);
    std::string success_method;
    
    // Launch face authentication in separate thread (if enrolled)
    std::future<bool> face_future;
    if (face_enrolled) {
        face_future = std::async(std::launch::async, [&]() -> bool {
            try {
                std::ifstream model_file_inner(model_path);
                if (!model_file_inner.is_open()) {
                    return false;
                }
                
                // Parse model
                Json::Value model_data;
                Json::Reader reader;
                if (!reader.parse(model_file_inner, model_data)) {
                    logger.error(std::string("Failed to parse face model for user ") + username);
                    return false;
                }
                
                // Initialize camera
                auto device = config.getString("camera", "device").value_or("/dev/video0");
                Camera camera(device);
                
                auto width = config.getInt("camera", "width").value_or(640);
                auto height = config.getInt("camera", "height").value_or(480);
                
                if (!camera.open(width, height)) {
                    logger.error("Failed to open camera");
                    return false;
                }
                
                // Initialize face detector
                FaceDetector detector;
                std::string recognition_model = std::string(CONFIG_DIR) + "/models/face_recognition_sface_2021dec.onnx";
                
                if (!detector.loadModels(recognition_model)) {
                    logger.error("Failed to load face recognition model");
                    return false;
                }
                
                double threshold = config.getDouble("recognition", "threshold").value_or(0.6);
                int tracking_interval = config.getInt("face_detection", "tracking_interval").value_or(10);
                
                auto start = std::chrono::steady_clock::now();
                while (!cancel_flag.load() && std::chrono::duration_cast<std::chrono::seconds>(
                       std::chrono::steady_clock::now() - start).count() < timeout) {
                    
                    cv::Mat frame;
                    if (!camera.read(frame)) {
                        continue;
                    }
                    
                    // Preprocess for better detection
                    frame = detector.preprocessFrame(frame);
                    
                    // Detect faces (with tracking optimization)
                    auto faces = detector.detectOrTrackFaces(frame, tracking_interval);
                    if (faces.empty()) {
                        continue;
                    }
                    
                    // Encode faces
                    auto encodings = detector.encodeFaces(frame, faces);
                    if (encodings.empty()) {
                        continue;
                    }
                    
                    // Compare with stored encodings
                    std::vector<faceid::FaceEncoding> stored_encodings;
                    
                    // Try new multi-face format first
                    if (model_data.isMember("faces") && !model_data["faces"].empty()) {
                        const Json::Value& faces_data = model_data["faces"];
                        
                        // Load all encodings from all faces
                        for (const auto& face_id : faces_data.getMemberNames()) {
                            const Json::Value& face_data = faces_data[face_id];
                            if (face_data.isMember("encodings")) {
                                const Json::Value& encodings_array = face_data["encodings"];
                                
                                for (const auto& enc_json : encodings_array) {
                                    faceid::FaceEncoding stored_encoding(128);
                                    for (int i = 0; i < 128 && i < static_cast<int>(enc_json.size()); i++) {
                                        stored_encoding[i] = enc_json[i].asFloat();
                                    }
                                    stored_encodings.push_back(stored_encoding);
                                }
                            }
                        }
                    }
                    // Fallback to old single-face format (backward compatibility)
                    else if (model_data.isMember("encodings") && !model_data["encodings"].empty()) {
                        const Json::Value& stored_encodings_json = model_data["encodings"];
                        
                        for (const auto& enc_json : stored_encodings_json) {
                            faceid::FaceEncoding stored_encoding(128);
                            for (int i = 0; i < 128 && i < static_cast<int>(enc_json.size()); i++) {
                                stored_encoding[i] = enc_json[i].asFloat();
                            }
                            stored_encodings.push_back(stored_encoding);
                        }
                    }
                    
                    // Compare detected faces with all stored encodings
                    for (const auto& stored_encoding : stored_encodings) {
                        for (const auto& detected_encoding : encodings) {
                            double distance = detector.compareFaces(detected_encoding, stored_encoding);
                            
                            // SFace compareFaces returns distance (lower = more similar)
                            // Default threshold 0.6 works well
                            if (distance < threshold) {
                                logger.info(std::string("Face matched for user ") + username + 
                                          " (distance: " + std::to_string(distance) + ")");
                                return true;
                            }
                        }
                    }
                }
                
                return false;
            } catch (const std::exception& e) {
                logger.error(std::string("Face auth exception: ") + e.what());
                return false;
            }
        });
    }
    
    // Launch fingerprint authentication in separate thread (if available)
    std::future<bool> fingerprint_future;
    if (fingerprint_available) {
        fingerprint_future = std::async(std::launch::async, [&]() -> bool {
            try {
                return fingerprint.authenticate(username, timeout, cancel_flag);
            } catch (const std::exception& e) {
                logger.error(std::string("Fingerprint auth exception: ") + e.what());
                return false;
            }
        });
    }
    
    // Wait for first successful authentication
    auto check_start = std::chrono::steady_clock::now();
    while (std::chrono::duration_cast<std::chrono::seconds>(
           std::chrono::steady_clock::now() - check_start).count() < timeout) {
        
        // Check face authentication
        if (face_enrolled && face_future.valid()) {
            auto status = face_future.wait_for(std::chrono::milliseconds(100));
            if (status == std::future_status::ready && face_future.get()) {
                cancel_flag.store(true);  // Cancel fingerprint
                success_method = "face";
                auth_success.store(true);
                break;
            }
        }
        
        // Check fingerprint authentication
        if (fingerprint_available && fingerprint_future.valid()) {
            auto status = fingerprint_future.wait_for(std::chrono::milliseconds(100));
            if (status == std::future_status::ready && fingerprint_future.get()) {
                cancel_flag.store(true);  // Cancel face
                success_method = "fingerprint";
                auth_success.store(true);
                break;
            }
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    // Ensure cancellation
    cancel_flag.store(true);
    
    // Calculate duration
    auto auth_end = std::chrono::steady_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(auth_end - auth_start).count();
    
    if (auth_success.load()) {
        syslog(LOG_INFO, "Authentication successful for user %s via %s", username, success_method.c_str());
        logger.auditAuthSuccess(username, success_method, duration_ms);
        closelog();
        return true;
    }
    
    syslog(LOG_WARNING, "Face+fingerprint authentication failed for user %s", username);
    logger.auditAuthFailure(username, "face+fingerprint", "timeout_or_no_match");
    closelog();
    return false;
}

extern "C" {

PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh, int flags,
                                   int argc, const char **argv) {
    // Unused parameters (required by PAM interface)
    (void)flags;
    (void)argc;
    (void)argv;
    
    const char* username = nullptr;
    int ret = pam_get_user(pamh, &username, nullptr);
    
    if (ret != PAM_SUCCESS || username == nullptr) {
        return PAM_USER_UNKNOWN;
    }
    
    if (authenticate_user(username)) {
        return PAM_SUCCESS;
    }
    
    return PAM_AUTH_ERR;
}

PAM_EXTERN int pam_sm_setcred(pam_handle_t *pamh, int flags,
                              int argc, const char **argv) {
    // Unused parameters (required by PAM interface)
    (void)pamh;
    (void)flags;
    (void)argc;
    (void)argv;
    return PAM_SUCCESS;
}

PAM_EXTERN int pam_sm_acct_mgmt(pam_handle_t *pamh, int flags,
                                int argc, const char **argv) {
    // Unused parameters (required by PAM interface)
    (void)pamh;
    (void)flags;
    (void)argc;
    (void)argv;
    return PAM_SUCCESS;
}

} // extern "C"
