#include <security/pam_appl.h>
#include <security/pam_modules.h>
#include <syslog.h>
#include <fstream>
#include <thread>
#include <atomic>
#include <future>
#include <mutex>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include "../config.h"
#include "../logger.h"
#include "../fingerprint_auth.h"
#include "../lid_detector.h"
#include "../display_detector.h"
#include "../models/model_cache.h"

// Suppress external library warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include "../camera.h"
#include "../face_detector.h"
#pragma GCC diagnostic pop

#include "config_paths.h"

using namespace faceid;

// Lock file path for system-wide singleton
// DISABLED: File-based lock caused permission issues
// Commenting out the lock mechanism - returning to pre-lock behavior
// static const char* LOCK_FILE_PATH = "/run/lock/pam_faceid.lock";

// No-op lock class - authentication proceeds without locking
// DISABLED: File-based lock caused permission issues with polkit
// This restores the behavior before lock implementation was added
class SystemWideLock {
public:
    SystemWideLock() {}
    ~SystemWideLock() {}
    
    // Always return true - no actual locking performed
    bool tryAcquire() { return true; }
    bool acquireWithTimeout(int timeout_seconds = 10) { 
        (void)timeout_seconds; // Unused parameter
        return true; 
    }
    void release() {}
    bool isLocked() const { return true; }
};

static bool authenticate_user(const char* username) {
    const auto auth_start = std::chrono::steady_clock::now();
    
    // Initialize logger
    Logger& logger = Logger::getInstance();
    logger.setLogFile("/var/log/faceid.log");
    openlog("pam_faceid", LOG_PID, LOG_AUTH);
    
    // Load configuration
    Config& config = Config::getInstance();
    const std::string config_path = std::string(CONFIG_DIR) + "/faceid.conf";
    if (!config.load(config_path)) {
        syslog(LOG_ERR, "Failed to load configuration");
        logger.auditAuthFailure(username, "biometric", "config_load_failed");
        closelog();
        return false;
    }
    
    // Check lid state
    const bool check_lid = config.getBool("authentication", "check_lid_state").value_or(true);
    if (check_lid) {
        LidDetector lid_detector;
        const LidState lid_state = lid_detector.getLidState();
        
        if (lid_state == LidState::CLOSED) {
            logger.info(std::string("Lid is CLOSED, skipping biometric authentication for user ") + username);
            logger.auditAuthFailure(username, "biometric", "lid_closed");
            syslog(LOG_INFO, "Lid closed, skipping biometric auth for user %s", username);
            closelog();
            return false;
        }
        
        if (lid_state == LidState::OPEN) {
            logger.debug("Lid is OPEN (" + lid_detector.getDetectionMethod() + "), proceeding with biometric authentication");
            syslog(LOG_DEBUG, "Lid open, proceeding with biometric auth");
        } else {
            logger.warning("Could not determine lid state (" + lid_detector.getLastError() + "), proceeding with biometric auth");
            syslog(LOG_WARNING, "Unknown lid state, proceeding with biometric auth");
        }
    }
    
    // Check display state
    const bool check_display = config.getBool("authentication", "check_display_state").value_or(true);
    if (check_display) {
        DisplayDetector display_detector;
        DisplayState display_state = display_detector.getDisplayState();
        
        // If on lock screen, add delay before checking display state
        if (display_detector.isLockScreenGreeter() || display_detector.isScreenLocked()) {
            const int delay_ms = config.getInt("authentication", "lock_screen_delay_ms").value_or(1000);
            if (delay_ms > 0) {
                logger.debug("Lock screen detected, waiting " + std::to_string(delay_ms) + "ms before checking display state");
                std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
                display_state = display_detector.getDisplayState();
            }
        }
        
        // Check if using external monitor only (laptop screen off, external on)
        const bool skip_external_only = config.getBool("authentication", "skip_external_monitor_only").value_or(true);
        if (skip_external_only && display_detector.isExternalMonitorOnly()) {
            logger.info(std::string("External monitor only detected (laptop screen off), skipping biometric authentication for user ") + username);
            logger.auditAuthFailure(username, "biometric", "external_monitor_only");
            syslog(LOG_INFO, "External monitor only, skipping biometric auth for user %s", username);
            closelog();
            return false;
        }
        
        if (display_state == DisplayState::OFF) {
            logger.info("Display is OFF (" + display_detector.getDetectionMethod() + 
                       "), skipping biometric authentication for user " + username);
            logger.auditAuthFailure(username, "biometric", "display_off");
            syslog(LOG_INFO, "Display off, skipping biometric auth for user %s", username);
            closelog();
            return false;
        }
        
        if (display_state == DisplayState::ON) {
            logger.debug("Display is ON (" + display_detector.getDetectionMethod() + "), proceeding with biometric authentication");
            syslog(LOG_DEBUG, "Display on, proceeding with biometric auth");
        } else {
            logger.warning("Could not determine display state (" + display_detector.getLastError() + "), proceeding with biometric auth");
            syslog(LOG_WARNING, "Unknown display state, proceeding with biometric auth");
        }
    }
    
    logger.auditAuthAttempt(username, "face+fingerprint");
    
    // Check for face enrollment
    auto& cache = ModelCache::getInstance();
    const bool face_enrolled = cache.hasUserModel(username);
    
    if (!face_enrolled) {
        logger.info(std::string("No face model found for user ") + username);
    } else {
        logger.debug(std::string("Face model(s) found for user ") + username);
    }
    
    // Get fingerprint configuration
    const int fingerprint_delay_ms = config.getInt("authentication", "fingerprint_delay_ms").value_or(500);
    FingerprintAuth fingerprint;
    const bool fingerprint_enabled = config.getBool("authentication", "enable_fingerprint").value_or(true);
    
    // If neither method is available, fail early
    if (!face_enrolled && !fingerprint_enabled) {
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
                // Load model using ModelCache
                auto& cache = ModelCache::getInstance();
                BinaryFaceModel model;
                if (!cache.loadUserModel(username, model)) {
                    logger.error(std::string("Failed to load face model for user ") + username);
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
                
                if (!detector.loadModels()) {  // Use default MODELS_DIR/sface path
                    logger.error("Failed to load face recognition model");
                    return false;
                }
                
                double threshold = config.getDouble("recognition", "threshold").value_or(0.6);
                int tracking_interval = config.getInt("face_detection", "tracking_interval").value_or(10);
                
                auto start = std::chrono::steady_clock::now();
                while (!cancel_flag.load() && std::chrono::duration_cast<std::chrono::seconds>(
                       std::chrono::steady_clock::now() - start).count() < timeout) {
                    
                    faceid::Image frame;
                    if (!camera.read(frame)) {
                        continue;
                    }
                    
                    // Preprocess for better detection
                    faceid::Image processed_frame = detector.preprocessFrame(frame.view());
                    
                    // Detect faces (with tracking optimization)
                    auto faces = detector.detectOrTrackFaces(processed_frame.view(), tracking_interval);
                    if (faces.empty()) {
                        continue;
                    }
                    
                    // Encode faces
                    auto encodings = detector.encodeFaces(processed_frame.view(), faces);
                    if (encodings.empty()) {
                        continue;
                    }
                    
                    // Compare with stored encodings from binary model
                    std::vector<faceid::FaceEncoding> stored_encodings = model.encodings;
                    
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
    // Delayed launch to give face auth a head start (face is typically faster)
    std::future<bool> fingerprint_future;
    std::atomic<bool> fingerprint_started(false);
    
    if (fingerprint_enabled) {
        fingerprint_future = std::async(std::launch::async, [&]() -> bool {
            try {
                // Wait for configured delay before initializing fingerprint
                if (fingerprint_delay_ms > 0) {
                    logger.debug(std::string("Delaying fingerprint init by ") + 
                               std::to_string(fingerprint_delay_ms) + "ms (face auth head start)");
                    std::this_thread::sleep_for(std::chrono::milliseconds(fingerprint_delay_ms));
                    
                    // Check if face auth already succeeded
                    if (cancel_flag.load()) {
                        logger.debug("Face auth succeeded, skipping fingerprint initialization");
                        return false;
                    }
                }
                
                // Initialize fingerprint (delayed)
                bool fingerprint_available = fingerprint.initialize() && fingerprint.isAvailable();
                fingerprint_started.store(true);
                
                if (!fingerprint_available) {
                    logger.info("Fingerprint authentication not available");
                    return false;
                }
                
                logger.debug("Fingerprint reader initialized, starting authentication");
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
        if (fingerprint_enabled && fingerprint_future.valid()) {
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
    
    // Provide detailed failure reason
    std::string failure_reason;
    if (!face_enrolled && !fingerprint_enabled) {
        failure_reason = "no_methods_available";
        syslog(LOG_WARNING, "Authentication failed for user %s: no face or fingerprint enrolled", username);
    } else if (face_enrolled && !fingerprint_enabled) {
        failure_reason = "face_timeout_or_no_match";
        syslog(LOG_WARNING, "Face authentication failed for user %s: timeout or no match", username);
    } else if (!face_enrolled && fingerprint_enabled) {
        failure_reason = "fingerprint_timeout_or_no_match";
        syslog(LOG_WARNING, "Fingerprint authentication failed for user %s: timeout or no match", username);
    } else {
        failure_reason = "both_timeout_or_no_match";
        syslog(LOG_WARNING, "Face+fingerprint authentication failed for user %s: timeout or no match", username);
    }
    
    logger.auditAuthFailure(username, "face+fingerprint", failure_reason);
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
    
    // Debug: Log that PAM module was called
    openlog("pam_faceid", LOG_PID, LOG_AUTH);
    syslog(LOG_INFO, "pam_faceid: authenticate called (PID: %d)", getpid());
    
    // Try to acquire system-wide lock with timeout
    SystemWideLock lock;
    if (!lock.acquireWithTimeout(10)) {
        syslog(LOG_WARNING, "pam_faceid: failed to acquire lock - another authentication may be in progress or timed out");
        
        // Send message to user via PAM conversation
        struct pam_conv *conv;
        if (pam_get_item(pamh, PAM_CONV, (const void **)&conv) == PAM_SUCCESS && conv && conv->conv) {
            struct pam_message msg;
            const struct pam_message *msgp = &msg;
            struct pam_response *resp = nullptr;
            
            msg.msg_style = PAM_ERROR_MSG;
            msg.msg = const_cast<char*>("FaceID: Another authentication is in progress. Please wait and try again.");
            
            conv->conv(1, &msgp, &resp, conv->appdata_ptr);
            if (resp) {
                free(resp);
            }
        }
        
        closelog();
        return PAM_AUTH_ERR;
    }
    
    // Don't close syslog yet - authenticate_user() needs it
    
    const char* username = nullptr;
    int ret = pam_get_user(pamh, &username, nullptr);
    
    if (ret != PAM_SUCCESS || username == nullptr) {
        closelog();
        return PAM_USER_UNKNOWN;
    }
    
    bool success = authenticate_user(username);
    
    // Lock will be automatically released by RAII destructor
    
    closelog();
    
    if (success) {
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
