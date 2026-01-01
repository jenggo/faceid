#include <security/pam_appl.h>
#include <security/pam_modules.h>
#include <syslog.h>
#include <thread>
#include <atomic>
#include <future>
#include <unistd.h>
#include <systemd/sd-login.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>
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

// System-wide file-based lock to prevent concurrent authentication attempts
// Uses POSIX file locking (fcntl) for robust cross-process synchronization
class SystemWideLock {
private:
    int lock_fd;
    bool is_locked;
    static constexpr const char* LOCK_FILE_PATH = "/var/run/faceid.lock";
    
public:
    SystemWideLock() : lock_fd(-1), is_locked(false) {}
    
    ~SystemWideLock() {
        if (is_locked) {
            release();
        }
    }
    
    // Acquire lock with blocking wait (returns true on success, false on error)
    // This will WAIT until the lock is available (no timeout)
    bool acquire() {
        // Open or create lock file
        lock_fd = open(LOCK_FILE_PATH, O_CREAT | O_RDWR, 0666);
        if (lock_fd == -1) {
            syslog(LOG_ERR, "pam_faceid: Failed to open lock file %s: %s", 
                   LOCK_FILE_PATH, strerror(errno));
            return false;
        }
        
        // Write current PID to lock file for debugging
        pid_t pid = getpid();
        std::string pid_str = std::to_string(pid) + "\n";
        if (write(lock_fd, pid_str.c_str(), pid_str.length()) == -1) {
            syslog(LOG_WARNING, "pam_faceid: Failed to write PID to lock file: %s", 
                   strerror(errno));
        }
        
        // Set up file lock structure
        struct flock fl;
        fl.l_type = F_WRLCK;    // Exclusive write lock
        fl.l_whence = SEEK_SET; // Lock from beginning of file
        fl.l_start = 0;         // Start at byte 0
        fl.l_len = 0;           // Lock entire file
        fl.l_pid = pid;
        
        syslog(LOG_INFO, "pam_faceid: Attempting to acquire system-wide lock (PID: %d)", pid);
        
        // F_SETLKW: Set lock and WAIT (blocks until lock is available)
        if (fcntl(lock_fd, F_SETLKW, &fl) == -1) {
            syslog(LOG_ERR, "pam_faceid: Failed to acquire lock: %s", strerror(errno));
            close(lock_fd);
            lock_fd = -1;
            return false;
        }
        
        is_locked = true;
        syslog(LOG_INFO, "pam_faceid: System-wide lock acquired successfully (PID: %d)", pid);
        return true;
    }
    
    // Acquire lock with timeout (for backwards compatibility)
    // timeout_seconds: Maximum time to wait for lock
    bool acquireWithTimeout(int timeout_seconds = 10) {
        // Open or create lock file
        lock_fd = open(LOCK_FILE_PATH, O_CREAT | O_RDWR, 0666);
        if (lock_fd == -1) {
            syslog(LOG_ERR, "pam_faceid: Failed to open lock file %s: %s", 
                   LOCK_FILE_PATH, strerror(errno));
            return false;
        }
        
        pid_t pid = getpid();
        std::string pid_str = std::to_string(pid) + "\n";
        if (write(lock_fd, pid_str.c_str(), pid_str.length()) == -1) {
            syslog(LOG_WARNING, "pam_faceid: Failed to write PID to lock file: %s", 
                   strerror(errno));
        }
        
        // Set up file lock structure
        struct flock fl;
        fl.l_type = F_WRLCK;
        fl.l_whence = SEEK_SET;
        fl.l_start = 0;
        fl.l_len = 0;
        fl.l_pid = pid;
        
        syslog(LOG_INFO, "pam_faceid: Attempting to acquire lock with %d second timeout (PID: %d)", 
               timeout_seconds, pid);
        
        // Try to acquire lock with polling (for timeout support)
        auto start_time = std::chrono::steady_clock::now();
        while (true) {
            // F_SETLK: Try to set lock without blocking
            if (fcntl(lock_fd, F_SETLK, &fl) != -1) {
                is_locked = true;
                syslog(LOG_INFO, "pam_faceid: Lock acquired successfully (PID: %d)", pid);
                return true;
            }
            
            if (errno != EACCES && errno != EAGAIN) {
                syslog(LOG_ERR, "pam_faceid: Lock acquisition failed: %s", strerror(errno));
                close(lock_fd);
                lock_fd = -1;
                return false;
            }
            
            // Check timeout
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time).count();
            if (elapsed >= timeout_seconds) {
                syslog(LOG_WARNING, "pam_faceid: Lock acquisition timeout after %ld seconds", elapsed);
                close(lock_fd);
                lock_fd = -1;
                return false;
            }
            
            // Sleep briefly before retrying
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    
    // Release the lock
    void release() {
        if (!is_locked || lock_fd == -1) {
            return;
        }
        
        // Set up unlock structure
        struct flock fl;
        fl.l_type = F_UNLCK;    // Unlock
        fl.l_whence = SEEK_SET;
        fl.l_start = 0;
        fl.l_len = 0;
        fl.l_pid = getpid();
        
        // Release the lock
        if (fcntl(lock_fd, F_SETLK, &fl) == -1) {
            syslog(LOG_ERR, "pam_faceid: Failed to release lock: %s", strerror(errno));
        } else {
            syslog(LOG_INFO, "pam_faceid: System-wide lock released (PID: %d)", getpid());
        }
        
        close(lock_fd);
        lock_fd = -1;
        is_locked = false;
    }
};

// Check if we should skip biometric authentication
// Returns true if biometric should be skipped, false if it should proceed
static bool should_skip_biometric(pam_handle_t *pamh, const char* username, const std::string& camera_device) {
    // Check 1: Password already in PAM stack?
    // This happens when:
    // - SDDM/GDM pre-provided password
    // - Previous PAM module already prompted
    // - try_first_pass flag used
    const char *password = nullptr;
    int ret = pam_get_item(pamh, PAM_AUTHTOK, (const void **)&password);
    
    if (ret == PAM_SUCCESS && password != nullptr && strlen(password) > 0) {
        syslog(LOG_INFO, "pam_faceid: Password already in PAM stack for user %s, skipping biometric", username);
        return true;
    }
    
    // Check 2: SSH/remote session detected?
    // No point trying biometric in headless/SSH sessions (no camera)
    int is_remote = sd_session_is_remote(nullptr);
    if (is_remote > 0) {
        syslog(LOG_INFO, "pam_faceid: Remote SSH session detected for user %s, skipping biometric", username);
        return true;
    }
    
    // Check 3: Camera device accessible?
    // Check the configured camera device (from config), not hardcoded /dev/video0
    // Polkit and other restricted contexts may not have camera visible
    if (access(camera_device.c_str(), F_OK) != 0) {
        syslog(LOG_INFO, "pam_faceid: Camera device %s not accessible (in use or permission issue), skipping biometric for user %s", 
               camera_device.c_str(), username);
        return true;
    }
    
    // Proceed with biometric authentication
    syslog(LOG_DEBUG, "pam_faceid: Camera device %s accessible, proceeding with biometric", camera_device.c_str());
    return false;
}

static bool authenticate_user(const char* username) {
    const auto auth_start = std::chrono::steady_clock::now();
    
    // Set environment variable to signal Logger we're in PAM context
    // This prevents stderr warnings that break pkttyagent authentication
    setenv("FACEID_PAM_CONTEXT", "1", 1);
    
    // Initialize logger
    Logger& logger = Logger::getInstance();
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
    std::atomic<bool> face_finished(false);
    std::atomic<bool> fingerprint_finished(false);
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
                    face_finished.store(true);
                    return false;
                }
                
                // Load ALL users' models for verification (prevent false positives)
                std::vector<BinaryFaceModel> all_users = cache.loadAllUsersParallel(4);
                logger.debug("Loaded " + std::to_string(all_users.size()) + " user models for verification");
                
                // Initialize camera
                auto device = config.getString("camera", "device").value_or("/dev/video0");
                Camera camera(device);
                
                auto width = config.getInt("camera", "width").value_or(640);
                auto height = config.getInt("camera", "height").value_or(480);
                
                if (!camera.open(width, height)) {
                    logger.error("Failed to open camera");
                    face_finished.store(true);
                    return false;
                }
                
                // Initialize face detector
                FaceDetector detector;
                
                if (!detector.loadModels()) {  // Use default MODELS_DIR/sface path
                    logger.error("Failed to load face recognition model");
                    face_finished.store(true);
                    return false;
                }
                
                double threshold = config.getDouble("recognition", "threshold").value_or(0.6);
                
                // Get detection confidence threshold from config
                float detection_confidence = config.getDouble("face_detection", "confidence").value_or(0.31);
                
                logger.debug(std::string("Starting face detection with cascading detection (confidence: ") + 
                           std::to_string(detection_confidence) + ")");
                syslog(LOG_DEBUG, "pam_faceid: Using cascading detection with confidence: %.3f", detection_confidence);
                
                auto start = std::chrono::steady_clock::now();
                while (!cancel_flag.load() && std::chrono::duration_cast<std::chrono::seconds>(
                       std::chrono::steady_clock::now() - start).count() < timeout) {
                    
                    faceid::Image frame;
                    if (!camera.read(frame)) {
                        continue;
                    }
                    
                    // Use cascading detection for robust face detection across all lighting conditions
                    // This automatically tries 3 stages: standard CLAHE, aggressive CLAHE, and fallback detector
                    auto cascade_result = detector.detectFacesCascade(frame.view(), false, detection_confidence);
                    
                    if (cascade_result.faces.empty()) {
                        continue;
                    }
                    
                    // Log which cascade stage was used for detection
                    if (cascade_result.stage_used > 1) {
                        logger.debug(std::string("Face detected using cascade stage ") + 
                                   std::to_string(cascade_result.stage_used) + 
                                   " (brightness: " + std::to_string(cascade_result.avg_brightness) + ")");
                        syslog(LOG_DEBUG, "pam_faceid: Cascade stage %d used (brightness: %.2f)", 
                               cascade_result.stage_used, cascade_result.avg_brightness);
                    }
                    
                    // Encode faces using the preprocessed frame from cascade
                    auto encodings = detector.encodeFaces(cascade_result.processed_frame.view(), 
                                                         cascade_result.faces);
                    if (encodings.empty()) {
                        continue;
                    }
                    
                    // Deduplicate faces - filter out multiple detections of the same person
                    // This prevents false positives from the same face detected at different angles/positions
                    auto unique_indices = FaceDetector::deduplicateFaces(cascade_result.faces, encodings, 0.15);
                    
                    // Filter to only unique faces
                    std::vector<FaceEncoding> unique_encodings;
                    for (size_t idx : unique_indices) {
                        if (idx < encodings.size()) {
                            unique_encodings.push_back(encodings[idx]);
                        }
                    }
                    
                    // Compare detected faces against ALL users to find best match
                    for (const auto& detected_encoding : unique_encodings) {
                        double best_distance = 999.0;
                        std::string best_match_user = "";
                        
                        // Compare against all enrolled users
                        for (const auto& user_model : all_users) {
                            for (const auto& stored_encoding : user_model.encodings) {
                                double distance = detector.compareFaces(detected_encoding, stored_encoding);
                                if (distance < best_distance) {
                                    best_distance = distance;
                                    best_match_user = user_model.username;
                                }
                            }
                        }
                        
                        // Only accept if:
                        // 1. Distance is below threshold
                        // 2. Best match is the current user (not another user)
                        if (best_distance < threshold && best_match_user == username) {
                            logger.info(std::string("Face matched for user ") + username + 
                                      " (distance: " + std::to_string(best_distance) + 
                                      ", cascade stage: " + std::to_string(cascade_result.stage_used) + ")");
                            syslog(LOG_INFO, "pam_faceid: Face match success (distance: %.3f, cascade stage: %d)", 
                                   best_distance, cascade_result.stage_used);
                            return true;
                        } else if (best_distance < threshold && best_match_user != username) {
                            // Face matched a different user - log security event
                            logger.warning(std::string("Face matched different user '") + best_match_user + 
                                         "' instead of '" + username + "' (distance: " + 
                                         std::to_string(best_distance) + "), rejecting authentication");
                        }
                    }
                }
                
                face_finished.store(true);
                return false;
            } catch (const std::exception& e) {
                logger.error(std::string("Face auth exception: ") + e.what());
                face_finished.store(true);
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
                        fingerprint_finished.store(true);
                        return false;
                    }
                }
                
                // Initialize fingerprint (delayed)
                bool fingerprint_available = fingerprint.initialize() && fingerprint.isAvailable();
                fingerprint_started.store(true);
                
                if (!fingerprint_available) {
                    logger.info("Fingerprint authentication not available");
                    fingerprint_finished.store(true);
                    return false;
                }
                
                logger.debug("Fingerprint reader initialized, starting authentication");
                bool result = fingerprint.authenticate(username, timeout, cancel_flag);
                fingerprint_finished.store(true);
                return result;
            } catch (const std::exception& e) {
                logger.error(std::string("Fingerprint auth exception: ") + e.what());
                fingerprint_finished.store(true);
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
        
        // Early exit if both methods have finished (failed)
        bool face_done = !face_enrolled || face_finished.load();
        bool fingerprint_done = !fingerprint_enabled || fingerprint_finished.load();
        if (face_done && fingerprint_done && !auth_success.load()) {
            syslog(LOG_INFO, "pam_faceid: Both authentication methods finished without success, exiting early");
            break;
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
    
    // Get process info for debugging
    char exe_path[256] = {0};
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path)-1);
    if (len != -1) {
        exe_path[len] = '\0';
    }
    
    syslog(LOG_INFO, "pam_faceid: authenticate called (PID: %d, UID: %d, GID: %d, exe: %s)", 
           getpid(), getuid(), getgid(), len != -1 ? exe_path : "unknown");
    
    // Get username first (before lock acquisition)
    const char* username = nullptr;
    int ret = pam_get_user(pamh, &username, nullptr);
    
    if (ret != PAM_SUCCESS || username == nullptr) {
        closelog();
        return PAM_USER_UNKNOWN;
    }
    
    // Load configuration EARLY (before skip checks) to read camera device
    Config& config = Config::getInstance();
    const std::string config_path = std::string(CONFIG_DIR) + "/faceid.conf";
    if (!config.load(config_path)) {
        syslog(LOG_ERR, "pam_faceid: Failed to load configuration from %s", config_path.c_str());
        closelog();
        return PAM_AUTH_ERR;
    }
    
    // Get configured camera device
    auto camera_device = config.getString("camera", "device").value_or("/dev/video0");
    syslog(LOG_DEBUG, "pam_faceid: Using camera device: %s", camera_device.c_str());
    
    // Check if we should skip biometric authentication (before acquiring lock)
    // This allows fast-path exit for SSH, no camera, password already provided, etc.
    if (should_skip_biometric(pamh, username, camera_device)) {
        syslog(LOG_INFO, "pam_faceid: Skipping biometric auth (no lock acquired)");
        closelog();
        return PAM_AUTH_ERR;  // Let next PAM module handle authentication
    }
    
    // Only acquire lock if we're actually going to perform biometric authentication
    // This prevents unnecessary lock contention for cases where biometric is not needed
    syslog(LOG_DEBUG, "pam_faceid: Biometric auth required, acquiring system-wide lock");
    SystemWideLock lock;
    if (!lock.acquire()) {
        syslog(LOG_ERR, "pam_faceid: Failed to acquire system-wide lock (error: %s)", strerror(errno));
        
        // Send error message to user via PAM conversation
        struct pam_conv *conv;
        if (pam_get_item(pamh, PAM_CONV, (const void **)&conv) == PAM_SUCCESS && conv && conv->conv) {
            struct pam_message msg;
            const struct pam_message *msgp = &msg;
            struct pam_response *resp = nullptr;
            
            msg.msg_style = PAM_ERROR_MSG;
            msg.msg = const_cast<char*>("FaceID: Failed to acquire authentication lock. Please try again.");
            
            conv->conv(1, &msgp, &resp, conv->appdata_ptr);
            if (resp) {
                free(resp);
            }
        }
        
        closelog();
        return PAM_AUTH_ERR;
    }
    
    // Don't close syslog yet - authenticate_user() needs it
    syslog(LOG_DEBUG, "pam_faceid: Lock acquired, proceeding with authentication");
    
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
