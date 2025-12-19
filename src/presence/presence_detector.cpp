#include "presence_detector.h"
#include "../logger.h"
#include "../face_detector.h"
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <algorithm>
#include <cctype>

namespace faceid {

PresenceDetector::PresenceDetector(
    const std::string& camera_device,
    std::chrono::seconds inactive_threshold,
    std::chrono::seconds scan_interval,
    int max_scan_failures,
    std::chrono::minutes max_idle_time)
    : camera_device_(camera_device)
    , detector_initialized_(false)
    , current_state_(State::ACTIVELY_PRESENT)
    , last_activity_(std::chrono::steady_clock::now())
    , state_entry_time_(std::chrono::steady_clock::now())
    , start_time_(std::chrono::steady_clock::now())
    , scan_failures_(0)
    , pause_count_(0)
    , inactive_threshold_ms_(std::chrono::duration_cast<std::chrono::milliseconds>(inactive_threshold).count())
    , scan_interval_ms_(std::chrono::duration_cast<std::chrono::milliseconds>(scan_interval).count())
    , max_scan_failures_(max_scan_failures)
    , max_idle_time_ms_(std::chrono::duration_cast<std::chrono::milliseconds>(max_idle_time).count())
    , last_interrupt_check_(std::chrono::steady_clock::now() - std::chrono::seconds(1)) {
}

PresenceDetector::~PresenceDetector() {
    stop();
}

bool PresenceDetector::initialize() {
    Logger& logger = Logger::getInstance();
    
    try {
        // Detect display server (X11 vs Wayland)
        is_wayland_ = detectDisplayServer();
        logger.info(std::string("Display server: ") + (is_wayland_ ? "Wayland" : "X11"));
        
        logger.info("Presence detector initialized successfully (lazy loading enabled)");
        return true;
    } catch (const std::exception& e) {
        logger.error(std::string("Failed to initialize presence detector: ") + e.what());
        return false;
    }
}

bool PresenceDetector::start() {
    if (running_.load()) {
        return true; // Already running
    }
    
    if (!detector_initialized_) {
        if (!initialize()) {
            return false;
        }
    }
    
    Logger& logger = Logger::getInstance();
    logger.info("Starting presence detection service");
    
    running_.store(true);
    detection_thread_ = std::thread(&PresenceDetector::detectionLoop, this);
    
    return true;
}

void PresenceDetector::stop() {
    if (!running_.load()) {
        return;
    }
    
    Logger& logger = Logger::getInstance();
    logger.info("Stopping presence detection service");
    
    running_.store(false);
    
    if (detection_thread_.joinable()) {
        detection_thread_.join();
    }
    
    // Release camera
    std::lock_guard<std::mutex> lock(camera_mutex_);
    if (camera_ && camera_->isOpened()) {
        camera_->release();
        camera_.reset();
    }
}

bool PresenceDetector::isUserPresent() const {
    return current_state_ == State::ACTIVELY_PRESENT;
}

std::string PresenceDetector::getStateString() const {
    switch (current_state_) {
        case State::ACTIVELY_PRESENT:   return "ACTIVELY_PRESENT";
        case State::IDLE_WITH_SCANNING: return "IDLE_WITH_SCANNING";
        case State::AWAY_CONFIRMED:     return "AWAY_CONFIRMED";
        default:                        return "UNKNOWN";
    }
}

void PresenceDetector::notifyActivity() {
    last_activity_ = std::chrono::steady_clock::now();
    
    // If we were away or scanning, transition back to active
    if (current_state_ != State::ACTIVELY_PRESENT) {
        transitionTo(State::ACTIVELY_PRESENT);
    }
}

void PresenceDetector::pauseForAuthentication() {
    std::lock_guard<std::mutex> lock(pause_mutex_);
    pause_count_++;
    
    if (pause_count_ == 1) {
        paused_for_auth_.store(true);
        Logger::getInstance().debug("Presence detection paused for authentication");
        
        // Release camera for PAM auth
        std::lock_guard<std::mutex> cam_lock(camera_mutex_);
        if (camera_ && camera_->isOpened()) {
            camera_->release();
        }
    }
}

void PresenceDetector::resumeAfterAuthentication() {
    std::lock_guard<std::mutex> lock(pause_mutex_);
    pause_count_--;
    
    if (pause_count_ == 0) {
        paused_for_auth_.store(false);
        Logger::getInstance().debug("Presence detection resumed after authentication");
    }
}

void PresenceDetector::detectionLoop() {
    guard_.updateState();
    
    while (running_.load()) {
        // Check if paused for authentication
        if (paused_for_auth_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        // Update guard conditions
        guard_.updateState();
        
        // Check schedule first
        if (!isWithinSchedule()) {
            // Outside scheduled hours/days, pause detection
            // If we were scanning, transition to ACTIVELY_PRESENT (assume user present)
            if (current_state_ == State::IDLE_WITH_SCANNING || 
                current_state_ == State::AWAY_CONFIRMED) {
                Logger::getInstance().info("Outside schedule - pausing presence detection");
                transitionTo(State::ACTIVELY_PRESENT);
            }
            std::this_thread::sleep_for(std::chrono::seconds(60));  // Check every minute
            continue;
        }
        
        if (!guard_.shouldRunPresenceDetection()) {
            // Guard conditions not met, pause detection
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }
        
        // Update state machine
        updateStateMachine();
        
        // Sleep based on current state
        if (current_state_ == State::IDLE_WITH_SCANNING) {
            std::this_thread::sleep_for(std::chrono::milliseconds(scan_interval_ms_));
        } else {
            // Not scanning, just monitoring activity
            // Use longer interval to reduce memory churn from /proc/interrupts reads
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }
}

void PresenceDetector::updateStateMachine() {
    auto now = std::chrono::steady_clock::now();
    auto inactive_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_activity_).count();
    
    Logger& logger = Logger::getInstance();
    
    // Debug: Log state and activity every 5 seconds
    static auto last_debug_log = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_debug_log).count() >= 5) {
        bool has_activity = hasRecentActivity();
        // Use static buffer to avoid string allocation overhead
        static char log_buf[256];
        snprintf(log_buf, sizeof(log_buf), 
                "State: %s, Inactive time: %lds, Has recent activity: %s, Threshold: %ds",
                getStateString().c_str(), 
                (long)(inactive_time / 1000),
                has_activity ? "YES" : "NO",
                inactive_threshold_ms_ / 1000);
        logger.info(log_buf);
        last_debug_log = now;
    }
    
    switch (current_state_) {
        case State::ACTIVELY_PRESENT:
            // Check if user has been inactive
            if (hasRecentActivity()) {
                // User still active, update timestamp
                last_activity_ = now;
            } else if (inactive_time > inactive_threshold_ms_) {
                // User has been inactive, start scanning
                logger.info("Inactivity detected! Transitioning to scanning mode");
                transitionTo(State::IDLE_WITH_SCANNING);
                scan_failures_ = 0;
            }
            break;
            
        case State::IDLE_WITH_SCANNING: {
            // FIRST: Check if user became active (keyboard/mouse input)
            // This takes priority over face scanning
            if (hasRecentActivity()) {
                logger.info("User activity detected during scanning, returning to active state");
                transitionTo(State::ACTIVELY_PRESENT);
                last_activity_ = now;
                scan_failures_ = 0;  // Reset failure counter
                consecutive_shutter_closed_scans_ = 0;  // Reset shutter counter
                break;
            }
            
            // Scan for face
            static char scan_buf[128];
            snprintf(scan_buf, sizeof(scan_buf), "Scanning for face... (failures so far: %d)", scan_failures_);
            logger.info(scan_buf);
            bool face_detected = detectFace();
            
            // Check if failure was due to closed shutter
            if (!face_detected && last_shutter_state_ == ShutterState::CLOSED) {
                consecutive_shutter_closed_scans_++;
                static char shutter_buf[128];
                snprintf(shutter_buf, sizeof(shutter_buf), 
                        "Camera shutter is CLOSED (scan #%d) - pausing presence detection",
                        consecutive_shutter_closed_scans_);
                logger.warning(shutter_buf);
                
                // Don't count as "away" - user may have closed shutter for privacy
                // Just pause scanning and stay in IDLE_WITH_SCANNING state
                
                // If shutter has been closed for configured timeout, lock anyway
                if (consecutive_shutter_closed_scans_ * scan_interval_ms_ > shutter_timeout_ms_) {
                    static char timeout_buf[128];
                    snprintf(timeout_buf, sizeof(timeout_buf), 
                            "Camera shutter closed for %d+ minutes - locking anyway",
                            shutter_timeout_ms_ / 60000);
                    logger.info(timeout_buf);
                    transitionTo(State::AWAY_CONFIRMED);
                    lockScreen();
                }
                break;
            }
            
            // Shutter is open, reset counter
            consecutive_shutter_closed_scans_ = 0;
            
            if (face_detected) {
                // Face detected! User is still here
                successful_detections_++;
                logger.info("Face detected! Returning to active state");
                transitionTo(State::ACTIVELY_PRESENT);
                last_activity_ = now;
                scan_failures_ = 0;  // Reset failure counter
                
                // Check for peek (shoulder surfing) using cached frame
                // This avoids reopening the camera after it was just released
                if (no_peek_enabled_ && !last_captured_frame_.empty()) {
                    bool peek = detectPeek(last_captured_frame_);
                    updatePeekState(peek);
                    // Clear cached frame after use
                    last_captured_frame_ = cv::Mat();
                }
            } else {
                // No face detected (and shutter is open)
                failed_detections_++;
                scan_failures_++;
                static char failure_buf[128];
                snprintf(failure_buf, sizeof(failure_buf), 
                        "No face detected (failure %d of %d)",
                        scan_failures_, max_scan_failures_);
                logger.info(failure_buf);
                
                if (scan_failures_ >= max_scan_failures_) {
                    // Too many failures, user is away
                    static char away_buf[128];
                    snprintf(away_buf, sizeof(away_buf), 
                            "User confirmed away after %d failed scans - locking screen",
                            scan_failures_);
                    logger.info(away_buf);
                    transitionTo(State::AWAY_CONFIRMED);
                    lockScreen();
                }
            }
            
            // Also check timeout
            if (inactive_time > max_idle_time_ms_) {
                static char idle_buf[128];
                snprintf(idle_buf, sizeof(idle_buf), 
                        "User confirmed away after %ld seconds idle - locking screen",
                        (long)(inactive_time / 1000));
                logger.info(idle_buf);
                transitionTo(State::AWAY_CONFIRMED);
                lockScreen();
            }
            break;
        }
            
        case State::AWAY_CONFIRMED:
            // Waiting for input activity to transition back
            if (hasRecentActivity()) {
                transitionTo(State::ACTIVELY_PRESENT);
                last_activity_ = now;
            }
            break;
    }
}

void PresenceDetector::transitionTo(State new_state) {
    if (new_state == current_state_) {
        return;
    }
    
    state_transitions_++;
    
    Logger& logger = Logger::getInstance();
    logger.info("State transition: " + getStateString() + " -> " + 
                [new_state]() {
                    switch (new_state) {
                        case State::ACTIVELY_PRESENT: return std::string("ACTIVELY_PRESENT");
                        case State::IDLE_WITH_SCANNING: return std::string("IDLE_WITH_SCANNING");
                        case State::AWAY_CONFIRMED: return std::string("AWAY_CONFIRMED");
                    }
                    return std::string("UNKNOWN");
                }());
    
    // Close camera when leaving IDLE_WITH_SCANNING state (save power/privacy)
    if (current_state_ == State::IDLE_WITH_SCANNING && 
        (new_state == State::ACTIVELY_PRESENT || new_state == State::AWAY_CONFIRMED)) {
        std::lock_guard<std::mutex> lock(camera_mutex_);
        if (camera_ && camera_->isOpened()) {
            camera_->release();
            camera_.reset();
            logger.info("Camera released (no longer scanning)");
        }
    }
    
    current_state_ = new_state;
    state_entry_time_ = std::chrono::steady_clock::now();
}

bool PresenceDetector::ensureDetectorInitialized() {
    if (detector_initialized_) {
        return true;
    }
    
    Logger& logger = Logger::getInstance();
    logger.info("Lazy loading YuNet face detector...");
    
    try {
        std::string model_path = "/etc/faceid/models/face_detection_yunet_2023mar.onnx";
        
        // Check if model file exists
        std::ifstream model_file(model_path);
        if (!model_file.good()) {
            char log_buf[512];
            snprintf(log_buf, sizeof(log_buf), 
                    "YuNet model not found at %s", model_path.c_str());
            logger.error(log_buf);
            return false;
        }
        model_file.close();
        
        // Create YuNet detector with optimized settings
        // Parameters: model_path, config, input_size, score_threshold, nms_threshold, top_k
        detector_ = cv::FaceDetectorYN::create(
            model_path,
            "",                      // No config file needed
            cv::Size(320, 240),      // Input size (must match resize in detectFace)
            0.6f,                    // Score threshold
            0.3f,                    // NMS threshold
            5000                     // Top K
        );
        
        if (detector_.empty()) {
            logger.error("Failed to create YuNet detector");
            return false;
        }
        
        detector_initialized_ = true;
        logger.info("YuNet face detector loaded successfully");
        return true;
        
    } catch (const std::exception& e) {
        char log_buf[512];
        snprintf(log_buf, sizeof(log_buf), 
                "Exception loading YuNet detector: %s", e.what());
        logger.error(log_buf);
        return false;
    }
}

bool PresenceDetector::detectFace() {
    try {
        total_scans_++;
        
        cv::Mat frame = captureFrame();
        if (frame.empty()) {
            failed_detections_++;
            return false;
        }
        
        // Check if camera shutter is closed
        ShutterState shutter = detectShutterState(frame);
        if (shutter == ShutterState::CLOSED) {
            char log_buf[256];
            snprintf(log_buf, sizeof(log_buf), 
                    "Camera shutter closed, skipping face detection (closed count: %d)",
                    consecutive_shutter_closed_scans_);
            Logger::getInstance().info(log_buf);
            failed_detections_++;
            return false;
        }
        
        if (shutter == ShutterState::UNCERTAIN) {
            Logger::getInstance().debug("Camera image is very dark - shutter might be closed");
            // Continue with detection anyway - might just be a dark room
        }
        
        // Lazy load detector if not already loaded
        if (!ensureDetectorInitialized()) {
            Logger::getInstance().error("Cannot detect face: YuNet detector not available");
            failed_detections_++;
            return false;
        }
        
        // Shutter is open, proceed with face detection
        // Resize for faster detection (YuNet expects 320x240 as configured)
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(320, 240));
        
        // Detect faces using YuNet
        cv::Mat faces;
        detector_->detect(resized, faces);
        
        // faces.rows contains the number of detected faces
        // Each row has: x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, confidence
        bool detected = (faces.rows > 0);
        
        if (detected) {
            // Log confidence of first detected face
            float confidence = faces.at<float>(0, 14);
            Logger::getInstance().debug("Face detected with confidence: " + 
                                      std::to_string(confidence));
        }
        
        return detected;
        
    } catch (const std::exception& e) {
        Logger::getInstance().error(std::string("Face detection error: ") + e.what());
        last_captured_frame_ = cv::Mat();  // Clear cached frame
        return false;
    }
}

cv::Mat PresenceDetector::captureFrame() {
    std::lock_guard<std::mutex> lock(camera_mutex_);
    
    // Initialize camera if not already open
    if (!camera_ || !camera_->isOpened()) {
        camera_ = std::make_unique<cv::VideoCapture>();
        
        // Open with V4L2 backend for better hardware support
        camera_->open(camera_device_, cv::CAP_V4L2);
        
        if (!camera_->isOpened()) {
            Logger::getInstance().error("Failed to open camera: " + camera_device_);
            return cv::Mat();
        }
        
        Logger& logger = Logger::getInstance();
        logger.info("Camera opened with V4L2 backend");
        
        // Set resolution (smaller for presence detection = faster processing)
        camera_->set(cv::CAP_PROP_FRAME_WIDTH, 640);
        camera_->set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        
        // Set MJPEG format for hardware acceleration on most webcams
        camera_->set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
        
        // Set FPS to reduce CPU load (we don't need high framerate for presence)
        camera_->set(cv::CAP_PROP_FPS, 15);
        
        // Disable auto-exposure for faster capture (optional)
        camera_->set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);  // Manual mode
        
        // Log actual settings
        logger.info("Camera resolution: " + 
                   std::to_string((int)camera_->get(cv::CAP_PROP_FRAME_WIDTH)) + "x" +
                   std::to_string((int)camera_->get(cv::CAP_PROP_FRAME_HEIGHT)));
        logger.info("Camera FPS: " + std::to_string((int)camera_->get(cv::CAP_PROP_FPS)));
        
        int fourcc = (int)camera_->get(cv::CAP_PROP_FOURCC);
        char fourcc_str[5] = {
            (char)(fourcc & 0xFF),
            (char)((fourcc >> 8) & 0xFF),
            (char)((fourcc >> 16) & 0xFF),
            (char)((fourcc >> 24) & 0xFF),
            '\0'
        };
        logger.info("Camera codec: " + std::string(fourcc_str));
    }
    
    cv::Mat frame;
    if (!camera_->read(frame)) {
        Logger::getInstance().error("Failed to capture frame");
        return cv::Mat();
    }
    
    return frame;
}

bool PresenceDetector::hasRecentActivity() const {
    // NEW IMPLEMENTATION: Use /dev/input/event* modification time
    // This is much faster and more reliable than /proc/interrupts
    
    time_t last_activity = getLastInputDeviceActivity();
    
    if (last_activity == 0) {
        return false;  // Could not determine
    }
    
    time_t now = time(nullptr);
    time_t idle_seconds = now - last_activity;
    
    // Apply mouse jitter threshold
    if (idle_seconds == 0 && last_device_was_mouse_) {
        // Check if mouse moved very recently (within jitter threshold)
        auto now_steady = std::chrono::steady_clock::now();
        auto since_last_mouse = std::chrono::duration_cast<std::chrono::milliseconds>(
            now_steady - last_mouse_activity_time_).count();
        
        if (since_last_mouse < mouse_jitter_threshold_ms_) {
            // Too soon after last mouse activity, might be jitter - ignore
            return false;
        }
    }
    
    // Activity detected if idle < 2 seconds
    bool activity = (idle_seconds < 2);
    
    if (activity) {
        Logger& logger = Logger::getInstance();
        logger.debug("ACTIVITY DETECTED: Input activity within last " + 
                    std::to_string(idle_seconds) + " seconds");
    }
    
    return activity;
}

bool PresenceDetector::detectDisplayServer() {
    // Method 1: Check environment variables (works if service has them)
    const char* wayland_display = getenv("WAYLAND_DISPLAY");
    if (wayland_display && strlen(wayland_display) > 0) {
        return true;  // Wayland
    }
    
    const char* session_type = getenv("XDG_SESSION_TYPE");
    if (session_type && strcmp(session_type, "wayland") == 0) {
        return true;  // Wayland
    }
    
    // Method 2: Check via loginctl (more reliable for systemd services)
    FILE* pipe = popen("loginctl show-session $(loginctl list-sessions --no-legend | awk '{print $1}' | head -1) -p Type --value 2>/dev/null | head -1", "r");
    if (pipe) {
        char buffer[32];
        bool wayland = false;
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            std::string type(buffer);
            // Remove whitespace
            type.erase(type.find_last_not_of(" \n\r\t") + 1);
            if (type == "wayland") {
                wayland = true;
            }
        }
        pclose(pipe);  // Only close once
        if (wayland) {
            return true;  // Wayland
        }
    }
    
    return false;  // X11 or unknown (assume X11)
}

std::chrono::steady_clock::time_point PresenceDetector::getLastInputActivity() const {
    auto now = std::chrono::steady_clock::now();
    Logger& logger = Logger::getInstance();
    
    // Use /proc/interrupts for both X11 and Wayland (most reliable method)
    // Check for input device interrupt changes
    FILE* pipe = popen("grep -E 'i8042|keyboard|mouse' /proc/interrupts 2>/dev/null | awk '{sum+=$2} END {print sum}'", "r");
    if (pipe) {
        char buffer[128];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            pclose(pipe);
            try {
                unsigned long interrupt_count = std::stoul(buffer);
                unsigned long prev_count = last_interrupt_count_;
                
                logger.info("Interrupt check: current=" + std::to_string(interrupt_count) + 
                           ", previous=" + std::to_string(prev_count) + 
                           ", delta=" + std::to_string(interrupt_count - prev_count));
                
                if (interrupt_count > last_interrupt_count_) {
                    // Input activity detected - interrupts increased
                    last_interrupt_count_ = interrupt_count;
                    logger.info("ACTIVITY DETECTED: Interrupts increased from " + 
                               std::to_string(prev_count) + " to " + std::to_string(interrupt_count));
                    return now;
                }
                // No new interrupts - calculate time since last activity
                // We don't know exact time, but we know it's been at least a few seconds
                // Return a time in the past to indicate idle
                logger.info("NO ACTIVITY: Interrupts unchanged at " + std::to_string(interrupt_count));
                return now - std::chrono::seconds(60);
            } catch (...) {
                // Parsing failed
                logger.error("Failed to parse interrupt count");
            }
        } else {
            pclose(pipe);
        }
    }
    
    // Fallback: Try X11 xprintidle if available
    if (!is_wayland_) {
        FILE* pipe2 = popen("xprintidle 2>/dev/null", "r");
        if (pipe2) {
            char buffer[128];
            if (fgets(buffer, sizeof(buffer), pipe2) != nullptr) {
                pclose(pipe2);
                try {
                    long idle_ms = std::stol(buffer);
                    return now - std::chrono::milliseconds(idle_ms);
                } catch (...) {
                    // Parsing failed
                }
            } else {
                pclose(pipe2);
            }
        }
    }
    
    // Final fallback: Return time in past to indicate idle
    return now - std::chrono::hours(1);
}

PresenceDetector::Statistics PresenceDetector::getStatistics() const {
    auto now = std::chrono::steady_clock::now();
    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
    
    return Statistics{
        .totalScans = total_scans_.load(),
        .facesDetected = successful_detections_.load(),
        .failedScans = failed_detections_.load(),
        .stateTransitions = state_transitions_.load(),
        .uptimeSeconds = static_cast<int>(uptime)
    };
}

// NEW: Input device detection using /dev/input mtime
time_t PresenceDetector::getLastInputDeviceActivity() const {
    // Cache activity checks to reduce /proc/interrupts reads and memory churn
    auto now = std::chrono::steady_clock::now();
    auto time_since_last_check = std::chrono::duration_cast<std::chrono::seconds>(
        now - last_activity_check_).count();
    
    // Return cached value if checked recently (within activity_cache_duration_)
    if (time_since_last_check < activity_cache_duration_.count()) {
        return cached_last_activity_;
    }
    
    // Time to perform actual check
    last_activity_check_ = now;
    
    // Use /proc/interrupts to track input activity
    // This works for PS/2 (i8042), USB, and Bluetooth (xhci_hcd) input devices
    
    Logger& logger = Logger::getInstance();
    
    std::ifstream interrupts("/proc/interrupts");
    if (!interrupts.is_open()) {
        logger.debug("Failed to open /proc/interrupts");
        return cached_last_activity_;  // Return cached value on error
    }
    
    unsigned long long total_count = 0;
    static std::string line;  // Static to avoid reallocation
    line.clear();
    line.reserve(256);  // Pre-allocate to reduce reallocations
    
    while (std::getline(interrupts, line)) {
        // Look for input-related interrupts:
        // - i8042: PS/2 keyboard/mouse controller (laptop built-in keyboard/touchpad)
        // - amd_gpio: GPIO-based i2c devices (modern touchscreens/touchpads)
        // NOTE: Removed xhci_hcd and usb because they include camera and other non-input USB devices
        // which cause false positive activity detection
        if (line.find("i8042") != std::string::npos ||
            line.find("amd_gpio") != std::string::npos) {
            
            // Parse interrupt counts from all CPUs and sum them
            // Use C-style parsing to avoid istringstream allocation overhead
            const char* ptr = line.c_str();
            
            // Skip IRQ number (everything before first colon)
            while (*ptr && *ptr != ':') ptr++;
            if (*ptr == ':') ptr++;
            
            // Sum all CPU interrupt counts
            unsigned long long count;
            while (*ptr) {
                while (*ptr == ' ' || *ptr == '\t') ptr++;  // Skip whitespace
                if (*ptr >= '0' && *ptr <= '9') {
                    char* end;
                    count = strtoull(ptr, &end, 10);
                    total_count += count;
                    ptr = end;
                } else {
                    break;  // Reached device name part
                }
            }
        }
    }
    
    interrupts.close();  // Explicit close to free resources immediately
    
    // If no interrupts found or count is 0, return cached value
    if (total_count == 0) {
        logger.debug("No input interrupts found in /proc/interrupts");
        return cached_last_activity_;
    }
    
    // Store the interrupt count (static for persistence across calls)
    static unsigned long long last_interrupt_count = 0;
    static time_t last_update_time = time(nullptr);
    
    // If interrupt count increased, update timestamp  
    if (total_count > last_interrupt_count) {
        last_interrupt_count = total_count;
        last_update_time = time(nullptr);
    }
    
    // Update cache
    cached_last_activity_ = last_update_time;
    
    return last_update_time;
}

// NEW: Schedule checking
bool PresenceDetector::isWithinSchedule() const {
    // If schedule is disabled, always return true (active all the time)
    if (!schedule_enabled_) {
        return true;
    }
    
    Logger& logger = Logger::getInstance();
    
    // Get current day of week and time
    time_t now = time(nullptr);
    struct tm* local_time = localtime(&now);
    
    // tm_wday: 0=Sunday, 1=Monday, ..., 6=Saturday
    // Convert to: 1=Monday, 7=Sunday
    int current_day = (local_time->tm_wday == 0) ? 7 : local_time->tm_wday;
    
    // Current time in HHMM format
    int current_time = local_time->tm_hour * 100 + local_time->tm_min;
    
    // Check if current day is in active days
    bool day_active = false;
    for (int day : active_days_) {
        if (day == current_day) {
            day_active = true;
            break;
        }
    }
    
    if (!day_active) {
        logger.debug("Outside schedule: Current day " + std::to_string(current_day) + 
                    " not in active days");
        return false;
    }
    
    // Check if current time is within active range
    bool time_active = (current_time >= schedule_time_start_ && 
                       current_time <= schedule_time_end_);
    
    if (!time_active) {
        logger.debug("Outside schedule: Current time " + std::to_string(current_time) + 
                    " not in range " + std::to_string(schedule_time_start_) + 
                    "-" + std::to_string(schedule_time_end_));
        return false;
    }
    
    logger.debug("Within schedule: Day " + std::to_string(current_day) + 
                ", Time " + std::to_string(current_time));
    return true;
}


// NEW: Lock screen trigger
void PresenceDetector::lockScreen() {
    Logger& logger = Logger::getInstance();
    logger.info("Attempting to lock screen...");
    
    // Method 1: loginctl lock-sessions (works from systemd service, locks ALL sessions)
    // This is the most reliable method for systemd services running as root
    int ret = system("loginctl lock-sessions 2>/dev/null");
    if (ret == 0) {
        logger.info("Screen locked successfully via loginctl lock-sessions");
        return;
    }
    
    // Method 2: Lock specific session (try to get active session)
    ret = system("loginctl lock-session $(loginctl list-sessions --no-legend | awk '{print $1}' | head -1) 2>/dev/null");
    if (ret == 0) {
        logger.info("Screen locked successfully via loginctl lock-session");
        return;
    }
    
    // Method 3: KDE-specific via user's D-Bus session (requires env variables)
    // This will fail from systemd service but worth trying
    ret = system("su - $(loginctl list-sessions --no-legend | awk '{print $3}' | head -1) -c 'qdbus org.freedesktop.ScreenSaver /ScreenSaver Lock' 2>/dev/null");
    if (ret == 0) {
        logger.info("Screen locked successfully via user D-Bus session");
        return;
    }
    
    logger.error("Failed to lock screen - all methods failed");
}

// NEW: Camera shutter detection
PresenceDetector::ShutterState PresenceDetector::detectShutterState(const cv::Mat& frame) {
    if (frame.empty()) {
        return ShutterState::UNCERTAIN;
    }
    
    // Calculate mean brightness (RGB average)
    cv::Scalar mean_color = cv::mean(frame);
    double brightness = (mean_color[0] + mean_color[1] + mean_color[2]) / 3.0;
    
    // Convert to grayscale for variance check
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    
    // Calculate standard deviation (variance indicator)
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    
    Logger& logger = Logger::getInstance();
    
    // Debug logging (throttled to every 5 seconds)
    static auto last_log = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_log).count() >= 5) {
        logger.debug("Shutter check: brightness=" + std::to_string(brightness) + 
                    ", stddev=" + std::to_string(stddev[0]));
        last_log = now;
    }
    
    // Check if image is pure black (shutter closed)
    if (brightness < shutter_brightness_threshold_ && 
        stddev[0] < shutter_variance_threshold_) {
        return ShutterState::CLOSED;
    }
    
    // Check if very dark but with some variance (might be dark room)
    if (brightness < 15.0) {
        return ShutterState::UNCERTAIN;
    }
    
    return ShutterState::OPEN;
}

// No-peek detection (detects shoulder surfing - additional faces behind user)
bool PresenceDetector::detectPeek(const cv::Mat& frame) {
    if (!no_peek_enabled_ || frame.empty()) {
        return false;
    }
    
    Logger& logger = Logger::getInstance();
    
    try {
        // Resize for faster detection (YuNet expects 320x240 as configured)
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(320, 240));
        
        // Detect faces using YuNet
        cv::Mat faces;
        detector_->detect(resized, faces);
        
        if (faces.rows == 0) {
            return false;  // No faces at all
        }
        
        // Convert YuNet results to cv::Rect for analysis
        std::vector<cv::Rect> face_rects;
        double scale_x = static_cast<double>(frame.cols) / 320.0;
        double scale_y = static_cast<double>(frame.rows) / 240.0;
        
        for (int i = 0; i < faces.rows; i++) {
            float x = faces.at<float>(i, 0) * scale_x;
            float y = faces.at<float>(i, 1) * scale_y;
            float w = faces.at<float>(i, 2) * scale_x;
            float h = faces.at<float>(i, 3) * scale_y;
            float confidence = faces.at<float>(i, 14);
            
            // Only count faces with reasonable confidence
            if (confidence > 0.6) {
                cv::Rect face(
                    static_cast<int>(x),
                    static_cast<int>(y),
                    static_cast<int>(w),
                    static_cast<int>(h)
                );
                
                // Filter out faces that are too small (too far away to see screen)
                double face_size_percent = static_cast<double>(w) / frame.cols;
                if (face_size_percent >= min_face_size_percent_) {
                    face_rects.push_back(face);
                }
            }
        }
        
        // Check if we have multiple distinct faces
        if (face_rects.size() < 2) {
            return false;  // Only one person (or none after filtering)
        }
        
        // Use FaceDetector helper to count distinct faces
        // (This filters out same person detected multiple times due to movement)
        int distinct_count = FaceDetector::countDistinctFaces(face_rects, min_face_distance_pixels_);
        
        bool peek = (distinct_count >= 2);
        
        if (peek) {
            logger.warning("NO PEEK: Detected " + std::to_string(distinct_count) + 
                          " distinct faces (potential shoulder surfing)");
        }
        
        return peek;
        
    } catch (const std::exception& e) {
        logger.error(std::string("Peek detection error: ") + e.what());
        return false;
    }
}

void PresenceDetector::updatePeekState(bool peek_detected) {
    if (!no_peek_enabled_) {
        return;
    }
    
    Logger& logger = Logger::getInstance();
    auto now = std::chrono::steady_clock::now();
    
    if (peek_detected) {
        // Peek detected in current frame
        if (peek_state_ == PeekState::NO_PEEK) {
            // First detection
            peek_state_ = PeekState::PEEK_DETECTED;
            peek_first_detected_ = now;
            consecutive_peek_detections_ = 1;
            logger.info("Peek DETECTED (first time)");
        } else {
            // Already detected, check if we should confirm
            consecutive_peek_detections_++;
            
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - peek_first_detected_).count();
            
            if (duration_ms >= peek_detection_delay_ms_ && 
                peek_state_ != PeekState::PEEK_CONFIRMED) {
                // Confirmed: peek has persisted long enough
                peek_state_ = PeekState::PEEK_CONFIRMED;
                logger.warning("Peek CONFIRMED - blanking screen");
                blankScreen();
            }
        }
        
        peek_last_seen_ = now;
        
    } else {
        // No peek in current frame
        if (peek_state_ != PeekState::NO_PEEK) {
            // Check if peek disappeared long enough to unblank
            auto time_since_last_peek_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - peek_last_seen_).count();
            
            if (time_since_last_peek_ms >= unblank_delay_ms_) {
                // Peek has been gone for grace period
                logger.info("Peek cleared - unblanking screen");
                peek_state_ = PeekState::NO_PEEK;
                consecutive_peek_detections_ = 0;
                unblankScreen();
            }
        }
    }
}

void PresenceDetector::blankScreen() {
    if (screen_blanked_) {
        return;  // Already blanked
    }
    
    Logger& logger = Logger::getInstance();
    logger.warning("BLANKING screen due to peek detection");
    
    // Try multiple methods to blank screen
    bool success = false;
    
    // Method 1: DPMS off via xset (works on X11)
    int ret = system("DISPLAY=:0 xset dpms force off 2>/dev/null");
    if (ret == 0) {
        success = true;
        logger.info("Screen blanked via xset (X11)");
    }
    
    // Method 2: KDE Plasma screen blanking (Wayland)
    if (!success) {
        ret = system("qdbus org.kde.KWin.ScreenSaver2 /ScreenSaver setActive true 2>/dev/null");
        if (ret == 0) {
            success = true;
            logger.info("Screen blanked via KDE ScreenSaver (Wayland)");
        }
    }
    
    // Method 3: GNOME (Wayland/X11)
    if (!success) {
        ret = system("dbus-send --session --type=method_call --dest=org.gnome.ScreenSaver "
                    "/org/gnome/ScreenSaver org.gnome.ScreenSaver.SetActive boolean:true 2>/dev/null");
        if (ret == 0) {
            success = true;
            logger.info("Screen blanked via GNOME ScreenSaver");
        }
    }
    
    // Method 4: Generic wlr-randr for wlroots compositors (Sway, etc.)
    if (!success) {
        ret = system("wlr-randr --output '*' --off 2>/dev/null");
        if (ret == 0) {
            success = true;
            logger.info("Screen blanked via wlr-randr");
        }
    }
    
    if (success) {
        screen_blanked_ = true;
    } else {
        logger.error("Failed to blank screen - all methods failed");
    }
}

void PresenceDetector::unblankScreen() {
    if (!screen_blanked_) {
        return;  // Not blanked
    }
    
    Logger& logger = Logger::getInstance();
    logger.info("UNBLANKING screen - peek cleared");
    
    // Try multiple methods to unblank screen
    bool success = false;
    
    // Method 1: DPMS on via xset (works on X11)
    int ret = system("DISPLAY=:0 xset dpms force on 2>/dev/null");
    if (ret == 0) {
        success = true;
        logger.info("Screen unblanked via xset (X11)");
    }
    
    // Method 2: KDE Plasma screen unblanking (Wayland)
    if (!success) {
        ret = system("qdbus org.kde.KWin.ScreenSaver2 /ScreenSaver setActive false 2>/dev/null");
        if (ret == 0) {
            success = true;
            logger.info("Screen unblanked via KDE ScreenSaver (Wayland)");
        }
    }
    
    // Method 3: GNOME (Wayland/X11)
    if (!success) {
        ret = system("dbus-send --session --type=method_call --dest=org.gnome.ScreenSaver "
                    "/org/gnome/ScreenSaver org.gnome.ScreenSaver.SetActive boolean:false 2>/dev/null");
        if (ret == 0) {
            success = true;
            logger.info("Screen unblanked via GNOME ScreenSaver");
        }
    }
    
    // Method 4: Generic wlr-randr for wlroots compositors
    if (!success) {
        ret = system("wlr-randr --output '*' --on 2>/dev/null");
        if (ret == 0) {
            success = true;
            logger.info("Screen unblanked via wlr-randr");
        }
    }
    
    // Method 5: Simple mouse wiggle to wake screen (fallback)
    if (!success) {
        ret = system("xdotool mousemove_relative -- 1 0 2>/dev/null");
        if (ret == 0) {
            success = true;
            logger.info("Screen unblanked via mouse wiggle");
        }
    }
    
    if (success) {
        screen_blanked_ = false;
    } else {
        logger.error("Failed to unblank screen - may need manual intervention");
    }
}

} // namespace faceid
