#include "presence_detector.h"
#include "../logger.h"
#include <fstream>
#include <sys/stat.h>

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
        // Set YuNet model path (will be installed to /etc/faceid/models/)
        std::string model_path = "/etc/faceid/models/face_detection_yunet_2023mar.onnx";
        
        // Check if model exists
        std::ifstream model_file(model_path);
        if (!model_file.good()) {
            logger.error("YuNet model not found at: " + model_path);
            logger.error("Please run: sudo cp models/face_detection_yunet_2023mar.onnx /etc/faceid/models/");
            return false;
        }
        
        // Initialize YuNet face detector
        detector_ = cv::FaceDetectorYN::create(
            model_path,
            "",                    // config (empty for ONNX)
            cv::Size(320, 240),   // input size (matches our resize)
            0.6f,                 // score threshold (lower = more strict)
            0.3f,                 // nms threshold
            5000                  // top_k
        );
        
        detector_initialized_ = true;
        logger.info("YuNet face detector initialized successfully");
        
        // Detect display server (X11 vs Wayland)
        is_wayland_ = detectDisplayServer();
        logger.info(std::string("Display server: ") + (is_wayland_ ? "Wayland" : "X11"));
        
        logger.info("Presence detector initialized successfully");
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
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
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
        logger.info("State: " + getStateString() + 
                   ", Inactive time: " + std::to_string(inactive_time / 1000) + "s" +
                   ", Has recent activity: " + (has_activity ? "YES" : "NO") +
                   ", Threshold: " + std::to_string(inactive_threshold_ms_ / 1000) + "s");
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
            
        case State::IDLE_WITH_SCANNING:
            // FIRST: Check if user became active (keyboard/mouse input)
            // This takes priority over face scanning
            if (hasRecentActivity()) {
                logger.info("User activity detected during scanning, returning to active state");
                transitionTo(State::ACTIVELY_PRESENT);
                last_activity_ = now;
                scan_failures_ = 0;  // Reset failure counter
                break;
            }
            
            // Scan for face
            logger.info("Scanning for face... (failures so far: " + std::to_string(scan_failures_) + ")");
            if (detectFace()) {
                // Face detected! User is still here
                successful_detections_++;
                logger.info("Face detected! Returning to active state");
                transitionTo(State::ACTIVELY_PRESENT);
                last_activity_ = now;
                scan_failures_ = 0;  // Reset failure counter
            } else {
                // No face detected
                failed_detections_++;
                scan_failures_++;
                logger.info("No face detected (failure " + std::to_string(scan_failures_) + 
                           " of " + std::to_string(max_scan_failures_) + ")");
                
                if (scan_failures_ >= max_scan_failures_) {
                    // Too many failures, user is away
                    logger.info("User confirmed away after " + 
                               std::to_string(scan_failures_) + " failed scans - locking screen");
                    transitionTo(State::AWAY_CONFIRMED);
                    // TODO: Trigger screen lock
                }
            }
            
            // Also check timeout
            if (inactive_time > max_idle_time_ms_) {
                logger.info("User confirmed away after " + 
                           std::to_string(inactive_time / 1000) + " seconds idle");
                transitionTo(State::AWAY_CONFIRMED);
                // TODO: Trigger screen lock
            }
            break;
            
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

bool PresenceDetector::detectFace() {
    total_scans_++;
    
    try {
        cv::Mat frame = captureFrame();
        if (frame.empty()) {
            return false;
        }
        
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
    // Check for keyboard/mouse activity by looking at interrupt count changes
    // Cache the result to avoid spawning processes too frequently
    
    auto now = std::chrono::steady_clock::now();
    
    // Only check interrupts every 500ms to reduce process spawning
    if (now - last_interrupt_check_ < interrupt_check_interval_) {
        // Use cached result - no change detected since last check
        return false;
    }
    
    last_interrupt_check_ = now;
    
    FILE* pipe = popen("grep -E 'i8042|keyboard|mouse' /proc/interrupts 2>/dev/null | awk '{sum+=$2} END {print sum}'", "r");
    if (!pipe) {
        return false;
    }
    
    char buffer[128];
    if (fgets(buffer, sizeof(buffer), pipe) == nullptr) {
        pclose(pipe);
        return false;
    }
    pclose(pipe);
    
    try {
        unsigned long interrupt_count = std::stoul(buffer);
        unsigned long prev_count = last_interrupt_count_;
        
        // Log for debugging (throttled to every 2 seconds)
        Logger& logger = Logger::getInstance();
        static auto last_log = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_log).count() >= 2) {
            logger.info("Interrupt check: current=" + std::to_string(interrupt_count) + 
                       ", previous=" + std::to_string(prev_count) + 
                       ", delta=" + std::to_string(interrupt_count - prev_count));
            last_log = now;
        }
        
        // Check if interrupts increased (indicates activity)
        if (interrupt_count > prev_count) {
            logger.info("ACTIVITY DETECTED: Interrupts increased from " + 
                       std::to_string(prev_count) + " to " + std::to_string(interrupt_count));
            last_interrupt_count_ = interrupt_count;
            return true;
        }
        
        // Update the count even if no change (for next comparison)
        last_interrupt_count_ = interrupt_count;
        
        // No change in interrupts = no activity
        return false;
    } catch (...) {
        return false;
    }
}

bool PresenceDetector::detectDisplayServer() {
    // Check for Wayland session
    const char* wayland_display = getenv("WAYLAND_DISPLAY");
    if (wayland_display && strlen(wayland_display) > 0) {
        return true;  // Wayland
    }
    
    // Check for XDG_SESSION_TYPE
    const char* session_type = getenv("XDG_SESSION_TYPE");
    if (session_type && strcmp(session_type, "wayland") == 0) {
        return true;  // Wayland
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

} // namespace faceid
