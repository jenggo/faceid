#include "presence_detector.h"
#include "../logger.h"
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
            logger.info("Scanning for face... (failures so far: " + std::to_string(scan_failures_) + ")");
            bool face_detected = detectFace();
            
            // Check if failure was due to closed shutter
            if (!face_detected && last_shutter_state_ == ShutterState::CLOSED) {
                consecutive_shutter_closed_scans_++;
                logger.warning("Camera shutter is CLOSED (scan #" + 
                              std::to_string(consecutive_shutter_closed_scans_) + 
                              ") - pausing presence detection");
                
                // Don't count as "away" - user may have closed shutter for privacy
                // Just pause scanning and stay in IDLE_WITH_SCANNING state
                
                // If shutter has been closed for configured timeout, lock anyway
                if (consecutive_shutter_closed_scans_ * scan_interval_ms_ > shutter_timeout_ms_) {
                    logger.info("Camera shutter closed for " + 
                               std::to_string(shutter_timeout_ms_ / 60000) + 
                               "+ minutes - locking anyway");
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
            } else {
                // No face detected (and shutter is open)
                failed_detections_++;
                scan_failures_++;
                logger.info("No face detected (failure " + std::to_string(scan_failures_) + 
                           " of " + std::to_string(max_scan_failures_) + ")");
                
                if (scan_failures_ >= max_scan_failures_) {
                    // Too many failures, user is away
                    logger.info("User confirmed away after " + 
                               std::to_string(scan_failures_) + " failed scans - locking screen");
                    transitionTo(State::AWAY_CONFIRMED);
                    lockScreen();
                }
            }
            
            // Also check timeout
            if (inactive_time > max_idle_time_ms_) {
                logger.info("User confirmed away after " + 
                           std::to_string(inactive_time / 1000) + " seconds idle - locking screen");
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

bool PresenceDetector::detectFace() {
    total_scans_++;
    
    try {
        cv::Mat frame = captureFrame();
        if (frame.empty()) {
            return false;
        }
        
        // CHECK SHUTTER STATE BEFORE DETECTING FACE
        ShutterState shutter = detectShutterState(frame);
        last_shutter_state_ = shutter;
        
        if (shutter == ShutterState::CLOSED) {
            Logger::getInstance().warning("Camera shutter is CLOSED - pausing face detection");
            // Return special value to indicate shutter closed (not a failure)
            // This will be handled in updateStateMachine()
            return false;  // Will be handled specially by checking last_shutter_state_
        }
        
        if (shutter == ShutterState::UNCERTAIN) {
            Logger::getInstance().debug("Camera image is very dark - shutter might be closed");
            // Continue with detection anyway - might just be a dark room
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
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            pclose(pipe);
            std::string type(buffer);
            // Remove whitespace
            type.erase(type.find_last_not_of(" \n\r\t") + 1);
            if (type == "wayland") {
                return true;  // Wayland
            }
        }
        pclose(pipe);
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
    // Use /proc/interrupts to track input activity
    // This works for both PS/2 (i8042) and USB input devices
    
    std::ifstream interrupts("/proc/interrupts");
    if (!interrupts.is_open()) {
        return 0;
    }
    
    unsigned long long max_count = 0;
    std::string line;
    
    while (std::getline(interrupts, line)) {
        // Look for input-related interrupts:
        // - i8042: PS/2 keyboard/mouse controller
        // - usb-hid: USB input devices
        if (line.find("i8042") != std::string::npos ||
            line.find("usb") != std::string::npos) {
            
            // Parse interrupt counts from all CPUs
            std::istringstream iss(line);
            std::string irq_num;
            iss >> irq_num;  // Skip IRQ number
            
            unsigned long long count;
            while (iss >> count) {
                if (count > max_count) {
                    max_count = count;
                }
            }
        }
    }
    
    // If no interrupts found or count is 0, return 0
    if (max_count == 0) {
        return 0;
    }
    
    // Store the interrupt count
    static unsigned long long last_interrupt_count = 0;
    static time_t last_update_time = time(nullptr);
    
    // If interrupt count increased, update timestamp
    if (max_count > last_interrupt_count) {
        last_interrupt_count = max_count;
        last_update_time = time(nullptr);
    }
    
    return last_update_time;
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

} // namespace faceid
