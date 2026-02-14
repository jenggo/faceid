#include "detector.h"
#include "../logger.h"
#include "../daemon/config.h"
#include "../models/model_cache.h"
#include <libyuv.h>
#include <fstream>
#include <thread>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/file.h>
#include <unistd.h>
#include <fcntl.h>
#include <glob.h>
#include <cmath>

// Lock file pattern for detecting PAM authentication
// Must match the path used in pam/pam_faceid.cpp SystemWideLock class
static const char* PAM_LOCK_PATTERN = "/run/faceid/faceid-*.lock";

namespace faceid {

// Helper function: Fast GRAY to BGR conversion using libyuv
static Image convertGrayToBGRLibyuv(const ImageView& src_gray) {
    Image dst_bgr(src_gray.width(), src_gray.height(), 3);
    
    // Convert grayscale to ARGB first, then to RGB24
    Image argb_temp(src_gray.width(), src_gray.height(), 4);
    
    // J400 (grayscale) to ARGB
    libyuv::J400ToARGB(src_gray.data(), src_gray.stride(), 
                       argb_temp.data(), argb_temp.stride(), 
                       src_gray.width(), src_gray.height());
    
    // ARGB to RGB24 (BGR)
    libyuv::ARGBToRGB24(argb_temp.data(), argb_temp.stride(), 
                        dst_bgr.data(), dst_bgr.stride(), 
                        dst_bgr.width(), dst_bgr.height());
    
    return dst_bgr;
}

PresenceDetector::PresenceDetector(
    const std::string& camera_device,
    std::chrono::seconds inactive_threshold,
    std::chrono::seconds scan_interval,
    int max_scan_failures,
    std::chrono::minutes max_idle_time)
    : camera_device_(camera_device)
    , last_enrollment_load_(std::chrono::steady_clock::now() - std::chrono::hours(24))
    , current_state_(State::ACTIVELY_PRESENT)
    , last_activity_(std::chrono::steady_clock::now())
    , state_entry_time_(std::chrono::steady_clock::now())
    , start_time_(std::chrono::steady_clock::now())
    , scan_failures_(0)
    , pause_count_(0)
    , inactive_threshold_ms_(std::chrono::duration_cast<std::chrono::milliseconds>(inactive_threshold).count())
    , scan_interval_ms_(std::chrono::duration_cast<std::chrono::milliseconds>(scan_interval).count())
    , max_scan_failures_(max_scan_failures)
    , max_idle_time_ms_(std::chrono::duration_cast<std::chrono::milliseconds>(max_idle_time).count()) {
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
        
        // Initialize InputMonitor for event-driven activity detection
        input_monitor_ = std::make_unique<InputMonitor>();
        if (!input_monitor_->initialize()) {
            logger.error("Failed to initialize InputMonitor");
            return false;
        }
        
        // Set activity callback
        input_monitor_->setActivityCallback([this]() {
            notifyActivity();
        });
        
        // Initialize PeekDetector (lazy-loaded with FaceDetector)
        peek_detector_ = std::make_unique<PeekDetector>(face_detector_.get());
        peek_detector_->setGazeThresholds(gaze_yaw_threshold_, gaze_pitch_threshold_);
        peek_detector_->setFaceFilters(min_face_size_percent_, min_face_distance_pixels_);
        
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
    
    Logger& logger = Logger::getInstance();
    logger.info("Starting presence detection service");
    
    // Start InputMonitor
    if (input_monitor_) {
        input_monitor_->start();
    }
    
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
    
    // Stop InputMonitor
    if (input_monitor_) {
        input_monitor_->stop();
    }
    
    if (detection_thread_.joinable()) {
        detection_thread_.join();
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
    has_recent_activity_.store(true);
    
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

// Check if PAM authentication is in progress by testing if any lock file is locked
bool PresenceDetector::checkPAMLockFile() {
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));
    
    int ret = glob(PAM_LOCK_PATTERN, GLOB_TILDE, NULL, &glob_result);
    if (ret != 0) {
        return false; // No matching files found
    }
    
    bool any_locked = false;
    
    // Check each lock file
    for (size_t i = 0; i < glob_result.gl_pathc; i++) {
        const char* lock_file = glob_result.gl_pathv[i];
        
        int fd = open(lock_file, O_RDONLY);
        if (fd == -1) {
            continue;
        }
        
        // Try to acquire shared lock (non-blocking)
        int result = flock(fd, LOCK_SH | LOCK_NB);
        
        if (result == -1 && errno == EWOULDBLOCK) {
            // PAM holds exclusive lock
            any_locked = true;
            close(fd);
            break;
        } else if (result == 0) {
            // We got the shared lock, release it
            flock(fd, LOCK_UN);
        }
        
        close(fd);
    }
    
    globfree(&glob_result);
    return any_locked;
}

void PresenceDetector::detectionLoop() {
    guard_.updateState();
    
    while (running_.load()) {
        // Check if PAM authentication is in progress
        bool pam_lock_exists = checkPAMLockFile();
        if (pam_lock_exists && !paused_for_auth_.load()) {
            Logger::getInstance().debug("PAM authentication lock detected, pausing presence detection");
            pauseForAuthentication();
        } else if (!pam_lock_exists && paused_for_auth_.load() && pause_count_ == 1) {
            Logger::getInstance().debug("PAM authentication lock released, resuming presence detection");
            resumeAfterAuthentication();
        }
        
        // Check if paused for authentication
        if (paused_for_auth_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        // Check schedule first
        if (!isWithinSchedule()) {
            if (current_state_ == State::IDLE_WITH_SCANNING || 
                current_state_ == State::AWAY_CONFIRMED) {
                Logger::getInstance().info("Outside schedule - pausing presence detection");
                transitionTo(State::ACTIVELY_PRESENT);
            }
            // Sleep in short intervals
            for (int i = 0; i < 60 && running_.load(); i++) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            continue;
        }
        
        // Update guard conditions
        guard_.updateState();
        
        if (!guard_.shouldRunPresenceDetection()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }
        
        // Update state machine
        updateStateMachine();
        
        // Sleep based on current state
        if (current_state_ == State::IDLE_WITH_SCANNING) {
            std::this_thread::sleep_for(std::chrono::milliseconds(scan_interval_ms_));
        } else {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

void PresenceDetector::updateStateMachine() {
    auto now = std::chrono::steady_clock::now();
    auto inactive_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_activity_).count();
    
    Logger& logger = Logger::getInstance();
    
    // Debug: Log state every 5 seconds
    static auto last_debug_log = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_debug_log).count() >= 5) {
        bool has_activity = hasRecentActivity();
        logger.debug("State: " + getStateString() + 
                     ", Inactive time: " + std::to_string(inactive_time / 1000) + "s" +
                     ", Has activity: " + (has_activity ? "YES" : "NO"));
        last_debug_log = now;
    }
    
    switch (current_state_) {
        case State::ACTIVELY_PRESENT:
            // Check if user has been inactive
            if (hasRecentActivity()) {
                last_activity_ = now;
            } else if (inactive_time > inactive_threshold_ms_) {
                logger.info("Inactivity detected! Transitioning to scanning mode");
                transitionTo(State::IDLE_WITH_SCANNING);
                scan_failures_ = 0;
            }
            break;
            
        case State::IDLE_WITH_SCANNING: {
            // Check if user became active
            if (hasRecentActivity()) {
                logger.info("User activity detected during scanning, returning to active state");
                transitionTo(State::ACTIVELY_PRESENT);
                last_activity_ = now;
                scan_failures_ = 0;
                consecutive_shutter_closed_scans_ = 0;
                break;
            }
            
            // Scan for face with recognition
            logger.info("Scanning for authorized user... (failures so far: " + 
                       std::to_string(scan_failures_) + ")");
            bool authorized = detectAndRecognizeFace();
            
            // Check if failure was due to closed shutter
            if (!authorized && last_shutter_state_ == ShutterState::CLOSED) {
                consecutive_shutter_closed_scans_++;
                logger.warning("Camera shutter is CLOSED (scan #" + 
                              std::to_string(consecutive_shutter_closed_scans_) + 
                              ") - pausing presence detection");
                
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
            
            if (authorized) {
                // Authorized user detected!
                successful_detections_++;
                logger.info("Authorized user detected! Returning to active state");
                transitionTo(State::ACTIVELY_PRESENT);
                last_activity_ = now;
                scan_failures_ = 0;
            } else {
                // No authorized user detected
                failed_detections_++;
                scan_failures_++;
                logger.info("No authorized user detected (failure " + 
                           std::to_string(scan_failures_) + " of " + 
                           std::to_string(max_scan_failures_) + ")");
                
                if (scan_failures_ >= max_scan_failures_) {
                    logger.info("User confirmed away after " + 
                               std::to_string(scan_failures_) + 
                               " failed scans - locking screen");
                    transitionTo(State::AWAY_CONFIRMED);
                    lockScreen();
                }
            }
            
            // Also check timeout
            if (inactive_time > max_idle_time_ms_) {
                logger.info("User confirmed away after " + 
                           std::to_string(inactive_time / 1000) + 
                           " seconds idle - locking screen");
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
    std::string new_state_str;
    switch (new_state) {
        case State::ACTIVELY_PRESENT: new_state_str = "ACTIVELY_PRESENT"; break;
        case State::IDLE_WITH_SCANNING: new_state_str = "IDLE_WITH_SCANNING"; break;
        case State::AWAY_CONFIRMED: new_state_str = "AWAY_CONFIRMED"; break;
    }
    
    logger.info("State transition: " + getStateString() + " -> " + new_state_str);
    
    current_state_ = new_state;
    state_entry_time_ = std::chrono::steady_clock::now();
}

bool PresenceDetector::ensureDetectorInitialized() {
    if (!face_detector_) {
        face_detector_ = std::make_unique<FaceDetector>();
        if (!face_detector_->loadModels()) {
            Logger::getInstance().error("Failed to load face detection models");
            return false;
        }
        Logger::getInstance().info("Face detector initialized (lazy load)");
    }
    return true;
}

// CRITICAL SECURITY FUNCTION: Detect AND Recognize face
bool PresenceDetector::detectAndRecognizeFace() {
    try {
        total_scans_++;
        
        // Ensure detector is initialized
        if (!ensureDetectorInitialized()) {
            return false;
        }
        
        // Step 1: Capture frame and close camera IMMEDIATELY
        Image frame = captureFrameAndClose();
        if (frame.empty()) {
            return false;
        }
        
        // Step 2: Check if camera shutter is closed
        ShutterState shutter = detectShutterState(frame.view());
        if (shutter == ShutterState::CLOSED) {
            Logger::getInstance().info("Camera shutter closed, skipping face detection");
            last_shutter_state_ = ShutterState::CLOSED;
            return false;
        }
        
        if (shutter == ShutterState::UNCERTAIN) {
            Logger::getInstance().debug("Camera image is very dark - shutter might be closed");
        }
        last_shutter_state_ = shutter;
        
        // Step 3: Detect faces using cascade
        Image bgr_frame = std::move(frame);
        if (bgr_frame.channels() != 3) {
            bgr_frame = convertGrayToBGRLibyuv(bgr_frame.view());
        }
        
        auto cascade_result = face_detector_->detectFacesCascade(bgr_frame.view(), false);
        
        if (cascade_result.faces.empty()) {
            Logger::getInstance().debug("No faces detected");
            return false; // No face detected
        }
        
        Logger::getInstance().debug("Face(s) detected (stage " + 
                                   std::to_string(cascade_result.stage_used) + 
                                   "), checking authorization...");
        
        // Step 4: Recognize faces (SECURITY: required!)
        if (!recognition_required_) {
            Logger::getInstance().warning("Recognition disabled - accepting any face (INSECURE!)");
            // Check for peek if enabled
            if (no_peek_enabled_) {
                checkForPeek(cascade_result);
            }
            return true;
        }
        
        // Encode faces
        auto encodings = face_detector_->encodeFaces(
            cascade_result.processed_frame.view(),
            cascade_result.faces
        );
        
        if (encodings.empty()) {
            Logger::getInstance().error("Failed to encode detected faces");
            return false;
        }
        
        // Step 5: Match against enrolled users
        bool authorized = false;
        for (const auto& encoding : encodings) {
            if (matchAgainstEnrolledUsers(encoding)) {
                authorized = true;
                break;
            }
        }
        
        if (!authorized) {
            // SECURITY: Unknown person detected!
            Logger::getInstance().warning("SECURITY ALERT: Unauthorized person detected!");
            lockScreen(); // IMMEDIATE LOCK!
            return false;
        }
        
        // Step 6: Authorized user - check for peek if enabled
        if (no_peek_enabled_) {
            checkForPeek(cascade_result);
        }
        
        return true; // Authorized user present
        
    } catch (const std::exception& e) {
        Logger::getInstance().error(std::string("Face detection/recognition error: ") + e.what());
        return false;
    }
}

// NEW: Capture frame and close camera immediately (privacy fix!)
Image PresenceDetector::captureFrameAndClose() {
    Logger& logger = Logger::getInstance();
    
    Camera camera(camera_device_);
    
    if (!camera.open(presence_camera_width_, presence_camera_height_)) {
        logger.error("Failed to open camera: " + camera_device_);
        return Image();
    }
    
    // Warm-up for IR cameras (5 frames to stabilize illuminator)
    Image warmup;
    for (int i = 0; i < 5; i++) {
        if (!camera.read(warmup)) {
            logger.warning("Warm-up frame " + std::to_string(i + 1) + " failed");
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Capture actual frame
    Image frame;
    if (!camera.read(frame)) {
        logger.error("Failed to capture frame");
        camera.close();
        return Image();
    }
    
    // CRITICAL: Close camera immediately! (Privacy + fixes "camera on too long")
    camera.close();
    
    return frame; // Return by move
}

// Face recognition: Match against enrolled users
bool PresenceDetector::matchAgainstEnrolledUsers(const FaceEncoding& encoding) {
    Logger& logger = Logger::getInstance();
    
    // Load enrolled users only once at first use (not periodically)
    // MEMORY FIX: Avoid reloading large datasets repeatedly
    if (enrolled_encodings_.empty()) {
        auto now = std::chrono::steady_clock::now();
        
        // Load all enrolled users
        auto& cache = ModelCache::getInstance();
        auto all_users = cache.loadAllUsersParallel(4);
        
        enrolled_encodings_.clear();
        enrolled_encodings_.reserve(all_users.size() * 5); // Pre-allocate to avoid reallocation
        
        for (const auto& user : all_users) {
            // Flatten all encodings from all poses
            for (const auto& pose_encodings : user.sample_encodings) {
                enrolled_encodings_.insert(
                    enrolled_encodings_.end(),
                    pose_encodings.begin(),
                    pose_encodings.end()
                );
            }
        }
        
        last_enrollment_load_ = now;
        logger.info("Loaded " + std::to_string(enrolled_encodings_.size()) + 
                   " encodings from " + std::to_string(all_users.size()) + " users (cached for session)");
    }
    
    if (enrolled_encodings_.empty()) {
        logger.warning("No enrolled users found - cannot verify authorization!");
        return false;
    }
    
    // Match against enrolled encodings
    // Load threshold from config
    auto& config = Config::instance();
    double threshold = config.recognition.threshold;
    
    for (const auto& enrolled : enrolled_encodings_) {
        double distance = face_detector_->compareFaces(encoding, enrolled);
        if (distance < threshold) {
            logger.debug("Match found! Distance: " + std::to_string(distance));
            return true; // Match found!
        }
    }
    
    logger.debug("No match found (best distance above threshold)");
    return false; // No match
}

// Enhanced peek detection with notifications
void PresenceDetector::checkForPeek(const FaceDetector::CascadeResult& cascade_result) {
    if (!peek_detector_) {
        return;
    }
    
    PeekAnalysis peek = peek_detector_->analyze(
        cascade_result.processed_frame.view(),
        cascade_result.faces
    );
    
    Logger& logger = Logger::getInstance();
    
    if (peek.watchers > 0) {
        // Someone is looking at screen!
        logger.warning("PEEK DETECTED: " + std::to_string(peek.watchers) + 
                      " person(s) watching screen (passing by: " + 
                      std::to_string(peek.passing_by) + ")");
        
        // Send notification
        if (peek_show_notification_) {
            std::string msg = std::to_string(peek.watchers) + 
                             " person(s) watching your screen!";
            NotificationHelper::sendSecurityAlert(msg);
        }
        
        // Optional actions (configurable)
        if (peek_blur_screen_ || peek_blank_screen_) {
            blankScreen();
        }
        
        updatePeekState(true);
    } else {
        updatePeekState(false);
    }
}

void PresenceDetector::updatePeekState(bool peek_detected) {
    if (!no_peek_enabled_) {
        return;
    }
    
    Logger& logger = Logger::getInstance();
    auto now = std::chrono::steady_clock::now();
    
    if (peek_detected) {
        if (peek_state_ == PeekState::NO_PEEK) {
            peek_state_ = PeekState::PEEK_DETECTED;
            peek_first_detected_ = now;
            consecutive_peek_detections_ = 1;
            logger.info("Peek DETECTED (first time)");
        } else {
            consecutive_peek_detections_++;
            
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - peek_first_detected_).count();
            
            if (duration_ms >= peek_detection_delay_ms_ && 
                peek_state_ != PeekState::PEEK_CONFIRMED) {
                peek_state_ = PeekState::PEEK_CONFIRMED;
                logger.warning("Peek CONFIRMED - screen actions triggered");
            }
        }
        
        peek_last_seen_ = now;
    } else {
        if (peek_state_ != PeekState::NO_PEEK) {
            auto time_since_last_peek_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - peek_last_seen_).count();
            
            if (time_since_last_peek_ms >= unblank_delay_ms_) {
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
        return;
    }
    
    Logger& logger = Logger::getInstance();
    logger.warning("BLANKING screen due to peek detection");
    
    // Method 1: loginctl (works on Wayland/X11)
    int ret = system("loginctl lock-sessions 2>/dev/null");
    if (ret == 0) {
        screen_blanked_ = true;
        logger.info("Screen blanked via loginctl");
        return;
    }
    
    // Method 2: KDE D-Bus (Wayland-specific)
    if (is_wayland_) {
        ret = system("qdbus org.freedesktop.ScreenSaver /ScreenSaver Lock 2>/dev/null");
        if (ret == 0) {
            screen_blanked_ = true;
            logger.info("Screen blanked via KDE D-Bus");
            return;
        }
    }
    
    logger.error("Failed to blank screen");
}

void PresenceDetector::unblankScreen() {
    if (!screen_blanked_) {
        return;
    }
    
    Logger& logger = Logger::getInstance();
    logger.info("Unblanking screen");
    
    // Note: Screen unlocking typically requires user interaction
    // We can only trigger wake, not unlock
    screen_blanked_ = false;
}

// NEW: Simplified with InputMonitor (event-driven, not polling!)
bool PresenceDetector::hasRecentActivity() const {
    if (!input_monitor_ || !input_monitor_->isRunning()) {
        return false;
    }
    
    auto last_activity = input_monitor_->getLastActivity();
    auto now = std::chrono::steady_clock::now();
    auto idle_time = std::chrono::duration_cast<std::chrono::seconds>(
        now - last_activity).count();
    
    return (idle_time < 5); // Activity within last 5 seconds
}

bool PresenceDetector::detectDisplayServer() {
    // Method 1: Check environment variables
    const char* wayland_display = getenv("WAYLAND_DISPLAY");
    if (wayland_display && strlen(wayland_display) > 0) {
        return true; // Wayland
    }
    
    const char* session_type = getenv("XDG_SESSION_TYPE");
    if (session_type && strcmp(session_type, "wayland") == 0) {
        return true; // Wayland
    }
    
    // Method 2: Check via loginctl
    FILE* pipe = popen("loginctl show-session $(loginctl list-sessions --no-legend | awk '{print $1}' | head -1) -p Type --value 2>/dev/null | head -1", "r");
    if (pipe) {
        char buffer[32];
        bool wayland = false;
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            std::string type(buffer);
            type.erase(type.find_last_not_of(" \n\r\t") + 1);
            if (type == "wayland") {
                wayland = true;
            }
        }
        pclose(pipe);
        if (wayland) {
            return true;
        }
    }
    
    return false; // X11 or unknown
}

bool PresenceDetector::isWithinSchedule() const {
    if (!schedule_enabled_) {
        return true;
    }
    
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
        return false;
    }
    
    // Check if current time is within active range
    return (current_time >= schedule_time_start_ && 
            current_time <= schedule_time_end_);
}

void PresenceDetector::lockScreen() {
    Logger& logger = Logger::getInstance();
    logger.info("Attempting to lock screen...");
    
    // Method 1: loginctl (most reliable)
    int ret = system("loginctl lock-sessions 2>/dev/null");
    if (ret == 0) {
        logger.info("Screen locked via loginctl lock-sessions");
        return;
    }
    
    // Method 2: Lock specific session
    ret = system("loginctl lock-session $(loginctl list-sessions --no-legend | awk '{print $1}' | head -1) 2>/dev/null");
    if (ret == 0) {
        logger.info("Screen locked via loginctl lock-session");
        return;
    }
    
    // Method 3: KDE-specific
    if (is_wayland_) {
        ret = system("qdbus org.freedesktop.ScreenSaver /ScreenSaver Lock 2>/dev/null");
        if (ret == 0) {
            logger.info("Screen locked via KDE D-Bus");
            return;
        }
    }
    
    logger.error("Failed to lock screen - all methods failed");
}

PresenceDetector::ShutterState PresenceDetector::detectShutterState(const ImageView& frame) {
    if (frame.empty()) {
        return ShutterState::UNCERTAIN;
    }
    
    // Calculate mean brightness
    double sum = 0.0;
    int pixel_count = 0;
    const uint8_t* data = frame.data();
    int stride = frame.stride();
    int channels = frame.channels();
    
    for (int y = 0; y < frame.height(); y++) {
        const uint8_t* row = data + y * stride;
        for (int x = 0; x < frame.width(); x++) {
            for (int c = 0; c < channels; c++) {
                sum += row[x * channels + c];
            }
            pixel_count += channels;
        }
    }
    
    double brightness = (pixel_count > 0) ? (sum / pixel_count) : 0.0;
    
    // Calculate standard deviation
    double variance_sum = 0.0;
    for (int y = 0; y < frame.height(); y++) {
        const uint8_t* row = data + y * stride;
        for (int x = 0; x < frame.width(); x++) {
            for (int c = 0; c < channels; c++) {
                double pixel_val = row[x * channels + c];
                double diff = pixel_val - brightness;
                variance_sum += diff * diff;
            }
        }
    }
    
    double stddev = (pixel_count > 0) ? std::sqrt(variance_sum / pixel_count) : 0.0;
    
    // Check if image is pure black (shutter closed)
    if (brightness < shutter_brightness_threshold_ && 
        stddev < shutter_variance_threshold_) {
        return ShutterState::CLOSED;
    }
    
    // Check if very dark
    if (brightness < 15.0) {
        return ShutterState::UNCERTAIN;
    }
    
    return ShutterState::OPEN;
}

void PresenceDetector::setGazeThresholds(float yaw, float pitch) {
    gaze_yaw_threshold_ = yaw;
    gaze_pitch_threshold_ = pitch;
    if (peek_detector_) {
        peek_detector_->setGazeThresholds(yaw, pitch);
    }
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
