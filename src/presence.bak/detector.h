#ifndef FACEID_PRESENCE_DETECTOR_H
#define FACEID_PRESENCE_DETECTOR_H

#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <memory>
#include "guard.h"
#include "input_monitor.h"
#include "peek_detector.h"
#include "notification.h"
#include "../camera.h"
#include "../image.h"
#include "../face_detector.h"

namespace faceid {

class PresenceDetector {
public:
    enum class State {
        ACTIVELY_PRESENT,   // User is typing/clicking, no scanning needed
        IDLE_WITH_SCANNING, // User inactive, scanning for face
        AWAY_CONFIRMED      // User confirmed away, screen locked
    };
    
    enum class ShutterState {
        OPEN,       // Camera shutter is open
        CLOSED,     // Camera shutter is closed (physical privacy)
        UNCERTAIN   // Very dark image, might be closed or dark room
    };
    
    enum class PeekState {
        NO_PEEK,            // Only authorized user visible
        PEEK_DETECTED,      // Additional face detected (shoulder surfing)
        PEEK_CONFIRMED      // Peek persisted for configured delay
    };
    
    PresenceDetector(
        const std::string& camera_device = "/dev/video0",
        std::chrono::seconds inactive_threshold = std::chrono::seconds(30),
        std::chrono::seconds scan_interval = std::chrono::seconds(2),
        int max_scan_failures = 3,
        std::chrono::minutes max_idle_time = std::chrono::minutes(15)
    );
    ~PresenceDetector();
    
    // Initialize detector
    bool initialize();
    
    // Start/stop detection service
    bool start();
    void stop();
    bool isRunning() const { return running_.load(); }
    
    // Query current state
    bool isUserPresent() const;
    State getCurrentState() const { return current_state_; }
    std::string getStateString() const;
    
    // Activity notification (from input monitoring)
    void notifyActivity();
    
    // Pause/resume for authentication (called by PAM)
    void pauseForAuthentication();
    void resumeAfterAuthentication();
    
    // Configuration
    void setInactiveThreshold(int ms) { inactive_threshold_ms_ = ms; }
    void setScanInterval(int ms) { scan_interval_ms_ = ms; }
    void setMaxScanFailures(int count) { max_scan_failures_ = count; }
    void setMaxIdleTime(int ms) { max_idle_time_ms_ = ms; }
    void setShutterBrightnessThreshold(double threshold) { shutter_brightness_threshold_ = threshold; }
    void setShutterVarianceThreshold(double threshold) { shutter_variance_threshold_ = threshold; }
    void setShutterTimeout(int ms) { shutter_timeout_ms_ = ms; }
    
    // Presence camera resolution configuration
    void setPresenceCameraResolution(int width, int height) { 
        presence_camera_width_ = width; 
        presence_camera_height_ = height; 
    }
    
    // No-peek configuration
    void enableNoPeek(bool enable) { no_peek_enabled_ = enable; }
    void setGazeDetectionEnabled(bool enable) { gaze_detection_enabled_ = enable; }
    void setGazeThresholds(float yaw, float pitch);
    void setMinFaceDistance(int pixels) { min_face_distance_pixels_ = pixels; }
    void setMinFaceSizePercent(double percent) { min_face_size_percent_ = percent; }
    void setPeekDetectionDelay(int ms) { peek_detection_delay_ms_ = ms; }
    void setPeekShowNotification(bool enable) { peek_show_notification_ = enable; }
    void setPeekBlurScreen(bool enable) { peek_blur_screen_ = enable; }
    void setPeekBlankScreen(bool enable) { peek_blank_screen_ = enable; }
    void setUnblankDelay(int ms) { unblank_delay_ms_ = ms; }
    
    // Face recognition configuration
    void setRecognitionRequired(bool required) { recognition_required_ = required; }
    
    // Schedule configuration
    void enableSchedule(bool enable) { schedule_enabled_ = enable; }
    void setActiveDays(const std::vector<int>& days) { active_days_ = days; }
    void setActiveTimeRange(int start_hhmm, int end_hhmm) { 
        schedule_time_start_ = start_hhmm; 
        schedule_time_end_ = end_hhmm; 
    }
    
    // Query peek state
    PeekState getPeekState() const { return peek_state_; }
    bool isScreenBlanked() const { return screen_blanked_; }
    
    // Statistics
    struct Statistics {
        int totalScans;
        int facesDetected;
        int failedScans;
        int stateTransitions;
        int uptimeSeconds;
    };
    
    Statistics getStatistics() const;
    int getTotalScans() const { return total_scans_.load(); }
    int getSuccessfulDetections() const { return successful_detections_.load(); }
    int getFailedDetections() const { return failed_detections_.load(); }
    
private:
    // Main detection thread
    void detectionLoop();
    bool checkPAMLockFile();  // Check if PAM authentication is in progress
    
    // State machine updates
    void updateStateMachine();
    void transitionTo(State new_state);
    
    // Face detection and recognition (NEW: Security-critical!)
    bool detectAndRecognizeFace();
    Image captureFrameAndClose();  // NEW: Immediate camera close
    bool ensureDetectorInitialized();  // Lazy load detector
    bool matchAgainstEnrolledUsers(const FaceEncoding& encoding);
    
    // Camera shutter detection
    ShutterState detectShutterState(const ImageView& frame);
    
    // No-peek detection (NEW: Enhanced with gaze)
    void checkForPeek(const FaceDetector::CascadeResult& cascade_result);
    void updatePeekState(bool peek_detected);
    void blankScreen();
    void unblankScreen();
    
    // Input activity monitoring (NEW: libevdev-based)
    bool hasRecentActivity() const;
    
    // Schedule checking
    bool isWithinSchedule() const;
    
    // Lock screen trigger
    void lockScreen();
    
    // Display server detection
    bool detectDisplayServer();
    
    // Camera
    std::string camera_device_;
    
    // NEW: InputMonitor for event-driven input detection
    std::unique_ptr<InputMonitor> input_monitor_;
    std::atomic<bool> has_recent_activity_{false};
    
    // Face detection with tracking support (lazy-loaded)
    std::unique_ptr<FaceDetector> face_detector_;
    
    // NEW: PeekDetector for gaze-aware peek detection
    std::unique_ptr<PeekDetector> peek_detector_;
    
    // Face recognition (NEW: Security critical!)
    bool recognition_required_ = true;  // Default: ALWAYS require recognition
    std::vector<FaceEncoding> enrolled_encodings_;  // Cached enrolled users
    std::chrono::steady_clock::time_point last_enrollment_load_;
    std::chrono::minutes enrollment_cache_duration_{10};  // Reload every 10 minutes
    
    // Guard conditions
    PresenceGuard guard_;
    
    // State machine
    State current_state_;
    std::chrono::steady_clock::time_point last_activity_;
    std::chrono::steady_clock::time_point state_entry_time_;
    std::chrono::steady_clock::time_point start_time_;
    int scan_failures_;
    
    // Thread control
    std::atomic<bool> running_{false};
    std::atomic<bool> paused_for_auth_{false};
    int pause_count_;
    std::mutex pause_mutex_;
    std::thread detection_thread_;
    
    // Configuration
    int inactive_threshold_ms_ = 30000;  // 30 seconds
    int scan_interval_ms_ = 2000;        // 2 seconds
    int max_scan_failures_ = 3;          // 3 consecutive failures
    int max_idle_time_ms_ = 900000;      // 15 minutes
    
    // Camera shutter detection
    double shutter_brightness_threshold_ = 10.0;   // Max brightness for "closed"
    double shutter_variance_threshold_ = 2.0;      // Max stddev for "closed"
    int shutter_timeout_ms_ = 300000;              // 5 minutes
    int consecutive_shutter_closed_scans_ = 0;
    ShutterState last_shutter_state_ = ShutterState::OPEN;
    
    // Presence camera resolution (configurable)
    int presence_camera_width_ = 640;
    int presence_camera_height_ = 480;
    
    // Display server detection (set once at startup)
    bool is_wayland_ = false;
    
    // Statistics
    std::atomic<int> total_scans_{0};
    std::atomic<int> successful_detections_{0};
    std::atomic<int> failed_detections_{0};
    std::atomic<int> state_transitions_{0};
    
    // No-peek detection (NEW: Enhanced with gaze)
    bool no_peek_enabled_ = false;
    bool gaze_detection_enabled_ = true;  // NEW: Use gaze direction
    float gaze_yaw_threshold_ = 30.0f;    // NEW: degrees
    float gaze_pitch_threshold_ = 30.0f;  // NEW: degrees
    int min_face_distance_pixels_ = 80;
    double min_face_size_percent_ = 0.08;
    int peek_detection_delay_ms_ = 2000;
    bool peek_show_notification_ = true;   // NEW: Send desktop notification
    bool peek_blur_screen_ = false;        // NEW: Optional blur
    bool peek_blank_screen_ = false;       // NEW: Optional blank
    int unblank_delay_ms_ = 3000;
    
    PeekState peek_state_ = PeekState::NO_PEEK;
    bool screen_blanked_ = false;
    std::chrono::steady_clock::time_point peek_first_detected_;
    std::chrono::steady_clock::time_point peek_last_seen_;
    int consecutive_peek_detections_ = 0;
    
    // Schedule configuration
    bool schedule_enabled_ = false;
    std::vector<int> active_days_;  // 1=Monday, 7=Sunday
    int schedule_time_start_ = 0;   // HHMM format (e.g., 900 = 9:00 AM)
    int schedule_time_end_ = 2359;  // HHMM format (e.g., 1700 = 5:00 PM)
};

} // namespace faceid

#endif // FACEID_PRESENCE_DETECTOR_H
