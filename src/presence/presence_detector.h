#ifndef FACEID_PRESENCE_DETECTOR_H
#define FACEID_PRESENCE_DETECTOR_H

#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>   // OpenCL support
#include "presence_guard.h"

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
    void setMouseJitterThreshold(int ms) { mouse_jitter_threshold_ms_ = ms; }
    void setShutterBrightnessThreshold(double threshold) { shutter_brightness_threshold_ = threshold; }
    void setShutterVarianceThreshold(double threshold) { shutter_variance_threshold_ = threshold; }
    void setShutterTimeout(int ms) { shutter_timeout_ms_ = ms; }
    
    // No-peek configuration
    void enableNoPeek(bool enable) { no_peek_enabled_ = enable; }
    void setMinFaceDistance(int pixels) { min_face_distance_pixels_ = pixels; }
    void setMinFaceSizePercent(double percent) { min_face_size_percent_ = percent; }
    void setPeekDetectionDelay(int ms) { peek_detection_delay_ms_ = ms; }
    void setUnblankDelay(int ms) { unblank_delay_ms_ = ms; }
    
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
    
    // State machine updates
    void updateStateMachine();
    void transitionTo(State new_state);
    
    // Face detection
    bool detectFace();
    cv::Mat captureFrame();
    bool ensureDetectorInitialized();  // Lazy load YuNet detector
    
    // Camera shutter detection
    ShutterState detectShutterState(const cv::Mat& frame);
    
    // No-peek detection
    bool detectPeek(const cv::Mat& frame);
    void updatePeekState(bool peek_detected);
    void blankScreen();
    void unblankScreen();
    
    // Input activity monitoring (NEW implementation)
    bool hasRecentActivity() const;
    time_t getLastInputDeviceActivity() const;
    
    // Schedule checking
    bool isWithinSchedule() const;
    
    // Lock screen trigger
    void lockScreen();
    
    // Legacy methods (keeping for compatibility)
    std::chrono::steady_clock::time_point getLastInputActivity() const;
    bool detectDisplayServer();  // Detect X11 vs Wayland
    
    // Camera
    std::string camera_device_;
    std::unique_ptr<cv::VideoCapture> camera_;
    std::mutex camera_mutex_;
    cv::Mat last_captured_frame_;  // Cache for peek detection (avoids reopening camera)
    
    // Detection (YuNet)
    // LibFaceDetection has embedded models - no detector instance needed
    // No detector initialization needed - LibFaceDetection has embedded models
    
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
    
    // NEW: Mouse jitter filtering
    int mouse_jitter_threshold_ms_ = 300;  // Ignore mouse within 300ms
    mutable bool last_device_was_mouse_ = false;
    mutable std::chrono::steady_clock::time_point last_mouse_activity_time_;
    
    // NEW: Camera shutter detection
    double shutter_brightness_threshold_ = 10.0;   // Max brightness for "closed"
    double shutter_variance_threshold_ = 2.0;      // Max stddev for "closed"
    int shutter_timeout_ms_ = 300000;              // 5 minutes
    int consecutive_shutter_closed_scans_ = 0;
    ShutterState last_shutter_state_ = ShutterState::OPEN;
    
    // Display server detection (set once at startup)
    bool is_wayland_ = false;
    mutable unsigned long last_interrupt_count_ = 0;  // For /proc/interrupts method
    mutable std::chrono::steady_clock::time_point last_interrupt_check_;  // Cache timestamp
    mutable std::chrono::milliseconds interrupt_check_interval_{500};  // Check every 500ms
    
    // Activity detection caching (to reduce /proc/interrupts reads)
    mutable time_t cached_last_activity_ = 0;
    mutable std::chrono::steady_clock::time_point last_activity_check_;
    mutable std::chrono::seconds activity_cache_duration_{2};  // Cache for 2 seconds
    
    // Statistics
    std::atomic<int> total_scans_{0};
    std::atomic<int> successful_detections_{0};
    std::atomic<int> failed_detections_{0};
    std::atomic<int> state_transitions_{0};
    
    // No-peek detection
    bool no_peek_enabled_ = false;
    int min_face_distance_pixels_ = 80;
    double min_face_size_percent_ = 0.08;
    int peek_detection_delay_ms_ = 2000;
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
