#ifndef FACEID_PRESENCE_DETECTOR_H
#define FACEID_PRESENCE_DETECTOR_H

#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>  // For FaceDetectorYN
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
    
    // Camera shutter detection
    ShutterState detectShutterState(const cv::Mat& frame);
    
    // Input activity monitoring (NEW implementation)
    bool hasRecentActivity() const;
    time_t getLastInputDeviceActivity() const;
    
    // Lock screen trigger
    void lockScreen();
    
    // Legacy methods (keeping for compatibility)
    std::chrono::steady_clock::time_point getLastInputActivity() const;
    bool detectDisplayServer();  // Detect X11 vs Wayland
    
    // Camera
    std::string camera_device_;
    std::unique_ptr<cv::VideoCapture> camera_;
    std::mutex camera_mutex_;
    
    // Detection (YuNet)
    cv::Ptr<cv::FaceDetectorYN> detector_;
    bool detector_initialized_;
    
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
    
    // Statistics
    std::atomic<int> total_scans_{0};
    std::atomic<int> successful_detections_{0};
    std::atomic<int> failed_detections_{0};
    std::atomic<int> state_transitions_{0};
};

} // namespace faceid

#endif // FACEID_PRESENCE_DETECTOR_H
