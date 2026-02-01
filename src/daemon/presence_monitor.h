#pragma once

#include <memory>
#include <string>
#include <atomic>
#include <thread>
#include <chrono>

/**
 * Presence Monitor - Continuous face detection for auto-lock
 * 
 * Monitors face presence and auto-locks the session when user walks away.
 * Integrates with systemd-logind to lock the session.
 * 
 * Features:
 * - Continuous background scanning
 * - Configurable lock timeout
 * - Thread-safe state management
 * - Auto-unlock on face re-detection
 */
class PresenceMonitor {
public:
    static PresenceMonitor& instance();
    
    /**
     * Start presence monitoring (async, runs in background thread)
     */
    bool start_monitoring();
    
    /**
     * Stop presence monitoring
     */
    void stop_monitoring();
    
    /**
     * Check if monitoring is active
     */
    bool is_monitoring() const { return monitoring_active_; }
    
    /**
     * Get last detection time (seconds since epoch, or 0 if not detected)
     */
    std::time_t get_last_detection() const { return last_detection_time_; }
    
    /**
     * Force immediate lock (for testing/debugging)
     */
    bool force_lock();
    
    /**
     * Get current presence state
     */
    bool is_user_present() const { return user_present_; }
    
    ~PresenceMonitor();
    
private:
    PresenceMonitor() = default;
    
    std::atomic<bool> monitoring_active_{false};
    std::atomic<bool> should_stop_{false};
    std::atomic<bool> user_present_{false};
    std::atomic<std::time_t> last_detection_time_{0};
    
    std::unique_ptr<std::thread> monitor_thread_;
    
    // Configuration (from Config)
    bool enabled_ = false;
    int lock_after_seconds_ = 30;
    int scan_interval_ms_ = 500;
    
    // Monitor loop (runs in background thread)
    void monitor_loop();
    
    // Load configuration
    void load_config();
    
    // Perform face detection check
    bool detect_face();
    
    // Lock the session via systemd-logind
    bool lock_session();
    
    // Log presence change
    void log_presence_change(bool present);
};
