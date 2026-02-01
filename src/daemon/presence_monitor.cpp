#include "presence_monitor.h"
#include "config.h"
#include "face_detector_manager.h"
#include <syslog.h>
#include <thread>
#include <chrono>
#include <ctime>
#include <cerrno>
#include <systemd/sd-bus.h>

PresenceMonitor& PresenceMonitor::instance() {
    static PresenceMonitor instance;
    return instance;
}

bool PresenceMonitor::start_monitoring() {
    if (monitoring_active_.exchange(true)) {
        syslog(LOG_WARNING, "Presence monitoring already active");
        return false;
    }
    
    load_config();
    
    if (!enabled_) {
        syslog(LOG_INFO, "Presence detection disabled in config");
        monitoring_active_ = false;
        return false;
    }
    
    should_stop_ = false;
    
    // Start monitoring thread
    try {
        monitor_thread_ = std::make_unique<std::thread>(&PresenceMonitor::monitor_loop, this);
        syslog(LOG_INFO, "✓ Presence monitoring started (lock after %d seconds)", 
               lock_after_seconds_);
        return true;
    } catch (const std::exception& e) {
        syslog(LOG_ERR, "Failed to start presence monitor: %s", e.what());
        monitoring_active_ = false;
        return false;
    }
}

void PresenceMonitor::stop_monitoring() {
    if (!monitoring_active_.exchange(false)) {
        return;  // Not active
    }
    
    should_stop_ = true;
    
    if (monitor_thread_ && monitor_thread_->joinable()) {
        monitor_thread_->join();
    }
    
    syslog(LOG_INFO, "Presence monitoring stopped");
}

void PresenceMonitor::load_config() {
    const Config& config = Config::instance();
    
    // Load presence detection settings
    enabled_ = config.presence.enabled;
    lock_after_seconds_ = config.presence.lock_after;
    scan_interval_ms_ = config.presence.scan_interval * 100;  // Convert to milliseconds
    
    syslog(LOG_DEBUG, "Presence config: enabled=%d, lock_after=%ds, interval=%dms",
           enabled_, lock_after_seconds_, scan_interval_ms_);
}

void PresenceMonitor::monitor_loop() {
    syslog(LOG_DEBUG, "Presence monitor thread started");
    
    std::time_t absence_start = 0;
    
    while (!should_stop_) {
        // Scan for face
        bool face_detected = detect_face();
        
        if (face_detected) {
            last_detection_time_ = std::time(nullptr);
            
            if (!user_present_.exchange(true)) {
                // Transition: absent → present
                log_presence_change(true);
            }
            absence_start = 0;  // Reset absence timer
        } else {
            if (user_present_.exchange(false)) {
                // Transition: present → absent
                log_presence_change(false);
                absence_start = std::time(nullptr);
            } else if (absence_start > 0) {
                // Check if absence timeout exceeded
                std::time_t now = std::time(nullptr);
                int absence_time = static_cast<int>(now - absence_start);
                
                if (absence_time >= lock_after_seconds_) {
                    syslog(LOG_INFO, "Absence timeout reached (%ds), locking session", 
                           absence_time);
                    if (!lock_session()) {
                        syslog(LOG_ERR, "Failed to lock session");
                    }
                    absence_start = 0;  // Reset to avoid repeated lock attempts
                }
            }
        }
        
        // Sleep before next scan
        std::this_thread::sleep_for(std::chrono::milliseconds(scan_interval_ms_));
    }
    
    syslog(LOG_DEBUG, "Presence monitor thread exiting");
}

bool PresenceMonitor::detect_face() {
    // Use face detector to check for presence
    FaceDetectorManager& face_mgr = FaceDetectorManager::instance();
    
    if (!face_mgr.is_ready()) {
        return false;
    }
    
    // Quick check: just detect face without verifying identity
    // Return true if any face detected (confidence > 0.5)
    // This is a placeholder - real implementation would:
    // 1. Get frame from camera
    // 2. Run YuNet detection
    // 3. Return if any face found (no identification needed)
    
    // For now, return false (no face detection without ML models)
    return false;
}

bool PresenceMonitor::lock_session() {
    // Use systemd-logind to lock session
    sd_bus* bus = nullptr;
    sd_bus_error error = SD_BUS_ERROR_NULL;
    
    int ret = sd_bus_open_user(&bus);
    if (ret < 0) {
        syslog(LOG_ERR, "Failed to connect to D-Bus: %s", strerror(-ret));
        return false;
    }
    
    // Call org.freedesktop.login1.Session.Lock() method
    ret = sd_bus_call_method(
        bus,
        "org.freedesktop.login1",
        "/org/freedesktop/login1/session/self",
        "org.freedesktop.login1.Session",
        "Lock",
        &error,
        nullptr,  // No reply needed
        "");      // No parameters
    
    sd_bus_unref(bus);
    
    if (ret < 0) {
        syslog(LOG_WARNING, "Failed to lock session: %s", error.message);
        sd_bus_error_free(&error);
        return false;
    }
    
    sd_bus_error_free(&error);
    syslog(LOG_INFO, "Session locked via systemd-logind");
    return true;
}

void PresenceMonitor::log_presence_change(bool present) {
    if (present) {
        syslog(LOG_INFO, "Face detected - user present");
    } else {
        syslog(LOG_INFO, "Face lost - user absent (will lock in %d seconds)", 
               lock_after_seconds_);
    }
}

bool PresenceMonitor::force_lock() {
    syslog(LOG_WARNING, "Force locking session (debug/testing)");
    return lock_session();
}

PresenceMonitor::~PresenceMonitor() {
    stop_monitoring();
}
