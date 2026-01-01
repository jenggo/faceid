#include "guard.h"
#include "../systemd_helper.h"
#include "../lid_detector.h"
#include <unistd.h>
#include <mutex>

namespace faceid {

PresenceGuard::PresenceGuard()
    : lid_open_(false)
    , camera_shutter_open_(false)
    , screen_unlocked_(false)
    , last_update_(std::chrono::steady_clock::time_point::min())
    , last_session_check_(std::chrono::steady_clock::time_point::min())
    , cached_lock_state_(true)  // Assume unlocked initially
    , last_lock_state_check_(std::chrono::steady_clock::time_point::min()) {
}

void PresenceGuard::updateState() {
    lid_open_ = checkLidState();
    camera_shutter_open_ = checkCameraShutter();
    screen_unlocked_ = checkScreenLock();
    last_update_ = std::chrono::steady_clock::now();
}

bool PresenceGuard::shouldRunPresenceDetection() const {
    return lid_open_ && camera_shutter_open_ && screen_unlocked_;
}

std::string PresenceGuard::getFailureReason() const {
    if (!lid_open_) return "lid_closed";
    if (!camera_shutter_open_) return "camera_shutter_closed";
    if (!screen_unlocked_) return "screen_locked";
    return "all_conditions_met";
}

bool PresenceGuard::checkLidState() {
    // Use existing LidDetector
    LidDetector detector;
    LidState state = detector.getLidState();
    return (state == LidState::OPEN);
}

bool PresenceGuard::checkCameraShutter() {
    // Check if camera device is accessible
    // This is a simple check - assumes if /dev/video0 exists and is readable,
    // the shutter is open
    
    // Try primary camera device
    if (access("/dev/video0", R_OK) == 0) {
        return true;
    }
    
    // Try secondary camera device
    if (access("/dev/video1", R_OK) == 0) {
        return true;
    }
    
    return false;
}

bool PresenceGuard::checkScreenLock() {
    // Lock mutex to protect cache access
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    // Return cached state if checked recently (every 2 seconds max)
    // This prevents querying D-Bus 10 times per second
    auto now = std::chrono::steady_clock::now();
    auto time_since_lock_check = std::chrono::duration_cast<std::chrono::seconds>(
        now - last_lock_state_check_).count();
    
    if (time_since_lock_check < LOCK_STATE_CACHE_SECONDS) {
        return cached_lock_state_;
    }
    
    // Time to refresh - check actual lock state
    bool is_unlocked = true;  // Default: assume unlocked
    
    // Cache session ID (only lookup every 30 seconds)
    auto time_since_session_check = std::chrono::duration_cast<std::chrono::seconds>(
        now - last_session_check_).count();
    
    if (cached_session_id_.empty() || time_since_session_check > 30) {
        auto session_id = SystemdHelper::getActiveSessionId();
        if (session_id.has_value()) {
            cached_session_id_ = *session_id;
        }
        last_session_check_ = now;
    }
    
    // Fast check: Use cached session ID with D-Bus (Wayland-compatible)
    if (!cached_session_id_.empty()) {
        if (SystemdHelper::isSessionLocked(cached_session_id_)) {
            // Session is locked
            is_unlocked = false;
            
            // Update cache and return
            cached_lock_state_ = is_unlocked;
            last_lock_state_check_ = now;
            return cached_lock_state_;
        } else {
            // Session is unlocked
            is_unlocked = true;
            
            // Update cache and return
            cached_lock_state_ = is_unlocked;
            last_lock_state_check_ = now;
            return cached_lock_state_;
        }
    }
    
    // Fallback: KDE-specific check (for older systems without loginctl LockedHint)
    // NOTE: kscreenlocker_daemon is ALWAYS running in KDE, even when unlocked!
    // We must check for kscreenlocker_greet which only runs when the lock screen is active
    if (SystemdHelper::isProcessRunning("kscreenlocker_greet")) {
        // kscreenlocker_greet is running = screen is locked
        is_unlocked = false;
    }
    
    // Update cache and return
    cached_lock_state_ = is_unlocked;
    last_lock_state_check_ = now;
    return cached_lock_state_;
}

} // namespace faceid
