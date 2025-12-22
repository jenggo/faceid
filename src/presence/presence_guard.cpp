#include "presence_guard.h"
#include "../lid_detector.h"
#include <unistd.h>

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
    // Return cached state if checked recently (every 2 seconds max)
    // This prevents spawning loginctl processes 10 times per second
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
        FILE* pipe = popen("loginctl list-sessions --no-legend 2>/dev/null | awk '{print $1}' | head -1", "r");
        if (pipe) {
            char buffer[128];
            if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                cached_session_id_ = buffer;
                // Remove newline
                if (!cached_session_id_.empty() && cached_session_id_.back() == '\n') {
                    cached_session_id_.pop_back();
                }
            }
            pclose(pipe);  // Always close pipe
        }
        last_session_check_ = now;
    }
    
    // Fast check: Use cached session ID with loginctl (Wayland-compatible)
    if (!cached_session_id_.empty()) {
        std::string cmd = "loginctl show-session " + cached_session_id_ + 
                         " -p LockedHint --value 2>/dev/null";
        FILE* pipe = popen(cmd.c_str(), "r");
        if (pipe) {
            char buffer[16];
            if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                std::string result(buffer);
                // Remove whitespace/newline
                result.erase(result.find_last_not_of(" \n\r\t") + 1);
                if (result == "yes") {
                    // Session is locked
                    is_unlocked = false;
                } else {
                    // "no" or empty means unlocked
                    is_unlocked = true;
                }
                
                pclose(pipe);  // Close before return
                
                // Update cache and return
                cached_lock_state_ = is_unlocked;
                last_lock_state_check_ = now;
                return cached_lock_state_;
            }
            pclose(pipe);  // Close if fgets failed
        }
    }
    
    // Fallback: KDE-specific check (for older systems without loginctl LockedHint)
    // NOTE: kscreenlocker_daemon is ALWAYS running in KDE, even when unlocked!
    // We must check for kscreenlocker_greet which only runs when the lock screen is active
    FILE* pipe = popen("pgrep -x kscreenlocker_greet >/dev/null 2>&1", "r");
    if (pipe) {
        int status = pclose(pipe);
        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            // kscreenlocker_greet is running = screen is locked
            is_unlocked = false;
        }
    }
    
    // Update cache and return
    cached_lock_state_ = is_unlocked;
    last_lock_state_check_ = now;
    return cached_lock_state_;
}

} // namespace faceid
