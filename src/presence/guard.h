#ifndef FACEID_PRESENCE_GUARD_H
#define FACEID_PRESENCE_GUARD_H

#include <chrono>
#include <string>
#include <mutex>

namespace faceid {

class PresenceGuard {
public:
    PresenceGuard();
    ~PresenceGuard() = default;
    
    // Update all guard conditions
    void updateState();
    
    // Check if all conditions are met to run presence detection
    bool shouldRunPresenceDetection() const;
    bool checkGuardConditions() { updateState(); return shouldRunPresenceDetection(); }
    
    // Individual guard checks
    bool isLidOpen() const { return lid_open_; }
    bool isCameraShutterOpen() const { return camera_shutter_open_; }
    bool isScreenUnlocked() const { return screen_unlocked_; }
    
    // Get last update time
    std::chrono::steady_clock::time_point getLastUpdate() const { 
        return last_update_; 
    }
    
    // Get reasons for guard failure (for logging)
    std::string getFailureReason() const;

private:
    bool lid_open_;
    bool camera_shutter_open_;
    bool screen_unlocked_;
    std::chrono::steady_clock::time_point last_update_;
    
    // Cache for screen lock check (avoid calling D-Bus constantly)
    std::string cached_session_id_;
    std::chrono::steady_clock::time_point last_session_check_;
    
    // Cache the actual lock state result to avoid excessive D-Bus calls
    bool cached_lock_state_;
    std::chrono::steady_clock::time_point last_lock_state_check_;
    static constexpr int LOCK_STATE_CACHE_SECONDS = 2;  // Check lock state every 2 seconds max
    
    // Mutex to protect cache access from multiple threads
    mutable std::mutex cache_mutex_;
    
    // Individual check implementations
    bool checkLidState();
    bool checkCameraShutter();
    bool checkScreenLock();
};

} // namespace faceid

#endif // FACEID_PRESENCE_GUARD_H
