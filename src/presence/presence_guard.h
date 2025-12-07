#ifndef FACEID_PRESENCE_GUARD_H
#define FACEID_PRESENCE_GUARD_H

#include <chrono>
#include <string>

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
    
    // Cache for screen lock check (avoid spawning processes constantly)
    std::string cached_session_id_;
    std::chrono::steady_clock::time_point last_session_check_;
    
    // Individual check implementations
    bool checkLidState();
    bool checkCameraShutter();
    bool checkScreenLock();
};

} // namespace faceid

#endif // FACEID_PRESENCE_GUARD_H
