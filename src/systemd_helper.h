#ifndef FACEID_SYSTEMD_HELPER_H
#define FACEID_SYSTEMD_HELPER_H

#include <string>
#include <optional>
#include <systemd/sd-bus.h>

namespace faceid {

// RAII wrapper for sd_bus
class SDBusWrapper {
public:
    SDBusWrapper();
    ~SDBusWrapper();
    
    // Delete copy operations
    SDBusWrapper(const SDBusWrapper&) = delete;
    SDBusWrapper& operator=(const SDBusWrapper&) = delete;
    
    sd_bus* get() const { return bus_; }
    bool isValid() const { return bus_ != nullptr; }
    
private:
    sd_bus* bus_;
};

// Helper class for systemd-logind and session queries
class SystemdHelper {
public:
    // Get the active session ID for the current user
    static std::optional<std::string> getActiveSessionId();
    
    // Check if a session is locked
    static bool isSessionLocked(const std::string& session_id);
    
    // Get lid closed state from systemd-logind
    static std::optional<bool> getLidClosed();
    
    // Check if a process is running by name (uses /proc instead of ps/pgrep)
    static bool isProcessRunning(const std::string& process_name);
    
    // Get current username (cached, uses getpwuid instead of whoami)
    static std::string getCurrentUsername();
    
    // Check if GNOME screensaver is active (uses D-Bus session bus)
    static bool isGnomeScreenSaverActive();
    
private:
    // D-Bus call helper
    static std::optional<std::string> getStringProperty(
        const char* destination,
        const char* path,
        const char* interface,
        const char* member
    );
    
    static std::optional<bool> getBoolProperty(
        const char* destination,
        const char* path,
        const char* interface,
        const char* member
    );
};

} // namespace faceid

#endif // FACEID_SYSTEMD_HELPER_H
