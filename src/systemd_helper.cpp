#include "systemd_helper.h"
#include "logger.h"
#include <unistd.h>
#include <pwd.h>
#include <dirent.h>
#include <fstream>
#include <cstring>

namespace faceid {

// SDBusWrapper implementation
SDBusWrapper::SDBusWrapper() : bus_(nullptr) {
    int r = sd_bus_open_system(&bus_);
    if (r < 0) {
        // Use thread-safe error message construction
        char error_buf[256];
        strerror_r(-r, error_buf, sizeof(error_buf));
        Logger::getInstance().error("Failed to connect to system bus: " + std::string(error_buf));
        bus_ = nullptr;
    }
}

SDBusWrapper::~SDBusWrapper() {
    if (bus_) {
        sd_bus_unref(bus_);
    }
}

// SystemdHelper implementation
std::optional<std::string> SystemdHelper::getActiveSessionId() {
    SDBusWrapper bus;
    if (!bus.isValid()) {
        return std::nullopt;
    }
    
    sd_bus_error error = SD_BUS_ERROR_NULL;
    sd_bus_message* reply = nullptr;
    
    // First, try to get the active session for the current user's seat
    // This is more reliable than just picking the first session
    uid_t current_uid = getuid();
    
    // Call ListSessions to find sessions for current user
    int r = sd_bus_call_method(
        bus.get(),
        "org.freedesktop.login1",
        "/org/freedesktop/login1",
        "org.freedesktop.login1.Manager",
        "ListSessions",
        &error,
        &reply,
        ""
    );
    
    if (r < 0) {
        Logger::getInstance().debug("Failed to list sessions: " + std::string(error.message));
        sd_bus_error_free(&error);
        return std::nullopt;
    }
    
    // Parse the array of sessions and find the active one for our user
    r = sd_bus_message_enter_container(reply, 'a', "(susso)");
    if (r < 0) {
        sd_bus_message_unref(reply);
        return std::nullopt;
    }
    
    std::string session_id;
    std::string fallback_session_id;  // First matching session as fallback
    
    while (sd_bus_message_enter_container(reply, 'r', "susso") > 0) {
        const char* id = nullptr;
        uint32_t uid = 0;
        const char* name = nullptr;
        const char* seat = nullptr;
        const char* path = nullptr;
        
        r = sd_bus_message_read(reply, "susso", &id, &uid, &name, &seat, &path);
        if (r < 0) {
            Logger::getInstance().debug("Failed to parse session entry, stopping iteration");
            sd_bus_message_exit_container(reply);
            break;  // Break instead of continue - stream is corrupted
        }
        
        // Find sessions for our user
        if (uid == current_uid && id != nullptr && path != nullptr) {
            // Save first matching session as fallback
            if (fallback_session_id.empty()) {
                fallback_session_id = id;
            }
            
            // Check if this session is active
            sd_bus_error session_error = SD_BUS_ERROR_NULL;
            sd_bus_message* session_reply = nullptr;
            
            int session_r = sd_bus_get_property(
                bus.get(),
                "org.freedesktop.login1",
                path,
                "org.freedesktop.login1.Session",
                "Active",
                &session_error,
                &session_reply,
                "b"
            );
            
            if (session_r >= 0) {
                int active = 0;
                sd_bus_message_read(session_reply, "b", &active);
                sd_bus_message_unref(session_reply);
                
                if (active) {
                    // Found active session for our user
                    session_id = id;
                    sd_bus_message_exit_container(reply);
                    break;
                }
            } else {
                sd_bus_error_free(&session_error);
            }
        }
        
        sd_bus_message_exit_container(reply);
    }
    
    sd_bus_message_unref(reply);
    
    // Use active session if found, otherwise fallback to first session
    if (!session_id.empty()) {
        return session_id;
    }
    
    if (!fallback_session_id.empty()) {
        return fallback_session_id;
    }
    
    return std::nullopt;
}

bool SystemdHelper::isSessionLocked(const std::string& session_id) {
    if (session_id.empty()) {
        return false;
    }
    
    SDBusWrapper bus;
    if (!bus.isValid()) {
        return false;
    }
    
    sd_bus_error error = SD_BUS_ERROR_NULL;
    sd_bus_message* reply = nullptr;
    
    std::string session_path = "/org/freedesktop/login1/session/" + session_id;
    
    int r = sd_bus_get_property(
        bus.get(),
        "org.freedesktop.login1",
        session_path.c_str(),
        "org.freedesktop.login1.Session",
        "LockedHint",
        &error,
        &reply,
        "b"
    );
    
    if (r < 0) {
        Logger::getInstance().debug("Failed to get LockedHint: " + std::string(error.message));
        sd_bus_error_free(&error);
        return false;
    }
    
    int locked = 0;
    r = sd_bus_message_read(reply, "b", &locked);
    sd_bus_message_unref(reply);
    
    if (r < 0) {
        return false;
    }
    
    return locked != 0;
}

std::optional<bool> SystemdHelper::getLidClosed() {
    SDBusWrapper bus;
    if (!bus.isValid()) {
        return std::nullopt;
    }
    
    sd_bus_error error = SD_BUS_ERROR_NULL;
    sd_bus_message* reply = nullptr;
    
    int r = sd_bus_get_property(
        bus.get(),
        "org.freedesktop.login1",
        "/org/freedesktop/login1",
        "org.freedesktop.login1.Manager",
        "LidClosed",
        &error,
        &reply,
        "b"
    );
    
    if (r < 0) {
        Logger::getInstance().debug("Failed to get LidClosed: " + std::string(error.message));
        sd_bus_error_free(&error);
        return std::nullopt;
    }
    
    int closed = 0;
    r = sd_bus_message_read(reply, "b", &closed);
    sd_bus_message_unref(reply);
    
    if (r < 0) {
        return std::nullopt;
    }
    
    return closed != 0;
}

bool SystemdHelper::isProcessRunning(const std::string& process_name) {
    // Check /proc filesystem for process (no popen needed)
    DIR* proc_dir = opendir("/proc");
    if (!proc_dir) {
        return false;
    }
    
    struct dirent* entry;
    while ((entry = readdir(proc_dir)) != nullptr) {
        // Skip non-numeric directories
        if (entry->d_type != DT_DIR) {
            continue;
        }
        
        // Check if directory name is numeric (PID)
        bool is_pid = true;
        for (const char* p = entry->d_name; *p; p++) {
            if (*p < '0' || *p > '9') {
                is_pid = false;
                break;
            }
        }
        
        if (!is_pid) {
            continue;
        }
        
        // Read cmdline
        std::string cmdline_path = std::string("/proc/") + entry->d_name + "/comm";
        std::ifstream cmdline_file(cmdline_path);
        if (cmdline_file.is_open()) {
            std::string comm;
            std::getline(cmdline_file, comm);
            
            // Check if this matches our process name
            if (comm == process_name) {
                closedir(proc_dir);
                return true;
            }
        }
    }
    
    closedir(proc_dir);
    return false;
}

std::string SystemdHelper::getCurrentUsername() {
    static std::string cached_username;
    
    if (!cached_username.empty()) {
        return cached_username;
    }
    
    // Get username from UID (cached at process start)
    uid_t uid = getuid();
    struct passwd* pw = getpwuid(uid);
    
    if (pw && pw->pw_name) {
        cached_username = pw->pw_name;
        return cached_username;
    }
    
    return "";
}

bool SystemdHelper::isGnomeScreenSaverActive() {
    // Open session bus for GNOME ScreenSaver
    sd_bus* bus = nullptr;
    int r = sd_bus_open_user(&bus);
    if (r < 0) {
        return false;
    }
    
    sd_bus_error error = SD_BUS_ERROR_NULL;
    sd_bus_message* reply = nullptr;
    
    r = sd_bus_call_method(
        bus,
        "org.gnome.ScreenSaver",
        "/org/gnome/ScreenSaver",
        "org.gnome.ScreenSaver",
        "GetActive",
        &error,
        &reply,
        ""
    );
    
    if (r < 0) {
        sd_bus_error_free(&error);
        sd_bus_unref(bus);
        return false;
    }
    
    int active = 0;
    sd_bus_message_read(reply, "b", &active);
    sd_bus_message_unref(reply);
    sd_bus_unref(bus);
    
    return active != 0;
}

} // namespace faceid
