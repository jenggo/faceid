#include "validation.h"
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <syslog.h>
#include <filesystem>

namespace validation {

bool is_root() {
    return getuid() == 0 || geteuid() == 0;
}

bool has_user_session() {
    return getenv("XDG_RUNTIME_DIR") != nullptr;
}

std::string get_xdg_runtime_dir() {
    const char* xdg = getenv("XDG_RUNTIME_DIR");
    return xdg ? std::string(xdg) : "";
}

std::string get_config_dir() {
    const char* xdg_config = getenv("XDG_CONFIG_HOME");
    if (xdg_config && xdg_config[0] == '/') {
        return std::string(xdg_config) + "/faceid";
    }
    
    const char* home = getenv("HOME");
    if (!home) {
        syslog(LOG_ERR, "Failed to get HOME environment variable");
        return "";
    }
    
    return std::string(home) + "/.config/faceid";
}

std::string get_data_dir() {
    const char* xdg_data = getenv("XDG_DATA_HOME");
    if (xdg_data && xdg_data[0] == '/') {
        return std::string(xdg_data) + "/faceid";
    }
    
    const char* home = getenv("HOME");
    if (!home) {
        syslog(LOG_ERR, "Failed to get HOME environment variable");
        return "";
    }
    
    return std::string(home) + "/.local/share/faceid";
}

std::string get_cache_dir() {
    const char* xdg_cache = getenv("XDG_CACHE_HOME");
    if (xdg_cache && xdg_cache[0] == '/') {
        return std::string(xdg_cache) + "/faceid";
    }
    
    const char* home = getenv("HOME");
    if (!home) {
        syslog(LOG_ERR, "Failed to get HOME environment variable");
        return "";
    }
    
    return std::string(home) + "/.cache/faceid";
}

bool is_user_level_path(const std::string& path) {
    // Reject system paths
    if (path.find("/etc/") == 0 || path.find("/usr/") == 0 || 
        path.find("/opt/") == 0 || path.find("/root/") == 0) {
        return false;
    }
    
    // Require home directory or XDG paths
    return path.find(getenv("HOME")) == 0 || 
           path.find(getenv("XDG_RUNTIME_DIR")) == 0 ||
           path.find(getenv("XDG_CONFIG_HOME")) == 0 ||
           path.find(getenv("XDG_DATA_HOME")) == 0 ||
           path.find(getenv("XDG_CACHE_HOME")) == 0;
}

void init_logging(const char* program_name) {
    openlog(program_name, LOG_PID, LOG_AUTH);
}

} // namespace validation
