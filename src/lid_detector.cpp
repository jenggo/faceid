#include "lid_detector.h"
#include "systemd_helper.h"
#include "logger.h"
#include <fstream>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>

namespace faceid {

LidDetector::LidDetector() {
    detection_method_ = "none";
}

LidState LidDetector::getLidState() const {
    // Try different detection methods in order of reliability
    
    // Method 1: /proc/acpi/button/lid/*/state
    LidState state = detectViaProc();
    if (state != LidState::UNKNOWN) {
        detection_method_ = "proc_acpi";
        return state;
    }
    
    // Method 2: /sys/class/input/input*/device/lid_state or similar
    state = detectViaSysfs();
    if (state != LidState::UNKNOWN) {
        detection_method_ = "sysfs";
        return state;
    }
    
    // Method 3: systemd-logind via D-Bus (more complex, last resort)
    state = detectViaSystemdLogind();
    if (state != LidState::UNKNOWN) {
        detection_method_ = "systemd_logind";
        return state;
    }
    
    last_error_ = "No lid detection method available";
    detection_method_ = "none";
    return LidState::UNKNOWN;
}

bool LidDetector::isLidClosed() const {
    return getLidState() == LidState::CLOSED;
}

std::string LidDetector::getLastError() const {
    return last_error_;
}

std::string LidDetector::getDetectionMethod() const {
    return detection_method_;
}

LidState LidDetector::detectViaProc() const {
    // Try /proc/acpi/button/lid/*/state
    // Format: "state:      open" or "state:      closed"
    DIR* dir = opendir("/proc/acpi/button/lid");
    if (!dir) {
        return LidState::UNKNOWN;
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] == '.') {
            continue;
        }
        
        std::string state_path = std::string("/proc/acpi/button/lid/") + 
                                entry->d_name + "/state";
        
        std::ifstream state_file(state_path);
        if (state_file.is_open()) {
            std::string line;
            if (std::getline(state_file, line)) {
                closedir(dir);
                
                // Parse line format: "state:      open" or "state:      closed"
                // Extract the second word after splitting by whitespace
                size_t colon_pos = line.find(':');
                if (colon_pos != std::string::npos) {
                    std::string value_part = line.substr(colon_pos + 1);
                    
                    // Trim leading whitespace
                    size_t start = value_part.find_first_not_of(" \t");
                    if (start != std::string::npos) {
                        value_part = value_part.substr(start);
                    }
                    
                    // Check value
                    if (value_part.find("open") == 0) {
                        Logger::getInstance().debug("Lid detected as OPEN via " + state_path);
                        return LidState::OPEN;
                    } else if (value_part.find("closed") == 0) {
                        Logger::getInstance().debug("Lid detected as CLOSED via " + state_path);
                        return LidState::CLOSED;
                    }
                }
            }
        }
    }
    
    closedir(dir);
    return LidState::UNKNOWN;
}

LidState LidDetector::detectViaSysfs() const {
    // Try various sysfs paths that might contain lid state
    const char* sysfs_paths[] = {
        "/sys/devices/virtual/input/input0/lid_state",
        "/sys/class/input/input0/lid_state",
        "/sys/devices/platform/lis3lv02d/lid_state",
        nullptr
    };
    
    for (int i = 0; sysfs_paths[i] != nullptr; i++) {
        std::ifstream lid_file(sysfs_paths[i]);
        if (lid_file.is_open()) {
            std::string value;
            if (lid_file >> value) {
                if (value == "1" || value == "open") {
                    Logger::getInstance().debug(std::string("Lid detected as OPEN via ") + sysfs_paths[i]);
                    return LidState::OPEN;
                } else if (value == "0" || value == "closed") {
                    Logger::getInstance().debug(std::string("Lid detected as CLOSED via ") + sysfs_paths[i]);
                    return LidState::CLOSED;
                }
            }
        }
    }
    
    // Try searching /sys/class/input/input*/lid_state dynamically
    DIR* dir = opendir("/sys/class/input");
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            if (strncmp(entry->d_name, "input", 5) == 0) {
                std::string lid_path = std::string("/sys/class/input/") + 
                                      entry->d_name + "/lid_state";
                
                std::ifstream lid_file(lid_path);
                if (lid_file.is_open()) {
                    std::string value;
                    if (lid_file >> value) {
                        closedir(dir);
                        
                        if (value == "1" || value == "open") {
                            Logger::getInstance().debug("Lid detected as OPEN via " + lid_path);
                            return LidState::OPEN;
                        } else if (value == "0" || value == "closed") {
                            Logger::getInstance().debug("Lid detected as CLOSED via " + lid_path);
                            return LidState::CLOSED;
                        }
                    }
                }
            }
        }
        closedir(dir);
    }
    
    return LidState::UNKNOWN;
}

LidState LidDetector::detectViaSystemdLogind() const {
    // Use D-Bus API to query systemd-logind for lid state
    auto lid_closed = SystemdHelper::getLidClosed();
    
    if (!lid_closed.has_value()) {
        return LidState::UNKNOWN;
    }
    
    if (*lid_closed) {
        Logger::getInstance().debug("Lid detected as CLOSED via systemd-logind D-Bus");
        return LidState::CLOSED;
    } else {
        Logger::getInstance().debug("Lid detected as OPEN via systemd-logind D-Bus");
        return LidState::OPEN;
    }
}

} // namespace faceid
