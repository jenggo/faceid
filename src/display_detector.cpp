#include "display_detector.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

namespace faceid {

DisplayDetector::DisplayDetector() : is_wayland_(false) {
    // Detect display server type
    const char* wayland_display = getenv("WAYLAND_DISPLAY");
    const char* xdg_session_type = getenv("XDG_SESSION_TYPE");
    
    is_wayland_ = (wayland_display && strlen(wayland_display) > 0) ||
                  (xdg_session_type && strcmp(xdg_session_type, "wayland") == 0);
}

DisplayState DisplayDetector::getDisplayState() {
    // PRIORITY 1: Check if being called from lock screen greeter
    // Lock screen greeters should ONLY use biometrics if display is actually on
    if (isLockScreenGreeter()) {
        // We're being called from lock screen, check if display is actually on
        if (isDisplayBlanked()) {
            detection_method_ = "lock_screen_display_blanked";
            return DisplayState::OFF;
        }
        // Lock screen is showing and display is on - allow biometrics
        detection_method_ = "lock_screen_display_on";
        return DisplayState::ON;
    }
    
    // PRIORITY 2: Check if screen is locked (not from greeter, e.g., from sudo/su)
    if (isScreenLocked()) {
        // When locked, check if display is actually blanked/off
        if (isDisplayBlanked()) {
            detection_method_ = "screen_locked_and_blanked";
            return DisplayState::OFF;
        }
        // Screen is locked but display is still on (lock screen visible)
        detection_method_ = "screen_locked_display_on";
        return DisplayState::ON;
    }
    
    // PRIORITY 3: Not locked, check if display is powered off
    if (isDisplayBlanked()) {
        detection_method_ = "display_blanked";
        return DisplayState::OFF;
    }
    
    detection_method_ = "display_active";
    return DisplayState::ON;
}

bool DisplayDetector::isScreenLocked() {
    // Check systemd session LockedHint (most reliable)
    FILE* pipe = popen("loginctl show-session $(loginctl | grep $(whoami) | awk '{print $1}') -p LockedHint --value 2>/dev/null", "r");
    if (pipe) {
        char buffer[128];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            pclose(pipe);
            std::string result(buffer);
            result.erase(result.find_last_not_of(" \t\n\r") + 1);  // Trim trailing whitespace
            if (result == "yes" || result == "true" || result == "1") {
                return true;
            }
        } else {
            pclose(pipe);
        }
    }
    
    // Check for KDE lock screen GREETER (not daemon)
    pipe = popen("ps aux | grep -w '[k]screenlocker_greet' 2>/dev/null", "r");
    if (pipe) {
        char buffer[128];
        bool has_output = (fgets(buffer, sizeof(buffer), pipe) != nullptr);
        pclose(pipe);
        if (has_output) {
            return true;
        }
    }
    
    return false;
}

bool DisplayDetector::isLockScreenGreeter() {
    // Check environment variables that indicate we're running from a lock screen greeter
    const char* pam_service = getenv("PAM_SERVICE");
    if (pam_service) {
        // Common lock screen greeter service names
        if (strcmp(pam_service, "kde") == 0 ||
            strcmp(pam_service, "kde-fingerprint") == 0 ||
            strcmp(pam_service, "sddm") == 0 ||
            strcmp(pam_service, "lightdm") == 0 ||
            strcmp(pam_service, "gdm-password") == 0) {
            return true;
        }
    }
    
    // Check if called from known lock screen processes
    if (checkKDELockScreen() || checkGNOMELockScreen()) {
        return true;
    }
    
    return false;
}

bool DisplayDetector::isDisplayBlanked() {
    // Method 1: Check DRM power state for laptop's built-in display (eDP) FIRST
    // This is the MOST RELIABLE method on KDE Plasma and modern Linux systems
    // NOTE: Only check eDP specifically - external DP/HDMI ports may be "Off" when disconnected
    std::ifstream drm_state("/sys/class/drm/card0/card0-eDP-1/dpms");
    if (drm_state.is_open()) {
        std::string state;
        std::getline(drm_state, state);
        if (state.find("Off") != std::string::npos || state.find("off") != std::string::npos) {
            return true;
        }
        // eDP is On, screen is definitely on
        return false;
    }
    
    // Method 2: Check backlight state as fallback
    // Note: On some systems (KDE Plasma), backlight may not go to 0 even when display is "Off"
    std::ifstream backlight("/sys/class/backlight/intel_backlight/actual_brightness");
    if (backlight.is_open()) {
        int brightness = 0;
        backlight >> brightness;
        if (brightness == 0) {
            return true;  // Screen is definitely off
        }
    }
    
    // Method 3: Check DPMS state via xset (X11 only)
    if (!is_wayland_) {
        FILE* pipe = popen("DISPLAY=:0 xset q 2>/dev/null | grep 'Monitor is' | awk '{print $3}'", "r");
        if (pipe) {
            char buffer[128];
            if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                pclose(pipe);
                std::string result(buffer);
                if (result.find("Off") != std::string::npos || result.find("off") != std::string::npos) {
                    return true;
                }
                return false;
            }
            pclose(pipe);
        }
    }
    
    // Could not determine, assume screen is on (safe default)
    return false;
}

bool DisplayDetector::checkKDELockScreen() {
    // Check if kscreenlocker_greet process exists (the actual lock screen UI)
    // NOT kscreenlocker (which is a daemon that runs all the time)
    FILE* pipe = popen("ps aux | grep -w '[k]screenlocker_greet' 2>/dev/null", "r");
    if (pipe) {
        char buffer[128];
        bool has_output = (fgets(buffer, sizeof(buffer), pipe) != nullptr);
        pclose(pipe);
        return has_output;
    }
    
    return false;
}

bool DisplayDetector::checkGNOMELockScreen() {
    // Check for GNOME Shell's screen shield
    FILE* pipe = popen("gdbus call --session --dest org.gnome.ScreenSaver --object-path /org/gnome/ScreenSaver --method org.gnome.ScreenSaver.GetActive 2>/dev/null", "r");
    if (pipe) {
        char buffer[128];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            pclose(pipe);
            return std::string(buffer).find("true") != std::string::npos;
        }
        pclose(pipe);
    }
    
    // Check if gnome-screensaver is running
    pipe = popen("pgrep -x gnome-screensav 2>/dev/null", "r");
    if (pipe) {
        char buffer[128];
        bool has_output = (fgets(buffer, sizeof(buffer), pipe) != nullptr);
        pclose(pipe);
        return has_output;
    }
    
    return false;
}

} // namespace faceid
