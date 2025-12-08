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
    // Try multiple methods to detect screen lock
    
    // Method 1: Check systemd session LockedHint (most reliable)
    FILE* pipe = popen("loginctl show-session $(loginctl | grep $(whoami) | awk '{print $1}') -p LockedHint --value 2>/dev/null", "r");
    if (pipe) {
        char buffer[128];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            pclose(pipe);
            std::string result(buffer);
            // Remove trailing newline
            if (!result.empty() && result.back() == '\n') {
                result.pop_back();
            }
            if (result == "yes" || result == "true" || result == "1") {
                return true;
            }
        } else {
            pclose(pipe);
        }
    }
    
    // Method 2: Check for KDE lock screen GREETER (not daemon!)
    // The greeter only runs when actually locked, daemon is always running
    // Use ps to get exact process name to avoid false positives from daemon
    pipe = popen("ps aux | grep -w '[k]screenlocker_greet' 2>/dev/null", "r");
    if (pipe) {
        char buffer[128];
        bool has_output = (fgets(buffer, sizeof(buffer), pipe) != nullptr);
        pclose(pipe);
        if (has_output) {
            return true;
        }
    }
    
    // Method 3: Check for screensaver/lock processes (GNOME)
    // REMOVED: pgrep -f gnome-screensav matches background daemons that always run
    // Use D-Bus check in checkGNOMELockScreen() instead
    
    // Method 4: Check XDG screensaver status (works on many DEs)
    // REMOVED: xdg-screensaver "enabled" means "can be activated", NOT "currently locked"
    // This method causes false positives and should not be used
    
    return false;
}

bool DisplayDetector::isDisplayBlanked() {
    // Method 1: Check DRM power state for laptop's built-in display (eDP) FIRST
    // This is the MOST RELIABLE method on KDE Plasma and modern Linux systems
    // NOTE: Only check eDP specifically - external DP/HDMI ports may be "Off" when disconnected
    // which would cause false positives if we check all displays with wildcards
    FILE* pipe = popen("cat /sys/class/drm/card*/card*-eDP-*/dpms 2>/dev/null | head -1", "r");
    if (pipe) {
        char buffer[128];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            pclose(pipe);
            std::string result(buffer);
            if (result.find("Off") != std::string::npos || 
                result.find("off") != std::string::npos) {
                return true;
            }
            // eDP is On, screen is definitely on
            return false;
        } else {
            pclose(pipe);
        }
    }
    
    // Method 2: Check backlight state as fallback
    // Note: On some systems (KDE Plasma), backlight may not go to 0 even when display is "Off"
    // so we use this as secondary check only
    pipe = popen("cat /sys/class/backlight/*/actual_brightness 2>/dev/null | head -1", "r");
    if (pipe) {
        char buffer[128];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            pclose(pipe);
            try {
                int brightness = std::stoi(buffer);
                if (brightness == 0) {
                    return true;  // Screen is definitely off
                }
                // Brightness > 0, but we already checked DRM above
                // Don't return false here - continue to other methods
            } catch (...) {
                // Ignore parse errors, try other methods
            }
        } else {
            pclose(pipe);
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
                if (result.find("Off") != std::string::npos || 
                    result.find("off") != std::string::npos) {
                    return true;
                }
                return false;
            } else {
                pclose(pipe);
            }
        }
    }
    
    // Could not determine, assume screen is on (safe default)
    return false;
}

bool DisplayDetector::checkWaylandDisplay() {
    // For Wayland, we need to check compositor-specific methods
    // This is more complex and compositor-dependent
    return false; // TODO: Implement Wayland-specific checks
}

bool DisplayDetector::checkX11Display() {
    // Check DPMS state
    return isDisplayBlanked();
}

bool DisplayDetector::checkSystemdSession() {
    // Check if session is active
    FILE* pipe = popen("loginctl show-session $(loginctl | grep $(whoami) | awk '{print $1}') -p Active --value 2>/dev/null", "r");
    if (pipe) {
        char buffer[128];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            pclose(pipe);
            std::string result(buffer);
            if (!result.empty() && result.back() == '\n') {
                result.pop_back();
            }
            return (result == "yes" || result == "true" || result == "1");
        }
        pclose(pipe);
    }
    return false;
}

bool DisplayDetector::checkDPMS() {
    return isDisplayBlanked();
}

bool DisplayDetector::isLockScreenGreeter() {
    // Check if we're being called from a lock screen greeter process
    // This helps us differentiate between:
    // 1. Lock screen authentication (greeter process)
    // 2. Sudo/su authentication while screen is locked
    
    // Method 1: Check if KDE lock screen greeter is in our process tree
    if (checkKDELockScreen()) {
        return true;
    }
    
    // Method 2: Check if GNOME lock screen is in our process tree
    if (checkGNOMELockScreen()) {
        return true;
    }
    
    // Method 3: Check PAM_SERVICE environment variable
    const char* pam_service = getenv("PAM_SERVICE");
    if (pam_service) {
        std::string service(pam_service);
        // Common lock screen PAM services
        if (service.find("kde-screen-locker") != std::string::npos ||
            service.find("kscreenlocker") != std::string::npos ||
            service.find("gnome-screensaver") != std::string::npos ||
            service.find("lightdm") != std::string::npos ||
            service.find("gdm-password") != std::string::npos) {
            return true;
        }
    }
    
    // Method 4: Check PAM_TTY for typical lock screen patterns
    const char* pam_tty = getenv("PAM_TTY");
    if (pam_tty) {
        std::string tty(pam_tty);
        if (tty == ":0" || tty.find("login") != std::string::npos) {
            // Might be a lock screen, but not definitive
            // Additional check: is screen actually locked?
            return isScreenLocked();
        }
    }
    
    return false;
}

bool DisplayDetector::checkKDELockScreen() {
    // Check if kscreenlocker_greet process exists (the actual lock screen UI)
    // NOT kscreenlocker (which is a daemon that runs all the time)
    
    // Method 1: Check for the greeter process specifically
    // Use ps with grep to get exact binary name, avoiding daemon false positives
    FILE* pipe = popen("ps aux | grep -w '[k]screenlocker_greet' 2>/dev/null", "r");
    if (pipe) {
        char buffer[128];
        bool has_output = (fgets(buffer, sizeof(buffer), pipe) != nullptr);
        pclose(pipe);
        if (has_output) {
            // KDE lock screen UI is active
            return true;
        }
    }
    
    // Method 3: Check DRM power state for laptop's built-in display (eDP) only
    // BUG FIX: Don't check all displays - disconnected DP/HDMI ports report "Off"
    // which causes false positives. Only check the eDP (laptop screen).
    pipe = popen("cat /sys/class/drm/card*/card*-eDP-*/dpms 2>/dev/null | head -1", "r");
    if (pipe) {
        char buffer[128];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            pclose(pipe);
            std::string result(buffer);
            if (result.find("Off") != std::string::npos || 
                result.find("off") != std::string::npos) {
                return true;
            }
            // eDP display is On
            return false;
        } else {
            pclose(pipe);
        }
    }
    
    // If we can't determine, assume display is on (conservative approach)
    return false;
}

bool DisplayDetector::checkGNOMELockScreen() {
    // Check for GNOME Shell's screen shield
    FILE* pipe = popen("gdbus call --session --dest org.gnome.ScreenSaver --object-path /org/gnome/ScreenSaver --method org.gnome.ScreenSaver.GetActive 2>/dev/null", "r");
    if (pipe) {
        char buffer[128];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            pclose(pipe);
            std::string result(buffer);
            if (result.find("true") != std::string::npos) {
                return true;
            }
        } else {
            pclose(pipe);
        }
    }
    
    // Check if gnome-screensaver is running
    pipe = popen("pgrep -x gnome-screensav 2>/dev/null", "r");
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

} // namespace faceid
