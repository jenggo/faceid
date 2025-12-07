#ifndef FACEID_DISPLAY_DETECTOR_H
#define FACEID_DISPLAY_DETECTOR_H

#include <string>

namespace faceid {

enum class DisplayState {
    ON,       // Display is powered on and active
    OFF,      // Display is powered off (DPMS off, screen locked with blank)
    UNKNOWN   // Could not determine state
};

class DisplayDetector {
public:
    DisplayDetector();
    
    // Get current display power state
    DisplayState getDisplayState();
    
    // Check if screen is locked
    bool isScreenLocked();
    
    // Check if display is blanked/off via DPMS
    bool isDisplayBlanked();
    
    // Check if being called from lock screen greeter (KDE/GNOME)
    bool isLockScreenGreeter();
    
    // Get the detection method used
    std::string getDetectionMethod() const { return detection_method_; }
    
    // Get last error message
    std::string getLastError() const { return last_error_; }
    
private:
    std::string detection_method_;
    std::string last_error_;
    bool is_wayland_;
    
    // Detection methods
    bool checkWaylandDisplay();
    bool checkX11Display();
    bool checkSystemdSession();
    bool checkDPMS();
    bool checkKDELockScreen();
    bool checkGNOMELockScreen();
};

} // namespace faceid

#endif // FACEID_DISPLAY_DETECTOR_H
