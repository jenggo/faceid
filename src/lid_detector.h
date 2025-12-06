#ifndef FACEID_LID_DETECTOR_H
#define FACEID_LID_DETECTOR_H

#include <string>

namespace faceid {

enum class LidState {
    OPEN,
    CLOSED,
    UNKNOWN
};

class LidDetector {
public:
    LidDetector();
    
    // Get current lid state
    LidState getLidState() const;
    
    // Check if lid is closed
    bool isLidClosed() const;
    
    // Get last error message
    std::string getLastError() const;
    
    // Get detection method used
    std::string getDetectionMethod() const;

private:
    // Try different methods to detect lid state
    LidState detectViaProc() const;
    LidState detectViaSysfs() const;
    LidState detectViaSystemdLogind() const;
    
    mutable std::string last_error_;
    mutable std::string detection_method_;
};

} // namespace faceid

#endif // FACEID_LID_DETECTOR_H
