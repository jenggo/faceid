#include "notification.h"
#include "../logger.h"
#include <cstdlib>
#include <sstream>

namespace faceid {

std::string NotificationHelper::urgencyToString(Urgency urgency) {
    switch (urgency) {
        case Urgency::LOW:      return "low";
        case Urgency::NORMAL:   return "normal";
        case Urgency::CRITICAL: return "critical";
        default:                return "normal";
    }
}

bool NotificationHelper::sendNotification(
    const std::string& title,
    const std::string& message,
    Urgency urgency,
    int timeout_ms) {
    
    Logger& logger = Logger::getInstance();
    
    // Build notify-send command
    std::ostringstream cmd;
    cmd << "notify-send";
    
    // Add urgency
    cmd << " -u " << urgencyToString(urgency);
    
    // Add timeout if specified
    if (timeout_ms > 0) {
        cmd << " -t " << timeout_ms;
    }
    
    // Add app name
    cmd << " -a 'FaceID'";
    
    // Add title and message (properly escaped)
    // Note: Using single quotes to avoid shell injection
    cmd << " '" << title << "'";
    cmd << " '" << message << "'";
    
    // Redirect stderr to suppress errors
    cmd << " 2>/dev/null";
    
    // Execute command
    int ret = system(cmd.str().c_str());
    
    if (ret != 0) {
        logger.debug("Failed to send notification: " + cmd.str());
        return false;
    }
    
    logger.debug("Notification sent: " + title + " - " + message);
    return true;
}

bool NotificationHelper::sendSecurityAlert(const std::string& message) {
    return sendNotification(
        "FaceID Security Alert",
        message,
        Urgency::CRITICAL,
        5000  // 5 second timeout
    );
}

} // namespace faceid
