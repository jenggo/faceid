#ifndef FACEID_PRESENCE_NOTIFICATION_H
#define FACEID_PRESENCE_NOTIFICATION_H

#include <string>

namespace faceid {

/**
 * Desktop Notification Helper
 * 
 * Provides cross-desktop notification support via notify-send
 * Compatible with KDE Plasma, GNOME, and other freedesktop-compliant desktops
 */
class NotificationHelper {
public:
    enum class Urgency {
        LOW,
        NORMAL,
        CRITICAL
    };
    
    /**
     * Send a desktop notification
     * 
     * @param title Notification title
     * @param message Notification message body
     * @param urgency Notification urgency level
     * @param timeout Display timeout in milliseconds (0 = default)
     * @return true if notification was sent successfully
     */
    static bool sendNotification(
        const std::string& title,
        const std::string& message,
        Urgency urgency = Urgency::NORMAL,
        int timeout_ms = 0
    );
    
    /**
     * Send a security alert notification (critical urgency)
     * 
     * @param message Alert message
     * @return true if notification was sent successfully
     */
    static bool sendSecurityAlert(const std::string& message);
    
private:
    static std::string urgencyToString(Urgency urgency);
};

} // namespace faceid

#endif // FACEID_PRESENCE_NOTIFICATION_H
