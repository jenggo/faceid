#pragma once

#include <string>
#include <security/pam_appl.h>

/**
 * Context Detector - Detects PAM authentication context
 * 
 * Determines what triggered the authentication:
 * - lockscreen: KDE kscreenlocker, GNOME lock-screen, SDDM
 * - sudo: sudo command authentication
 * - polkit: PolicyKit authentication
 * - login: Desktop session login
 * - unknown: Couldn't determine context
 */
class ContextDetector {
public:
    /**
     * Detect PAM context from handle and environment
     * Returns one of: "lockscreen", "sudo", "polkit", "login", "unknown"
     */
    static std::string detect_context(pam_handle_t* pamh);
    
private:
    static std::string detect_service(pam_handle_t* pamh);
    static std::string detect_from_environment();
    static std::string detect_from_parent_process();
};
