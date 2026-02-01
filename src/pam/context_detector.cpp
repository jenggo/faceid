#include "context_detector.h"
#include <syslog.h>
#include <cstring>
#include <unistd.h>
#include <fstream>
#include <sstream>

std::string ContextDetector::detect_context(pam_handle_t* pamh) {
    // Try to detect from PAM service name first (most reliable)
    std::string context = detect_service(pamh);
    if (context != "unknown") {
        return context;
    }
    
    // Fall back to environment variables
    context = detect_from_environment();
    if (context != "unknown") {
        return context;
    }
    
    // Final fallback: check parent process
    context = detect_from_parent_process();
    return context;
}

std::string ContextDetector::detect_service(pam_handle_t* pamh) {
    const char* service = nullptr;
    
    if (pam_get_item(pamh, PAM_SERVICE, (const void**)&service) != PAM_SUCCESS || !service) {
        return "unknown";
    }
    
    syslog(LOG_DEBUG, "pam_faceid: Detected PAM service: %s", service);
    
    // KDE lock screen
    if (strcmp(service, "kde") == 0 || 
        strcmp(service, "kscreenlocker") == 0 ||
        strstr(service, "kde-unlock") != nullptr) {
        return "lockscreen";
    }
    
    // GNOME lock screen
    if (strcmp(service, "gnome-screensaver") == 0 ||
        strstr(service, "gnome-lock") != nullptr) {
        return "lockscreen";
    }
    
    // SDDM
    if (strcmp(service, "sddm") == 0 ||
        strcmp(service, "sddm-greeter") == 0) {
        return "lockscreen";  // Could also be login, but lockscreen is more conservative
    }
    
    // sudo
    if (strcmp(service, "sudo") == 0) {
        return "sudo";
    }
    
    // PolicyKit
    if (strcmp(service, "polkit-1") == 0 ||
        strstr(service, "polkit") != nullptr) {
        return "polkit";
    }
    
    // GDM/SDDM login (not in lock screen)
    if (strcmp(service, "gdm-password") == 0 ||
        strcmp(service, "gdm") == 0 ||
        strcmp(service, "login") == 0) {
        return "login";  // We don't support login screen (no face yet)
    }
    
    return "unknown";
}

std::string ContextDetector::detect_from_environment() {
    // Check for sudo
    if (getenv("SUDO_USER") != nullptr) {
        syslog(LOG_DEBUG, "pam_faceid: Detected sudo from environment");
        return "sudo";
    }
    
    // Check for polkit
    if (getenv("POLKIT_CALLER_UID") != nullptr) {
        syslog(LOG_DEBUG, "pam_faceid: Detected polkit from environment");
        return "polkit";
    }
    
    // Check for lock screen environment
    if (getenv("LOCKSCREEN") != nullptr ||
        getenv("KDE_LOCK_SCREEN") != nullptr ||
        getenv("GNOME_LOCK_SCREEN") != nullptr) {
        syslog(LOG_DEBUG, "pam_faceid: Detected lock screen from environment");
        return "lockscreen";
    }
    
    return "unknown";
}

std::string ContextDetector::detect_from_parent_process() {
    // Read /proc/self/comm to get current process name
    pid_t ppid = getppid();
    
    std::string cmdline_path = "/proc/" + std::to_string(ppid) + "/comm";
    std::ifstream comm_file(cmdline_path);
    
    if (!comm_file.is_open()) {
        return "unknown";
    }
    
    std::string parent_process;
    if (std::getline(comm_file, parent_process)) {
        // Remove trailing newline
        if (!parent_process.empty() && parent_process.back() == '\n') {
            parent_process.pop_back();
        }
        
        syslog(LOG_DEBUG, "pam_faceid: Parent process: %s", parent_process.c_str());
        
        // Check known lock screen processes
        if (parent_process.find("kscreenlocker") != std::string::npos ||
            parent_process.find("kde") != std::string::npos) {
            return "lockscreen";
        }
        
        if (parent_process.find("gnome-screensaver") != std::string::npos ||
            parent_process.find("gnome-shell") != std::string::npos) {
            return "lockscreen";
        }
        
        if (parent_process.find("sddm") != std::string::npos) {
            return "lockscreen";
        }
    }
    
    return "unknown";
}
