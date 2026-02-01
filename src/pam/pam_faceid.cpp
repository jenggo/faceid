#include <security/pam_appl.h>
#include <security/pam_modules.h>
#include <syslog.h>
#include <unistd.h>
#include <string.h>
#include <pwd.h>
#include "dbus_client.h"
#include "context_detector.h"

extern "C" {

/**
 * pam_sm_authenticate - PAM authentication module entry point
 * 
 * This is the refactored D-Bus client version that delegates to faceid-daemon.
 * Much simpler than the old monolithic version:
 * 1. Get username
 2. Detect authentication context (lockscreen, sudo, polkit, etc.)
 * 3. Call daemon via D-Bus VerifyStart
 * 4. Wait for signals (VerifyStatus)
 * 5. Return success/failure
 */
PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh, int flags,
                                   int argc, const char **argv) {
    // Silence unused parameter warnings
    (void)flags;
    (void)argc;
    (void)argv;
    
    openlog("pam_faceid", LOG_PID, LOG_AUTH);
    syslog(LOG_INFO, "pam_faceid: authenticate called (PID: %d, UID: %d)", 
           getpid(), getuid());
    
    // ═══════════════════════════════════════════════════════════════
    // Step 1: Get username
    // ═══════════════════════════════════════════════════════════════
    const char* username = nullptr;
    int ret = pam_get_user(pamh, &username, nullptr);
    
    if (ret != PAM_SUCCESS || !username) {
        syslog(LOG_ERR, "pam_faceid: Failed to get username");
        closelog();
        return PAM_USER_UNKNOWN;
    }
    
    syslog(LOG_DEBUG, "pam_faceid: Authenticating user: %s", username);
    
    // ═══════════════════════════════════════════════════════════════
    // Step 2: Detect authentication context
    // ═══════════════════════════════════════════════════════════════
    std::string context = ContextDetector::detect_context(pamh);
    
    syslog(LOG_DEBUG, "pam_faceid: Detected context: %s", context.c_str());
    
    // ═══════════════════════════════════════════════════════════════
    // Step 3: Connect to daemon and start verification
    // ═══════════════════════════════════════════════════════════════
    DBusAuthClient client;
    
    if (!client.connect()) {
        syslog(LOG_ERR, "pam_faceid: Failed to connect to D-Bus");
        closelog();
        return PAM_AUTH_ERR;
    }
    
    if (!client.is_daemon_available()) {
        syslog(LOG_WARNING, "pam_faceid: FaceID daemon not available on D-Bus");
        closelog();
        return PAM_AUTH_ERR;
    }
    
    // Callback to log signals
    auto status_callback = [username](const std::string& status, bool done) {
        syslog(LOG_DEBUG, "pam_faceid: VerifyStatus signal - %s (done: %s)",
               status.c_str(), done ? "yes" : "no");
    };
    
    // ═══════════════════════════════════════════════════════════════
    // Step 4: Call daemon to start verification
    // ═══════════════════════════════════════════════════════════════
    if (!client.verify_start(username, context, status_callback)) {
        syslog(LOG_ERR, "pam_faceid: Failed to call VerifyStart on daemon");
        closelog();
        return PAM_AUTH_ERR;
    }
    
    // ═══════════════════════════════════════════════════════════════
    // Step 5: Wait for verification signals
    // ═══════════════════════════════════════════════════════════════
    // Default timeout: 10 seconds (configurable in daemon)
    int timeout_ms = 10000;
    
    if (!client.wait_for_verification(timeout_ms)) {
        syslog(LOG_WARNING, "pam_faceid: Verification failed or timeout");
        client.verify_stop();  // Clean up
        closelog();
        return PAM_AUTH_ERR;
    }
    
    // ═══════════════════════════════════════════════════════════════
    // Step 6: Check result and return
    // ═══════════════════════════════════════════════════════════════
    if (client.was_successful()) {
        syslog(LOG_INFO, "pam_faceid: Authentication successful for user %s", username);
        closelog();
        return PAM_SUCCESS;
    } else {
        syslog(LOG_WARNING, "pam_faceid: Authentication failed for user %s", username);
        closelog();
        return PAM_AUTH_ERR;
    }
}

/**
 * pam_sm_setcred - Set credentials (optional)
 * Required by PAM interface but not used for biometric auth
 */
PAM_EXTERN int pam_sm_setcred(pam_handle_t *pamh, int flags,
                              int argc, const char **argv) {
    // Silence unused parameter warnings
    (void)pamh;
    (void)flags;
    (void)argc;
    (void)argv;
    
    return PAM_SUCCESS;
}

/**
 * pam_sm_acct_mgmt - Account management (optional)
 * Required by PAM interface but not used for biometric auth
 */
PAM_EXTERN int pam_sm_acct_mgmt(pam_handle_t *pamh, int flags,
                               int argc, const char **argv) {
    // Silence unused parameter warnings
    (void)pamh;
    (void)flags;
    (void)argc;
    (void)argv;
    
    return PAM_SUCCESS;
}

}  // extern "C"
