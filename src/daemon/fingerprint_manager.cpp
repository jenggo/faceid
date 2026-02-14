#include "fingerprint_manager.h"
#include "config.h"
#include <syslog.h>

namespace faceid {
namespace daemon {

FingerprintManager& FingerprintManager::instance() {
    static FingerprintManager instance;
    return instance;
}

bool FingerprintManager::initialize() {
    if (initialized_) {
        syslog(LOG_INFO, "FingerprintManager already initialized");
        return reader_available_.load();
    }
    
    // Check if fingerprint enabled in config
    const auto& config = Config::instance();
    bool enabled = config.fingerprint_enabled.value_or(true);
    
    if (!enabled) {
        syslog(LOG_INFO, "Fingerprint authentication disabled in config");
        reader_available_.store(false);
        initialized_ = true;
        return false;
    }
    
    // Initialize FingerprintAuth (fprintd D-Bus connection)
    syslog(LOG_INFO, "Initializing fingerprint device...");
    bool success = fingerprint_auth_.initialize();
    
    if (success) {
        syslog(LOG_INFO, "Fingerprint device initialized successfully");
        reader_available_.store(true);
    } else {
        syslog(LOG_WARNING, "Fingerprint device not available (fprintd may not be running)");
        reader_available_.store(false);
    }
    
    initialized_ = true;
    return success;
}

float FingerprintManager::verify_fingerprint(const std::string& username, 
                                             std::atomic<bool>& cancel_flag) {
    // Check if reader available
    if (!reader_available_.load()) {
        syslog(LOG_WARNING, "Fingerprint verification requested but device unavailable");
        return -1.0f;
    }
    
    // Lock mutex for fprintd D-Bus call
    std::lock_guard<std::mutex> lock(fingerprint_mutex_);
    
    // Get timeout from config
    const auto& config = Config::instance();
    int timeout_seconds = config.fingerprint_timeout.value_or(30);
    
    syslog(LOG_INFO, "Starting fingerprint verification for user: %s (timeout: %ds)", 
           username.c_str(), timeout_seconds);
    
    // Call FingerprintAuth::authenticate (blocking with timeout)
    bool success = fingerprint_auth_.authenticate(username, timeout_seconds, cancel_flag);
    
    if (cancel_flag.load()) {
        syslog(LOG_INFO, "Fingerprint verification cancelled for user: %s", username.c_str());
        return -1.0f;  // Cancelled
    }
    
    if (success) {
        syslog(LOG_INFO, "Fingerprint verification succeeded for user: %s", username.c_str());
        return 1.0f;  // Match
    } else {
        std::string error = fingerprint_auth_.getLastError();
        if (error.find("timeout") != std::string::npos) {
            syslog(LOG_INFO, "Fingerprint verification timeout for user: %s", username.c_str());
        } else if (error.find("no match") != std::string::npos || 
                   error.find("not recognized") != std::string::npos) {
            syslog(LOG_INFO, "Fingerprint did not match for user: %s", username.c_str());
            return 0.0f;  // No match (not error)
        } else {
            syslog(LOG_WARNING, "Fingerprint verification error for user %s: %s", 
                   username.c_str(), error.c_str());
        }
        return -1.0f;  // Error or timeout
    }
}

bool FingerprintManager::is_available() const {
    return reader_available_.load();
}

faceid::FingerprintAuth& FingerprintManager::get_fingerprint_auth() {
    return fingerprint_auth_;
}

std::string FingerprintManager::get_status() const {
    if (!initialized_) {
        return "not_initialized";
    }
    if (reader_available_.load()) {
        return "available";
    }
    return "unavailable";
}

} // namespace daemon
} // namespace faceid
