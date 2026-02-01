#include "fingerprint_manager.h"
#include <syslog.h>

FingerprintManager& FingerprintManager::instance() {
    static FingerprintManager instance;
    return instance;
}

bool FingerprintManager::initialize() {
    // TODO: Initialize libfprint device discovery
    // For now, mark as unavailable since libfprint integration is WIP
    
    syslog(LOG_INFO, "Fingerprint manager initialized (placeholder)");
    reader_available_ = false;  // TODO: set to true when libfprint available
    return true;
}

float FingerprintManager::verify_fingerprint(const std::string& username) {
    if (!reader_available_) {
        syslog(LOG_DEBUG, "Fingerprint reader not available");
        return -1.0f;
    }
    
    // TODO: Implement actual fingerprint verification
    // This will:
    // 1. Wait for fingerprint touch
    // 2. Compare against enrolled fingerprints
    // 3. Return confidence score
    
    syslog(LOG_DEBUG, "Fingerprint verification for user: %s (placeholder)", username.c_str());
    return -1.0f;  // Not available yet
}

std::string FingerprintManager::get_status() const {
    return reader_available_ ? "available" : "unavailable";
}

FingerprintManager::~FingerprintManager() {
    // Cleanup when daemon shuts down
}
