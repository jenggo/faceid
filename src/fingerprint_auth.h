#ifndef FACEID_FINGERPRINT_AUTH_H
#define FACEID_FINGERPRINT_AUTH_H

#include <string>
#include <memory>
#include <atomic>

namespace faceid {

class FingerprintAuth {
public:
    FingerprintAuth();
    ~FingerprintAuth();
    
    // Initialize fingerprint device
    bool initialize();
    
    // Authenticate user (blocking call with timeout)
    // Returns true if fingerprint matches, false otherwise
    bool authenticate(const std::string& username, int timeout_seconds, std::atomic<bool>& cancel_flag);
    
    // Check if fingerprint authentication is available
    bool isAvailable() const;
    
    // Get last error message
    std::string getLastError() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    bool available_ = false;
    std::string last_error_;
};

} // namespace faceid

#endif // FACEID_FINGERPRINT_AUTH_H
