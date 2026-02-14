#ifndef FACEID_FINGERPRINT_AUTH_H
#define FACEID_FINGERPRINT_AUTH_H

#include <string>
#include <memory>
#include <atomic>
#include <functional>

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
    
    // Enroll fingerprint for user (blocking call with timeout)
    // Collects multiple samples and creates persistent fingerprint data
    // Returns true if enrollment completed, false on error or cancellation
    // Callback is invoked for progress updates with format: ("sample_n/total", progress_percentage)
    bool enroll(const std::string& username, int timeout_seconds, std::atomic<bool>& cancel_flag,
                std::function<void(const std::string&, int)> progress_callback = nullptr);
    
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
