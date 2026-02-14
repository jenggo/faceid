#ifndef FACEID_DAEMON_FINGERPRINT_MANAGER_H
#define FACEID_DAEMON_FINGERPRINT_MANAGER_H

#include <atomic>
#include <mutex>
#include <string>
#include "fingerprint_auth.h"

namespace faceid {
namespace daemon {

/**
 * Manages fingerprint authentication device and operations.
 * Thread-safe singleton wrapping FingerprintAuth (fprintd D-Bus client).
 */
class FingerprintManager {
public:
    // Get singleton instance
    static FingerprintManager& instance();
    
    /**
     * Initialize fingerprint device via fprintd.
     * @return true if fingerprint reader available, false otherwise
     */
    bool initialize();
    
    /**
     * Verify user's fingerprint.
     * Blocking call with configurable timeout.
     * 
     * @param username User to authenticate
     * @param cancel_flag Atomic bool to cancel verification
     * @return 1.0 if match, 0.0 if no match, -1.0 on error
     */
    float verify_fingerprint(const std::string& username, 
                            std::atomic<bool>& cancel_flag);
    
    /**
     * Check if fingerprint reader is available.
     * @return true if reader available and initialized
     */
    bool is_available() const;
    
    /**
     * Get underlying FingerprintAuth instance (for enrollment).
     * @return Reference to FingerprintAuth
     */
    faceid::FingerprintAuth& get_fingerprint_auth();
    
    /**
     * Get status string for debugging.
     * @return Status description
     */
    std::string get_status() const;
    
    ~FingerprintManager() = default;

private:
    FingerprintManager() = default;
    FingerprintManager(const FingerprintManager&) = delete;
    FingerprintManager& operator=(const FingerprintManager&) = delete;
    
    faceid::FingerprintAuth fingerprint_auth_;
    std::mutex fingerprint_mutex_;
    std::atomic<bool> reader_available_{false};
    bool initialized_{false};
};

} // namespace daemon
} // namespace faceid

#endif // FACEID_DAEMON_FINGERPRINT_MANAGER_H
