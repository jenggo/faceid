#pragma once

#include <systemd/sd-bus.h>
#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include <vector>
#include <future>

/**
 * Authentication Manager
 * 
 * Handles D-Bus VerifyStart/VerifyStop methods
 * Implements state machine and signal emission
 * Supports context-aware biometric method selection
 */
class AuthManager {
public:
    enum class AuthStatus {
        IDLE,
        VERIFYING,
        COMPLETED,
        FAILED
    };

    AuthManager();
    ~AuthManager();

    /**
     * Start authentication for a user with PAM context
     * Returns: true if started, false if already verifying
     * 
     * Context determines which biometric methods are used:
     * - "lockscreen" → face only (unless enabled in config)
     * - "sudo" → face first, fingerprint in parallel (if enabled)
     * - "polkit" → face first, fingerprint in parallel (if enabled)
     * - "default" → face only
     */
    bool verify_start(const std::string& username, const std::string& context);

    /**
     * Stop ongoing authentication
     */
    void verify_stop();

    /**
     * Check if authentication is in progress
     */
    bool is_verifying() const { return status_ == AuthStatus::VERIFYING; }

    /**
     * Get current status
     */
    AuthStatus get_status() const { return status_; }

    /**
     * Emit VerifyStatus signal via D-Bus
     */
    void emit_verify_status(sd_bus* bus, const std::string& result, bool done);

    /**
     * Register D-Bus methods on the given bus
     */
    static int register_methods(sd_bus* bus);

private:
    std::atomic<AuthStatus> status_{AuthStatus::IDLE};
    std::string current_username_;
    std::string current_context_;
    
    // Biometric method selection based on context
    bool should_use_face(const std::string& context) const;
    bool should_use_fingerprint(const std::string& context) const;
    
    // Verification thread helpers
    void run_face_verification(sd_bus* bus, const std::string& username);
    void run_fingerprint_verification(sd_bus* bus, const std::string& username);
    
    std::vector<std::thread> verification_threads_;
    std::atomic<bool> cancel_verification_{false};
};
