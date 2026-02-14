#pragma once

#include <systemd/sd-bus.h>
#include <string>
#include <atomic>
#include <thread>
#include <memory>

/**
 * Enrollment Manager - Handles face model enrollment
 * 
 * Manages the enrollment process:
 * - Starts enrollment for a username
 * - Captures multiple faces from different angles
 * - Trains ML model
 * - Saves model to ~/.local/share/faceid/models/
 * 
 * Thread-safe with state machine.
 */
class EnrollmentManager {
public:
    enum class EnrollStatus {
        IDLE,
        ENROLLING,
        COMPLETED,
        FAILED
    };

    EnrollmentManager();
    ~EnrollmentManager();

    /**
     * Start enrollment for a user
     * Returns: true if started, false if already enrolling
     * bus: D-Bus connection for emitting signals
     */
    bool enroll_start(sd_bus* bus, const std::string& username);

    /**
     * Stop ongoing enrollment
     */
    void enroll_stop();

    /**
     * Check if enrollment is in progress
     */
    bool is_enrolling() const { return status_ == EnrollStatus::ENROLLING; }

    /**
     * Get current status
     */
    EnrollStatus get_status() const { return status_; }

    /**
     * Emit EnrollStatus signal via D-Bus
     * Parameters: username, status (in_progress/success/error/etc), progress (0-100), message
     */
    void emit_enroll_status(sd_bus* bus, const std::string& username, 
                           const std::string& status, int progress, 
                           const std::string& message);
    
    /**
     * Enroll user's fingerprint via fprintd.
     * @param bus D-Bus connection for progress signals
     * @param username User to enroll
     */
    void enroll_fingerprint(sd_bus* bus, const std::string& username);

private:
    std::atomic<EnrollStatus> status_{EnrollStatus::IDLE};
    std::string enrolling_username_;
    std::atomic<bool> cancel_enrollment_{false};
    
    std::unique_ptr<std::thread> enrollment_thread_;
    
    // Enrollment phases (runs in background thread)
    void run_enrollment(sd_bus* bus, const std::string& username);
    void capture_training_frames(int& progress);
    bool train_model(const std::string& username, int& progress);
    bool save_model(const std::string& username);
    
    // Utility
    std::string get_model_path(const std::string& username);
};
