#pragma once

#include <systemd/sd-bus.h>
#include <string>
#include <memory>
#include "auth_manager.h"
#include "enroll_manager.h"

/**
 * D-Bus server wrapper for org.freedesktop.FaceID.Device interface
 */
class DBusServer {
public:
    /**
     * Create and initialize D-Bus server
     * Connects to session bus, registers object path, claims bus name
     */
    static std::unique_ptr<DBusServer> create();

    ~DBusServer();

    /**
     * Wait for next D-Bus event with timeout
     * Returns true if event was processed, false on timeout
     */
    bool wait_event(int timeout_ms);

    /**
     * Get the underlying sd_bus pointer for direct access if needed
     */
    sd_bus* get_bus() { return bus_; }

    /**
     * Check if bus connection is valid
     */
    bool is_connected() const { return bus_ != nullptr; }

    /**
     * Get auth manager (for emitting signals)
     */
    AuthManager* get_auth_manager() { return auth_manager_.get(); }

    /**
     * Get enrollment manager (for emitting enrollment signals)
     */
    EnrollmentManager* get_enroll_manager() { return enroll_manager_.get(); }

private:
    sd_bus* bus_ = nullptr;
    std::unique_ptr<AuthManager> auth_manager_;
    std::unique_ptr<EnrollmentManager> enroll_manager_;

    // Private constructor - use create() factory method
    DBusServer(sd_bus* bus) : bus_(bus) {}

    /**
     * Register all D-Bus methods and signals
     */
    int register_interfaces();
};
