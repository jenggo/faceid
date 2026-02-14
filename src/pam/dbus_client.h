#pragma once

#include <string>
#include <functional>
#include <systemd/sd-bus.h>

/**
 * D-Bus Client for PAM authentication
 * 
 * Handles communication with the faceid-daemon over D-Bus
 * This client is used by both PAM module and enrollment tool
 */
class DBusAuthClient {
public:
    DBusAuthClient();
    ~DBusAuthClient();
    
    /**
     * Connect to the faceid daemon on session bus
     * Returns true if connection established, false otherwise
     */
    // Connect to the D-Bus session bus for the current process (legacy)
    bool connect();

    // Connect to a specific user's session bus by username.
    // This attempts to open the user bus at /run/user/<uid>/bus which allows
    // PAM (running as root) to reach the user's faceid-daemon even when
    // the PAM process does not share the user's session environment.
    bool connect_for_user(const std::string& username);
    
    /**
     * Check if daemon is available on D-Bus
     */
    bool is_daemon_available();
    
    /**
     * Start verification with username and PAM context
     * callback(status_str, done_bool) is called when signals arrive
     * Returns true if call sent successfully
     */
    bool verify_start(const std::string& username, const std::string& context,
                     std::function<void(const std::string&, bool)> status_callback);
    
    /**
     * Stop ongoing verification
     */
    void verify_stop();
    
    /**
     * Wait for verification signals with timeout (milliseconds)
     * Returns true if verification completed, false on timeout or error
     */
    bool wait_for_verification(int timeout_ms);
    
    /**
     * Get last received verification status
     */
    std::string get_last_status() const { return last_status_; }
    
    /**
     * Check if last verification was successful
     */
    bool was_successful() const { return is_successful_; }
    
private:
    sd_bus* bus_ = nullptr;
    sd_bus_slot* signal_slot_ = nullptr;
    std::string last_status_;
    bool is_successful_ = false;
    std::function<void(const std::string&, bool)> status_callback_;
    
    // D-Bus signal handler (static, uses context pointer)
    static int signal_handler(sd_bus_message* msg, void* userdata, sd_bus_error* err);
};
