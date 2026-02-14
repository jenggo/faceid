#include "dbus_client.h"
#include <systemd/sd-bus.h>
#include <syslog.h>
#include <cstring>
#include <cerrno>
#include <pwd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <errno.h>
#include <string>

// Helper to convert D-Bus method call errors
static int handle_dbus_error(int ret, sd_bus_error* error, const char* context) {
    if (ret >= 0) return ret;
    
    if (error && error->name) {
        syslog(LOG_ERR, "D-Bus %s error: %s: %s", context, error->name, error->message);
    } else {
        syslog(LOG_ERR, "D-Bus %s error: %s", context, strerror(-ret));
    }
    return ret;
}

// Signal handler wrapper - converts D-Bus signal to callback
int DBusAuthClient::signal_handler(sd_bus_message* msg, void* userdata, sd_bus_error* err) {
    DBusAuthClient* client = static_cast<DBusAuthClient*>(userdata);
    
    const char* status = nullptr;
    int done = 0;
    
    int ret = sd_bus_message_read(msg, "sb", &status, &done);
    if (ret < 0) {
        syslog(LOG_ERR, "Failed to parse VerifyStatus signal: %s", strerror(-ret));
        return ret;
    }
    
    client->last_status_ = status;
    client->is_successful_ = (done != 0 && strcmp(status, "success") == 0);
    
    if (client->status_callback_) {
        client->status_callback_(status, done != 0);
    }
    
    syslog(LOG_DEBUG, "PAM received VerifyStatus: %s (done=%d)", status, done);
    return 0;
}

DBusAuthClient::DBusAuthClient() : bus_(nullptr), signal_slot_(nullptr) {
}

DBusAuthClient::~DBusAuthClient() {
    if (signal_slot_) {
        sd_bus_slot_unref(signal_slot_);
    }
    if (bus_) {
        sd_bus_unref(bus_);
    }
}

bool DBusAuthClient::connect() {
    // Use system bus - always available regardless of session context
    int ret = sd_bus_open_system(&bus_);
    if (ret >= 0) {
        syslog(LOG_DEBUG, "Connected to D-Bus system bus");
        return true;
    }

    syslog(LOG_ERR, "Failed to connect to system bus via sd_bus_open_system(): %s", strerror(-ret));
    return false;
}

bool DBusAuthClient::connect_for_user(const std::string& username) {
    // Use system bus for all user authentication
    // The daemon (running as root on system bus) handles per-user authentication
    // No need to connect to user's session bus
    return connect();
}


bool DBusAuthClient::is_daemon_available() {
    if (!bus_) {
        return false;
    }
    
    // Query bus to check if service is available
    sd_bus_message* reply = nullptr;
    sd_bus_error error = SD_BUS_ERROR_NULL;
    
    int ret = sd_bus_call_method(bus_,
        "org.freedesktop.DBus",
        "/org/freedesktop/DBus",
        "org.freedesktop.DBus",
        "ListNames",
        &error,
        &reply,
        "");
    
    if (ret < 0) {
        syslog(LOG_DEBUG, "Failed to query D-Bus for service list: %s", strerror(-ret));
        return false;
    }
    
    // Read array of strings from reply
    char** names = nullptr;
    ret = sd_bus_message_read_strv(reply, &names);
    if (ret < 0) {
        syslog(LOG_DEBUG, "Failed to parse service list: %s", strerror(-ret));
        sd_bus_message_unref(reply);
        sd_bus_error_free(&error);
        return false;
    }
    
    // Check if org.freedesktop.FaceID is in the list
    bool found = false;
    for (size_t i = 0; names && names[i]; i++) {
        if (strcmp(names[i], "org.freedesktop.FaceID") == 0) {
            found = true;
            break;
        }
    }
    
    // Free the string array
    if (names) {
        for (size_t i = 0; names[i]; i++) {
            free((void*)names[i]);
        }
        free(names);
    }
    
    sd_bus_message_unref(reply);
    sd_bus_error_free(&error);
    
    if (!found) {
        syslog(LOG_INFO, "FaceID daemon not found on D-Bus (disabled or not running)");
    }
    
    return found;
}

bool DBusAuthClient::verify_start(const std::string& username, const std::string& context,
                                  std::function<void(const std::string&, bool)> status_callback) {
    if (!bus_) {
        syslog(LOG_ERR, "Not connected to D-Bus");
        return false;
    }
    
    status_callback_ = status_callback;
    sd_bus_error error = SD_BUS_ERROR_NULL;
    
    // Register signal handler first
    int ret = sd_bus_match_signal(bus_,
        &signal_slot_,
        "org.freedesktop.FaceID",           // Service
        "/org/freedesktop/FaceID/Device",   // Object path
        "org.freedesktop.FaceID.Device",    // Interface
        "VerifyStatus",                      // Signal name
        signal_handler,
        this);  // userdata
    
    if (ret < 0) {
        syslog(LOG_ERR, "Failed to register signal handler: %s", strerror(-ret));
        return false;
    }
    
    // Call VerifyStart on the daemon (NoReply, so method returns immediately)
    ret = sd_bus_call_method(bus_,
        "org.freedesktop.FaceID",           // Service
        "/org/freedesktop/FaceID/Device",   // Object path
        "org.freedesktop.FaceID.Device",    // Interface
        "VerifyStart",                       // Method name
        &error,
        nullptr,  // No reply expected (NoReply method)
        "ss",     // Argument types: string, string
        username.c_str(),
        context.c_str());
    
    if (ret < 0) {
        handle_dbus_error(ret, &error, "VerifyStart");
        sd_bus_error_free(&error);
        sd_bus_slot_unref(signal_slot_);
        signal_slot_ = nullptr;
        return false;
    }
    
    syslog(LOG_INFO, "Started FaceID verification for user %s (context: %s)", 
           username.c_str(), context.c_str());
    return true;
}

void DBusAuthClient::verify_stop() {
    if (!bus_) return;
    
    sd_bus_error error = SD_BUS_ERROR_NULL;
    
    sd_bus_call_method(bus_,
        "org.freedesktop.FaceID",
        "/org/freedesktop/FaceID/Device",
        "org.freedesktop.FaceID.Device",
        "VerifyStop",
        &error,
        nullptr,  // NoReply
        "");
    
    sd_bus_error_free(&error);
    syslog(LOG_DEBUG, "Sent VerifyStop to daemon");
}

bool DBusAuthClient::wait_for_verification(int timeout_ms) {
    if (!bus_) return false;
    
    // Wait for signals from the daemon
    // This is a simplified version - in production you'd use sd_bus_process in a loop
    uint64_t timeout_usec = timeout_ms * 1000;
    
    int ret = sd_bus_wait(bus_, timeout_usec);
    if (ret < 0) {
        syslog(LOG_ERR, "D-Bus wait error: %s", strerror(-ret));
        return false;
    }
    
    if (ret == 0) {
        // Timeout
        syslog(LOG_WARNING, "FaceID verification timeout after %d ms", timeout_ms);
        return false;
    }
    
    // Process one message
    ret = sd_bus_process(bus_, nullptr);
    if (ret < 0) {
        syslog(LOG_ERR, "Failed to process D-Bus message: %s", strerror(-ret));
        return false;
    }
    
    // Return true if we got a success status
    return is_successful_;
}
