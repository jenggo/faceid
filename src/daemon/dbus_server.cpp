#include "dbus_server.h"
#include "auth_manager.h"
#include <syslog.h>
#include <cstring>
#include <cerrno>

// ============================================================================
// D-Bus Method Handlers - These are called by the D-Bus framework
// ============================================================================

// Global pointer to auth manager for use in C-style handlers
static AuthManager* g_auth_manager = nullptr;

/**
 * VerifyStart method handler
 * Called when PAM client calls org.freedesktop.FaceID.Device.VerifyStart
 * NoReply method - returns immediately, signals emit asynchronously
 */
static int method_verify_start(sd_bus_message* m, void* userdata, sd_bus_error* ret_error) {
    const char* username = nullptr;
    const char* context = nullptr;
    
    int r = sd_bus_message_read(m, "ss", &username, &context);
    if (r < 0) {
        sd_bus_error_set_const(ret_error, "org.freedesktop.DBus.Error.InvalidArgs", 
                              "Invalid arguments");
        return r;
    }
    
    DBusServer* server = static_cast<DBusServer*>(userdata);
    AuthManager* auth_mgr = server->get_auth_manager();
    
    if (!auth_mgr->verify_start(username, context)) {
        sd_bus_error_set_const(ret_error, "org.freedesktop.FaceID.Error.AlreadyVerifying",
                              "Authentication already in progress");
        return -EBUSY;
    }
    
    // NoReply method - don't send reply, async signals will follow
    return 0;
}

/**
 * VerifyStop method handler
 * Called when PAM client calls org.freedesktop.FaceID.Device.VerifyStop
 * NoReply method - cancels ongoing authentication
 */
static int method_verify_stop(sd_bus_message* m, void* userdata, sd_bus_error* ret_error) {
    DBusServer* server = static_cast<DBusServer*>(userdata);
    AuthManager* auth_mgr = server->get_auth_manager();
    
    auth_mgr->verify_stop();
    
    // NoReply method - don't send reply
    return 0;
}

/**
 * GetStatus method handler
 * Returns daemon status: camera ready, models loaded, version
 */
static int method_get_status(sd_bus_message* m, void* userdata, sd_bus_error* ret_error) {
    // For now, return simple status dict
    // TODO: Replace with actual status from resource managers
    return sd_bus_reply_method_return(m, "a{sv}",
        6,  // 3 key-value pairs in the dict
        "version", "s", "2.0.0",
        "camera_ready", "b", 1,  // true
        "models_loaded", "i", 1
    );
}

// ============================================================================
// D-Bus Virtual Table - Defines the interface
// ============================================================================

static const sd_bus_vtable device_vtable[] = {
    SD_BUS_VTABLE_START(0),
    
    // Methods
    SD_BUS_METHOD("VerifyStart", "ss", "", method_verify_start, 
                  SD_BUS_VTABLE_UNPRIVILEGED),
    SD_BUS_METHOD("VerifyStop", "", "", method_verify_stop,
                  SD_BUS_VTABLE_UNPRIVILEGED),
    SD_BUS_METHOD("GetStatus", "", "a{sv}", method_get_status,
                  SD_BUS_VTABLE_UNPRIVILEGED),
    
    // Signals
    SD_BUS_SIGNAL("VerifyStatus", "sb", 0),
    
    // Properties (for now, just return static values)
    SD_BUS_PROPERTY("Version", "s", nullptr, 0, SD_BUS_VTABLE_PROPERTY_CONST),
    SD_BUS_PROPERTY("CameraReady", "b", nullptr, 0, SD_BUS_VTABLE_PROPERTY_CONST),
    SD_BUS_PROPERTY("ModelsLoaded", "i", nullptr, 0, SD_BUS_VTABLE_PROPERTY_CONST),
    
    SD_BUS_VTABLE_END
};

// ============================================================================
// DBusServer Implementation
// ============================================================================

std::unique_ptr<DBusServer> DBusServer::create() {
    sd_bus* bus = nullptr;
    int ret = sd_bus_open_user(&bus);
    
    if (ret < 0) {
        syslog(LOG_ERR, "Failed to open session bus: %s", strerror(-ret));
        return nullptr;
    }

    auto server = std::unique_ptr<DBusServer>(new DBusServer(bus));
    
    // Create auth manager
    server->auth_manager_ = std::make_unique<AuthManager>();
    g_auth_manager = server->auth_manager_.get();
    
    if (server->register_interfaces() < 0) {
        syslog(LOG_ERR, "Failed to register D-Bus interfaces");
        return nullptr;
    }

    // Claim the bus name
    ret = sd_bus_request_name(bus, "org.freedesktop.FaceID", 0);
    if (ret < 0) {
        syslog(LOG_ERR, "Failed to request bus name: %s", strerror(-ret));
        return nullptr;
    }

    syslog(LOG_INFO, "✓ D-Bus service registered (org.freedesktop.FaceID)");
    return server;
}

DBusServer::~DBusServer() {
    if (bus_) {
        sd_bus_release_name(bus_, "org.freedesktop.FaceID");
        sd_bus_unref(bus_);
    }
}

bool DBusServer::wait_event(int timeout_ms) {
    if (!bus_) return false;
    
    return sd_bus_wait(bus_, timeout_ms * 1000) > 0;
}

int DBusServer::register_interfaces() {
    // Register the Device object and its virtual table
    int ret = sd_bus_add_object_vtable(bus_,
        nullptr,                                    // slot (we don't use it)
        "/org/freedesktop/FaceID/Device",          // object path
        "org.freedesktop.FaceID.Device",           // interface name
        device_vtable,                              // virtual table with methods
        this);                                      // userdata (passed to handlers)
    
    if (ret < 0) {
        syslog(LOG_ERR, "Failed to register object vtable: %s", strerror(-ret));
        return ret;
    }
    
    syslog(LOG_INFO, "✓ D-Bus interface registered with method handlers");
    return 0;
}
