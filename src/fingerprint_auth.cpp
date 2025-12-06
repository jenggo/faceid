#include "fingerprint_auth.h"
#include "logger.h"
#include <gio/gio.h>
#include <thread>
#include <chrono>
#include <atomic>

namespace faceid {

// State for D-Bus verification
struct VerifyState {
    bool success = false;
    bool completed = false;
    bool device_claimed = false;
    std::string error_message;
    std::string device_path;
    GDBusConnection* connection = nullptr;
    GDBusProxy* device_proxy = nullptr;
    GMainLoop* loop = nullptr;
};

class FingerprintAuth::Impl {
public:
    VerifyState state;
    std::string error_message;
    
    ~Impl() {
        cleanup();
    }
    
    void cleanup() {
        if (state.device_proxy) {
            // Release device if claimed
            if (state.device_claimed) {
                GError* error = nullptr;
                g_dbus_proxy_call_sync(state.device_proxy, "Release",
                                      nullptr, G_DBUS_CALL_FLAGS_NONE,
                                      -1, nullptr, &error);
                if (error) {
                    g_error_free(error);
                }
                state.device_claimed = false;
            }
            g_object_unref(state.device_proxy);
            state.device_proxy = nullptr;
        }
        
        if (state.connection) {
            g_object_unref(state.connection);
            state.connection = nullptr;
        }
    }
};

FingerprintAuth::FingerprintAuth() : impl_(std::make_unique<Impl>()) {
    // Check if fprintd is available via D-Bus
    GError* error = nullptr;
    
    impl_->state.connection = g_bus_get_sync(G_BUS_TYPE_SYSTEM, nullptr, &error);
    if (error) {
        last_error_ = std::string("Failed to connect to system bus: ") + error->message;
        g_error_free(error);
        available_ = false;
        Logger::getInstance().warning("Fingerprint: " + last_error_);
        return;
    }
    
    // Check if fprintd service exists
    GDBusProxy* manager_proxy = g_dbus_proxy_new_sync(
        impl_->state.connection,
        G_DBUS_PROXY_FLAGS_NONE,
        nullptr,
        "net.reactivated.Fprint",
        "/net/reactivated/Fprint/Manager",
        "net.reactivated.Fprint.Manager",
        nullptr,
        &error
    );
    
    if (error) {
        last_error_ = std::string("Failed to connect to fprintd: ") + error->message;
        g_error_free(error);
        available_ = false;
        Logger::getInstance().warning("Fingerprint: " + last_error_);
        return;
    }
    
    // Get default device
    GVariant* result = g_dbus_proxy_call_sync(
        manager_proxy,
        "GetDefaultDevice",
        nullptr,
        G_DBUS_CALL_FLAGS_NONE,
        -1,
        nullptr,
        &error
    );
    
    if (error) {
        last_error_ = std::string("Failed to get fingerprint device: ") + error->message;
        g_error_free(error);
        g_object_unref(manager_proxy);
        available_ = false;
        Logger::getInstance().warning("Fingerprint: " + last_error_);
        return;
    }
    
    // Extract device path
    const gchar* device_path = nullptr;
    g_variant_get(result, "(&o)", &device_path);
    impl_->state.device_path = device_path;
    g_variant_unref(result);
    g_object_unref(manager_proxy);
    
    available_ = true;
    Logger::getInstance().info("Fingerprint authentication available via fprintd");
}

FingerprintAuth::~FingerprintAuth() = default;

bool FingerprintAuth::initialize() {
    if (!available_) {
        return false;
    }
    
    Logger::getInstance().info("Fingerprint device initialized successfully");
    return true;
}

// Signal handler for VerifyStatus and VerifyFingerSelected
static void on_verify_signal(GDBusProxy* proxy, gchar* sender_name, gchar* signal_name,
                             GVariant* parameters, gpointer user_data) {
    VerifyState* state = static_cast<VerifyState*>(user_data);
    
    if (g_strcmp0(signal_name, "VerifyStatus") == 0) {
        const gchar* status = nullptr;
        gboolean done = FALSE;
        g_variant_get(parameters, "(&sb)", &status, &done);
        
        Logger::getInstance().debug(std::string("Verify status: ") + status + " done=" + (done ? "true" : "false"));
        
        if (g_strcmp0(status, "verify-match") == 0) {
            state->success = true;
        } else if (g_strcmp0(status, "verify-no-match") == 0) {
            state->success = false;
        }
        
        if (done) {
            state->completed = true;
            if (state->loop) {
                g_main_loop_quit(state->loop);
            }
        }
    } else if (g_strcmp0(signal_name, "VerifyFingerSelected") == 0) {
        const gchar* finger = nullptr;
        g_variant_get(parameters, "(&s)", &finger);
        Logger::getInstance().debug(std::string("Finger selected: ") + finger);
    }
}

bool FingerprintAuth::authenticate(const std::string& username, int timeout_seconds, std::atomic<bool>& cancel_flag) {
    if (!available_) {
        return false;
    }
    
    Logger::getInstance().debug("Starting fingerprint authentication for user: " + username);
    
    GError* error = nullptr;
    
    // Create device proxy
    impl_->state.device_proxy = g_dbus_proxy_new_sync(
        impl_->state.connection,
        G_DBUS_PROXY_FLAGS_NONE,
        nullptr,
        "net.reactivated.Fprint",
        impl_->state.device_path.c_str(),
        "net.reactivated.Fprint.Device",
        nullptr,
        &error
    );
    
    if (error) {
        Logger::getInstance().warning(std::string("Failed to create device proxy: ") + error->message);
        g_error_free(error);
        return false;
    }
    
    // Check enrolled fingers
    GVariant* props = g_dbus_proxy_call_sync(
        impl_->state.device_proxy,
        "ListEnrolledFingers",
        g_variant_new("(s)", username.c_str()),
        G_DBUS_CALL_FLAGS_NONE,
        -1,
        nullptr,
        &error
    );
    
    if (error) {
        Logger::getInstance().warning(std::string("Failed to list enrolled fingers: ") + error->message);
        g_error_free(error);
        g_object_unref(impl_->state.device_proxy);
        impl_->state.device_proxy = nullptr;
        return false;
    }
    
    GVariantIter* iter = nullptr;
    g_variant_get(props, "(as)", &iter);
    int finger_count = g_variant_iter_n_children(iter);
    g_variant_iter_free(iter);
    g_variant_unref(props);
    
    if (finger_count == 0) {
        Logger::getInstance().warning("No enrolled fingerprints found for user: " + username);
        g_object_unref(impl_->state.device_proxy);
        impl_->state.device_proxy = nullptr;
        return false;
    }
    
    Logger::getInstance().debug(std::string("Found ") + std::to_string(finger_count) + " enrolled finger(s) for " + username);
    
    // Claim device
    GVariant* claim_result = g_dbus_proxy_call_sync(
        impl_->state.device_proxy,
        "Claim",
        g_variant_new("(s)", username.c_str()),
        G_DBUS_CALL_FLAGS_NONE,
        -1,
        nullptr,
        &error
    );
    
    if (error) {
        Logger::getInstance().warning(std::string("Failed to claim device: ") + error->message);
        g_error_free(error);
        g_object_unref(impl_->state.device_proxy);
        impl_->state.device_proxy = nullptr;
        return false;
    }
    
    g_variant_unref(claim_result);
    impl_->state.device_claimed = true;
    
    // Reset state
    impl_->state.success = false;
    impl_->state.completed = false;
    impl_->state.loop = g_main_loop_new(nullptr, FALSE);
    
    // Connect to signals
    gulong signal_id = g_signal_connect(impl_->state.device_proxy, "g-signal",
                                       G_CALLBACK(on_verify_signal), &impl_->state);
    
    // Start verification
    GVariant* verify_result = g_dbus_proxy_call_sync(
        impl_->state.device_proxy,
        "VerifyStart",
        g_variant_new("(s)", "any"), // "any" means match against any enrolled finger
        G_DBUS_CALL_FLAGS_NONE,
        -1,
        nullptr,
        &error
    );
    
    if (error) {
        Logger::getInstance().warning(std::string("Failed to start verification: ") + error->message);
        g_error_free(error);
        g_signal_handler_disconnect(impl_->state.device_proxy, signal_id);
        impl_->cleanup();
        return false;
    }
    
    g_variant_unref(verify_result);
    
    Logger::getInstance().debug("Verification started, waiting for fingerprint...");
    
    // Run main loop in a separate thread
    std::thread loop_thread([this]() {
        g_main_loop_run(impl_->state.loop);
    });
    
    // Wait for completion or timeout/cancel
    auto start = std::chrono::steady_clock::now();
    
    while (!impl_->state.completed && !cancel_flag.load()) {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start).count();
        
        if (elapsed >= timeout_seconds) {
            Logger::getInstance().debug("Fingerprint authentication timeout");
            break;
        }
        
        // Small sleep to avoid busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Stop verification
    GVariant* stop_result = g_dbus_proxy_call_sync(
        impl_->state.device_proxy,
        "VerifyStop",
        nullptr,
        G_DBUS_CALL_FLAGS_NONE,
        -1,
        nullptr,
        nullptr // Ignore errors on stop
    );
    
    if (stop_result) {
        g_variant_unref(stop_result);
    }
    
    // Cleanup
    if (impl_->state.loop) {
        if (g_main_loop_is_running(impl_->state.loop)) {
            g_main_loop_quit(impl_->state.loop);
        }
    }
    
    if (loop_thread.joinable()) {
        loop_thread.join();
    }
    
    if (impl_->state.loop) {
        g_main_loop_unref(impl_->state.loop);
        impl_->state.loop = nullptr;
    }
    
    g_signal_handler_disconnect(impl_->state.device_proxy, signal_id);
    impl_->cleanup();
    
    if (cancel_flag.load()) {
        Logger::getInstance().debug("Fingerprint authentication cancelled by flag");
        return false;
    }
    
    if (impl_->state.success) {
        Logger::getInstance().info("Fingerprint authentication successful");
        return true;
    }
    
    Logger::getInstance().debug("Fingerprint authentication failed");
    return false;
}

bool FingerprintAuth::isAvailable() const {
    return available_;
}

std::string FingerprintAuth::getLastError() const {
    return last_error_;
}

} // namespace faceid
