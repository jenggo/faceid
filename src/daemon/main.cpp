#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <syslog.h>
#include <systemd/sd-daemon.h>
#include <systemd/sd-bus.h>
#include <unistd.h>
#include <signal.h>
#include "validation.h"
#include "config.h"
#include "dbus_server.h"
#include "path_utils.h"

/**
 * FaceID Daemon - System-level D-Bus service for biometric authentication
 * 
 * This daemon runs as a systemd system service and provides D-Bus methods for:
 * - Face/fingerprint authentication
 * - Face enrollment
 * - Presence detection and auto-lock
 * - Model management
 * 
 * Architecture:
 * - Runs as root (system service)
 * - Uses D-Bus system bus (accessible from PAM and all users)
 * - Pre-loads camera and ML models
 * - Shared resource access for auth/enrollment/presence
 */

namespace {
    // Global bus pointer used in signal handlers
    sd_bus* g_bus = nullptr;
    bool g_running = true;

    void signal_handler(int sig) {
        if (sig == SIGTERM || sig == SIGINT) {
            syslog(LOG_INFO, "Received signal %d, shutting down", sig);
            g_running = false;
        }
    }
}

int main(int argc, char** argv) {
    // Setup syslog (before any logging)
    validation::init_logging("faceid-daemon");
    
    syslog(LOG_INFO, "FaceID Daemon starting (version 2.0.0)");

    // ═════════════════════════════════════════════════════════════════════
    // VALIDATION PHASE
    // ═════════════════════════════════════════════════════════════════════
    
    // Check 1: Must run as root (system service)
    if (!validation::is_root()) {
        std::cerr << "❌ faceid-daemon MUST run as root\n"
                  << "   This is a system-level service\n"
                  << "   Enable it with: systemctl enable faceid\n";
        syslog(LOG_ERR, "Rejected: Not running as root");
        return EXIT_FAILURE;
    }
    syslog(LOG_INFO, "✓ UID validation passed (running as root)");

     // ═════════════════════════════════════════════════════════════════════
     // INITIALIZATION PHASE
     // ═════════════════════════════════════════════════════════════════════

    // Load and validate configuration
    auto config_ptr = Config::load();
    if (!config_ptr) {
        std::cerr << "❌ Failed to load configuration\n";
        syslog(LOG_ERR, "Failed to load configuration");
        return EXIT_FAILURE;
    }
    syslog(LOG_INFO, "✓ Configuration loaded and validated");

    /* 
     * NOTE: We do NOT ensure user directories at startup anymore.
     * The daemon runs as root, so ensure_user_directories() would try to create 
     * /root/.local/share/faceid which fails if /root/.local doesn't exist.
     * Instead, we ensure target user directories on demand (in SaveEmbedding).
     */
    // if (!ensure_user_directories(&mkdir_err)) { ... } removed
    syslog(LOG_INFO, "✓ User data directories will be created on demand");

     // Setup signal handlers
     signal(SIGTERM, signal_handler);
     signal(SIGINT, signal_handler);
     syslog(LOG_INFO, "✓ Signal handlers registered");

    // Initialize D-Bus server
    auto dbus_server = DBusServer::create();
    if (!dbus_server || !dbus_server->is_connected()) {
        std::cerr << "❌ Failed to initialize D-Bus server\n";
        syslog(LOG_ERR, "Failed to initialize D-Bus server");
        return EXIT_FAILURE;
    }
    g_bus = dbus_server->get_bus();

    syslog(LOG_INFO, "✓ Daemon initialization complete (bus name: org.freedesktop.FaceID)");

    // ═════════════════════════════════════════════════════════════════════
    // MAIN EVENT LOOP
    // ═════════════════════════════════════════════════════════════════════

    syslog(LOG_INFO, "Entering main loop");
    while (g_running) {
        // Process D-Bus events with 1-second timeout
        int ret = sd_bus_wait(g_bus, 1000000);  // 1 second in microseconds
        if (ret < 0 && ret != -EINTR) {
            syslog(LOG_ERR, "sd_bus_wait failed: %s", strerror(-ret));
            break;
        }
        
        // Process pending D-Bus messages
        ret = sd_bus_process(g_bus, nullptr);
        if (ret < 0) {
            syslog(LOG_ERR, "sd_bus_process failed: %s", strerror(-ret));
            break;
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // SHUTDOWN PHASE
    // ═════════════════════════════════════════════════════════════════════

    syslog(LOG_INFO, "Shutting down gracefully");
    
    // TODO: Cleanup resources
    
    closelog();
    return EXIT_SUCCESS;
}
