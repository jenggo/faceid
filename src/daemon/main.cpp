#include <iostream>
#include <cstdlib>
#include <cstring>
#include <syslog.h>
#include <systemd/sd-daemon.h>
#include <systemd/sd-bus.h>
#include <unistd.h>
#include <signal.h>
#include "validation.h"
#include "dbus_server.h"

/**
 * FaceID Daemon - User-level D-Bus service for biometric authentication
 * 
 * This daemon runs as a systemd --user service and provides D-Bus methods for:
 * - Face/fingerprint authentication
 * - Face enrollment
 * - Presence detection and auto-lock
 * - Model management
 * 
 * Architecture:
 * - Runs as unprivileged user (rejects root)
 * - Uses D-Bus session bus (user-isolated)
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
    
    // Check 1: Must not be root
    if (validation::is_root()) {
        std::cerr << "❌ faceid-daemon MUST NOT run as root\n"
                  << "   This is a user-level service (systemd --user)\n"
                  << "   Enable it with: systemctl --user enable faceid-daemon\n";
        syslog(LOG_ERR, "Rejected: Running as root");
        return EXIT_FAILURE;
    }
    syslog(LOG_INFO, "✓ UID validation passed (UID: %d)", getuid());

    // Check 2: Must have user session
    if (!validation::has_user_session()) {
        std::cerr << "❌ No XDG_RUNTIME_DIR - not in a user session\n"
                  << "   faceid-daemon requires an active user session\n";
        syslog(LOG_ERR, "Rejected: No user session (XDG_RUNTIME_DIR missing)");
        return EXIT_FAILURE;
    }
    std::string xdg_runtime = validation::get_xdg_runtime_dir();
    syslog(LOG_INFO, "✓ User session validation passed (XDG_RUNTIME_DIR: %s)", 
           xdg_runtime.c_str());

    // Check 3: Verify config path is user-level
    std::string config_dir = validation::get_config_dir();
    if (!validation::is_user_level_path(config_dir)) {
        std::cerr << "❌ Config path is system-wide: " << config_dir << "\n"
                  << "   Expected user-level path like ~/.config/faceid/\n";
        syslog(LOG_ERR, "Rejected: Config path is system-wide: %s", config_dir.c_str());
        return EXIT_FAILURE;
    }
    syslog(LOG_INFO, "✓ Config path validation passed (config_dir: %s)", 
           config_dir.c_str());

    // ═════════════════════════════════════════════════════════════════════
    // INITIALIZATION PHASE
    // ═════════════════════════════════════════════════════════════════════

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
        // TODO: Process D-Bus events and handle timeout
        sleep(1);
    }

    // ═════════════════════════════════════════════════════════════════════
    // SHUTDOWN PHASE
    // ═════════════════════════════════════════════════════════════════════

    syslog(LOG_INFO, "Shutting down gracefully");
    
    // TODO: Cleanup resources
    
    closelog();
    return EXIT_SUCCESS;
}
