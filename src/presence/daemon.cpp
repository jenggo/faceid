/**
 * FaceID Presence Detection Daemon
 * 
 * Systemd/OpenRC service that monitors user presence through:
 * - Activity detection (keyboard/mouse via X11 idle time)
 * - Face detection (simple detection, no recognition)
 * - Smart guard conditions (lid, camera, screen lock)
 * 
 * State Machine:
 * 1. ACTIVELY_PRESENT: User is active (typing/mouse) - NO SCANNING
 * 2. IDLE_WITH_SCANNING: User idle 30+ seconds - SCAN every 2s
 * 3. AWAY_CONFIRMED: User away (3 failures or 15 min) - LOCK & STOP
 */

#include "detector.h"
#include "guard.h"
#include "../config.h"
#include "../logger.h"
#include "../adaptive_auth.h"
#include "../face_detector.h"
#include "../models/model_cache.h"
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>
#include <getopt.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
    std::atomic<bool> g_running{true};
    std::atomic<bool> g_reload_config{false};
    std::atomic<bool> g_optimization_worker_running{false};
    
    void signalHandler(int signal) {
        if (signal == SIGTERM || signal == SIGINT) {
            faceid::Logger::getInstance().info("Received shutdown signal");
            g_running = false;
            g_optimization_worker_running = false;
        } else if (signal == SIGHUP) {
            faceid::Logger::getInstance().info("Received reload signal");
            g_reload_config = true;
        }
    }
    
    void setupSignalHandlers() {
        struct sigaction sa;
        sa.sa_handler = signalHandler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        
        sigaction(SIGTERM, &sa, nullptr);
        sigaction(SIGINT, &sa, nullptr);
        sigaction(SIGHUP, &sa, nullptr);
    }
    
    void daemonize() {
        pid_t pid = fork();
        if (pid < 0) {
            std::cerr << "Failed to fork daemon process" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (pid > 0) {
            // Parent exits
            exit(EXIT_SUCCESS);
        }
        
        // Child continues
        if (setsid() < 0) {
            exit(EXIT_FAILURE);
        }
        
        // Fork again to prevent acquiring controlling terminal
        pid = fork();
        if (pid < 0) {
            exit(EXIT_FAILURE);
        }
        if (pid > 0) {
            exit(EXIT_SUCCESS);
        }
        
        // Set file permissions
        umask(0);
        
        // Change working directory to root
        if (chdir("/") < 0) {
            exit(EXIT_FAILURE);
        }
        
        // Close standard file descriptors
        close(STDIN_FILENO);
        close(STDOUT_FILENO);
        close(STDERR_FILENO);
    }
    
    void printUsage(const char* program) {
        std::cout << "Usage: " << program << " [OPTIONS]\n"
                  << "\nOptions:\n"
                  << "  -c, --config PATH    Configuration file path (default: /etc/faceid/faceid.conf)\n"
                  << "  -d, --daemon         Run as daemon (fork to background)\n"
                  << "  -h, --help           Show this help message\n"
                  << "  -v, --verbose        Enable verbose logging\n"
                  << "\nSignals:\n"
                  << "  SIGTERM/SIGINT       Graceful shutdown\n"
                  << "  SIGHUP               Reload configuration\n"
                  << std::endl;
    }
    
    struct DaemonConfig {
        std::string config_path = "/etc/faceid/faceid.conf";
        bool daemon_mode = false;
        bool verbose = false;
    };
    
    DaemonConfig parseArguments(int argc, char* argv[]) {
        DaemonConfig config;
        
        static struct option long_options[] = {
            {"config",  required_argument, nullptr, 'c'},
            {"daemon",  no_argument,       nullptr, 'd'},
            {"help",    no_argument,       nullptr, 'h'},
            {"verbose", no_argument,       nullptr, 'v'},
            {nullptr, 0, nullptr, 0}
        };
        
        int opt;
        while ((opt = getopt_long(argc, argv, "c:dhv", long_options, nullptr)) != -1) {
            switch (opt) {
                case 'c':
                    config.config_path = optarg;
                    break;
                case 'd':
                    config.daemon_mode = true;
                    break;
                case 'v':
                    config.verbose = true;
                    break;
                case 'h':
                    printUsage(argv[0]);
                    exit(EXIT_SUCCESS);
                default:
                    printUsage(argv[0]);
                    exit(EXIT_FAILURE);
            }
        }
        
        return config;
    }
    
    bool loadConfiguration(const std::string& config_path) {
        auto& config = faceid::Config::getInstance();
        if (!config.load(config_path)) {
            faceid::Logger::getInstance().error("Failed to load configuration from: " + config_path);
            return false;
        }
        
        faceid::Logger::getInstance().info("Configuration loaded from: " + config_path);
        return true;
    }
    
    // Adaptive authentication optimization worker thread
    // Monitors shared memory for optimization requests from PAM module
    // Runs optimization algorithm in background and updates config
    void optimizationWorkerThread(const std::string& config_path) {
        auto& logger = faceid::Logger::getInstance();
        logger.info("Adaptive authentication optimization worker started");
        
        faceid::AdaptiveAuthManager adaptive_mgr;
        if (!adaptive_mgr.initialize()) {
            logger.error("Failed to initialize adaptive auth manager in worker thread");
            return;
        }
        
        while (g_optimization_worker_running) {
            // Check for optimization request every second
            if (adaptive_mgr.hasOptimizationRequest()) {
                logger.info("Optimization request detected, starting background optimization");
                adaptive_mgr.startOptimization();
                
                try {
                    // Extract frame from shared memory
                    uint8_t* frame_buffer = new uint8_t[faceid::MAX_FRAME_SIZE];
                    int frame_width = 0, frame_height = 0, frame_channels = 0;
                    
                    if (!adaptive_mgr.getFrameData(frame_buffer, frame_width, frame_height, frame_channels)) {
                        logger.error("Failed to extract frame data from shared memory");
                        adaptive_mgr.failOptimization();
                        delete[] frame_buffer;
                        continue;
                    }
                    
                    logger.info("Extracted frame: " + std::to_string(frame_width) + "x" + 
                               std::to_string(frame_height) + "x" + std::to_string(frame_channels));
                    
                    // Initialize face detector
                    faceid::FaceDetector detector;
                    if (!detector.loadModels()) {
                        logger.error("Failed to load face detection models");
                        adaptive_mgr.failOptimization();
                        delete[] frame_buffer;
                        continue;
                    }
                    
                    // Create ImageView from frame buffer
                    faceid::ImageView frame_view(frame_buffer, frame_width, frame_height, frame_channels);
                    
                    // Run binary search for optimal confidence (0.1 - 0.9)
                    float optimal_confidence = -1.0f;
                    float low = 0.10f;
                    float high = 0.90f;
                    
                    // Binary search with 0.05 precision (faster for runtime optimization)
                    while (high - low > 0.05f) {
                        float mid = (low + high) / 2.0f;
                        
                        auto cascade_result = detector.detectFacesCascade(frame_view, false, mid);
                        
                        if (cascade_result.faces.size() == 1) {
                            // Found exactly 1 face - this is good
                            optimal_confidence = mid;
                            high = mid;  // Try lower confidence
                        } else if (cascade_result.faces.size() > 1) {
                            // Too many faces, increase confidence (be more strict)
                            low = mid;
                        } else {
                            // No faces, decrease confidence (be more lenient)
                            high = mid;
                        }
                    }
                    
                    // If we didn't find optimal, use the low value
                    if (optimal_confidence < 0.0f) {
                        auto cascade_result = detector.detectFacesCascade(frame_view, false, low);
                        if (cascade_result.faces.size() >= 1) {
                            optimal_confidence = low;
                        } else {
                            // No face detected even at low confidence - fail
                            logger.error("No face detected even at confidence " + std::to_string(low));
                            adaptive_mgr.failOptimization();
                            delete[] frame_buffer;
                            continue;
                        }
                    }
                    
                    logger.info("Found optimal confidence: " + std::to_string(optimal_confidence));
                    
                    // Now calculate optimal threshold based on enrolled models
                    // Load all user models
                    auto& cache = faceid::ModelCache::getInstance();
                    std::vector<faceid::BinaryFaceModel> all_users = cache.loadAllUsersParallel(4);
                    
                    if (all_users.empty()) {
                        logger.error("No enrolled users found for threshold calculation");
                        adaptive_mgr.failOptimization();
                        delete[] frame_buffer;
                        continue;
                    }
                    
                    // Detect and encode face with optimal confidence
                    auto cascade_result = detector.detectFacesCascade(frame_view, false, optimal_confidence);
                    if (cascade_result.faces.empty()) {
                        logger.error("Face disappeared after optimization");
                        adaptive_mgr.failOptimization();
                        delete[] frame_buffer;
                        continue;
                    }
                    
                    auto encodings = detector.encodeFaces(cascade_result.processed_frame.view(), 
                                                         cascade_result.faces);
                    if (encodings.empty()) {
                        logger.error("Failed to encode face");
                        adaptive_mgr.failOptimization();
                        delete[] frame_buffer;
                        continue;
                    }
                    
                    // Calculate max intra-user distance (distance between user's own faces)
                    float max_intra_distance = 0.0f;
                    for (const auto& user_model : all_users) {
                        for (size_t i = 0; i < user_model.encodings.size(); i++) {
                            for (size_t j = i + 1; j < user_model.encodings.size(); j++) {
                                double dist = detector.compareFaces(user_model.encodings[i], 
                                                                    user_model.encodings[j]);
                                if (dist > max_intra_distance) {
                                    max_intra_distance = dist;
                                }
                            }
                        }
                    }
                    
                    // Set threshold to max_intra_distance * 1.2 (same formula as enrollment)
                    float optimal_threshold = max_intra_distance * 1.2f;
                    
                    // Clamp threshold to reasonable range
                    if (optimal_threshold < 0.3f) optimal_threshold = 0.3f;
                    if (optimal_threshold > 0.8f) optimal_threshold = 0.8f;
                    
                    logger.info("Calculated optimal threshold: " + std::to_string(optimal_threshold) + 
                               " (max_intra_distance: " + std::to_string(max_intra_distance) + ")");
                    
                    // Update config file atomically
                    // Note: updateConfigFile is in cli_helpers.h, we need to implement similar logic here
                    // For now, signal completion with new values (config update will be done separately)
                    
                    logger.info("Optimization complete - confidence: " + std::to_string(optimal_confidence) + 
                               ", threshold: " + std::to_string(optimal_threshold));
                    
                    adaptive_mgr.completeOptimization(optimal_confidence, optimal_threshold);
                    
                    delete[] frame_buffer;
                    
                } catch (const std::exception& e) {
                    logger.error(std::string("Optimization failed with exception: ") + e.what());
                    adaptive_mgr.failOptimization();
                }
            }
            
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        logger.info("Adaptive authentication optimization worker stopped");
    }
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    auto daemon_config = parseArguments(argc, argv);
    
    // Initialize logger
    auto& logger = faceid::Logger::getInstance();
    
    // Daemonize if requested (before logging setup)
    if (daemon_config.daemon_mode) {
        daemonize();
    }
    
    logger.info("FaceID Presence Detection Daemon starting...");
    
    // Load configuration
    if (!loadConfiguration(daemon_config.config_path)) {
        return EXIT_FAILURE;
    }
    
    auto& config = faceid::Config::getInstance();
    
    // Configure logging from config file
    std::string log_file = config.getString("logging", "log_file").value_or("/var/log/faceid.log");
    std::string log_level_str = config.getString("logging", "log_level").value_or("INFO");
    
    logger.setLogFile(log_file);
    
    // Parse log level
    faceid::LogLevel log_level = faceid::LogLevel::INFO;
    if (log_level_str == "DEBUG") {
        log_level = faceid::LogLevel::DEBUG;
    } else if (log_level_str == "INFO") {
        log_level = faceid::LogLevel::INFO;
    } else if (log_level_str == "WARNING") {
        log_level = faceid::LogLevel::WARNING;
    } else if (log_level_str == "ERROR") {
        log_level = faceid::LogLevel::ERROR;
    }
    logger.setLogLevel(log_level);
    
    logger.info("Logging configured: file=" + log_file + ", level=" + log_level_str);
    
    // Read presence detection configuration
    bool enabled = config.getBool("presence_detection", "enabled").value_or(false);
    if (!enabled) {
        logger.info("Presence detection is disabled in configuration");
        return EXIT_SUCCESS;
    }
    
    int inactive_threshold = config.getInt("presence_detection", "inactive_threshold_seconds").value_or(30);
    int scan_interval = config.getInt("presence_detection", "scan_interval_seconds").value_or(2);
    int max_scan_failures = config.getInt("presence_detection", "max_scan_failures").value_or(3);
    int max_idle_time = config.getInt("presence_detection", "max_idle_time_minutes").value_or(15);
    int mouse_jitter_threshold = config.getInt("presence_detection", "mouse_jitter_threshold_ms").value_or(300);
    double shutter_brightness = config.getDouble("presence_detection", "shutter_brightness_threshold").value_or(10.0);
    double shutter_variance = config.getDouble("presence_detection", "shutter_variance_threshold").value_or(2.0);
    int shutter_timeout = config.getInt("presence_detection", "shutter_timeout_minutes").value_or(5);
    std::string camera_device = config.getString("camera", "device").value_or("/dev/video0");
    
    // Read no-peek configuration
    bool no_peek_enabled = config.getBool("no_peek", "enabled").value_or(false);
    int min_face_distance = config.getInt("no_peek", "min_face_distance_pixels").value_or(80);
    double min_face_size = config.getDouble("no_peek", "min_face_size_percent").value_or(0.08);
    int peek_delay = config.getInt("no_peek", "peek_detection_delay_seconds").value_or(2);
    int unblank_delay = config.getInt("no_peek", "unblank_delay_seconds").value_or(3);
    
    // Read schedule configuration
    bool schedule_enabled = config.getBool("schedule", "enabled").value_or(false);
    std::string active_days_str = config.getString("schedule", "active_days").value_or("1,2,3,4,5");
    int time_start = config.getInt("schedule", "time_start").value_or(0);
    int time_end = config.getInt("schedule", "time_end").value_or(2359);
    
    // Parse active days (comma-separated)
    std::vector<int> active_days;
    std::istringstream iss(active_days_str);
    std::string token;
    while (std::getline(iss, token, ',')) {
        try {
            int day = std::stoi(token);
            if (day >= 1 && day <= 7) {
                active_days.push_back(day);
            }
        } catch (...) {
            // Skip invalid values
        }
    }
    if (active_days.empty()) {
        // Default to weekdays if parsing failed
        active_days = {1, 2, 3, 4, 5};
    }
    
    logger.info("Presence detection configuration:");
    logger.info("  Inactive threshold: " + std::to_string(inactive_threshold) + "s");
    logger.info("  Scan interval: " + std::to_string(scan_interval) + "s");
    logger.info("  Max failures: " + std::to_string(max_scan_failures));
    logger.info("  Max idle time: " + std::to_string(max_idle_time) + " min");
    logger.info("  Mouse jitter threshold: " + std::to_string(mouse_jitter_threshold) + "ms");
    logger.info("  Shutter brightness threshold: " + std::to_string(shutter_brightness));
    logger.info("  Shutter variance threshold: " + std::to_string(shutter_variance));
    logger.info("  Shutter timeout: " + std::to_string(shutter_timeout) + " min");
    logger.info("  Camera device: " + camera_device);
    
    logger.info("No-peek detection configuration:");
    logger.info("  Enabled: " + std::string(no_peek_enabled ? "YES" : "NO"));
    logger.info("  Min face distance: " + std::to_string(min_face_distance) + " pixels");
    logger.info("  Min face size: " + std::to_string(min_face_size * 100) + "%");
    logger.info("  Peek detection delay: " + std::to_string(peek_delay) + "s");
    logger.info("  Unblank delay: " + std::to_string(unblank_delay) + "s");
    
    logger.info("Schedule configuration:");
    logger.info("  Enabled: " + std::string(schedule_enabled ? "YES" : "NO"));
    if (schedule_enabled) {
        std::string days_str;
        for (size_t i = 0; i < active_days.size(); i++) {
            days_str += std::to_string(active_days[i]);
            if (i < active_days.size() - 1) days_str += ",";
        }
        logger.info("  Active days: " + days_str + " (1=Mon, 7=Sun)");
        logger.info("  Active time: " + std::to_string(time_start) + "-" + std::to_string(time_end));
    }
    
    // Setup signal handlers
    setupSignalHandlers();
    
    // Initialize presence guard
    faceid::PresenceGuard guard;
    
    // Initialize presence detector
    faceid::PresenceDetector detector(
        camera_device,
        std::chrono::seconds(inactive_threshold),
        std::chrono::seconds(scan_interval),
        max_scan_failures,
        std::chrono::minutes(max_idle_time)
    );
    
    // Configure additional options
    detector.setMouseJitterThreshold(mouse_jitter_threshold);
    detector.setShutterBrightnessThreshold(shutter_brightness);
    detector.setShutterVarianceThreshold(shutter_variance);
    detector.setShutterTimeout(shutter_timeout * 60 * 1000);  // Convert minutes to milliseconds
    
    // Configure no-peek detection
    detector.enableNoPeek(no_peek_enabled);
    detector.setMinFaceDistance(min_face_distance);
    detector.setMinFaceSizePercent(min_face_size);
    detector.setPeekDetectionDelay(peek_delay * 1000);  // Convert seconds to milliseconds
    detector.setUnblankDelay(unblank_delay * 1000);  // Convert seconds to milliseconds
    
    // Configure schedule
    detector.enableSchedule(schedule_enabled);
    detector.setActiveDays(active_days);
    detector.setActiveTimeRange(time_start, time_end);
    
    if (!detector.start()) {
        logger.error("Failed to start presence detector");
        return EXIT_FAILURE;
    }
    
    logger.info("Presence detection daemon started successfully");
    
    // Start adaptive authentication optimization worker thread
    g_optimization_worker_running = true;
    std::thread optimization_worker(optimizationWorkerThread, daemon_config.config_path);
    logger.info("Adaptive authentication optimization worker thread started");
    
    // Main loop: Monitor guard conditions
    while (g_running) {
        // Check if configuration reload requested
        if (g_reload_config) {
            logger.info("Reloading configuration...");
            if (loadConfiguration(daemon_config.config_path)) {
                // Reread enabled flag
                enabled = config.getBool("presence_detection", "enabled").value_or(false);
                if (!enabled) {
                    logger.info("Presence detection disabled via config reload, shutting down...");
                    break;
                }
            }
            g_reload_config = false;
        }
        
        // Check guard conditions (and update them)
        guard.checkGuardConditions();
        
        // Detector automatically handles guard state internally
        // Sleep for 2 seconds between guard checks (aligned with lock state cache)
        // This significantly reduces process spawning overhead
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        // Optionally log statistics periodically (every 5 minutes)
        // Only log if presence detection is actively running (guard conditions met)
        static auto last_stats_time = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::minutes>(now - last_stats_time).count() >= 5) {
            // Only log stats if guard conditions are met (detection is actually running)
            if (guard.shouldRunPresenceDetection()) {
                auto stats = detector.getStatistics();
                logger.info("Presence detection statistics:");
                logger.info("  Total scans: " + std::to_string(stats.totalScans));
                logger.info("  Successful detections: " + std::to_string(stats.facesDetected));
                logger.info("  Failed scans: " + std::to_string(stats.failedScans));
                logger.info("  State transitions: " + std::to_string(stats.stateTransitions));
                logger.info("  Uptime: " + std::to_string(stats.uptimeSeconds / 3600) + "h " + 
                           std::to_string((stats.uptimeSeconds % 3600) / 60) + "m");
            }
            last_stats_time = now;
        }
    }
    
    // Graceful shutdown
    logger.info("Shutting down presence detection daemon...");
    
    // Stop optimization worker thread
    g_optimization_worker_running = false;
    if (optimization_worker.joinable()) {
        optimization_worker.join();
    }
    
    detector.stop();
    logger.info("Daemon stopped");
    
    return EXIT_SUCCESS;
}
