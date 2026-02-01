#pragma once

#include <yaml-cpp/yaml.h>
#include <string>
#include <map>
#include <vector>
#include <memory>

/**
 * FaceID Configuration System
 * 
 * 3-tier loading strategy:
 * 1. Hardcoded defaults (in code)
 * 2. User config: ~/.config/faceid/config.yaml
 * 3. Advanced tuning: ~/.config/faceid/advanced.yaml (optional)
 * 4. Expert settings: ~/.config/faceid/expert.yaml (optional)
 * 
 * Later files override earlier ones.
 */

struct CameraConfig {
    std::string device = "auto";  // "auto" or device path like "/dev/video2"
    int width = 640;
    int height = 360;
    int fps = 15;
};

struct LockscreenAuthConfig {
    bool fingerprint = false;
};

struct SudoAuthConfig {
    bool fingerprint = true;
};

struct AuthenticationConfig {
    bool face = true;
    bool fingerprint = true;
    int timeout = 5;  // seconds
    LockscreenAuthConfig lockscreen;
    SudoAuthConfig sudo;
};

struct RecognitionConfig {
    double threshold = 0.4;
    int temporal_frames = 3;
};

struct PreprocessingConfig {
    bool gamma_correction = true;
    bool adaptive_clahe = true;
    double clahe_clip_limit = 2.0;
};

struct QualityConfig {
    int min_face_size = 80;
    int max_yaw_angle = 30;
    int max_pitch_angle = 20;
    int max_roll_angle = 20;
    double min_confidence = 0.8;
};

struct PresenceDetectionConfig {
    bool enabled = false;
    int lock_after = 30;  // seconds
    int scan_interval = 2;
    int stability_frames = 3;
};

struct PerformanceConfig {
    int threads = 2;
};

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

struct LoggingConfig {
    LogLevel level = LogLevel::INFO;
};

/**
 * Main configuration container
 */
class Config {
public:
    CameraConfig camera;
    AuthenticationConfig authentication;
    RecognitionConfig recognition;
    PreprocessingConfig preprocessing;
    QualityConfig quality;
    PresenceDetectionConfig presence;
    PerformanceConfig performance;
    LoggingConfig logging;

    /**
     * Default constructor initializes all values to defaults
     */
    Config() = default;

    /**
     * Load configuration from files with 3-tier fallback
     * Returns: std::unique_ptr<Config> or nullptr on error
     */
    static std::unique_ptr<Config> load();

    /**
     * Get singleton instance (after load())
     */
    static Config& instance();

    /**
     * Convert LogLevel enum to string
     */
    static std::string log_level_to_string(LogLevel level);
    
    /**
     * Convert string to LogLevel enum
     */
    static LogLevel string_to_log_level(const std::string& str);

private:
    /**
     * Load and merge YAML file
     * Returns: true on success, false on error
     */
    bool merge_yaml(const std::string& config_path);

    /**
     * Validate all configuration values
     * Returns: true if valid, false if invalid
     */
    bool validate();

    /**
     * Global singleton instance
     */
    static std::unique_ptr<Config> g_instance;
};
