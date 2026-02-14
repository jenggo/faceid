#include "config.h"
#include "validation.h"
#include <syslog.h>
#include <filesystem>
#include <fstream>

std::unique_ptr<Config> Config::g_instance = nullptr;

std::unique_ptr<Config> Config::load() {
    Config config;  // Create default config
    
    // Load config files in order: tier 1 → tier 2 → tier 3
    // Each later file overrides earlier ones
    
    std::string config_dir = "/etc/faceid";
    
    // Tier 1: Minimal config (required, but missing is OK - use defaults)
    std::string config_file = config_dir + "/config.yaml";
    if (std::filesystem::exists(config_file)) {
        syslog(LOG_INFO, "Loading config: %s", config_file.c_str());
        if (!config.merge_yaml(config_file)) {
            syslog(LOG_ERR, "Failed to parse config: %s", config_file.c_str());
            return nullptr;
        }
    } else {
        syslog(LOG_INFO, "No config.yaml found, using defaults");
    }
    
    // Tier 2: Advanced config (optional)
    std::string advanced_file = config_dir + "/advanced.yaml";
    if (std::filesystem::exists(advanced_file)) {
        syslog(LOG_INFO, "Loading advanced config: %s", advanced_file.c_str());
        if (!config.merge_yaml(advanced_file)) {
            syslog(LOG_ERR, "Failed to parse advanced config: %s", advanced_file.c_str());
            return nullptr;
        }
    }
    
    // Tier 3: Expert config (optional)
    std::string expert_file = config_dir + "/expert.yaml";
    if (std::filesystem::exists(expert_file)) {
        syslog(LOG_INFO, "Loading expert config: %s", expert_file.c_str());
        if (!config.merge_yaml(expert_file)) {
            syslog(LOG_ERR, "Failed to parse expert config: %s", expert_file.c_str());
            return nullptr;
        }
    }
    
    // Validate all settings
    if (!config.validate()) {
        syslog(LOG_ERR, "Configuration validation failed");
        return nullptr;
    }
    
    syslog(LOG_INFO, "Configuration loaded successfully");
    g_instance = std::make_unique<Config>(config);
    return std::make_unique<Config>(config);
}

Config& Config::instance() {
    if (!g_instance) {
        syslog(LOG_WARNING, "Config instance not initialized, using defaults");
        g_instance = std::make_unique<Config>();
    }
    return *g_instance;
}

std::string Config::log_level_to_string(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG:   return "debug";
        case LogLevel::INFO:    return "info";
        case LogLevel::WARNING: return "warning";
        case LogLevel::ERROR:   return "error";
        default:                return "info";
    }
}

LogLevel Config::string_to_log_level(const std::string& str) {
    if (str == "debug")   return LogLevel::DEBUG;
    if (str == "info")    return LogLevel::INFO;
    if (str == "warning") return LogLevel::WARNING;
    if (str == "error")   return LogLevel::ERROR;
    return LogLevel::INFO;  // default
}

bool Config::merge_yaml(const std::string& config_path) {
    try {
        YAML::Node yaml = YAML::LoadFile(config_path);
        
        // Camera configuration
        if (yaml["camera"]) {
            if (yaml["camera"]["device"])
                camera.device = yaml["camera"]["device"].as<std::string>();
            if (yaml["camera"]["width"])
                camera.width = yaml["camera"]["width"].as<int>();
            if (yaml["camera"]["height"])
                camera.height = yaml["camera"]["height"].as<int>();
            if (yaml["camera"]["fps"])
                camera.fps = yaml["camera"]["fps"].as<int>();
        }
        
        // Authentication configuration
        if (yaml["authentication"]) {
            if (yaml["authentication"]["face"])
                authentication.face = yaml["authentication"]["face"].as<bool>();
            if (yaml["authentication"]["fingerprint"])
                authentication.fingerprint = yaml["authentication"]["fingerprint"].as<bool>();
            if (yaml["authentication"]["timeout"])
                authentication.timeout = yaml["authentication"]["timeout"].as<int>();
            
            if (yaml["authentication"]["lockscreen"]) {
                if (yaml["authentication"]["lockscreen"]["fingerprint"])
                    authentication.lockscreen.fingerprint = 
                        yaml["authentication"]["lockscreen"]["fingerprint"].as<bool>();
            }
            
            if (yaml["authentication"]["sudo"]) {
                if (yaml["authentication"]["sudo"]["fingerprint"])
                    authentication.sudo.fingerprint = 
                        yaml["authentication"]["sudo"]["fingerprint"].as<bool>();
            }
        }
        
        // Recognition configuration
        if (yaml["recognition"]) {
            if (yaml["recognition"]["threshold"])
                recognition.threshold = yaml["recognition"]["threshold"].as<double>();
            if (yaml["recognition"]["temporal_frames"])
                recognition.temporal_frames = yaml["recognition"]["temporal_frames"].as<int>();
            if (yaml["recognition"]["min_face_quality"])
                recognition.min_face_quality = yaml["recognition"]["min_face_quality"].as<double>();
            if (yaml["recognition"]["timeout"])
                recognition.timeout = yaml["recognition"]["timeout"].as<int>();
            if (yaml["recognition"]["num_threads"])
                recognition.num_threads = yaml["recognition"]["num_threads"].as<int>();
            if (yaml["recognition"]["enable_head_pose_correction"])
                recognition.enable_head_pose_correction = yaml["recognition"]["enable_head_pose_correction"].as<bool>();
            if (yaml["recognition"]["enable_gamma_correction"])
                recognition.enable_gamma_correction = yaml["recognition"]["enable_gamma_correction"].as<bool>();
            if (yaml["recognition"]["enable_brightness_normalization"])
                recognition.enable_brightness_normalization = yaml["recognition"]["enable_brightness_normalization"].as<bool>();
            if (yaml["recognition"]["enable_adaptive_clahe"])
                recognition.enable_adaptive_clahe = yaml["recognition"]["enable_adaptive_clahe"].as<bool>();
            if (yaml["recognition"]["debug_face_quality"])
                recognition.debug_face_quality = yaml["recognition"]["debug_face_quality"].as<bool>();
        }
        
        // Face detection configuration
        if (yaml["face_detection"]) {
            if (yaml["face_detection"]["confidence_threshold"])
                face_detection.confidence_threshold = yaml["face_detection"]["confidence_threshold"].as<double>();
            if (yaml["face_detection"]["tracking_interval"])
                face_detection.tracking_interval = yaml["face_detection"]["tracking_interval"].as<int>();
        }
        
        // Preprocessing configuration
        if (yaml["preprocessing"]) {
            if (yaml["preprocessing"]["gamma_correction"])
                preprocessing.gamma_correction = yaml["preprocessing"]["gamma_correction"].as<bool>();
            if (yaml["preprocessing"]["adaptive_clahe"])
                preprocessing.adaptive_clahe = yaml["preprocessing"]["adaptive_clahe"].as<bool>();
            if (yaml["preprocessing"]["clahe_clip_limit"])
                preprocessing.clahe_clip_limit = yaml["preprocessing"]["clahe_clip_limit"].as<double>();
            if (yaml["preprocessing"]["brightness_normalization"])
                preprocessing.brightness_normalization = yaml["preprocessing"]["brightness_normalization"].as<bool>();
            if (yaml["preprocessing"]["enable_gamma_correction"])
                preprocessing.enable_gamma_correction = yaml["preprocessing"]["enable_gamma_correction"].as<bool>();
            if (yaml["preprocessing"]["enable_brightness_normalization"])
                preprocessing.enable_brightness_normalization = yaml["preprocessing"]["enable_brightness_normalization"].as<bool>();
            if (yaml["preprocessing"]["enable_adaptive_clahe"])
                preprocessing.enable_adaptive_clahe = yaml["preprocessing"]["enable_adaptive_clahe"].as<bool>();
        }
        
        // Quality thresholds
        if (yaml["quality_thresholds"]) {
            if (yaml["quality_thresholds"]["min_face_size"])
                quality.min_face_size = yaml["quality_thresholds"]["min_face_size"].as<int>();
            if (yaml["quality_thresholds"]["max_yaw_angle"])
                quality.max_yaw_angle = yaml["quality_thresholds"]["max_yaw_angle"].as<int>();
            if (yaml["quality_thresholds"]["max_pitch_angle"])
                quality.max_pitch_angle = yaml["quality_thresholds"]["max_pitch_angle"].as<int>();
            if (yaml["quality_thresholds"]["max_roll_angle"])
                quality.max_roll_angle = yaml["quality_thresholds"]["max_roll_angle"].as<int>();
            if (yaml["quality_thresholds"]["min_confidence"])
                quality.min_confidence = yaml["quality_thresholds"]["min_confidence"].as<double>();
        }
        
        // Temporal smoothing configuration
        if (yaml["temporal_smoothing"]) {
            if (yaml["temporal_smoothing"]["enabled"])
                temporal_smoothing.enabled = yaml["temporal_smoothing"]["enabled"].as<bool>();
            if (yaml["temporal_smoothing"]["frames"])
                temporal_smoothing.frames = yaml["temporal_smoothing"]["frames"].as<int>();
            if (yaml["temporal_smoothing"]["required_matches"])
                temporal_smoothing.required_matches = yaml["temporal_smoothing"]["required_matches"].as<int>();
        }
        
        // Enrollment configuration
        if (yaml["enrollment"]) {
            if (yaml["enrollment"]["extended_enrollment"])
                enrollment.extended_enrollment = yaml["enrollment"]["extended_enrollment"].as<bool>();
        }
        
        // Presence detection
        if (yaml["presence_detection"]) {
            if (yaml["presence_detection"]["enabled"])
                presence.enabled = yaml["presence_detection"]["enabled"].as<bool>();
            if (yaml["presence_detection"]["lock_after"])
                presence.lock_after = yaml["presence_detection"]["lock_after"].as<int>();
            if (yaml["presence_detection"]["scan_interval"])
                presence.scan_interval = yaml["presence_detection"]["scan_interval"].as<int>();
        }
        
         // Performance
         if (yaml["performance"]) {
             if (yaml["performance"]["threads"])
                 performance.threads = yaml["performance"]["threads"].as<int>();
         }
         
         // Logging
        if (yaml["logging"]) {
            if (yaml["logging"]["level"]) {
                std::string level_str = yaml["logging"]["level"].as<std::string>();
                logging.level = string_to_log_level(level_str);
            }
        }
        
        return true;
    } catch (const YAML::Exception& e) {
        syslog(LOG_ERR, "YAML parse error: %s", e.what());
        return false;
    }
}

bool Config::validate() {
    // Validate camera settings
    if (camera.width < 320 || camera.width > 1920) {
        syslog(LOG_ERR, "Invalid camera.width: %d (must be 320-1920)", camera.width);
        return false;
    }
    
    if (camera.height < 240 || camera.height > 1080) {
        syslog(LOG_ERR, "Invalid camera.height: %d (must be 240-1080)", camera.height);
        return false;
    }
    
    if (camera.fps < 5 || camera.fps > 60) {
        syslog(LOG_ERR, "Invalid camera.fps: %d (must be 5-60)", camera.fps);
        return false;
    }
    
    // Validate authentication settings
    if (authentication.timeout < 1 || authentication.timeout > 60) {
        syslog(LOG_ERR, "Invalid authentication.timeout: %d (must be 1-60)", 
               authentication.timeout);
        return false;
    }
    
    // Validate recognition settings
    if (recognition.threshold < 0.0 || recognition.threshold > 1.0) {
        syslog(LOG_ERR, "Invalid recognition.threshold: %f (must be 0.0-1.0)", 
               recognition.threshold);
        return false;
    }
    
    if (recognition.temporal_frames < 1 || recognition.temporal_frames > 10) {
        syslog(LOG_ERR, "Invalid recognition.temporal_frames: %d (must be 1-10)", 
               recognition.temporal_frames);
        return false;
    }
    
     // Validate presence detection
     if (presence.lock_after < 10 || presence.lock_after > 300) {
         syslog(LOG_ERR, "Invalid presence_detection.lock_after: %d (must be 10-300)", 
                presence.lock_after);
         return false;
     }
     
     syslog(LOG_INFO, "✓ Configuration validation passed");
     return true;
}
