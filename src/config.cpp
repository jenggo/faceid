#include "config.h"
#include "logger.h"
#include <fstream>
#include <algorithm>

namespace faceid {

Config& Config::getInstance() {
    static Config instance;
    return instance;
}

std::string Config::trim(const std::string& str) const {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

bool Config::load(const std::string& path) {
    validation_errors_.clear();
    
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    std::string current_section;

    while (std::getline(file, line)) {
        line = trim(line);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#' || line[0] == ';') {
            continue;
        }

        // Section header
        if (line[0] == '[' && line.back() == ']') {
            current_section = line.substr(1, line.length() - 2);
            continue;
        }

        // Key-value pair
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = trim(line.substr(0, pos));
            std::string value = trim(line.substr(pos + 1));
            data_[current_section][key] = value;
        }
    }
    
    // Validate configuration
    bool valid = validate();
    
    // Log validation errors
    if (!validation_errors_.empty()) {
        Logger::getInstance().warning("Configuration validation found " + 
            std::to_string(validation_errors_.size()) + " issue(s):");
        for (const auto& error : validation_errors_) {
            Logger::getInstance().warning("  - " + error);
        }
    }

    return valid;
}

std::optional<std::string> Config::getString(const std::string& section, const std::string& key) const {
    auto section_it = data_.find(section);
    if (section_it == data_.end()) {
        return std::nullopt;
    }

    auto key_it = section_it->second.find(key);
    if (key_it == section_it->second.end()) {
        return std::nullopt;
    }

    return key_it->second;
}

std::optional<int> Config::getInt(const std::string& section, const std::string& key) const {
    auto value = getString(section, key);
    if (!value) return std::nullopt;
    
    try {
        return std::stoi(*value);
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<double> Config::getDouble(const std::string& section, const std::string& key) const {
    auto value = getString(section, key);
    if (!value) return std::nullopt;
    
    try {
        return std::stod(*value);
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<bool> Config::getBool(const std::string& section, const std::string& key) const {
    auto value = getString(section, key);
    if (!value) return std::nullopt;
    
    std::string lower = *value;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "true" || lower == "yes" || lower == "1" || lower == "on") {
        return true;
    } else if (lower == "false" || lower == "no" || lower == "0" || lower == "off") {
        return false;
    }
    
    return std::nullopt;
}

void Config::set(const std::string& section, const std::string& key, const std::string& value) {
    data_[section][key] = value;
}

bool Config::save(const std::string& path) {
    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }

    for (const auto& [section, keys] : data_) {
        file << "[" << section << "]\n";
        for (const auto& [key, value] : keys) {
            file << key << " = " << value << "\n";
        }
        file << "\n";
    }

    return true;
}

bool Config::validateInt(const std::string& section, const std::string& key, int min_val, int max_val) {
    auto value = getInt(section, key);
    if (!value.has_value()) {
        return true;  // Optional value, not set
    }
    
    if (*value < min_val || *value > max_val) {
        validation_errors_.push_back(
            "[" + section + "]." + key + " = " + std::to_string(*value) + 
            " is out of range [" + std::to_string(min_val) + ", " + std::to_string(max_val) + "]"
        );
        return false;
    }
    
    return true;
}

bool Config::validateDouble(const std::string& section, const std::string& key, double min_val, double max_val) {
    auto value = getDouble(section, key);
    if (!value.has_value()) {
        return true;  // Optional value, not set
    }
    
    if (*value < min_val || *value > max_val) {
        validation_errors_.push_back(
            "[" + section + "]." + key + " = " + std::to_string(*value) + 
            " is out of range [" + std::to_string(min_val) + ", " + std::to_string(max_val) + "]"
        );
        return false;
    }
    
    return true;
}

bool Config::validate() {
    bool all_valid = true;
    
    // Check that required sections exist
    const std::vector<std::string> required_sections = {
        "camera", "recognition", "authentication"
    };
    
    for (const auto& section : required_sections) {
        if (data_.find(section) == data_.end()) {
            validation_errors_.push_back("Missing required config section: [" + section + "]");
            all_valid = false;
        }
    }
    
    // Camera validation
    all_valid &= validateInt("camera", "width", 160, 3840);
    all_valid &= validateInt("camera", "height", 120, 2160);
    
    // Recognition validation
    all_valid &= validateDouble("recognition", "threshold", 0.0, 1.0);
    all_valid &= validateInt("recognition", "timeout", 1, 60);
    
    // Face detection validation
    all_valid &= validateInt("face_detection", "tracking_interval", 0, 30);
    
    // Authentication validation
    all_valid &= validateInt("authentication", "lock_screen_delay_ms", 0, 10000);
    all_valid &= validateInt("authentication", "fingerprint_delay_ms", 0, 5000);
    all_valid &= validateInt("authentication", "frame_count", 1, 20);
    
    // Presence detection validation
    all_valid &= validateInt("presence_detection", "inactive_threshold_seconds", 1, 3600);
    all_valid &= validateInt("presence_detection", "scan_interval_seconds", 1, 300);
    all_valid &= validateInt("presence_detection", "max_scan_failures", 1, 20);
    all_valid &= validateInt("presence_detection", "max_idle_time_minutes", 1, 240);
    all_valid &= validateInt("presence_detection", "mouse_jitter_threshold_ms", 0, 5000);
    all_valid &= validateDouble("presence_detection", "shutter_brightness_threshold", 0.0, 255.0);
    all_valid &= validateDouble("presence_detection", "shutter_variance_threshold", 0.0, 100.0);
    all_valid &= validateInt("presence_detection", "shutter_timeout_minutes", 1, 60);
    
    // No peek validation
    all_valid &= validateInt("no_peek", "min_face_distance_pixels", 10, 500);
    all_valid &= validateDouble("no_peek", "min_face_size_percent", 0.01, 0.5);
    all_valid &= validateInt("no_peek", "peek_detection_delay_seconds", 0, 30);
    all_valid &= validateInt("no_peek", "unblank_delay_seconds", 0, 30);
    
    // Schedule validation
    auto time_start = getInt("schedule", "time_start");
    auto time_end = getInt("schedule", "time_end");
    
    if (time_start.has_value()) {
        if (*time_start < 0 || *time_start > 2359) {
            validation_errors_.push_back("[schedule].time_start must be in format HHMM (0000-2359)");
            all_valid = false;
        } else {
            // Validate hour and minute components
            int hour = *time_start / 100;
            int minute = *time_start % 100;
            if (hour > 23 || minute > 59) {
                validation_errors_.push_back("[schedule].time_start has invalid hour or minute");
                all_valid = false;
            }
        }
    }
    
    if (time_end.has_value()) {
        if (*time_end < 0 || *time_end > 2359) {
            validation_errors_.push_back("[schedule].time_end must be in format HHMM (0000-2359)");
            all_valid = false;
        } else {
            // Validate hour and minute components
            int hour = *time_end / 100;
            int minute = *time_end % 100;
            if (hour > 23 || minute > 59) {
                validation_errors_.push_back("[schedule].time_end has invalid hour or minute");
                all_valid = false;
            }
        }
    }
    
    // Logical consistency: time_start should be <= time_end
    if (time_start.has_value() && time_end.has_value()) {
        if (*time_start > *time_end && *time_end != 0) {
            validation_errors_.push_back(
                "[schedule].time_start (" + std::to_string(*time_start) + 
                ") must be <= time_end (" + std::to_string(*time_end) + ")");
            all_valid = false;
        }
    }
    
    return all_valid;
}

} // namespace faceid
