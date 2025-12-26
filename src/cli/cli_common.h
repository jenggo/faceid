#ifndef FACEID_CLI_COMMON_H
#define FACEID_CLI_COMMON_H

/**
 * CLI Common Utilities and Includes
 * 
 * Provides shared headers and utilities for CLI commands
 */

// ========== Standard Library Includes ==========
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <ctime>
#include <cstring>
#include <algorithm>
#include <optional>
#include <dirent.h>
#include <fnmatch.h>
#include <cstdio>

// ========== FaceID Library Includes ==========
#include "../config.h"
#include "../face_detector.h"
#include "../camera.h"
#include "../display.h"
#include "../image.h"
#include "config_paths.h"

// ========== Using Declarations ==========
using namespace faceid;

// ========== Common Constants ==========

/**
 * Default number of face samples to capture for enrollment
 */
constexpr int DEFAULT_ENROLLMENT_SAMPLES = 5;

/**
 * Default camera resolution width
 */
constexpr int DEFAULT_CAMERA_WIDTH = 640;

/**
 * Default camera resolution height
 */
constexpr int DEFAULT_CAMERA_HEIGHT = 480;

/**
 * Default camera device path
 */
const char* const DEFAULT_CAMERA_DEVICE = "/dev/video0";

/**
 * Preview window width for enrollment/testing
 */
constexpr int PREVIEW_WINDOW_WIDTH = 800;

/**
 * Preview window height for enrollment/testing
 */
constexpr int PREVIEW_WINDOW_HEIGHT = 600;

/**
 * Default face recognition threshold (cosine distance)
 * Faces with distance below this are considered matches
 */
constexpr double DEFAULT_RECOGNITION_THRESHOLD = 0.6;

/**
 * Default tracking interval (number of frames between re-detections)
 */
constexpr int DEFAULT_TRACKING_INTERVAL = 10;

// ========== Common Helper Functions ==========

namespace faceid {
namespace cli {

/**
 * Load FaceID configuration from default path
 * 
 * Attempts to load from CONFIG_DIR/faceid.conf
 * Falls back to sensible defaults if file not found
 * 
 * @return Reference to Config singleton with loaded values
 */
inline faceid::Config& loadDefaultConfig() {
    faceid::Config& config = faceid::Config::getInstance();
    std::string config_path = std::string(CONFIG_DIR) + "/faceid.conf";
    config.load(config_path);  // Silently fails if file not found (uses defaults)
    return config;
}

/**
 * Get the models directory path (for NCNN models like sface, RFB-320)
 * 
 * @return Full path to models directory
 */
inline std::string getModelsDir() {
    return std::string(MODELS_DIR);
}

/**
 * Get the faces directory path (for user enrollment data)
 * 
 * @return Full path to faces directory
 */
inline std::string getFacesDir() {
    return std::string(FACES_DIR);
}

/**
 * Get the config directory path
 * 
 * @return Full path to config directory
 */
inline std::string getConfigDir() {
    return std::string(CONFIG_DIR);
}

/**
 * Format timestamp for display
 * 
 * @param timestamp Unix timestamp (seconds since epoch)
 * @return Formatted date/time string (YYYY-MM-DD HH:MM:SS)
 */
inline std::string formatTimestamp(std::time_t timestamp) {
    if (timestamp <= 0) return "unknown";
    
    char buffer[80];
    struct tm* tm_info = std::localtime(&timestamp);
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", tm_info);
    return std::string(buffer);
}

/**
 * Check if a file exists
 * 
 * @param filepath Path to file to check
 * @return true if file exists, false otherwise
 */
inline bool fileExists(const std::string& filepath) {
    std::ifstream f(filepath);
    return f.good();
}

/**
 * Parse face_id from string (validate format)
 * 
 * Face IDs should be alphanumeric with underscores and hyphens
 * 
 * @param face_id String to validate
 * @return true if valid, false otherwise
 */
inline bool isValidFaceId(const std::string& face_id) {
    if (face_id.empty() || face_id.length() > 64) {
        return false;
    }
    
    for (char c : face_id) {
        if (!std::isalnum(c) && c != '_' && c != '-') {
            return false;
        }
    }
    return true;
}

/**
 * Validate username format
 * 
 * Usernames should be alphanumeric with underscores
 * 
 * @param username String to validate
 * @return true if valid, false otherwise
 */
inline bool isValidUsername(const std::string& username) {
    if (username.empty() || username.length() > 32) {
        return false;
    }
    
    for (char c : username) {
        if (!std::isalnum(c) && c != '_') {
            return false;
        }
    }
    return true;
}

/**
 * Find all binary model files for a given username
 * 
 * Searches for both patterns:
 * - username.bin (single model file)
 * - username.*.bin (multi-face model files)
 * 
 * @param username Username to search for
 * @return Vector of full file paths (empty if none found)
 */
inline std::vector<std::string> findUserModelFiles(const std::string& username) {
    std::vector<std::string> files;
    std::string faces_dir = getFacesDir();
    
    DIR* dir = opendir(faces_dir.c_str());
    if (!dir) {
        return files;  // Return empty vector if can't open directory
    }
    
    // Patterns to match: username.bin and username.*.bin
    std::vector<std::string> patterns = {
        username + ".bin",
        username + ".*.bin"
    };
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        
        // Check if filename matches any pattern
        for (const auto& pattern : patterns) {
            if (fnmatch(pattern.c_str(), filename.c_str(), 0) == 0) {
                files.push_back(faces_dir + "/" + filename);
                break;  // Don't check other patterns for this file
            }
        }
    }
    closedir(dir);
    
    // Sort for consistent ordering
    std::sort(files.begin(), files.end());
    return files;
}

/**
 * Get the primary model file for a user
 * 
 * Returns username.bin if it exists, otherwise returns the first
 * username.*.bin file found
 * 
 * @param username Username to search for
 * @return Full path to model file, or empty string if not found
 */
inline std::string getUserModelFile(const std::string& username) {
    std::string faces_dir = getFacesDir();
    std::string primary = faces_dir + "/" + username + ".bin";
    
    // Check primary file first
    if (fileExists(primary)) {
        return primary;
    }
    
    // Fall back to first multi-face file
    auto files = findUserModelFiles(username);
    return files.empty() ? "" : files[0];
}

/**
 * Get all enrolled usernames
 * 
 * Scans the faces directory for all .bin files and extracts usernames
 * 
 * @return Vector of unique usernames
 */
inline std::vector<std::string> getEnrolledUsers() {
    std::vector<std::string> users;
    std::string faces_dir = getFacesDir();
    
    DIR* dir = opendir(faces_dir.c_str());
    if (!dir) {
        return users;
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        
        // Match *.bin files
        if (filename.length() > 4 && filename.substr(filename.length() - 4) == ".bin") {
            // Extract username (everything before first dot)
            size_t dot_pos = filename.find('.');
            if (dot_pos != std::string::npos) {
                std::string username = filename.substr(0, dot_pos);
                
                // Add if not already in list
                if (std::find(users.begin(), users.end(), username) == users.end()) {
                    users.push_back(username);
                }
            }
        }
    }
    closedir(dir);
    
    std::sort(users.begin(), users.end());
    return users;
}

} // namespace cli
} // namespace faceid

#endif // FACEID_CLI_COMMON_H
