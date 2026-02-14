#include "path_utils.h"
#include "logger.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>
#include <sstream>

namespace path_utils {

// ============================================================================
// Username Validation
// ============================================================================

bool validate_username(const std::string& username, std::string* error_msg) {
    // Empty username is invalid
    if (username.empty()) {
        if (error_msg) *error_msg = "Username cannot be empty";
        return false;
    }
    
    // Check for path traversal attempts
    if (username.find('/') != std::string::npos) {
        if (error_msg) *error_msg = "Username cannot contain '/'";
        faceid::Logger::getInstance().error("SECURITY: Username contains path separator: " + username);
        return false;
    }
    
    if (username.find("..") != std::string::npos) {
        if (error_msg) *error_msg = "Username cannot contain '..'";
        faceid::Logger::getInstance().error("SECURITY: Username contains directory traversal: " + username);
        return false;
    }
    
    // Check for leading dot (hidden directory attempt)
    if (username[0] == '.') {
        if (error_msg) *error_msg = "Username cannot start with '.'";
        faceid::Logger::getInstance().error("SECURITY: Username starts with dot: " + username);
        return false;
    }
    
    // Check length limit
    if (username.length() > 32) {
        if (error_msg) *error_msg = "Username too long (max 32 chars)";
        return false;
    }
    
    return true;
}

// ============================================================================
// Path Functions (FHS Compliant)
// ============================================================================

std::string get_models_dir() {
    // FHS compliant: /var/lib/faceid/models
    return std::string(FACEID_MODELS);
}

std::string get_user_faces_dir(const std::string& username) {
    // Validate username first
    std::string error;
    if (!validate_username(username, &error)) {
        faceid::Logger::getInstance().error("Invalid username for faces dir: " + error);
        return "";
    }
    
    // FHS compliant: /var/lib/faceid/faces/<username>
    std::ostringstream oss;
    oss << FACEID_FACES << "/" << username;
    return oss.str();
}

std::string get_user_fingerprints_dir(const std::string& username) {
    // Validate username first
    std::string error;
    if (!validate_username(username, &error)) {
        faceid::Logger::getInstance().error("Invalid username for fingerprints dir: " + error);
        return "";
    }
    
    // FHS compliant: /var/lib/faceid/fingerprints/<username>
    std::ostringstream oss;
    oss << FACEID_FINGERPRINTS << "/" << username;
    return oss.str();
}

// ============================================================================
// Directory Creation
// ============================================================================

bool ensure_faceid_directories(std::string* out_err) {
    auto create_dir = [](const char* path, mode_t mode, std::string* err) -> bool {
        struct stat st;
        if (stat(path, &st) == 0) {
            if (!S_ISDIR(st.st_mode)) {
                if (err) *err = std::string("Path exists but is not a directory: ") + path;
                return false;
            }
            // Directory exists, ensure correct permissions
            chmod(path, mode);
            return true;
        }
        
        // Create directory
        if (mkdir(path, mode) != 0) {
            if (err) {
                *err = std::string("Failed to create directory ") + path + ": " + std::strerror(errno);
            }
            return false;
        }
        
        faceid::Logger::getInstance().info("Created directory: " + std::string(path));
        return true;
    };
    
    // Create base directories with 0755 (readable by all, writable by root)
    if (!create_dir(FACEID_VAR_LIB, 0755, out_err)) return false;
    
    // Create models directory with 0755 (world-readable for model files)
    if (!create_dir(FACEID_MODELS, 0755, out_err)) return false;
    
    // Create faces directory with 0700 (biometric data - owner only)
    if (!create_dir(FACEID_FACES, 0700, out_err)) return false;
    
    // Create fingerprints directory with 0700 (biometric data - owner only)
    if (!create_dir(FACEID_FINGERPRINTS, 0700, out_err)) return false;
    
    // Create log directory with 0755
    if (!create_dir(FACEID_VAR_LOG, 0755, out_err)) return false;
    
    faceid::Logger::getInstance().info("FHS-compliant FaceID directories created successfully");
    return true;
}

} // namespace path_utils

// ============================================================================
// Legacy Compatibility Functions (DEPRECATED)
// ============================================================================
// These functions are kept for backward compatibility during transition.
// They will be removed in a future release.

std::string get_user_models_dir() {
    faceid::Logger::getInstance().warning("DEPRECATED: get_user_models_dir() called. Use get_models_dir() instead.");
    return path_utils::get_models_dir();
}

bool ensure_user_directories(std::string* out_err, const std::string& username) {
    faceid::Logger::getInstance().warning("DEPRECATED: ensure_user_directories() called. Use ensure_faceid_directories() instead.");
    
    // First ensure base directories
    if (!path_utils::ensure_faceid_directories(out_err)) {
        return false;
    }
    
    // If username provided, ensure their face directory exists
    if (!username.empty()) {
        std::string faces_dir = path_utils::get_user_faces_dir(username);
        if (faces_dir.empty()) {
            if (out_err) *out_err = "Invalid username: " + username;
            return false;
        }
        
        struct stat st;
        if (stat(faces_dir.c_str(), &st) != 0) {
            if (mkdir(faces_dir.c_str(), 0700) != 0) {
                if (out_err) {
                    *out_err = std::string("Failed to create faces dir: ") + std::strerror(errno);
                }
                return false;
            }
            faceid::Logger::getInstance().info("Created user face directory: " + faces_dir);
        }
    }
    
    return true;
}
