#pragma once
#include <string>

// ============================================================================
// FHS-Compliant Path Constants
// ============================================================================
// All FaceID data is stored in FHS-compliant system-wide locations:
// - /etc/faceid/           → Config files only (YAML)
// - /var/lib/faceid/       → All mutable data (models, faces, fingerprints)
// - /var/log/faceid/       → Log files
// ============================================================================

// Base directories (FHS compliant)
constexpr char FACEID_ETC[] = "/etc/faceid";
constexpr char FACEID_VAR_LIB[] = "/var/lib/faceid";
constexpr char FACEID_VAR_LOG[] = "/var/log/faceid";

// Subdirectories for different data types
constexpr char FACEID_MODELS[] = "/var/lib/faceid/models";
constexpr char FACEID_FACES[] = "/var/lib/faceid/faces";
constexpr char FACEID_FINGERPRINTS[] = "/var/lib/faceid/fingerprints";

// ============================================================================
// Path Access Functions (FHS Compliant)
// ============================================================================

namespace path_utils {

// Return the directory where recognition models are stored.
// FHS compliant: /var/lib/faceid/models
std::string get_models_dir();

// Return the directory where face enrollments are stored for a specific user.
// FHS compliant: /var/lib/faceid/faces/<username>
// Username is validated to prevent path traversal attacks.
std::string get_user_faces_dir(const std::string& username);

// Return the directory where fingerprint data is stored for a specific user.
// FHS compliant: /var/lib/faceid/fingerprints/<username>
std::string get_user_fingerprints_dir(const std::string& username);

// Ensure FaceID system directories exist with proper permissions.
// Creates directories with appropriate ownership (root:root).
// - Base dirs: 0755 (readable by all)
// - Biometric dirs: 0700 (owner only)
// Returns true on success.
bool ensure_faceid_directories(std::string* out_err = nullptr);

// Validate username to prevent path traversal attacks.
// Rejects usernames containing: '/', '..', or starting with '.'
// Returns true if username is safe to use in paths.
bool validate_username(const std::string& username, std::string* error_msg = nullptr);

} // namespace path_utils

