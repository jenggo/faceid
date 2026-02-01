#pragma once

#include <string>

/**
 * Daemon startup validation utilities
 * 
 * Ensures the daemon runs in proper user-level context:
 * - Not running as root
 * - User session exists
 * - Configuration paths are user-level
 */
namespace validation {

/**
 * Check if running as root (uid/euid == 0)
 * Returns: true if root, false if regular user
 */
bool is_root();

/**
 * Check if user session exists (XDG_RUNTIME_DIR environment variable set)
 * Returns: true if in user session, false otherwise
 */
bool has_user_session();

/**
 * Get XDG_RUNTIME_DIR value (e.g., /run/user/1000)
 * Returns: XDG_RUNTIME_DIR value or empty string if not set
 */
std::string get_xdg_runtime_dir();

/**
 * Get XDG_CONFIG_HOME, falling back to ~/.config
 * Returns: Configuration directory path
 */
std::string get_config_dir();

/**
 * Get XDG_DATA_HOME, falling back to ~/.local/share
 * Returns: Data directory path
 */
std::string get_data_dir();

/**
 * Get XDG_CACHE_HOME, falling back to ~/.cache
 * Returns: Cache directory path
 */
std::string get_cache_dir();

/**
 * Validate that config path is user-level (not /etc/, /usr/, etc)
 * Returns: true if valid user path, false if system path
 */
bool is_user_level_path(const std::string& path);

/**
 * Initialize logging to syslog
 * All subsequent validation messages logged via syslog
 */
void init_logging(const char* program_name);

} // namespace validation
