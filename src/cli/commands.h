#ifndef FACEID_CLI_COMMANDS_H
#define FACEID_CLI_COMMANDS_H

#include <string>

namespace faceid {

/**
 * Command Functions for FaceID CLI
 * 
 * Each command function returns:
 *   - 0 on success
 *   - 1 on failure
 */

/**
 * List all available camera devices
 * 
 * @return 0 if devices found, 1 if no devices or error
 */
int cmd_devices();

/**
 * Add a new face model for a user
 * 
 * @param username The username to enroll
 * @param face_id  The face identifier (e.g., "default", "glasses", "with_hat")
 *                 Defaults to "default" if not specified
 * @return 0 on success, 1 on failure
 */
int cmd_add(const std::string& username, const std::string& face_id = "default");

/**
 * Remove a face model from a user's profile
 * 
 * @param username The username whose face to remove
 * @param face_id  The face identifier to remove (empty to remove all faces)
 *                 Defaults to empty string (remove all)
 * @return 0 on success, 1 on failure
 */
int cmd_remove(const std::string& username, const std::string& face_id = "");

/**
 * List enrolled users and their faces
 * 
 * @param username If specified, list all faces for this user
 *                 If empty, list all enrolled users
 *                 Defaults to empty string (list all users)
 * @return 0 on success, 1 on failure
 */
int cmd_list(const std::string& username = "");

/**
 * Show live camera view with real-time face detection
 * 
 * Press 'q' or ESC to quit
 * 
 * @return 0 on success, 1 on failure
 */
int cmd_show();

/**
 * Test face recognition with live visual display
 *
 * Shows a live camera preview with face detection and recognition.
 * Compares detected faces against ALL enrolled users and displays the matched username.
 * Green box = Match found, Red box = Unknown face
 *
 * @param username Optional username for running integrity checks (can be empty)
 * @return 0 on success, 1 on error
 */
int cmd_test(const std::string& username);

/**
 * Benchmark recognition models
 * 
 * Tests all recognition models in the specified directory and reports performance metrics.
 * Requires a face to be visible in the camera frame.
 * 
 * @param test_dir Directory containing model files to benchmark
 * @param show_detail If true, shows detailed per-model testing output
 * @return 0 on success, 1 on error
 */
int cmd_bench(const std::string& test_dir, bool show_detail = false);

/**
 * Print usage information and command help
 */
void print_usage();

} // namespace faceid

#endif // FACEID_CLI_COMMANDS_H
