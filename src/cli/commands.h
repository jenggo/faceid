#ifndef FACEID_CLI_COMMANDS_H
#define FACEID_CLI_COMMANDS_H

#include <string>
#include <vector>

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
 * With --auto-adjust flag: automatically finds optimal detection confidence
 * and recognition threshold based on enrolled samples, then updates config file.
 *
 * @param username Optional username for running integrity checks (can be empty)
 * @param auto_adjust If true, perform auto-optimization and update config
 * @return 0 on success, 1 on error
 */
int cmd_test(const std::string& username, bool auto_adjust = false);

/**
 * Test face detection and recognition on static images
 *
 * Loads an enrollment image to use as reference, then detects all faces in a test image
 * and compares each against the enrolled reference face. Reports detailed debug information
 * including cosine distances, false positive rate, and threshold tuning recommendations.
 * Useful for testing and debugging recognition thresholds without needing camera interaction.
 *
 * @param args Command arguments: [0] = enrollment image path, [1] = test image path
 * @return 0 on success, 1 on error
 */
int cmd_test_image(const std::vector<std::string>& args);

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
 * Switch active model (detection or recognition)
 * 
 * Automatically detects if the model is for detection or recognition,
 * backs up the current model, and symlinks the new model as detection.* or recognition.*
 * 
 * @param model_name Model name (with or without extension)
 *                   Examples: "mnet-retinaface", "scrfd_500m-opt2.bin", "sface_2021dec_int8bq.ncnn"
 * @return 0 on success, 1 on error
 */
int cmd_use(const std::string& model_name);

/**
 * Print usage information and command help
 */
void print_usage();

} // namespace faceid

#endif // FACEID_CLI_COMMANDS_H
