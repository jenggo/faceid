#include "commands.h"
#include "cli_common.h"
#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <dirent.h>
#include <fnmatch.h>
#include "../models/binary_model.h"
#include "../models/model_cache.h"

namespace faceid {

using namespace faceid;

// Helper: Calculate L2 norm of a vector
static float calculateNorm(const std::vector<float>& vec) {
    float sum = 0.0f;
    for (float val : vec) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

// Helper: Check for NaN or Inf values
static bool hasInvalidValues(const std::vector<float>& vec) {
    for (float val : vec) {
        if (std::isnan(val) || std::isinf(val)) {
            return true;
        }
    }
    return false;
}

// Helper: Calculate cosine distance
static float cosineDistance(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    float dot = 0.0f;
    for (size_t i = 0; i < vec1.size(); i++) {
        dot += vec1[i] * vec2[i];
    }
    // Clamp dot product to [-1, 1] to handle floating point precision errors
    if (dot > 1.0f) dot = 1.0f;
    if (dot < -1.0f) dot = -1.0f;
    return 1.0f - dot;
}

// Perform integrity checks on face encodings
static bool checkEncodingIntegrity(const BinaryFaceModel& model, bool verbose = true) {
    if (verbose) {
        std::cout << "\n=== Encoding Integrity Check ===" << std::endl;
        std::cout << "Total encodings: " << model.encodings.size() << std::endl;
    }
    
    bool has_issues = false;
    
    // Check 1: Normalization
    bool all_normalized = true;
    for (const auto& enc : model.encodings) {
        float norm = calculateNorm(enc);
        if (std::abs(norm - 1.0f) > 0.01f) {
            all_normalized = false;
            break;
        }
    }
    
    if (!all_normalized) {
        std::cout << "✗ WARNING: Encodings are NOT properly normalized" << std::endl;
        std::cout << "  This may cause authentication issues" << std::endl;
        std::cout << "  Solution: Re-enroll with 'sudo faceid add " << model.username << "'" << std::endl;
        has_issues = true;
    } else if (verbose) {
        std::cout << "✓ All encodings are properly normalized" << std::endl;
    }
    
    // Check 2: Invalid values (NaN/Inf)
    bool has_invalid = false;
    for (const auto& enc : model.encodings) {
        if (hasInvalidValues(enc)) {
            has_invalid = true;
            break;
        }
    }
    
    if (has_invalid) {
        std::cout << "✗ CRITICAL: Encodings contain NaN or Inf values" << std::endl;
        std::cout << "  Solution: Re-enroll with 'sudo faceid add " << model.username << "'" << std::endl;
        has_issues = true;
    } else if (verbose) {
        std::cout << "✓ No NaN or Inf values found" << std::endl;
    }
    
    // Check 3: Size validation
    bool all_valid_size = true;
    for (const auto& enc : model.encodings) {
        if (enc.size() != FACE_ENCODING_DIM) {
            all_valid_size = false;
            break;
        }
    }
    
    if (!all_valid_size) {
        std::cout << "✗ CRITICAL: Some encodings have incorrect dimensions" << std::endl;
        std::cout << "  Expected: " << FACE_ENCODING_DIM << "D vectors (current model)" << std::endl;
        std::cout << "  Solution: Re-enroll with 'sudo faceid add " << model.username << "'" << std::endl;
        has_issues = true;
    } else if (verbose) {
        std::cout << "✓ All encodings have correct dimensions (" << FACE_ENCODING_DIM << "D)" << std::endl;
    }
    
    // Check 4: Self-similarity (optional, detailed check)
    if (verbose && model.encodings.size() > 0) {
        float self_dist = cosineDistance(model.encodings[0], model.encodings[0]);
        if (self_dist > 0.01f) {
            std::cout << "⚠ WARNING: Self-distance is " << self_dist << " (should be ~0.0)" << std::endl;
            has_issues = true;
        }
    }
    
    if (verbose) {
        if (has_issues) {
            std::cout << "\n⚠ Issues detected - please re-enroll for best results" << std::endl;
        } else {
            std::cout << "\n✓ All integrity checks passed" << std::endl;
        }
    }
    
    return !has_issues;
}

int cmd_test(const std::string& username) {
    std::cout << "Testing face recognition..." << std::endl;

    // Load ALL users for face matching (not just the specified user)
    auto& cache = ModelCache::getInstance();
    std::vector<BinaryFaceModel> all_models = cache.loadAllUsersParallel(4);

    if (all_models.empty()) {
        std::cerr << "Error: No face models found for any user" << std::endl;
        std::cerr << "Run: sudo faceid add <username>" << std::endl;
        return 1;
    }

    std::cout << "Loaded " << all_models.size() << " enrolled user(s)" << std::endl;

    // If username specified, run integrity checks for that user
    if (!username.empty()) {
        for (const auto& model : all_models) {
            if (model.username == username) {
                std::cout << "\nRunning integrity checks for user: " << username << std::endl;
                bool integrity_ok = checkEncodingIntegrity(model, true);
                if (!integrity_ok) {
                    std::cout << "\n⚠ Continuing with live test despite integrity issues..." << std::endl;
                }
                break;
            }
        }
    }
    
    // Load configuration
    faceid::Config& config = faceid::Config::getInstance();
    std::string config_path = std::string(CONFIG_DIR) + "/faceid.conf";
    config.load(config_path);
    
    auto device = config.getString("camera", "device").value_or("/dev/video0");
    auto width = config.getInt("camera", "width").value_or(640);
    auto height = config.getInt("camera", "height").value_or(480);
    double threshold = config.getDouble("recognition", "threshold").value_or(0.6);
    int tracking_interval = config.getInt("face_detection", "tracking_interval").value_or(10);
    
    std::cout << "Using camera: " << device << " (" << width << "x" << height << ")" << std::endl;
    std::cout << "Recognition threshold: " << threshold << std::endl;
    
    // Display unique usernames (models can have duplicate usernames from multiple files)
    std::vector<std::string> unique_usernames;
    for (const auto& model : all_models) {
        if (std::find(unique_usernames.begin(), unique_usernames.end(), model.username) == unique_usernames.end()) {
            unique_usernames.push_back(model.username);
        }
    }
    
    std::cout << "Enrolled users: ";
    for (size_t i = 0; i < unique_usernames.size(); i++) {
        std::cout << unique_usernames[i];
        if (i < unique_usernames.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl << std::endl;

    // Initialize camera
    Camera camera(device);
    if (!camera.open(width, height)) {
        std::cerr << "Error: Failed to open camera" << std::endl;
        return 1;
    }

    // Initialize face detector
    faceid::FaceDetector detector;

    if (!detector.loadModels()) {  // Use default MODELS_DIR/sface path
        std::cerr << "Error: Failed to load face recognition model" << std::endl;
        std::cerr << "Expected files: " << MODELS_DIR << "/sface.param and sface.bin" << std::endl;
        return 1;
    }

    // Create preview window
    faceid::Display display("FaceID - Face Recognition Test", width, height);

    std::cout << "Live preview started. Press 'q' or ESC to quit.\n" << std::endl;

    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();
    bool timing_displayed = false;  // Track if we've shown timing info

    while (display.isOpen()) {
        faceid::Image frame;
        if (!camera.read(frame)) {
            std::cerr << "Failed to read frame from camera" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // Preprocess and detect faces
        auto detect_start = std::chrono::high_resolution_clock::now();
        faceid::Image processed_frame = detector.preprocessFrame(frame.view());
        auto faces = detector.detectOrTrackFaces(processed_frame.view(), tracking_interval);
        auto detect_end = std::chrono::high_resolution_clock::now();
        double detection_time = std::chrono::duration<double, std::milli>(detect_end - detect_start).count();

        // Clone frame for drawing
        faceid::Image display_frame = frame.clone();

        // Process each detected face
        std::vector<std::string> matched_names(faces.size(), "");
        std::vector<double> matched_distances(faces.size(), 999.0);
        double recognition_time = 0.0;

        if (!faces.empty()) {
            auto recog_start = std::chrono::high_resolution_clock::now();
            auto encodings = detector.encodeFaces(processed_frame.view(), faces);

            // Match each face against all enrolled users
            for (size_t i = 0; i < encodings.size() && i < faces.size(); i++) {
                double best_distance = 999.0;
                std::string best_match = "";

                // Compare with all users
                for (const auto& model : all_models) {
                    for (const auto& stored_encoding : model.encodings) {
                        double distance = detector.compareFaces(stored_encoding, encodings[i]);
                        if (distance < best_distance) {
                            best_distance = distance;
                            best_match = model.username;
                        }
                    }
                }

                matched_distances[i] = best_distance;
                if (best_distance < threshold) {
                    matched_names[i] = best_match;
                }
            }
            auto recog_end = std::chrono::high_resolution_clock::now();
            recognition_time = std::chrono::duration<double, std::milli>(recog_end - recog_start).count();
            
            // Display timing info on first successful match
            if (!timing_displayed && !matched_names.empty() && !matched_names[0].empty()) {
                std::cout << "\n=== Performance Timing (First Match) ===" << std::endl;
                std::cout << "Face detection time:    " << std::fixed << std::setprecision(2) 
                         << detection_time << " ms" << std::endl;
                std::cout << "Face recognition time:  " << recognition_time << " ms" << std::endl;
                std::cout << "Total time:             " << (detection_time + recognition_time) << " ms" << std::endl;
                std::cout << "========================================\n" << std::endl;
                timing_displayed = true;
            }
        }

        // Draw detected faces with identity
        for (size_t i = 0; i < faces.size(); i++) {
            const auto& face = faces[i];
            
            // Debug: print coordinates for first face on first frame
            if (frame_count == 0 && i == 0) {
                std::cout << "DEBUG: Face bbox - x=" << face.x << " y=" << face.y 
                         << " w=" << face.width << " h=" << face.height 
                         << " (frame: " << display_frame.width() << "x" << display_frame.height() << ")" << std::endl;
            }

            // Color and label based on match status
            faceid::Color color = !matched_names[i].empty()
                ? faceid::Color::Green()  // Matched - green
                : faceid::Color::Red();   // No match - red

            std::string label = !matched_names[i].empty()
                ? matched_names[i] + " (" + std::to_string(static_cast<int>(matched_distances[i] * 100)) + "%)"
                : "Unknown (" + std::to_string(static_cast<int>(matched_distances[i] * 100)) + "%)";

            // Draw rectangle at ORIGINAL position (SDL will flip it correctly)
            faceid::drawRectangle(display_frame, face.x, face.y,
                                 face.width, face.height, color, 2);

            // Reverse text for SDL horizontal flip and position it above the box
            std::reverse(label.begin(), label.end());
            int text_width = label.length() * 8;
            // Position text: align with right edge of box before flip = left edge after flip
            int text_x = face.x + face.width - text_width;
            faceid::drawText(display_frame, label, text_x, face.y - 10, color, 1.0);
        }

        // Calculate FPS
        frame_count++;
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

        if (elapsed > 0) {
            double fps = static_cast<double>(frame_count) / elapsed;

            // Draw info banner at top
            faceid::drawFilledRectangle(display_frame, 0, 0, display_frame.width(), 70, faceid::Color::Black());

            // Face count
            std::string info_text = "Detected faces: " + std::to_string(faces.size());
            std::reverse(info_text.begin(), info_text.end());
            int info_width = info_text.length() * 8;
            faceid::drawText(display_frame, info_text, display_frame.width() - 10 - info_width, 10, faceid::Color::White(), 1.0);

            // FPS
            std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps));
            std::reverse(fps_text.begin(), fps_text.end());
            int fps_width = fps_text.length() * 8;
            faceid::drawText(display_frame, fps_text, display_frame.width() - 10 - fps_width, 25, faceid::Color::Green(), 1.0);

            // Threshold info
            std::string thresh_text = "Threshold: " + std::to_string(static_cast<int>(threshold * 100)) + "%";
            std::reverse(thresh_text.begin(), thresh_text.end());
            int thresh_width = thresh_text.length() * 8;
            faceid::drawText(display_frame, thresh_text, display_frame.width() - 10 - thresh_width, 45, faceid::Color::Gray(), 1.0);
        }

        // Draw help text at bottom
        faceid::drawFilledRectangle(display_frame, 0, display_frame.height() - 30,
                                   display_frame.width(), 30, faceid::Color::Black());
        std::string help_text = "Press 'q' or ESC to quit";
        std::reverse(help_text.begin(), help_text.end());
        int help_width = help_text.length() * 8;
        faceid::drawText(display_frame, help_text, display_frame.width() - 10 - help_width, display_frame.height() - 20,
                        faceid::Color::White(), 1.0);

        // Display the frame (SDL will flip horizontally)
        display.show(display_frame);

        // Check for quit key
        int key = display.waitKey(30);
        if (key == 'q' || key == 'Q' || key == 27) {  // q or ESC
            break;
        }
    }

    std::cout << "\nTest completed." << std::endl;
    std::cout << "Total frames processed: " << frame_count << std::endl;

    return 0;
}

} // namespace faceid
