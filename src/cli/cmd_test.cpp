#include "commands.h"
#include "cli_common.h"
#include "cli_helpers.h"
#include "dbus_client_cli.h"
#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <dirent.h>
#include <fnmatch.h>
#include "../models/binary_model.h"
#include "../models/model_cache.h"
#include "../daemon/config.h"
#include "config_paths.h"

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

// Helper: Format distance percentage with decimal precision for small values
static std::string formatDistancePercent(double distance) {
    double percent = distance * 100.0;
    
    if (percent < 1.0) {
        // Show one decimal place for very small distances (< 1%)
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.1f%%", percent);
        return std::string(buf);
    } else {
        // Show as integer for larger distances
        return std::to_string(static_cast<int>(percent)) + "%";
    }
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

// Perform integrity checks on face encodings
// Assumes daemon uses 512-dimensional embeddings (standard for modern face recognition)
static bool checkEncodingIntegrity(const BinaryFaceModel& model, bool verbose = true) {
    if (verbose) {
        std::cout << "\n=== Encoding Integrity Check ===" << std::endl;
        // V2 format: count total encodings across all poses
        int total_encodings = 0;
        for (const auto& pose_encodings : model.sample_encodings) {
            total_encodings += pose_encodings.size();
        }
        std::cout << "Total encodings: " << total_encodings << std::endl;
    }
    
    bool has_issues = false;
    const size_t DAEMON_ENCODING_DIM = 512;  // D-Bus daemon uses 512D embeddings
    
    // Flatten all encodings for checking
    std::vector<FaceEncoding> all_encodings;
    for (const auto& pose_encodings : model.sample_encodings) {
        all_encodings.insert(all_encodings.end(), 
                           pose_encodings.begin(), 
                           pose_encodings.end());
    }
    
    // Check 1: Normalization
    bool all_normalized = true;
    for (const auto& enc : all_encodings) {
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
    for (const auto& enc : all_encodings) {
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
    
    // Check 3: Size validation - check against daemon encoding dimension
    bool all_valid_size = true;
    for (const auto& enc : all_encodings) {
        if (enc.size() != DAEMON_ENCODING_DIM) {
            all_valid_size = false;
            break;
        }
    }
    
    if (!all_valid_size) {
        std::cout << "✗ CRITICAL: Some encodings have incorrect dimensions" << std::endl;
        std::cout << "  Expected: " << DAEMON_ENCODING_DIM << "D vectors (daemon standard)" << std::endl;
        std::cout << "  Found: " << (all_encodings.empty() ? 0 : all_encodings[0].size()) << "D vectors" << std::endl;
        std::cout << "  Solution: Re-enroll with 'sudo faceid add " << model.username << "'" << std::endl;
        has_issues = true;
    } else if (verbose) {
        std::cout << "✓ All encodings have correct dimensions (" << DAEMON_ENCODING_DIM << "D)" << std::endl;
    }
    
    // Check 4: Self-similarity (optional, detailed check)
    if (verbose && all_encodings.size() > 0) {
        float self_dist = cosineDistance(all_encodings[0], all_encodings[0]);
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

int cmd_test(const std::string& username, bool auto_adjust) {
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

    // Initialize D-Bus connection for face detection and embedding
    DBusClientCLI& dbus_client = DBusClientCLI::instance();
    if (!dbus_client.connect()) {
        std::cerr << "Error: Failed to connect to face recognition daemon" << std::endl;
        std::cerr << "Make sure the daemon is running with: sudo systemctl start faceid-daemon" << std::endl;
        return 1;
    }

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
    std::string config_path = std::string(CONFIG_DIR) + "/faceid.conf";
    auto config_ptr = ::Config::load();
    if (!config_ptr) {
        std::cerr << "Warning: Could not load config, using defaults" << std::endl;
    }
    ::Config& config = config_ptr ? *config_ptr : ::Config::instance();
    
    auto device = config.camera.device;
    auto width = config.camera.width;
    auto height = config.camera.height;
    double threshold = config.recognition.threshold;
    
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

    // Models already loaded earlier (line 148-152)

    // Run performance benchmark BEFORE GUI initialization (no display overhead)
    std::cout << "\nRunning performance benchmark (5 frames)..." << std::endl;
    std::vector<double> benchmark_detection_times;
    std::vector<double> benchmark_recognition_times;
    const int BENCHMARK_SAMPLES = 5;
    int attempts = 0;
    const int MAX_ATTEMPTS = 50; // Try up to 50 frames to get 5 successful detections
    
    while (benchmark_detection_times.size() < BENCHMARK_SAMPLES && attempts < MAX_ATTEMPTS) {
        attempts++;
        
        faceid::Image frame;
        if (!camera.read(frame)) {
            std::cerr << "Failed to read frame from camera" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        // Encode frame as PNG and send to daemon for detection
        auto detect_start = std::chrono::high_resolution_clock::now();
        std::vector<uint8_t> png_data = DBusClientCLI::encode_frame_as_jpeg(frame);
        auto detect_result = dbus_client.detect_face(png_data);
        auto detect_end = std::chrono::high_resolution_clock::now();
        double detection_time = std::chrono::duration<double, std::milli>(detect_end - detect_start).count();
        
        if (!detect_result.boxes.empty()) {
            // Generate embeddings and match faces
            auto recog_start = std::chrono::high_resolution_clock::now();
            
            std::vector<FaceEncoding> encodings;
            for (const auto& bbox : detect_result.boxes) {
                auto embedding_result = dbus_client.generate_embedding(png_data, bbox);
                if (!embedding_result.embedding.empty()) {
                    encodings.push_back(embedding_result.embedding);
                }
            }
            
            // Match against enrolled users
            bool matched = false;
            for (size_t i = 0; i < encodings.size() && i < detect_result.boxes.size(); i++) {
                double best_distance = 999.0;
                std::string best_match = "";

                for (const auto& model : all_models) {
                    // Flatten sample_encodings for matching
                    std::vector<FaceEncoding> model_flat_encodings;
                    for (const auto& pose_encodings : model.sample_encodings) {
                        model_flat_encodings.insert(model_flat_encodings.end(),
                                                   pose_encodings.begin(),
                                                   pose_encodings.end());
                    }
                    for (const auto& stored_encoding : model_flat_encodings) {
                        double distance = cosineDistance(stored_encoding, encodings[i]);
                        if (distance < best_distance) {
                            best_distance = distance;
                            best_match = model.username;
                        }
                    }
                }
                
                if (best_distance < threshold && !best_match.empty()) {
                    matched = true;
                    break;
                }
            }
            
            auto recog_end = std::chrono::high_resolution_clock::now();
            double recognition_time = std::chrono::duration<double, std::milli>(recog_end - recog_start).count();
            
            // Only count successful matches
            if (matched) {
                benchmark_detection_times.push_back(detection_time);
                benchmark_recognition_times.push_back(recognition_time);
                std::cout << "  Sample " << benchmark_detection_times.size() << "/" << BENCHMARK_SAMPLES 
                         << " - Detection: " << std::fixed << std::setprecision(2) << detection_time 
                         << "ms, Recognition: " << recognition_time << "ms" << std::endl;
            }
        }
    }
    
    // Display benchmark results
    if (benchmark_detection_times.size() == BENCHMARK_SAMPLES) {
        double avg_detection = 0.0;
        double avg_recognition = 0.0;
        
        for (size_t i = 0; i < BENCHMARK_SAMPLES; i++) {
            avg_detection += benchmark_detection_times[i];
            avg_recognition += benchmark_recognition_times[i];
        }
        avg_detection /= BENCHMARK_SAMPLES;
        avg_recognition /= BENCHMARK_SAMPLES;
        
        std::cout << "\n=== Performance Timing (Average of " << BENCHMARK_SAMPLES << " Matches) ===" << std::endl;
        std::cout << "Face detection time:    " << std::fixed << std::setprecision(2) 
                 << avg_detection << " ms" << std::endl;
        std::cout << "Face recognition time:  " << avg_recognition << " ms" << std::endl;
        std::cout << "Total time:             " << (avg_detection + avg_recognition) << " ms" << std::endl;
        std::cout << "Expected FPS:           " << std::setprecision(1) 
                 << (1000.0 / (avg_detection + avg_recognition)) << std::endl;
        std::cout << "========================================\n" << std::endl;
    } else {
        std::cout << "\nWarning: Could not collect enough samples (" 
                 << benchmark_detection_times.size() << "/" << BENCHMARK_SAMPLES << ")" << std::endl;
        std::cout << "Make sure your face is visible to the camera.\n" << std::endl;
    }

    // Variables for auto-adjust mode
    float optimal_confidence_value = -1.0f;
    float optimal_threshold_value = -1.0f;
    bool can_save_config = false;
    
    // Auto-adjust mode: Find optimal settings and update config
    if (auto_adjust) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "=== AUTO-ADJUSTMENT MODE ===" << std::endl;
        std::cout << "========================================" << std::endl;
        
        // Create temporary display for adjustment phase
        faceid::Display temp_display("FaceID - Auto-Adjustment", width, height);
        
        // Step 1: Wait for face detection with live preview
        std::cout << std::endl;
        std::cout << "=== Waiting for Face Detection ===" << std::endl;
        std::cout << "Position your face in the camera view..." << std::endl;
        std::cout << "Waiting for face... " << std::flush;
        
        faceid::Image reference_frame;
        bool face_found = false;
        
        while (!face_found) {
            faceid::Image frame;
            if (!camera.read(frame)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
            
            // Use D-Bus for face detection
            std::vector<uint8_t> png_data = DBusClientCLI::encode_frame_as_jpeg(frame);
            auto detect_result = dbus_client.detect_face(png_data);
            auto test_faces = detect_result.boxes;
            
            // Draw visualization
            faceid::Image display_frame = frame.clone();
            
            if (!test_faces.empty()) {
                // Draw green box for detected face
                for (const auto& face : test_faces) {
                    faceid::drawRectangle(display_frame, face.x, face.y, 
                                         face.width, face.height, faceid::Color::Green(), 2);
                }
                
                // Draw status text
                std::string status_text = "Face detected! Analyzing optimal settings...";
                faceid::drawFilledRectangle(display_frame, 0, 0, display_frame.width(), 40, faceid::Color::Black());
                
                // Note: cascade stage info not available from D-Bus, just show quality
                status_text = "Face detected (quality: " + std::to_string(static_cast<int>(detect_result.quality * 100)) + "%)";
                
                std::string status_reversed = status_text;
                std::reverse(status_reversed.begin(), status_reversed.end());
                int text_width = status_reversed.length() * 8;
                faceid::drawText(display_frame, status_reversed, display_frame.width() - 10 - text_width, 10, 
                               faceid::Color::Green(), 1.0);
                
                reference_frame = frame.clone();
                face_found = true;
            } else {
                // Draw orange status - waiting
                std::string status_text = "Waiting for face... Position yourself in frame";
                faceid::drawFilledRectangle(display_frame, 0, 0, display_frame.width(), 40, faceid::Color::Black());
                std::string status_reversed = status_text;
                std::reverse(status_reversed.begin(), status_reversed.end());
                int text_width = status_reversed.length() * 8;
                faceid::drawText(display_frame, status_reversed, display_frame.width() - 10 - text_width, 10, 
                               faceid::Color::Orange(), 1.0);
            }
            
            // Show frame
            temp_display.show(display_frame);
            
            // Check for quit
            int key = temp_display.waitKey(50);
            if (key == 'q' || key == 'Q' || key == 27 || !temp_display.isOpen()) {
                std::cout << "Cancelled by user" << std::endl;
                return 1;
            }
        }
        
        std::cout << "detected!" << std::endl;
        
        // Step 2: Use D-Bus detection - no need for manual confidence optimization
        std::cout << std::endl;
        std::cout << "=== Using D-Bus Face Detection ===" << std::endl;
        std::cout << "Using system daemon for face detection" << std::endl;
        std::cout << "This adapts automatically to your lighting conditions" << std::endl;
        
        // Test detection on reference frame
        std::vector<uint8_t> ref_png = DBusClientCLI::encode_frame_as_jpeg(reference_frame);
        auto detect_result = dbus_client.detect_face(ref_png);
        
        if (detect_result.boxes.empty()) {
            std::cerr << "✗ No faces detected in reference frame" << std::endl;
            return 1;
        }
        
        std::cout << "✓ Face detection successful" << std::endl;
        std::cout << "  Detected faces: " << detect_result.boxes.size() << std::endl;
        std::cout << "  Detection quality: " << std::fixed << std::setprecision(1) 
                  << (detect_result.quality * 100) << "%" << std::endl;
        
        // Use default confidence (daemon handles it automatically)
        float optimal_confidence = 0.5f;
        
        // Step 3: Capture samples in current conditions to calculate optimal threshold
        std::cout << std::endl;
        std::cout << "=== Capturing Test Samples ===" << std::endl;
        std::cout << "Capturing 3 samples with different poses..." << std::endl;
        std::cout << "This helps adapt to your current lighting and conditions." << std::endl;
        std::cout << std::endl;
        
        const int num_test_samples = 3;
        std::vector<std::vector<float>> test_encodings;
        const std::string test_prompts[] = {
            "Look straight at camera",
            "Turn head slightly",
            "Tilt head slightly"
        };
        
        for (int i = 0; i < num_test_samples; i++) {
            std::cout << "  Sample " << (i + 1) << "/" << num_test_samples << " (" << test_prompts[i] << ")... " << std::flush;
            
            // Wait for face with live preview
            bool captured = false;
            std::vector<float> captured_encoding;
            auto capture_start = std::chrono::steady_clock::now();
            
            while (!captured) {
                // Give user 3 seconds to adjust pose
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - capture_start).count();
                
                if (elapsed >= 3000) {
                    // Capture now using D-Bus detection
                    faceid::Image frame;
                    if (camera.read(frame)) {
                        std::vector<uint8_t> png_data = DBusClientCLI::encode_frame_as_jpeg(frame);
                        auto detect_result = dbus_client.detect_face(png_data);
                        
                        if (detect_result.boxes.size() == 1) {
                            auto embedding_result = dbus_client.generate_embedding(png_data, detect_result.boxes[0]);
                            if (!embedding_result.embedding.empty()) {
                                captured_encoding = embedding_result.embedding;
                                captured = true;
                                std::cout << "✓ OK" << std::endl;
                            }
                        }
                    }
                    
                    if (!captured) {
                        std::cout << "retry..." << std::flush;
                        capture_start = std::chrono::steady_clock::now();  // Retry
                    }
                } else {
                    // Show live preview during countdown using D-Bus
                    faceid::Image frame;
                    if (camera.read(frame)) {
                        std::vector<uint8_t> png_data = DBusClientCLI::encode_frame_as_jpeg(frame);
                        auto detect_result = dbus_client.detect_face(png_data);
                        auto faces = detect_result.boxes;
                        
                        faceid::Image display_frame = frame.clone();
                        
                        // Draw face box
                        for (const auto& face : faces) {
                            faceid::Color color = (faces.size() == 1) ? faceid::Color::Green() : faceid::Color::Red();
                            faceid::drawRectangle(display_frame, face.x, face.y, face.width, face.height, color, 2);
                        }
                        
                        // Draw countdown
                        int remaining_sec = (3000 - elapsed) / 1000 + 1;
                        std::string status_text = test_prompts[i] + std::string(" - Capturing in ") + std::to_string(remaining_sec) + "s...";
                        faceid::drawFilledRectangle(display_frame, 0, 0, display_frame.width(), 40, faceid::Color::Black());
                        std::string status_reversed = status_text;
                        std::reverse(status_reversed.begin(), status_reversed.end());
                        int text_width = status_reversed.length() * 8;
                        faceid::drawText(display_frame, status_reversed, display_frame.width() - 10 - text_width, 10,
                                       faceid::Color::Green(), 1.0);
                        
                        temp_display.show(display_frame);
                    }
                    
                    int key = temp_display.waitKey(50);
                    if (key == 'q' || key == 'Q' || key == 27 || !temp_display.isOpen()) {
                        std::cout << "Cancelled" << std::endl;
                        return 1;
                    }
                }
            }
            
            test_encodings.push_back(captured_encoding);
        }
        
        std::cout << std::endl;
        std::cout << "=== Calculating Optimal Recognition Threshold ===" << std::endl;
        std::cout << "Comparing test samples against enrolled face..." << std::endl;
        
        // Compare test samples against enrolled face (not against each other!)
        std::vector<float> test_distances;
        for (const auto& test_enc : test_encodings) {
            // Compare against ALL enrolled faces to find the best match
            for (const auto& model : all_models) {
                // Flatten sample_encodings for comparison
                std::vector<FaceEncoding> model_flat_encodings;
                for (const auto& pose_encodings : model.sample_encodings) {
                    model_flat_encodings.insert(model_flat_encodings.end(),
                                               pose_encodings.begin(),
                                               pose_encodings.end());
                }
                for (const auto& enrolled_enc : model_flat_encodings) {
                    float dist = cosineDistance(enrolled_enc, test_enc);
                    test_distances.push_back(dist);
                }
            }
        }
        
        float optimal_threshold = 0.4f;  // Default fallback
        if (!test_distances.empty()) {
            // Find maximum distance from test samples to enrolled face
            float max_enrolled_distance = *std::max_element(test_distances.begin(), test_distances.end());
            
            // Set threshold with 20% safety margin above max enrolled distance
            optimal_threshold = max_enrolled_distance * 1.2f;
            
            // Clamp to reasonable range
            if (optimal_threshold < 0.15f) optimal_threshold = 0.15f;
            if (optimal_threshold > 0.80f) optimal_threshold = 0.80f;
            
            std::cout << "✓ Optimal recognition threshold calculated: " << std::fixed << std::setprecision(2) 
                      << optimal_threshold << std::endl;
            std::cout << "  Based on " << test_distances.size() << " comparisons against enrolled face" << std::endl;
            std::cout << "  Max distance to enrolled face: " << std::fixed << std::setprecision(4) 
                      << max_enrolled_distance << " (" << std::setprecision(0) 
                      << (max_enrolled_distance * 100) << "%)" << std::endl;
        } else {
            std::cerr << "⚠ Could not calculate optimal threshold" << std::endl;
            std::cerr << "  Using default value: " << optimal_threshold << std::endl;
        }
        
        // Step 4: Apply settings to this session (don't save to config yet)
        std::cout << std::endl;
        std::cout << "=== Settings Applied ===" << std::endl;
        std::cout << "Optimal settings are now active in this test session:" << std::endl;
        std::cout << "  Detection confidence: " << std::fixed << std::setprecision(2) << optimal_confidence << std::endl;
        std::cout << "  Recognition threshold: " << std::fixed << std::setprecision(2) << optimal_threshold << std::endl;
        std::cout << std::endl;
        
        // Store optimal values for potential saving later
        optimal_confidence_value = optimal_confidence;
        optimal_threshold_value = optimal_threshold;
        
        // Update threshold for this session
        threshold = optimal_threshold;
        
        // Check if we have write permission to config
        std::ofstream test_write(config_path, std::ios::app);
        if (test_write.is_open()) {
            can_save_config = true;
            test_write.close();
        }
        
        if (can_save_config) {
            std::cout << "✓ Config file is writable" << std::endl;
            std::cout << "  Press 's' during live test to save these settings" << std::endl;
        } else {
            std::cout << "⚠ Config file is not writable (no permission)" << std::endl;
            std::cout << "  Settings will only apply to this session" << std::endl;
            std::cout << "  To save permanently, run with sudo" << std::endl;
        }
        
        std::cout << std::endl;
        std::cout << "Starting live test with optimized settings..." << std::endl;
        std::cout << "Watch the recognition results to verify improvement!" << std::endl;
        
        std::cout << "========================================\n" << std::endl;
    }

    // Create preview window
    faceid::Display display("FaceID - Face Recognition Test", width, height);

    std::cout << "Live preview started. Press 'q' or ESC to quit.\n" << std::endl;

    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();
    
    // Continuous adjustment variables (for auto-adjust mode)
    int frames_since_adjustment = 0;
    const int adjustment_interval = 60; // Adjust every 60 frames (~2 seconds at 30fps)
    bool is_adjusting = false;
    std::string adjustment_status = "";

    while (display.isOpen()) {
        faceid::Image frame;
        if (!camera.read(frame)) {
            std::cerr << "Failed to read frame from camera" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // Continuous on-the-fly adjustment (if auto-adjust enabled)
        if (auto_adjust && !is_adjusting && frames_since_adjustment >= adjustment_interval) {
            is_adjusting = true;
            adjustment_status = "Adjusting...";
            frames_since_adjustment = 0;
            
            // Capture 3 quick samples for threshold calculation
            std::vector<std::vector<float>> adjustment_encodings;
            adjustment_encodings.reserve(3);
            
            for (int sample = 0; sample < 3; sample++) {
                faceid::Image adj_frame;
                if (camera.read(adj_frame)) {
                    std::vector<uint8_t> png_data = DBusClientCLI::encode_frame_as_jpeg(adj_frame);
                    auto detect_result = dbus_client.detect_face(png_data);
                    
                    if (detect_result.boxes.size() == 1) {
                        auto embedding_result = dbus_client.generate_embedding(png_data, detect_result.boxes[0]);
                        if (!embedding_result.embedding.empty() && isValidFace(detect_result.boxes[0], adj_frame.width(), adj_frame.height(), embedding_result.embedding)) {
                            adjustment_encodings.push_back(embedding_result.embedding);
                        }
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Small delay between samples
            }
            
            // Calculate new optimal threshold if we got all 3 samples
            if (adjustment_encodings.size() == 3) {
                // Compare adjustment samples against enrolled face (not against each other!)
                double max_distance = 0.0;
                for (const auto& adj_enc : adjustment_encodings) {
                    // Compare against ALL enrolled faces to find the best match
                    for (const auto& model : all_models) {
                        // Flatten sample_encodings for comparison
                        std::vector<FaceEncoding> model_flat_encodings;
                        for (const auto& pose_encodings : model.sample_encodings) {
                            model_flat_encodings.insert(model_flat_encodings.end(),
                                                       pose_encodings.begin(),
                                                       pose_encodings.end());
                        }
                        for (const auto& enrolled_enc : model_flat_encodings) {
                            double dist = cosineDistance(enrolled_enc, adj_enc);
                            max_distance = std::max(max_distance, dist);
                        }
                    }
                }
                
                // Calculate new threshold with safety margin
                float new_threshold = static_cast<float>(max_distance * 1.2);
                new_threshold = std::max(0.15f, std::min(0.80f, new_threshold));
                
                // Update threshold immediately
                threshold = new_threshold;
                optimal_threshold_value = new_threshold;
                
                adjustment_status = "Adjusted: " + std::to_string(static_cast<int>(new_threshold * 100)) + "%";
            } else {
                adjustment_status = "Adjust failed";
            }
            
            is_adjusting = false;
        }

        // Use D-Bus detection for robust face detection in all lighting conditions
        std::vector<uint8_t> png_data = DBusClientCLI::encode_frame_as_jpeg(frame);
        auto detect_result = dbus_client.detect_face(png_data);
        auto faces = detect_result.boxes;

        // Clone frame for drawing
        faceid::Image display_frame = frame.clone();

        // Process each detected face
        std::vector<std::string> matched_names(faces.size(), "");
        std::vector<double> matched_distances(faces.size(), 999.0);

        if (!faces.empty()) {
            // Generate embeddings for all detected faces
            std::vector<FaceEncoding> encodings;
            for (const auto& bbox : faces) {
                auto embedding_result = dbus_client.generate_embedding(png_data, bbox);
                if (!embedding_result.embedding.empty()) {
                    encodings.push_back(embedding_result.embedding);
                }
            }
            
            // Deduplicate faces - filter out multiple detections of the same person
            // This prevents false positives from the same face detected at different angles/positions
            auto unique_indices = FaceDetector::deduplicateFaces(faces, encodings, 0.15);
            
            // Filter faces and encodings to only unique ones
            std::vector<Rect> unique_faces;
            std::vector<FaceEncoding> unique_encodings;
            for (size_t idx : unique_indices) {
                if (idx < faces.size() && idx < encodings.size()) {
                    unique_faces.push_back(faces[idx]);
                    unique_encodings.push_back(encodings[idx]);
                }
            }
            
            // Replace original faces/encodings with deduplicated ones
            faces = unique_faces;
            encodings = unique_encodings;
            
            // Update matched arrays to match new size
            matched_names.resize(faces.size(), "");
            matched_distances.resize(faces.size(), 999.0);

            // Match each face against all enrolled users
            for (size_t i = 0; i < encodings.size() && i < faces.size(); i++) {
                double best_distance = 999.0;
                double second_best_distance = 999.0;
                std::string best_match = "";
                std::string second_best_match = "";

                // Compare with all users
                for (const auto& model : all_models) {
                    // Flatten sample_encodings for matching
                    std::vector<FaceEncoding> model_flat_encodings;
                    for (const auto& pose_encodings : model.sample_encodings) {
                        model_flat_encodings.insert(model_flat_encodings.end(),
                                                   pose_encodings.begin(),
                                                   pose_encodings.end());
                    }
                    
                    for (const auto& stored_encoding : model_flat_encodings) {
                        double distance = cosineDistance(stored_encoding, encodings[i]);
                        
                        if (distance < best_distance) {
                            // Shift best to second best
                            second_best_distance = best_distance;
                            second_best_match = best_match;
                            // Update best
                            best_distance = distance;
                            best_match = model.username;
                        } else if (distance < second_best_distance) {
                            // Update second best
                            second_best_distance = distance;
                            second_best_match = model.username;
                        }
                    }
                }

                matched_distances[i] = best_distance;
                
                // Only accept match if:
                // 1. Best distance is below threshold
                // 2. For safety: if multiple users, ensure there's a margin between best and second best
                bool is_unique_match = true;
                
                // Check uniqueness only if we have a second match from a DIFFERENT user
                if (!second_best_match.empty() && second_best_match != best_match && second_best_distance < 999.0) {
                    const double MARGIN = 0.05; // 5% margin to ensure clear winner
                    is_unique_match = (second_best_distance - best_distance) > MARGIN;
                }
                
                if (best_distance < threshold && is_unique_match) {
                    matched_names[i] = best_match;
                } else if (best_distance < threshold && !is_unique_match) {
                    // Ambiguous match - too close to multiple users
                    matched_names[i] = ""; // Reject match
                }
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
                ? matched_names[i] + " (" + formatDistancePercent(matched_distances[i]) + ")"
                : "Unknown (" + formatDistancePercent(matched_distances[i]) + ")";

            // Draw rectangle with centering correction using shared helper
            int adjusted_x = faceid::drawFaceBoundingBox(display_frame, face, color, 2);

            // Draw facial landmarks if available (5-point landmarks from YOLO/YuNet)
            if (face.hasLandmarks()) {
                // Define colors for each landmark
                faceid::Color landmark_colors[] = {
                    faceid::Color(0, 255, 255),    // Left eye - Cyan
                    faceid::Color(0, 255, 255),    // Right eye - Cyan  
                    faceid::Color(255, 0, 0),      // Nose - Blue
                    faceid::Color(255, 0, 255),    // Left mouth - Magenta
                    faceid::Color(255, 0, 255)     // Right mouth - Magenta
                };
                
                for (size_t j = 0; j < face.landmarks.size() && j < 5; j++) {
                    const auto& pt = face.landmarks[j];
                    int px = static_cast<int>(pt.x);
                    int py = static_cast<int>(pt.y);
                    faceid::drawCircle(display_frame, px, py, 3, landmark_colors[j]);
                }
            }

            // Reverse text for SDL horizontal flip and position it above the box
            std::reverse(label.begin(), label.end());
            int text_width = label.length() * 8;
            // Position text: align with right edge of box before flip = left edge after flip
            // Use same centering adjustment as the box
            int text_x = adjusted_x + face.width - text_width;
            faceid::drawText(display_frame, label, text_x, face.y - 10, color, 1.0);
        }

        // Calculate FPS
        frame_count++;
        frames_since_adjustment++; // Track frames for continuous adjustment
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

        if (elapsed > 0) {
            double fps = static_cast<double>(frame_count) / elapsed;

            // Draw compact info banner at top using shared helper
            std::vector<std::string> info_items = {
                "Detected faces: " + std::to_string(faces.size()),
                "FPS: " + std::to_string(static_cast<int>(fps))
            };
            
            // Add distance if face detected
            if (!faces.empty() && !matched_distances.empty()) {
                double current_distance = matched_distances[0];
                info_items.push_back("Distance: " + formatDistancePercent(current_distance));
            }
            
            // Add threshold
            info_items.push_back("Threshold: " + std::to_string(static_cast<int>(threshold * 100)) + "%");
            
            // Add adjustment status if active
            if (auto_adjust && !adjustment_status.empty()) {
                info_items.push_back(adjustment_status);
            }
            
            faceid::drawCompactBanner(display_frame, info_items);
        }

        // Draw help text at bottom using shared helper
        std::string help_text = "Press 'q' or ESC to quit";
        if (auto_adjust && can_save_config) {
            help_text += " | Press 's' to save settings";
        }
        faceid::drawHelpBanner(display_frame, help_text);

        // Display the frame (SDL will flip horizontally)
        display.show(display_frame);

        // Check for quit key
        int key = display.waitKey(30);
        if (key == 'q' || key == 'Q' || key == 27) {  // q or ESC
            break;
        } else if (auto_adjust && can_save_config && (key == 's' || key == 'S')) {
            // Save optimal values to config
            std::cout << "\n========================================" << std::endl;
            std::cout << "Saving optimal settings to config..." << std::endl;
            
            if (updateConfigFile(config_path, optimal_confidence_value, optimal_threshold_value)) {
                std::cout << "✓ Settings saved successfully!" << std::endl;
                std::cout << "  Detection confidence: " << std::fixed << std::setprecision(2) 
                         << optimal_confidence_value << std::endl;
                std::cout << "  Recognition threshold: " << std::fixed << std::setprecision(2) 
                         << optimal_threshold_value << std::endl;
                std::cout << "========================================" << std::endl;
                
                // Mark that we've saved, so we don't need to prompt again
                can_save_config = false;
            } else {
                std::cerr << "✗ Failed to save settings to config" << std::endl;
            }
        }
    }

    std::cout << "\nTest completed." << std::endl;
    std::cout << "Total frames processed: " << frame_count << std::endl;

    return 0;
}

} // namespace faceid
