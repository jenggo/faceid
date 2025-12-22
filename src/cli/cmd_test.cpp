#include "commands.h"
#include "cli_common.h"
#include <chrono>
#include <vector>
#include <string>
#include <cmath>
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
        if (enc.size() != 128) {
            all_valid_size = false;
            break;
        }
    }
    
    if (!all_valid_size) {
        std::cout << "✗ CRITICAL: Some encodings have incorrect dimensions" << std::endl;
        std::cout << "  Expected: 128D vectors" << std::endl;
        std::cout << "  Solution: Re-enroll with 'sudo faceid add " << model.username << "'" << std::endl;
        has_issues = true;
    } else if (verbose) {
        std::cout << "✓ All encodings have correct dimensions (128D)" << std::endl;
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
    std::cout << "Testing face recognition for user: " << username << std::endl;
    
    // Load all face models for this user using ModelCache
    std::string models_dir = MODELS_DIR;
    
    auto& cache = ModelCache::getInstance();
    BinaryFaceModel model;
    if (!cache.loadUserModel(username, model)) {
        std::cerr << "Error: No face models found for user: " << username << std::endl;
        std::cerr << "Run: sudo faceid add " << username << std::endl;
        return 1;
    }
    
    std::vector<faceid::FaceEncoding> stored_encodings = model.encodings;
    
    if (stored_encodings.empty()) {
        std::cerr << "Error: No face encodings found" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << stored_encodings.size() << " face encoding(s)" << std::endl;
    
    // Run integrity checks
    bool integrity_ok = checkEncodingIntegrity(model, true);
    if (!integrity_ok) {
        std::cout << "\n⚠ Continuing with live test despite integrity issues..." << std::endl;
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
    
    std::cout << "Using camera: " << device << std::endl;
    std::cout << "Recognition threshold: " << threshold << std::endl;
    std::cout << std::endl;
    
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
    
    std::cout << "Please look at the camera..." << std::endl;
    std::cout << "Testing for 5 seconds..." << std::endl;
    
    auto start = std::chrono::steady_clock::now();
    bool recognized = false;
    double detection_time_ms = 0.0;
    double recognition_time_ms = 0.0;
    
    while (std::chrono::duration_cast<std::chrono::seconds>(
           std::chrono::steady_clock::now() - start).count() < 5) {
        
        faceid::Image frame;
        if (!camera.read(frame)) {
            continue;
        }
        
        faceid::Image processed_frame = detector.preprocessFrame(frame.view());
        
        // Time face detection
        auto detect_start = std::chrono::high_resolution_clock::now();
        auto faces = detector.detectOrTrackFaces(processed_frame.view(), tracking_interval);
        auto detect_end = std::chrono::high_resolution_clock::now();
        detection_time_ms = std::chrono::duration<double, std::milli>(detect_end - detect_start).count();
        
        if (faces.empty()) {
            std::cout << "." << std::flush;
            continue;
        }
        
        // Time face recognition
        auto recog_start = std::chrono::high_resolution_clock::now();
        auto encodings = detector.encodeFaces(processed_frame.view(), faces);
        if (encodings.empty()) {
            continue;
        }
        
        // Compare with all stored encodings
        double min_distance = 999.0;
        for (const auto& stored : stored_encodings) {
            double distance = detector.compareFaces(stored, encodings[0]);
            if (distance < min_distance) {
                min_distance = distance;
            }
        }
        auto recog_end = std::chrono::high_resolution_clock::now();
        recognition_time_ms = std::chrono::duration<double, std::milli>(recog_end - recog_start).count();
        
        std::cout << std::endl;
        std::cout << "Face detected! Distance: " << min_distance;
        
        if (min_distance < threshold) {
            std::cout << " ✓ MATCH" << std::endl;
            std::cout << std::endl;
            std::cout << "Performance:" << std::endl;
            std::cout << "  Detection:    " << std::fixed << std::setprecision(2) << detection_time_ms << " ms" << std::endl;
            std::cout << "  Recognition:  " << std::fixed << std::setprecision(2) << recognition_time_ms << " ms" << std::endl;
            std::cout << "  Total:        " << std::fixed << std::setprecision(2) << (detection_time_ms + recognition_time_ms) << " ms" << std::endl;
            recognized = true;
            break;
        } else {
            std::cout << " ✗ NO MATCH" << std::endl;
        }
    }
    
    std::cout << std::endl;
    if (recognized) {
        std::cout << "✓ Face recognition successful for user: " << username << std::endl;
        return 0;
    } else {
        std::cout << "✗ Face recognition failed" << std::endl;
        return 1;
    }
}

} // namespace faceid
