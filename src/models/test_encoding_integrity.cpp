// Test program to check encoding integrity and normalization
// Diagnoses false positive face recognition issues

#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>
#include <dirent.h>
#include <fnmatch.h>
#include "binary_model.h"
#include "config_paths.h"

// Find all model files for a user
std::vector<std::string> findUserModelFiles(const std::string& username) {
    std::vector<std::string> files;
    std::string models_dir = MODELS_DIR;
    
    DIR* dir = opendir(models_dir.c_str());
    if (!dir) {
        return files;
    }
    
    // Patterns: username.bin and username.*.bin
    std::vector<std::string> patterns = {
        username + ".bin",
        username + ".*.bin"
    };
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        
        // Check against all patterns
        for (const auto& pattern : patterns) {
            if (fnmatch(pattern.c_str(), filename.c_str(), 0) == 0) {
                files.push_back(models_dir + "/" + filename);
                break;  // Don't add same file twice
            }
        }
    }
    closedir(dir);
    
    std::sort(files.begin(), files.end());
    return files;
}


// Calculate L2 norm of a vector
float calculateNorm(const std::vector<float>& vec) {
    float sum = 0.0f;
    for (float val : vec) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

// Calculate dot product
float dotProduct(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    float sum = 0.0f;
    for (size_t i = 0; i < vec1.size(); i++) {
        sum += vec1[i] * vec2[i];
    }
    return sum;
}

// Calculate cosine distance
float cosineDistance(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    float dot = dotProduct(vec1, vec2);
    // Clamp dot product to [-1, 1] to handle floating point precision errors
    if (dot > 1.0f) dot = 1.0f;
    if (dot < -1.0f) dot = -1.0f;
    return 1.0f - dot;
}

// Check for NaN or Inf values
bool hasInvalidValues(const std::vector<float>& vec) {
    for (float val : vec) {
        if (std::isnan(val) || std::isinf(val)) {
            return true;
        }
    }
    return false;
}

// Get statistics for a vector
void printVectorStats(const std::vector<float>& vec, const std::string& name) {
    if (vec.empty()) {
        std::cout << name << ": EMPTY" << std::endl;
        return;
    }
    
    float min_val = vec[0], max_val = vec[0];
    float sum = 0.0f, sum_sq = 0.0f;
    
    for (float val : vec) {
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
        sum_sq += val * val;
    }
    
    float mean = sum / vec.size();
    float variance = (sum_sq / vec.size()) - (mean * mean);
    float stddev = std::sqrt(variance);
    float norm = calculateNorm(vec);
    
    std::cout << name << ": size=" << vec.size() 
              << ", min=" << std::fixed << std::setprecision(4) << min_val
              << ", max=" << max_val
              << ", mean=" << mean
              << ", stddev=" << stddev
              << ", norm=" << norm;
    
    if (hasInvalidValues(vec)) {
        std::cout << " ⚠ HAS NaN/Inf";
    }
    
    // Check if normalized
    if (std::abs(norm - 1.0f) < 0.01f) {
        std::cout << " ✓ normalized";
    } else {
        std::cout << " ✗ NOT normalized (should be ~1.0)";
    }
    
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    std::string username = "jenggo";
    if (argc > 1) {
        username = argv[1];
    }
    
    std::cout << "=== Face Encoding Integrity Test ===" << std::endl;
    std::cout << "User: " << username << std::endl;
    std::cout << std::endl;
    
    // Find all model files for this user
    auto model_files = findUserModelFiles(username);
    if (model_files.empty()) {
        std::cerr << "ERROR: No face models found for user: " << username << std::endl;
        return 1;
    }
    
    std::cout << "Found " << model_files.size() << " model file(s):" << std::endl;
    for (const auto& file : model_files) {
        std::cout << "  - " << file << std::endl;
    }
    std::cout << std::endl;
    
    // Load and merge all models
    faceid::BinaryFaceModel merged_model;
    merged_model.username = username;
    merged_model.valid = true;
    
    for (const auto& file : model_files) {
        faceid::BinaryFaceModel model;
        if (!faceid::BinaryModelLoader::loadUserModel(file, model) || !model.valid) {
            std::cerr << "WARNING: Failed to load " << file << std::endl;
            continue;
        }
        
        // Merge encodings and face_ids
        merged_model.encodings.insert(merged_model.encodings.end(),
                                     model.encodings.begin(), model.encodings.end());
        merged_model.face_ids.insert(merged_model.face_ids.end(),
                                    model.face_ids.begin(), model.face_ids.end());
    }
    
    std::cout << "Total encodings across all files: " << merged_model.encodings.size() << std::endl;
    std::cout << std::endl;
    
    // Use merged_model for analysis
    faceid::BinaryFaceModel& model = merged_model;
    
    // Analyze each encoding
    std::cout << "=== Encoding Analysis ===" << std::endl;
    for (size_t i = 0; i < model.encodings.size() && i < 10; i++) {
        printVectorStats(model.encodings[i], "Encoding[" + std::to_string(i) + "]");
    }
    
    if (model.encodings.size() > 10) {
        std::cout << "... (" << (model.encodings.size() - 10) << " more encodings)" << std::endl;
    }
    
    // Test self-similarity (encoding compared to itself)
    std::cout << "\n=== Self-Similarity Test ===" << std::endl;
    if (model.encodings.size() > 0) {
        float self_dot = dotProduct(model.encodings[0], model.encodings[0]);
        float self_dist = cosineDistance(model.encodings[0], model.encodings[0]);
        
        std::cout << "Encoding[0] compared to itself:" << std::endl;
        std::cout << "  Dot product: " << std::fixed << std::setprecision(6) << self_dot << std::endl;
        std::cout << "  Distance: " << self_dist << std::endl;
        
        if (self_dist < 0.0f) {
            std::cout << "  ✗ NEGATIVE DISTANCE! (dot product > 1.0)" << std::endl;
            std::cout << "  This indicates encodings are NOT properly normalized!" << std::endl;
        } else if (self_dist > 0.01f) {
            std::cout << "  ⚠ WARNING: Self-distance should be near 0.0" << std::endl;
        } else {
            std::cout << "  ✓ Self-distance looks good" << std::endl;
        }
    }
    
    // Test inter-encoding distances
    if (model.encodings.size() > 1) {
        std::cout << "\n=== Inter-Encoding Distances ===" << std::endl;
        std::cout << "Distance matrix (first 5 encodings):" << std::endl;
        
        size_t limit = std::min(size_t(5), model.encodings.size());
        for (size_t i = 0; i < limit; i++) {
            std::cout << "  [" << i << "]: ";
            for (size_t j = 0; j < limit; j++) {
                float dist = cosineDistance(model.encodings[i], model.encodings[j]);
                std::cout << std::setw(7) << std::fixed << std::setprecision(3) << dist << " ";
            }
            std::cout << std::endl;
        }
        
        // Find min/max distances between different encodings
        float min_dist = 999.0f, max_dist = -999.0f;
        for (size_t i = 0; i < model.encodings.size(); i++) {
            for (size_t j = i + 1; j < model.encodings.size(); j++) {
                float dist = cosineDistance(model.encodings[i], model.encodings[j]);
                if (dist < min_dist) min_dist = dist;
                if (dist > max_dist) max_dist = dist;
            }
        }
        
        std::cout << "\nDistance range between different encodings:" << std::endl;
        std::cout << "  Min: " << min_dist << std::endl;
        std::cout << "  Max: " << max_dist << std::endl;
        
        if (min_dist < 0.0f) {
            std::cout << "\n✗ CRITICAL: Negative distances found!" << std::endl;
            std::cout << "  This will cause FALSE POSITIVES in face matching!" << std::endl;
            std::cout << "  Root cause: Encodings are not L2-normalized" << std::endl;
        }
    }
    
    // Diagnostic summary
    std::cout << "\n=== Diagnostic Summary ===" << std::endl;
    
    bool has_issues = false;
    
    // Check for normalization
    bool all_normalized = true;
    for (const auto& enc : model.encodings) {
        float norm = calculateNorm(enc);
        if (std::abs(norm - 1.0f) > 0.01f) {
            all_normalized = false;
            break;
        }
    }
    
    if (!all_normalized) {
        std::cout << "✗ Encodings are NOT properly normalized (norm != 1.0)" << std::endl;
        std::cout << "  Solution: Re-enroll faces or normalize encodings" << std::endl;
        has_issues = true;
    } else {
        std::cout << "✓ All encodings are properly normalized" << std::endl;
    }
    
    // Check for invalid values
    bool has_invalid = false;
    for (const auto& enc : model.encodings) {
        if (hasInvalidValues(enc)) {
            has_invalid = true;
            break;
        }
    }
    
    if (has_invalid) {
        std::cout << "✗ Some encodings contain NaN or Inf values" << std::endl;
        std::cout << "  Solution: Re-enroll faces" << std::endl;
        has_issues = true;
    } else {
        std::cout << "✓ No NaN or Inf values found" << std::endl;
    }
    
    if (!has_issues) {
        std::cout << "\n✓ Encodings appear healthy" << std::endl;
        std::cout << "  If you're still seeing false positives, check:" << std::endl;
        std::cout << "  1. Threshold value (currently using < 0.35 from config)" << std::endl;
        std::cout << "  2. Face detection quality (lighting, angle, etc.)" << std::endl;
    }
    
    return has_issues ? 1 : 0;
}
