#ifndef FACEID_CLI_HELPERS_H
#define FACEID_CLI_HELPERS_H

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <algorithm>
#include "../face_detector.h"
#include "../camera.h"
#include "../display.h"

namespace faceid {

// Helper: Calculate cosine distance between two face encodings
static inline float cosineDistance(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    float dot = 0.0f;
    for (size_t i = 0; i < vec1.size(); i++) {
        dot += vec1[i] * vec2[i];
    }
    // Clamp dot product to [-1, 1] to handle floating point precision errors
    if (dot > 1.0f) dot = 1.0f;
    if (dot < -1.0f) dot = -1.0f;
    return 1.0f - dot;
}

// Helper: Validate if a detected face is likely a real face
static inline bool isValidFace(const Rect& face, int img_width, int img_height, 
                               const std::vector<float>& encoding) {
    // Check 1: Face size (should be 10-80% of image width)
    float face_width_ratio = (float)face.width / img_width;
    if (face_width_ratio < 0.10f || face_width_ratio > 0.80f) {
        return false;
    }
    
    // Check 2: Aspect ratio (faces should be roughly 1:1 to 1:1.5 - width:height)
    float aspect_ratio = (float)face.width / face.height;
    if (aspect_ratio < 0.6f || aspect_ratio > 1.8f) {
        return false;
    }
    
    // Check 3: Position (face center should be in middle 80% of image)
    float face_center_x = (face.x + face.width / 2.0f) / img_width;
    float face_center_y = (face.y + face.height / 2.0f) / img_height;
    if (face_center_x < 0.1f || face_center_x > 0.9f || 
        face_center_y < 0.1f || face_center_y > 0.9f) {
        return false;
    }
    
    // Check 4: Encoding quality (L2 norm should be close to 1.0 for normalized embeddings)
    if (!encoding.empty()) {
        float norm = 0.0f;
        for (float val : encoding) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        
        // Good face encodings should have norm between 0.90 and 1.10
        if (norm < 0.90f || norm > 1.10f) {
            return false;
        }
    }
    
    return true;
}

// Helper: Find optimal detection confidence from camera feed
// NOTE: This is used during enrollment (faceid add) where the two-phase capture
// already handles waiting for face detection. Don't call this directly from test command.
static inline float findOptimalDetectionConfidence(Camera& camera, FaceDetector& detector, Display& display) {
    std::cout << std::endl;
    std::cout << "=== Auto-Detecting Optimal Confidence ===" << std::endl;
    std::cout << "Analyzing your face to find best detection settings..." << std::endl;
    std::cout << std::endl;
    
    // Capture a reference frame
    faceid::Image frame;
    for (int attempts = 0; attempts < 10; attempts++) {
        if (!camera.read(frame)) {
            continue;
        }
        
        // Show preview
        display.show(frame);
        display.waitKey(30);
        
        faceid::Image processed_frame = detector.preprocessFrame(frame.view());
        
        // Quick check if any face is detected at default confidence
        auto test_faces = detector.detectFaces(processed_frame.view(), false, 0.5f);
        if (!test_faces.empty()) {
            break;
        }
    }
    
    if (frame.data() == nullptr) {
        std::cerr << "Failed to capture frame for confidence analysis" << std::endl;
        return -1.0f;
    }
    
    faceid::Image processed_frame = detector.preprocessFrame(frame.view());
    int img_width = frame.width();
    int img_height = frame.height();
    
    // Helper lambda to count valid faces at a given confidence
    auto countValidFaces = [&](float conf) -> int {
        auto faces = detector.detectFaces(processed_frame.view(), false, conf);
        auto encodings = detector.encodeFaces(processed_frame.view(), faces);
        
        int valid_count = 0;
        for (size_t i = 0; i < faces.size(); i++) {
            std::vector<float> encoding = (i < encodings.size()) ? encodings[i] : std::vector<float>();
            if (isValidFace(faces[i], img_width, img_height, encoding)) {
                valid_count++;
            }
        }
        return valid_count;
    };
    
    // Binary search for optimal confidence threshold
    float low = 0.30f;
    float high = 0.99f;
    float found_confidence = -1.0f;
    
    // First, do a coarse linear search to find a good starting range
    float coarse_step = 0.10f;
    for (float conf = low; conf <= high; conf += coarse_step) {
        int valid_count = countValidFaces(conf);
        
        if (valid_count == 1) {
            // Found a good candidate, now refine with binary search
            low = std::max(0.30f, conf - coarse_step);
            high = std::min(0.99f, conf + coarse_step);
            break;
        } else if (valid_count == 0) {
            // Went too high
            high = conf;
            break;
        }
    }
    
    // Binary search refinement with 0.01 precision
    while (high - low > 0.01f) {
        float mid = (low + high) / 2.0f;
        int valid_count = countValidFaces(mid);
        
        if (valid_count == 1) {
            found_confidence = mid;
            high = mid;  // Try to find lower confidence
        } else if (valid_count > 1) {
            // Too many faces, increase confidence
            low = mid;
        } else {
            // No faces, decrease confidence
            high = mid;
        }
    }
    
    // If not found yet, try the final candidate
    if (found_confidence < 0.0f) {
        int valid_count = countValidFaces(low);
        if (valid_count == 1) {
            found_confidence = low;
        }
    }
    
    if (found_confidence > 0.0f) {
        std::cout << "✓ Optimal detection confidence found: " << std::fixed << std::setprecision(2) 
                  << found_confidence << std::endl;
        std::cout << "  This will be used for your enrollment" << std::endl;
    } else {
        std::cerr << "⚠ Could not auto-detect optimal confidence" << std::endl;
        std::cerr << "  Using default value (0.5 for " << detector.getDetectionModelType() << ")" << std::endl;
        
        // Set reasonable default based on model type
        if (detector.getDetectionModelType() == "SCRFD" || detector.getDetectionModelType() == "UltraFace") {
            found_confidence = 0.5f;
        } else {
            found_confidence = 0.8f;  // RetinaFace, YuNet
        }
    }
    
    return found_confidence;
}

// Helper: Update config file with new confidence and threshold values
static inline bool updateConfigFile(const std::string& config_path, float confidence, float threshold) {
    std::cout << std::endl;
    std::cout << "=== Updating Configuration ===" << std::endl;
    
    // Read the entire config file
    std::ifstream infile(config_path);
    if (!infile.is_open()) {
        std::cerr << "Failed to open config file: " << config_path << std::endl;
        return false;
    }
    
    std::vector<std::string> lines;
    std::string line;
    bool in_recognition_section = false;
    bool confidence_updated = false;
    bool threshold_updated = false;
    
    while (std::getline(infile, line)) {
        // Check if we're in the [recognition] section
        if (line.find("[recognition]") != std::string::npos) {
            in_recognition_section = true;
            lines.push_back(line);
            continue;
        }
        
        // Check if we're leaving the recognition section
        if (in_recognition_section && line.find("[") != std::string::npos) {
            in_recognition_section = false;
        }
        
        // Update confidence value
        if (in_recognition_section && line.find("confidence") != std::string::npos && line.find("=") != std::string::npos) {
            std::ostringstream oss;
            oss << "confidence = " << std::fixed << std::setprecision(2) << confidence;
            lines.push_back(oss.str());
            confidence_updated = true;
            continue;
        }
        
        // Update threshold value
        if (in_recognition_section && line.find("threshold") != std::string::npos && line.find("=") != std::string::npos) {
            std::ostringstream oss;
            oss << "threshold = " << std::fixed << std::setprecision(2) << threshold;
            lines.push_back(oss.str());
            threshold_updated = true;
            continue;
        }
        
        lines.push_back(line);
    }
    infile.close();
    
    // If values weren't found, add them to the recognition section
    if (!confidence_updated || !threshold_updated) {
        in_recognition_section = false;
        for (size_t i = 0; i < lines.size(); i++) {
            if (lines[i].find("[recognition]") != std::string::npos) {
                in_recognition_section = true;
                if (!confidence_updated) {
                    std::ostringstream oss;
                    oss << "confidence = " << std::fixed << std::setprecision(2) << confidence;
                    lines.insert(lines.begin() + i + 1, oss.str());
                    i++;
                    confidence_updated = true;
                }
                if (!threshold_updated) {
                    std::ostringstream oss;
                    oss << "threshold = " << std::fixed << std::setprecision(2) << threshold;
                    lines.insert(lines.begin() + i + 1, oss.str());
                    threshold_updated = true;
                }
                break;
            }
        }
    }
    
    // Try to write back to file
    std::ofstream outfile(config_path);
    if (!outfile.is_open()) {
        // No write permission - show recommendations instead
        std::cout << "⚠ Cannot write to config file (no permission)" << std::endl;
        std::cout << std::endl;
        std::cout << "=== Recommended Configuration ===" << std::endl;
        std::cout << "Please update your config file manually:" << std::endl;
        std::cout << std::endl;
        std::cout << "File: " << config_path << std::endl;
        std::cout << std::endl;
        std::cout << "[recognition]" << std::endl;
        std::cout << "confidence = " << std::fixed << std::setprecision(2) << confidence << std::endl;
        std::cout << "threshold = " << std::fixed << std::setprecision(2) << threshold << std::endl;
        std::cout << std::endl;
        std::cout << "Or run with sudo to update automatically:" << std::endl;
        std::cout << "  sudo faceid test <username> --auto-adjust" << std::endl;
        std::cout << std::endl;
        return false;  // Return false but don't fail - caller can continue
    }
    
    for (const auto& l : lines) {
        outfile << l << std::endl;
    }
    outfile.close();
    
    std::cout << "✓ Configuration updated successfully!" << std::endl;
    std::cout << "  File: " << config_path << std::endl;
    std::cout << "  Detection confidence: " << std::fixed << std::setprecision(2) << confidence << std::endl;
    std::cout << "  Recognition threshold: " << std::fixed << std::setprecision(2) << threshold << std::endl;
    
    return true;
}

} // namespace faceid

#endif // FACEID_CLI_HELPERS_H
