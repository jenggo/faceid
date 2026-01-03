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
#include <thread>
#include <chrono>
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

// Camera quality metrics for enrollment validation
struct CameraQualityMetrics {
    float brightness;           // Average luminance (0.0-1.0)
    float contrast;            // Contrast measure (0.0-1.0)
    float focus_score;         // Encoding L2 norm
    float sharpness;           // Laplacian variance
    float encoding_norm;       // L2 norm of face encoding
    bool well_lit;             // brightness in range [0.3, 0.7]
    bool well_positioned;      // face centered, 20-60% of frame width
    bool good_contrast;        // contrast > 0.2
    bool good_focus;           // focus_score > 0.9 && sharpness > threshold
};

// Result of frame consistency validation
struct ConsistencyResult {
    bool is_consistent;                        // Overall success
    std::vector<std::vector<float>> encodings; // All 5 consecutive encodings
    std::vector<Rect> face_rects;             // All 5 face rectangles
    std::vector<float> distances;             // Distances between consecutive pairs (size=4)
    int best_frame_index;                     // Index of best quality frame (0-4)
    float best_quality_score;                 // Quality score of best frame
    float average_distance;                   // Average of distances
    float max_distance;                       // Maximum distance (for outlier detection)
    int frames_captured;                      // Number of frames successfully captured
    int total_attempts;                       // Total attempts (for timeout detection)
};

// Get model-aware consistency threshold based on recognition model
static inline float getConsistencyThreshold(const FaceDetector& detector) {
    std::string model_name = detector.getModelName();
    
    // Convert to lowercase for case-insensitive matching
    std::string model_lower = model_name;
    std::transform(model_lower.begin(), model_lower.end(), model_lower.begin(), ::tolower);
    
    // Model-specific thresholds based on typical intra-person distances
    if (model_lower.find("sface") != std::string::npos) {
        return 0.12f;  // SFace (128D) - very tight clustering
    } else if (model_lower.find("mobilefacenet") != std::string::npos || 
               model_lower.find("mobilenet") != std::string::npos) {
        return 0.15f;  // MobileFaceNet (192D)
    } else if (model_lower.find("arcface") != std::string::npos && 
               model_lower.find("r34") != std::string::npos) {
        return 0.18f;  // ArcFace ResNet-34 (256D)
    } else if (model_lower.find("glint360k") != std::string::npos || 
               model_lower.find("webface") != std::string::npos) {
        return 0.20f;  // Glint360K/WebFace (512D) - larger models have more variance
    }
    
    // Default: conservative threshold for unknown models
    return 0.15f;
}

// Calculate sharpness using Laplacian variance
static inline float calculateSharpness(const faceid::Image& img, const Rect& face_rect) {
    // Extract face region
    int x1 = std::max(0, face_rect.x);
    int y1 = std::max(0, face_rect.y);
    int x2 = std::min(img.width(), face_rect.x + face_rect.width);
    int y2 = std::min(img.height(), face_rect.y + face_rect.height);
    
    if (x2 <= x1 || y2 <= y1) return 0.0f;
    
    // Convert to grayscale if needed
    const uint8_t* data = img.data();
    int channels = img.channels();
    int stride = img.stride();
    
    // Calculate Laplacian variance on face region
    double sum = 0.0;
    double sum_sq = 0.0;
    int count = 0;
    
    for (int y = y1 + 1; y < y2 - 1; y++) {
        for (int x = x1 + 1; x < x2 - 1; x++) {
            // Get pixel intensity (average channels if RGB)
            float intensity = 0.0f;
            if (channels == 1) {
                intensity = data[y * stride + x];
            } else {
                for (int c = 0; c < channels; c++) {
                    intensity += data[y * stride + x * channels + c];
                }
                intensity /= channels;
            }
            
            // Apply Laplacian kernel (simplified 3x3)
            float center = intensity;
            float neighbors = 0.0f;
            
            // Get 4-connected neighbors
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    
                    int ny = y + dy;
                    int nx = x + dx;
                    
                    float n_intensity = 0.0f;
                    if (channels == 1) {
                        n_intensity = data[ny * stride + nx];
                    } else {
                        for (int c = 0; c < channels; c++) {
                            n_intensity += data[ny * stride + nx * channels + c];
                        }
                        n_intensity /= channels;
                    }
                    neighbors += n_intensity;
                }
            }
            
            // Laplacian response
            float response = std::abs(center * 8.0f - neighbors);
            sum += response;
            sum_sq += response * response;
            count++;
        }
    }
    
    if (count == 0) return 0.0f;
    
    // Calculate variance
    double mean = sum / count;
    double variance = (sum_sq / count) - (mean * mean);
    
    return static_cast<float>(variance);
}

// Validate camera quality for enrollment
static inline CameraQualityMetrics validateCameraQuality(
    const faceid::Image& frame,
    const Rect& face,
    const std::vector<float>& encoding
) {
    CameraQualityMetrics metrics = {};
    
    // Calculate brightness (average luminance)
    double lum_sum = 0.0;
    int pixel_count = 0;
    const uint8_t* data = frame.data();
    int stride = frame.stride();
    int channels = frame.channels();
    
    for (int y = 0; y < frame.height(); y++) {
        for (int x = 0; x < frame.width(); x++) {
            // Calculate luminance using standard weights
            if (channels == 3) {
                uint8_t b = data[y * stride + x * 3 + 0];
                uint8_t g = data[y * stride + x * 3 + 1];
                uint8_t r = data[y * stride + x * 3 + 2];
                lum_sum += 0.299 * r + 0.587 * g + 0.114 * b;
            } else {
                lum_sum += data[y * stride + x];
            }
            pixel_count++;
        }
    }
    metrics.brightness = static_cast<float>(lum_sum / (pixel_count * 255.0));
    
    // Calculate contrast (standard deviation of luminance)
    double lum_mean = lum_sum / pixel_count;
    double variance_sum = 0.0;
    for (int y = 0; y < frame.height(); y++) {
        for (int x = 0; x < frame.width(); x++) {
            double lum;
            if (channels == 3) {
                uint8_t b = data[y * stride + x * 3 + 0];
                uint8_t g = data[y * stride + x * 3 + 1];
                uint8_t r = data[y * stride + x * 3 + 2];
                lum = 0.299 * r + 0.587 * g + 0.114 * b;
            } else {
                lum = data[y * stride + x];
            }
            variance_sum += (lum - lum_mean) * (lum - lum_mean);
        }
    }
    double std_dev = std::sqrt(variance_sum / pixel_count);
    metrics.contrast = static_cast<float>(std_dev / 255.0);
    
    // Calculate encoding norm (focus score)
    if (!encoding.empty()) {
        float norm_sq = 0.0f;
        for (float val : encoding) {
            norm_sq += val * val;
        }
        metrics.encoding_norm = std::sqrt(norm_sq);
        metrics.focus_score = metrics.encoding_norm;
    }
    
    // Calculate sharpness
    metrics.sharpness = calculateSharpness(frame, face);
    
    // Evaluate quality flags
    metrics.well_lit = (metrics.brightness >= 0.3f && metrics.brightness <= 0.7f);
    metrics.good_contrast = (metrics.contrast > 0.2f);
    
    // Face positioning: centered and 20-60% of frame width
    float face_width_ratio = static_cast<float>(face.width) / frame.width();
    float face_center_x = (face.x + face.width / 2.0f) / frame.width();
    float face_center_y = (face.y + face.height / 2.0f) / frame.height();
    metrics.well_positioned = (face_width_ratio >= 0.20f && face_width_ratio <= 0.60f &&
                               face_center_x >= 0.3f && face_center_x <= 0.7f &&
                               face_center_y >= 0.3f && face_center_y <= 0.7f);
    
    // Focus: encoding norm close to 1.0 and high sharpness
    metrics.good_focus = (metrics.encoding_norm >= 0.9f && metrics.encoding_norm <= 1.1f &&
                          metrics.sharpness > 50.0f);  // Sharpness threshold is heuristic
    
    return metrics;
}

// Calculate frame quality score (60% encoding norm + 40% sharpness)
static inline float calculateFrameQualityScore(
    float encoding_norm,
    float sharpness
) {
    // Normalize encoding norm to [0, 1] range (assuming ideal is 1.0)
    float norm_score = 1.0f - std::abs(1.0f - encoding_norm);
    norm_score = std::max(0.0f, std::min(1.0f, norm_score));
    
    // Normalize sharpness (assuming typical range is 0-200, good is > 50)
    float sharpness_score = std::min(1.0f, sharpness / 200.0f);
    
    // Combined score: 60% encoding quality + 40% sharpness
    return 0.6f * norm_score + 0.4f * sharpness_score;
}

// Validate frame consistency: capture 5 consecutive frames where face is stable
// Returns ConsistencyResult with encodings, rects, and quality metrics
// Implements auto-relax if user can't hold still (silent, as requested)
static inline ConsistencyResult validateFrameConsistency(
    Camera& camera,
    FaceDetector& detector,
    Display& display,
    float base_threshold,
    int sample_index,
    const std::string& prompt,
    int num_samples,
    float optimal_confidence,
    int tracking_interval
) {
    ConsistencyResult result = {};
    result.is_consistent = false;
    result.frames_captured = 0;
    result.total_attempts = 0;
    result.best_frame_index = -1;
    result.best_quality_score = 0.0f;
    
    const int MAX_ATTEMPTS = 150;  // 150 frames * 50ms = 7.5 seconds timeout
    const int REQUIRED_FRAMES = 5;
    
    float current_threshold = base_threshold;
    int relax_count = 0;
    const int MAX_RELAX = 3;
    const float RELAX_FACTOR = 1.25f;
    const float MAX_RELAX_FACTOR = 1.5f;
    
    std::vector<faceid::Image> captured_frames;
    
    while (result.total_attempts < MAX_ATTEMPTS && relax_count < MAX_RELAX) {
        result.total_attempts++;
        
        // Read frame
        faceid::Image frame;
        if (!camera.read(frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        
        // Preprocess and detect
        faceid::Image processed_frame = detector.preprocessFrame(frame.view());
        auto faces = detector.detectOrTrackFaces(processed_frame.view(), tracking_interval, optimal_confidence);
        
        // Only proceed if exactly 1 face detected
        if (faces.size() != 1) {
            // Reset if multiple faces or no face
            result.encodings.clear();
            result.face_rects.clear();
            result.distances.clear();
            result.frames_captured = 0;
            captured_frames.clear();
            
            // Draw feedback
            faceid::Image display_frame = frame.clone();
            for (const auto& face : faces) {
                faceid::Color color = faceid::Color::Red();
                faceid::drawRectangle(display_frame, face.x, face.y, face.width, face.height, color, 2);
            }
            
            std::string status_text = prompt + " - " + 
                (faces.empty() ? "Waiting for face..." : "Multiple faces detected");
            faceid::drawFilledRectangle(display_frame, 0, 0, display_frame.width(), 40, faceid::Color::Black());
            std::string status_text_reversed = status_text;
            std::reverse(status_text_reversed.begin(), status_text_reversed.end());
            int status_width = status_text_reversed.length() * 8;
            faceid::drawText(display_frame, status_text_reversed, 
                           display_frame.width() - 10 - status_width, 10, 
                           faceid::Color::Orange(), 1.0);
            
            // Progress bar
            int progress_width = (display_frame.width() * sample_index) / num_samples;
            faceid::drawFilledRectangle(display_frame, 0, display_frame.height() - 10, 
                                       progress_width, 10, faceid::Color::Green());
            
            display.show(display_frame);
            display.waitKey(50);
            continue;
        }
        
        // Encode face
        auto encodings = detector.encodeFaces(processed_frame.view(), faces);
        if (encodings.empty()) {
            continue;
        }
        
        const auto& face = faces[0];
        const auto& encoding = encodings[0];
        
        // If this is our first frame, just add it
        if (result.encodings.empty()) {
            result.encodings.push_back(encoding);
            result.face_rects.push_back(face);
            captured_frames.push_back(frame.clone());
            result.frames_captured = 1;
        } else {
            // Check consistency with previous frame
            float distance = cosineDistance(result.encodings.back(), encoding);
            
            if (distance < current_threshold) {
                // Consistent! Add to collection
                result.encodings.push_back(encoding);
                result.face_rects.push_back(face);
                result.distances.push_back(distance);
                captured_frames.push_back(frame.clone());
                result.frames_captured++;
                
                // Check if we have enough frames
                if (result.frames_captured >= REQUIRED_FRAMES) {
                    result.is_consistent = true;
                    
                    // Calculate statistics
                    float sum = 0.0f;
                    result.max_distance = 0.0f;
                    for (float d : result.distances) {
                        sum += d;
                        if (d > result.max_distance) {
                            result.max_distance = d;
                        }
                    }
                    result.average_distance = sum / result.distances.size();
                    
                    // Find best quality frame
                    for (int i = 0; i < REQUIRED_FRAMES; i++) {
                        float norm = 0.0f;
                        for (float val : result.encodings[i]) {
                            norm += val * val;
                        }
                        norm = std::sqrt(norm);
                        
                        float sharpness = calculateSharpness(captured_frames[i], result.face_rects[i]);
                        float quality_score = calculateFrameQualityScore(norm, sharpness);
                        
                        if (quality_score > result.best_quality_score) {
                            result.best_quality_score = quality_score;
                            result.best_frame_index = i;
                        }
                    }
                    
                    break;  // Success!
                }
            } else {
                // Not consistent - reset and try again
                result.encodings.clear();
                result.face_rects.clear();
                result.distances.clear();
                result.frames_captured = 0;
                captured_frames.clear();
                
                // Check for timeout and auto-relax (silent)
                if (result.total_attempts > 50 * (relax_count + 1)) {
                    relax_count++;
                    float new_threshold = base_threshold * std::pow(RELAX_FACTOR, relax_count);
                    if (new_threshold <= base_threshold * MAX_RELAX_FACTOR) {
                        current_threshold = new_threshold;
                    } else {
                        break;  // Max relax reached
                    }
                }
            }
        }
        
        // Draw live feedback
        faceid::Image display_frame = frame.clone();
        faceid::Color color = faceid::Color::Green();
        faceid::drawRectangle(display_frame, face.x, face.y, face.width, face.height, color, 2);
        
        // Draw landmarks if available
        if (face.hasLandmarks()) {
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
        
        // Status text with progress
        std::string status_text = prompt + " - Holding steady... " + 
                                 std::to_string(result.frames_captured) + "/" + 
                                 std::to_string(REQUIRED_FRAMES);
        
        faceid::drawFilledRectangle(display_frame, 0, 0, display_frame.width(), 40, faceid::Color::Black());
        std::string status_text_reversed = status_text;
        std::reverse(status_text_reversed.begin(), status_text_reversed.end());
        int status_width = status_text_reversed.length() * 8;
        faceid::drawText(display_frame, status_text_reversed, 
                       display_frame.width() - 10 - status_width, 10, 
                       faceid::Color::Green(), 1.0);
        
        // Progress bar showing overall sample progress
        int base_progress = (display_frame.width() * sample_index) / num_samples;
        int consistency_progress = (display_frame.width() * result.frames_captured) / (num_samples * REQUIRED_FRAMES);
        int total_progress = base_progress + consistency_progress;
        faceid::drawFilledRectangle(display_frame, 0, display_frame.height() - 10, 
                                   total_progress, 10, faceid::Color::Green());
        
        display.show(display_frame);
        int key = display.waitKey(50);
        if (key == 'q' || key == 'Q' || key == 27 || !display.isOpen()) {
            result.is_consistent = false;
            return result;
        }
    }
    
    return result;
}

// Helper: Find optimal detection confidence from camera feed
// Analyzes 10-15 frames and also validates camera quality
// NOTE: This is used during enrollment (faceid add) where the two-phase capture
// already handles waiting for face detection. Don't call this directly from test command.
static inline float findOptimalDetectionConfidence(Camera& camera, FaceDetector& detector, Display& display) {
    std::cout << std::endl;
    std::cout << "=== Auto-Detecting Optimal Settings ===" << std::endl;
    std::cout << "Analyzing camera conditions and finding best detection settings..." << std::endl;
    std::cout << std::endl;
    
    // Capture 10-15 frames for analysis
    const int NUM_ANALYSIS_FRAMES = 15;
    std::vector<faceid::Image> frames;
    std::vector<faceid::Image> processed_frames;
    
    std::cout << "Capturing " << NUM_ANALYSIS_FRAMES << " frames for analysis..." << std::endl;
    
    for (int attempts = 0; attempts < 50 && frames.size() < NUM_ANALYSIS_FRAMES; attempts++) {
        faceid::Image frame;
        if (!camera.read(frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        
        // Show preview
        display.show(frame);
        display.waitKey(30);
        
        faceid::Image processed_frame = detector.preprocessFrame(frame.view());
        
        // Quick check if any face is detected at lower confidence (0.3 to be more lenient)
        auto test_faces = detector.detectFaces(processed_frame.view(), false, 0.3f);
        if (!test_faces.empty()) {
            frames.push_back(frame.clone());
            processed_frames.push_back(std::move(processed_frame));
            std::cout << "  Frame " << frames.size() << "/" << NUM_ANALYSIS_FRAMES << " (detected " << test_faces.size() << " face(s))\r" << std::flush;
        }
    }
    std::cout << std::endl;
    
    if (frames.empty()) {
        std::cerr << "Failed to capture frames for confidence analysis" << std::endl;
        return -1.0f;
    }
    
    std::cout << "Captured " << frames.size() << " frames, analyzing..." << std::endl;
    
    // Use the first frame for dimensions
    int img_width = frames[0].width();
    int img_height = frames[0].height();
    
    // Collect camera quality metrics across all frames
    std::vector<float> brightness_values;
    std::vector<float> contrast_values;
    std::vector<float> encoding_norms;
    std::vector<float> sharpness_values;
    
    // Helper lambda to count valid faces at a given confidence across all frames
    auto countValidFacesMultiFrame = [&](float conf) -> std::pair<int, int> {
        int frames_with_one_face = 0;
        int total_valid_faces = 0;
        
        for (size_t f = 0; f < processed_frames.size(); f++) {
            auto faces = detector.detectFaces(processed_frames[f].view(), false, conf);
            auto encodings = detector.encodeFaces(processed_frames[f].view(), faces);
            
            int valid_count = 0;
            for (size_t i = 0; i < faces.size(); i++) {
                std::vector<float> encoding = (i < encodings.size()) ? encodings[i] : std::vector<float>();
                if (isValidFace(faces[i], img_width, img_height, encoding)) {
                    valid_count++;
                    
                    // Collect quality metrics from first valid face
                    if (valid_count == 1 && !encoding.empty()) {
                        CameraQualityMetrics metrics = validateCameraQuality(
                            frames[f], faces[i], encoding);
                        
                        brightness_values.push_back(metrics.brightness);
                        contrast_values.push_back(metrics.contrast);
                        encoding_norms.push_back(metrics.encoding_norm);
                        sharpness_values.push_back(metrics.sharpness);
                    }
                }
            }
            
            if (valid_count == 1) {
                frames_with_one_face++;
            }
            total_valid_faces += valid_count;
        }
        
        return {frames_with_one_face, total_valid_faces};
    };
    
    // Binary search for optimal confidence threshold
    float low = 0.30f;
    float high = 0.99f;
    float found_confidence = -1.0f;
    int best_consistent_frames = 0;
    
    // First, do a coarse linear search to find a good starting range
    float coarse_step = 0.10f;
    for (float conf = low; conf <= high; conf += coarse_step) {
        auto [consistent_frames, total_faces] = countValidFacesMultiFrame(conf);
        
        if (consistent_frames >= processed_frames.size() * 0.7) {  // 70% consistency
            // Found a good candidate, now refine with binary search
            low = std::max(0.30f, conf - coarse_step);
            high = std::min(0.99f, conf + coarse_step);
            best_consistent_frames = consistent_frames;
            found_confidence = conf;
            break;
        } else if (consistent_frames == 0) {
            // Went too high
            high = conf;
            break;
        }
    }
    
    // Binary search refinement with 0.01 precision
    while (high - low > 0.01f) {
        float mid = (low + high) / 2.0f;
        auto [consistent_frames, total_faces] = countValidFacesMultiFrame(mid);
        
        if (consistent_frames >= processed_frames.size() * 0.7) {
            found_confidence = mid;
            best_consistent_frames = consistent_frames;
            high = mid;  // Try to find lower confidence
        } else if (total_faces > consistent_frames * 2) {
            // Too many faces detected, increase confidence
            low = mid;
        } else {
            // Not enough detections, decrease confidence
            high = mid;
        }
    }
    
    // If not found yet, try the final candidate
    if (found_confidence < 0.0f) {
        auto [consistent_frames, total_faces] = countValidFacesMultiFrame(low);
        if (consistent_frames >= processed_frames.size() * 0.5) {  // Relax to 50%
            found_confidence = low;
            best_consistent_frames = consistent_frames;
        }
    }
    
    // Report results
    if (found_confidence > 0.0f) {
        std::cout << "✓ Optimal detection confidence found: " << std::fixed << std::setprecision(2) 
                  << found_confidence << std::endl;
        std::cout << "  Consistent detection in " << best_consistent_frames << "/" 
                  << processed_frames.size() << " frames" << std::endl;
        
        // Report camera quality if we collected metrics
        if (!brightness_values.empty()) {
            float avg_brightness = 0.0f, avg_contrast = 0.0f, avg_sharpness = 0.0f, avg_norm = 0.0f;
            for (size_t i = 0; i < brightness_values.size(); i++) {
                avg_brightness += brightness_values[i];
                avg_contrast += contrast_values[i];
                avg_sharpness += sharpness_values[i];
                avg_norm += encoding_norms[i];
            }
            avg_brightness /= brightness_values.size();
            avg_contrast /= contrast_values.size();
            avg_sharpness /= sharpness_values.size();
            avg_norm /= encoding_norms.size();
            
            std::cout << std::endl;
            std::cout << "Camera Quality Assessment:" << std::endl;
            std::cout << "  Brightness: " << std::fixed << std::setprecision(2) 
                      << (avg_brightness * 100) << "% " 
                      << (avg_brightness >= 0.3f && avg_brightness <= 0.7f ? "✓" : "⚠") << std::endl;
            std::cout << "  Contrast:   " << std::fixed << std::setprecision(2) 
                      << (avg_contrast * 100) << "% " 
                      << (avg_contrast > 0.2f ? "✓" : "⚠") << std::endl;
            std::cout << "  Sharpness:  " << std::fixed << std::setprecision(1) 
                      << avg_sharpness << " " 
                      << (avg_sharpness > 50.0f ? "✓" : "⚠") << std::endl;
            std::cout << "  Focus:      " << std::fixed << std::setprecision(3) 
                      << avg_norm << " " 
                      << (avg_norm >= 0.9f && avg_norm <= 1.1f ? "✓" : "⚠") << std::endl;
            
            // Warnings for poor conditions
            if (avg_brightness < 0.3f) {
                std::cout << std::endl;
                std::cout << "⚠ Low lighting detected - consider improving lighting for better results" << std::endl;
            } else if (avg_brightness > 0.7f) {
                std::cout << std::endl;
                std::cout << "⚠ Very bright lighting - consider reducing brightness to avoid overexposure" << std::endl;
            }
            
            if (avg_contrast < 0.2f) {
                std::cout << "⚠ Low contrast - check lighting or camera settings" << std::endl;
            }
            
            if (avg_sharpness < 50.0f) {
                std::cout << "⚠ Low sharpness - check camera focus or clean lens" << std::endl;
            }
        }
        
        std::cout << std::endl;
        std::cout << "Proceeding with enrollment..." << std::endl;
    } else {
        std::cerr << "⚠ Could not auto-detect optimal confidence" << std::endl;
        std::cerr << "  Using default value (0.8 for " << detector.getDetectionModelType() << ")" << std::endl;
        
        // Set reasonable default based on model type
        found_confidence = 0.8f;  // RetinaFace, YuNet, YOLOv5/v7/v8-Face
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
