#include "face_detector.h"
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <algorithm>
#include <cmath>

namespace faceid {

FaceDetector::FaceDetector() {
    // LibFaceDetection has embedded models - no initialization needed
}

bool FaceDetector::loadModels(const std::string& face_recognition_model_path) {
    try {
        // Load SFace recognition model (unchanged)
        sface_recognizer_ = cv::FaceRecognizerSF::create(
            face_recognition_model_path,
            ""  // config (empty for ONNX)
        );
        models_loaded_ = true;
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

std::vector<cv::Rect> FaceDetector::detectFaces(const cv::Mat& frame, bool downscale) {
    // Check cache first
    if (use_cache_) {
        uint64_t frame_hash = hashFrame(frame);
        auto it = detection_cache_.find(frame_hash);
        if (it != detection_cache_.end()) {
            return it->second;
        }
    }
    
    cv::Mat processed_frame = frame;
    double scale = 1.0;
    
    // Downscale for faster detection if enabled (optional - LibFaceDetection is already fast)
    if (downscale && frame.cols > 640) {
        scale = 640.0 / frame.cols;
        cv::resize(frame, processed_frame, cv::Size(), scale, scale, cv::INTER_AREA);
    }
    
    // LibFaceDetection detection
    // Allocate result buffer (fixed size required by LibFaceDetection)
    unsigned char result_buffer[0x20000];
    
    // Call LibFaceDetection
    // Note: LibFaceDetection expects BGR format (OpenCV default)
    int* pResults = facedetect_cnn(
        result_buffer,
        (unsigned char*)processed_frame.data,
        processed_frame.cols,
        processed_frame.rows,
        (int)processed_frame.step[0]
    );
    
    // Extract results
    std::vector<cv::Rect> faces;
    
    if (pResults) {
        int face_count = *pResults;  // First element is the count
        
        // IMPORTANT: facedetect_cnn returns data as short integers, not FaceRect structs!
        // Format: [count(int)] + N * [score(short), x(short), y(short), w(short), h(short), landmarks(10 shorts)]
        short* face_data = (short*)(pResults + 1);
        
        // Confidence threshold (score is stored as short, 0-100 scale)
        const int min_confidence_score = 60;  // 60/100 = 0.6
        
        for (int i = 0; i < face_count; i++) {
            // Each face has 16 shorts: score(1) + bbox(4) + landmarks(10) + padding(1)
            short* p = face_data + 16 * i;
            
            // Parse data (see facedetectcnn-model.cpp lines 228-239)
            int score = p[0];           // Score * 100 (0-100 range)
            int x = p[1];
            int y = p[2];
            int w = p[3];
            int h = p[4];
            // landmarks at p[5] through p[14]
            
            // Check confidence score
            if (score < min_confidence_score) {
                continue;  // Skip low confidence detections
            }
            
            // Scale back to original frame size if we downscaled
            if (scale != 1.0) {
                x = static_cast<int>(x / scale);
                y = static_cast<int>(y / scale);
                w = static_cast<int>(w / scale);
                h = static_cast<int>(h / scale);
            }
            
            // Convert to cv::Rect
            cv::Rect face(x, y, w, h);
            
            // Ensure rect is within frame bounds
            face &= cv::Rect(0, 0, frame.cols, frame.rows);
            if (face.width > 0 && face.height > 0) {
                faces.push_back(face);
            }
        }
    }
    
    // Cache result
    if (use_cache_) {
        uint64_t frame_hash = hashFrame(frame);
        detection_cache_[frame_hash] = faces;
        
        // Limit cache size
        if (detection_cache_.size() > 10) {
            detection_cache_.clear();
        }
    }
    
    return faces;
}

cv::Mat FaceDetector::alignFace(const cv::Mat& frame, const cv::Rect& face_rect) {
    // Extract face region
    cv::Mat face_img = frame(face_rect).clone();
    
    // SFace expects 112x112 aligned face
    cv::Mat aligned;
    cv::resize(face_img, aligned, cv::Size(112, 112));
    
    return aligned;
}

std::vector<FaceEncoding> FaceDetector::encodeFaces(
    const cv::Mat& frame,
    const std::vector<cv::Rect>& face_locations) {
    
    if (!models_loaded_ || face_locations.empty() || sface_recognizer_.empty()) {
        return {};
    }
    
    std::vector<FaceEncoding> encodings;
    
    for (const auto& face_rect : face_locations) {
        // Align face for SFace
        cv::Mat aligned = alignFace(frame, face_rect);
        
        // Extract feature using SFace
        cv::Mat encoding;
        sface_recognizer_->feature(aligned, encoding);
        
        encodings.push_back(encoding);
    }
    
    return encodings;
}

double FaceDetector::compareFaces(const FaceEncoding& encoding1, const FaceEncoding& encoding2) {
    // Manual cosine similarity calculation
    // Encodings can be either 1x128 or 128x1, both are valid
    
    // Ensure encodings are valid
    if (encoding1.empty() || encoding2.empty()) {
        return 999.0;  // Return large distance for invalid comparison
    }
    
    // Get total elements (should be 128 for both)
    int size1 = encoding1.rows * encoding1.cols;
    int size2 = encoding2.rows * encoding2.cols;
    
    if (size1 != size2 || size1 != 128) {
        return 999.0;  // Size mismatch
    }
    
    // Calculate dot product and norms (works for both row and column vectors)
    double dot_product = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;
    
    for (int i = 0; i < size1; i++) {
        float val1 = encoding1.at<float>(i);
        float val2 = encoding2.at<float>(i);
        dot_product += val1 * val2;
        norm1 += val1 * val1;
        norm2 += val2 * val2;
    }
    
    // Compute cosine similarity
    double cosine_sim = dot_product / (std::sqrt(norm1) * std::sqrt(norm2));
    
    // Convert to distance (lower = more similar)
    // Cosine similarity: 1 (identical) to -1 (opposite)
    // Distance: 0 (identical) to 2 (opposite)
    double distance = 1.0 - cosine_sim;
    
    return distance;
}

cv::Mat FaceDetector::preprocessFrame(const cv::Mat& frame) {
    cv::Mat processed;
    
    // Convert to RGB if needed
    if (frame.channels() == 4) {
        cv::cvtColor(frame, processed, cv::COLOR_BGRA2BGR);
    } else {
        processed = frame;
    }
    
    // Enhance contrast for better detection using optimized CLAHE
    cv::Mat lab;
    cv::cvtColor(processed, lab, cv::COLOR_BGR2Lab);
    
    std::vector<cv::Mat> lab_planes;
    cv::split(lab, lab_planes);
    
    // Apply CLAHE to L channel with optimized parameters
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(lab_planes[0], lab_planes[0]);
    
    cv::merge(lab_planes, lab);
    cv::cvtColor(lab, processed, cv::COLOR_Lab2BGR);
    
    return processed;
}

void FaceDetector::enableCache(bool enable) {
    use_cache_ = enable;
    if (!enable) {
        detection_cache_.clear();
    }
}

void FaceDetector::clearCache() {
    detection_cache_.clear();
}

uint64_t FaceDetector::hashFrame(const cv::Mat& frame) {
    // Simple hash based on frame dimensions and checksum of subset of pixels
    uint64_t hash = frame.rows * 10000ULL + frame.cols;
    
    // Sample some pixels for hash
    int step = std::max(1, frame.rows / 8);
    for (int i = 0; i < frame.rows; i += step) {
        for (int j = 0; j < frame.cols; j += step) {
            if (frame.channels() == 3) {
                cv::Vec3b pixel = frame.at<cv::Vec3b>(i, j);
                hash = hash * 31 + pixel[0] + pixel[1] + pixel[2];
            }
        }
    }
    
    return hash;
}

// Multi-face detection helpers for "no peek" feature
double FaceDetector::faceDistance(const cv::Rect& face1, const cv::Rect& face2) {
    // Calculate center points
    double cx1 = face1.x + face1.width / 2.0;
    double cy1 = face1.y + face1.height / 2.0;
    double cx2 = face2.x + face2.width / 2.0;
    double cy2 = face2.y + face2.height / 2.0;
    
    // Euclidean distance
    double dx = cx2 - cx1;
    double dy = cy2 - cy1;
    return std::sqrt(dx * dx + dy * dy);
}

bool FaceDetector::areDistinctFaces(const cv::Rect& face1, const cv::Rect& face2, int min_distance) {
    double distance = faceDistance(face1, face2);
    return distance >= min_distance;
}

double FaceDetector::getFaceSizePercent(const cv::Rect& face, int frame_width) {
    if (frame_width <= 0) return 0.0;
    return static_cast<double>(face.width) / frame_width;
}

int FaceDetector::countDistinctFaces(const std::vector<cv::Rect>& faces, int min_distance) {
    if (faces.empty()) return 0;
    if (faces.size() == 1) return 1;
    
    // Mark faces that are distinct
    std::vector<bool> is_distinct(faces.size(), true);
    
    // Compare each pair of faces
    for (size_t i = 0; i < faces.size(); i++) {
        if (!is_distinct[i]) continue;
        
        for (size_t j = i + 1; j < faces.size(); j++) {
            if (!is_distinct[j]) continue;
            
            // If faces are too close, mark the smaller one as not distinct
            if (!areDistinctFaces(faces[i], faces[j], min_distance)) {
                // Keep the larger face (closer to camera)
                int area_i = faces[i].width * faces[i].height;
                int area_j = faces[j].width * faces[j].height;
                
                if (area_i >= area_j) {
                    is_distinct[j] = false;
                } else {
                    is_distinct[i] = false;
                    break;  // Face i is not distinct, skip to next i
                }
            }
        }
    }
    
    // Count distinct faces
    int count = 0;
    for (bool distinct : is_distinct) {
        if (distinct) count++;
    }
    
    return count;
}

} // namespace faceid
