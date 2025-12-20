#include "face_detector.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <cmath>

namespace faceid {

FaceDetector::FaceDetector() {
    // LibFaceDetection has embedded models - no initialization needed
}

bool FaceDetector::loadModels(const std::string& face_recognition_model_path) {
    try {
        // Load NCNN SFace model
        // Expect model_path to be directory containing sface.param and sface.bin
        // Or model_path to be base path without extension (e.g., "/etc/faceid/models/sface")
        
        std::string param_path;
        std::string bin_path;
        
        // Check if path is a directory or file base
        if (face_recognition_model_path.find(".param") != std::string::npos) {
            // Full path to .param file given
            param_path = face_recognition_model_path;
            bin_path = face_recognition_model_path;
            bin_path.replace(bin_path.find(".param"), 6, ".bin");
        } else if (face_recognition_model_path.find(".onnx") != std::string::npos) {
            // ONNX path given - convert to NCNN paths
            // /etc/faceid/models/face_recognition_sface_2021dec.onnx
            // -> /etc/faceid/models/sface.param and sface.bin
            std::string base_dir = face_recognition_model_path.substr(0, face_recognition_model_path.find_last_of('/'));
            param_path = base_dir + "/sface.param";
            bin_path = base_dir + "/sface.bin";
        } else {
            // Assume it's a base path without extension
            param_path = face_recognition_model_path + ".param";
            bin_path = face_recognition_model_path + ".bin";
        }
        
        // Configure NCNN options for optimal CPU performance
        ncnn_net_.opt.use_vulkan_compute = false;  // Use CPU (more reliable for PAM)
        ncnn_net_.opt.num_threads = 4;  // Use 4 threads for good balance
        ncnn_net_.opt.use_fp16_packed = false;
        ncnn_net_.opt.use_fp16_storage = false;
        
        // Load model files
        int ret = ncnn_net_.load_param(param_path.c_str());
        if (ret != 0) {
            return false;
        }
        
        ret = ncnn_net_.load_model(bin_path.c_str());
        if (ret != 0) {
            return false;
        }
        
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
    
    // Always downscale for faster detection (LibFaceDetection is resolution-dependent)
    // Target: 320px width for ~7ms detection (vs 35ms at 640px)
    // Note: Detection accuracy is still excellent at lower resolutions
    const int target_width = 320;
    
    if (frame.cols > target_width) {
        scale = static_cast<double>(target_width) / frame.cols;
        cv::resize(frame, processed_frame, cv::Size(), scale, scale, cv::INTER_AREA);
    }
    
    // LibFaceDetection detection
    // Allocate result buffer (fixed size required by LibFaceDetection)
    // Note: Modern compilers optimize stack allocation efficiently
    // Stack allocation is actually faster than heap for small, fixed-size buffers
    unsigned char result_buffer[0x20000];
    
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

std::vector<cv::Rect> FaceDetector::detectOrTrackFaces(const cv::Mat& frame, int track_interval) {
    // Always detect if tracking disabled (track_interval == 0)
    if (track_interval == 0) {
        return detectFaces(frame);
    }
    
    // Detect if we haven't initialized tracking yet or interval reached
    if (!tracking_initialized_ || frames_since_detection_ >= track_interval) {
        // Run full detection
        std::vector<cv::Rect> faces = detectFaces(frame);
        
        // Initialize tracking
        if (!faces.empty()) {
            tracked_faces_ = faces;
            cv::cvtColor(frame, prev_gray_frame_, cv::COLOR_BGR2GRAY);
            tracking_initialized_ = true;
            frames_since_detection_ = 0;
        }
        
        return faces;
    }
    
    // Use tracking for intermediate frames
    frames_since_detection_++;
    return trackFaces(frame);
}

std::vector<cv::Rect> FaceDetector::trackFaces(const cv::Mat& current_frame) {
    if (tracked_faces_.empty() || prev_gray_frame_.empty()) {
        return {};
    }
    
    // Convert current frame to grayscale
    cv::Mat current_gray;
    cv::cvtColor(current_frame, current_gray, cv::COLOR_BGR2GRAY);
    
    // Track each face using sparse optical flow on face center points
    std::vector<cv::Rect> updated_faces;
    
    for (const auto& face : tracked_faces_) {
        // Get face center and corners
        std::vector<cv::Point2f> prev_points;
        prev_points.push_back(cv::Point2f(face.x + face.width/2, face.y + face.height/2));  // center
        prev_points.push_back(cv::Point2f(face.x, face.y));  // top-left
        prev_points.push_back(cv::Point2f(face.x + face.width, face.y + face.height));  // bottom-right
        
        // Track points using Lucas-Kanade optical flow
        std::vector<cv::Point2f> new_points;
        std::vector<uchar> status;
        std::vector<float> err;
        
        cv::calcOpticalFlowPyrLK(
            prev_gray_frame_, 
            current_gray,
            prev_points, 
            new_points,
            status,
            err,
            cv::Size(21, 21),  // window size
            3,                  // pyramid levels
            cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, 0.03)
        );
        
        // If tracking successful, update face position
        if (status[0] && status[1] && status[2]) {
            // Calculate movement delta from center point
            float dx = new_points[0].x - prev_points[0].x;
            float dy = new_points[0].y - prev_points[0].y;
            
            // Update face rectangle
            cv::Rect updated_face(
                face.x + static_cast<int>(dx),
                face.y + static_cast<int>(dy),
                face.width,
                face.height
            );
            
            // Ensure face is within frame bounds
            updated_face &= cv::Rect(0, 0, current_frame.cols, current_frame.rows);
            
            if (updated_face.width > 0 && updated_face.height > 0) {
                updated_faces.push_back(updated_face);
            }
        }
    }
    
    // Update tracking state
    tracked_faces_ = updated_faces;
    prev_gray_frame_ = current_gray;
    
    // If tracking lost all faces, force re-detection next frame
    if (updated_faces.empty()) {
        tracking_initialized_ = false;
    }
    
    return updated_faces;
}

void FaceDetector::resetTracking() {
    tracking_initialized_ = false;
    tracked_faces_.clear();
    prev_gray_frame_.release();
    frames_since_detection_ = 0;
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
    
    if (!models_loaded_ || face_locations.empty()) {
        return {};
    }
    
    std::vector<FaceEncoding> encodings;
    
    for (const auto& face_rect : face_locations) {
        // Align face for SFace (112x112)
        cv::Mat aligned = alignFace(frame, face_rect);
        
        // Convert to NCNN format (no manual normalization - model has built-in preprocessing)
        ncnn::Mat in = ncnn::Mat::from_pixels(
            aligned.data, 
            ncnn::Mat::PIXEL_BGR, 
            aligned.cols, 
            aligned.rows
        );
        
        // Create extractor and run inference
        ncnn::Extractor ex = ncnn_net_.create_extractor();
        ex.set_light_mode(true);  // Optimize for speed
        ex.input("data", in);
        
        // Extract features
        ncnn::Mat out;
        ex.extract("fc1", out);
        
        // Convert NCNN output to std::vector<float> and normalize
        FaceEncoding encoding(out.w);
        for (int i = 0; i < out.w; i++) {
            encoding[i] = out[i];
        }
        
        // L2 normalization
        float norm = 0.0f;
        for (float val : encoding) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        
        if (norm > 0) {
            for (float& val : encoding) {
                val /= norm;
            }
        }
        
        encodings.push_back(encoding);
    }
    
    return encodings;
}

double FaceDetector::compareFaces(const FaceEncoding& encoding1, const FaceEncoding& encoding2) {
    // Manual cosine similarity calculation for std::vector<float>
    
    // Ensure encodings are valid
    if (encoding1.empty() || encoding2.empty()) {
        return 999.0;  // Return large distance for invalid comparison
    }
    
    // Check size (should be 128 for both)
    if (encoding1.size() != encoding2.size() || encoding1.size() != 128) {
        return 999.0;  // Size mismatch
    }
    
    // Calculate dot product (encodings are already L2 normalized)
    double dot_product = 0.0;
    
    for (size_t i = 0; i < encoding1.size(); i++) {
        dot_product += encoding1[i] * encoding2[i];
    }
    
    // Since encodings are already normalized, dot product IS cosine similarity
    double cosine_sim = dot_product;
    
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
