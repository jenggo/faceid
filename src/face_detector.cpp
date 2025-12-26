#include "face_detector.h"
#include "clahe.h"
#include "optical_flow.h"
#include "config_paths.h"
#include <libyuv.h>
#include <algorithm>
#include <cmath>

namespace faceid {

// Helper function: Fast image resize using libyuv (3-5x faster than OpenCV)
// Supports BGR24 format
static Image resizeImage(const uint8_t* src_data, int src_width, int src_height, int src_stride, int dst_width, int dst_height) {
    // Allocate intermediate ARGB buffers
    Image src_argb(src_width, src_height, 4);
    libyuv::RGB24ToARGB(src_data, src_stride, src_argb.data(), src_argb.stride(), src_width, src_height);
    
    // Allocate destination in ARGB format
    Image dst_argb(dst_width, dst_height, 4);
    
    // Use libyuv for fast scaling (kFilterBilinear provides good quality with excellent speed)
    libyuv::ARGBScale(
        src_argb.data(), src_argb.stride(),
        src_argb.width(), src_argb.height(),
        dst_argb.data(), dst_argb.stride(),
        dst_argb.width(), dst_argb.height(),
        libyuv::kFilterBilinear
    );
    
    // Convert back to BGR
    Image result(dst_width, dst_height, 3);
    libyuv::ARGBToRGB24(dst_argb.data(), dst_argb.stride(), result.data(), result.stride(), dst_width, dst_height);
    return result;
}

// Helper function: Fast BGR to GRAY conversion using libyuv (2-3x faster than OpenCV)
static Image toGrayscale(const uint8_t* src_data, int src_width, int src_height, int src_stride) {
    Image dst_gray(src_width, src_height, 1);
    
    // Use libyuv's RGB24ToJ400 (grayscale) conversion
    // Note: OpenCV's BGR = RGB24, J400 is grayscale (full range 0-255)
    libyuv::RGB24ToJ400(src_data, src_stride, dst_gray.data(), dst_gray.stride(), src_width, src_height);
    return dst_gray;
}

FaceDetector::FaceDetector() {
    // LibFaceDetection has embedded models - no initialization needed
}

bool FaceDetector::loadModels(const std::string& model_base_path) {
    try {
        // Load NCNN SFace model
        // Default to MODELS_DIR/sface if no path provided
        std::string base_path = model_base_path.empty() 
            ? std::string(MODELS_DIR) + "/sface"
            : model_base_path;
        
        // Build paths to .param and .bin files
        std::string param_path = base_path + ".param";
        std::string bin_path = base_path + ".bin";
        
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
        
        // Validate model by attempting to create an extractor and check layer exists
        // This catches subtle corruption that load_param/load_model might miss
        try {
            ncnn::Extractor ex = ncnn_net_.create_extractor();
            // Just creating the extractor validates the model structure
            // We don't need to run full inference here
        } catch (...) {
            // Model loaded but can't create extractor - likely corrupted
            return false;
        }
        
        models_loaded_ = true;
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

std::vector<Rect> FaceDetector::detectFaces(const ImageView& frame, bool downscale) {
    (void)downscale;  // Parameter deprecated but kept for API compatibility
    // Check cache first
    if (use_cache_) {
        uint64_t frame_hash = hashFrame(frame);
        auto it = detection_cache_.find(frame_hash);
        if (it != detection_cache_.end()) {
            return it->second;
        }
    }
    
    Image processed_frame_owned;
    const uint8_t* detect_data = frame.data();
    int detect_width = frame.width();
    int detect_height = frame.height();
    int detect_stride = frame.stride();
    double scale = 1.0;
    
    // Always downscale for faster detection (LibFaceDetection is resolution-dependent)
    // Target: 320px width for ~7ms detection (vs 35ms at 640px)
    // Note: Detection accuracy is still excellent at lower resolutions
    const int target_width = 320;
    
    if (frame.width() > target_width) {
        scale = static_cast<double>(target_width) / frame.width();
        int new_width = target_width;
        int new_height = static_cast<int>(frame.height() * scale);
        processed_frame_owned = resizeImage(frame.data(), frame.width(), frame.height(), frame.stride(), new_width, new_height);
        detect_data = processed_frame_owned.data();
        detect_width = processed_frame_owned.width();
        detect_height = processed_frame_owned.height();
        detect_stride = processed_frame_owned.stride();
    }
    
    // LibFaceDetection detection
    // Allocate result buffer (fixed size required by LibFaceDetection)
    // Note: Modern compilers optimize stack allocation efficiently
    // Stack allocation is actually faster than heap for small, fixed-size buffers
    unsigned char result_buffer[0x20000];
    
    // Note: LibFaceDetection expects BGR format (OpenCV default)
    int* pResults = facedetect_cnn(
        result_buffer,
        (unsigned char*)detect_data,
        detect_width,
        detect_height,
        (int)detect_stride
    );
    
    // Extract results
    std::vector<Rect> faces;
    
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
            
            // Convert to Rect
            Rect face(x, y, w, h);
            
            // Ensure rect is within frame bounds
            face &= Rect(0, 0, frame.width(), frame.height());
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

std::vector<Rect> FaceDetector::detectOrTrackFaces(const ImageView& frame, int track_interval) {
    // Always detect if tracking disabled (track_interval == 0)
    if (track_interval == 0) {
        return detectFaces(frame);
    }
    
    // Detect if we haven't initialized tracking yet or interval reached
    if (!tracking_initialized_ || frames_since_detection_ >= track_interval) {
        // Run full detection
        std::vector<Rect> faces = detectFaces(frame);
        
        // Initialize tracking
        if (!faces.empty()) {
            tracked_faces_ = faces;
            prev_gray_frame_ = toGrayscale(frame.data(), frame.width(), frame.height(), frame.stride());
            tracking_initialized_ = true;
            frames_since_detection_ = 0;
        }
        
        return faces;
    }
    
    // Use tracking for intermediate frames
    frames_since_detection_++;
    return trackFaces(frame);
}

std::vector<Rect> FaceDetector::trackFaces(const ImageView& current_frame) {
    if (tracked_faces_.empty() || prev_gray_frame_.empty()) {
        return {};
    }
    
    // Convert current frame to grayscale
    Image current_gray = toGrayscale(current_frame.data(), current_frame.width(), current_frame.height(), current_frame.stride());
    
    // Track each face using sparse optical flow on face center points
    std::vector<Rect> updated_faces;
    
    // Wrap Image as GrayImage for OpenCV-free optical flow
    GrayImage prev_gray(prev_gray_frame_.data(), prev_gray_frame_.width(), prev_gray_frame_.height(), prev_gray_frame_.stride());
    GrayImage curr_gray(current_gray.data(), current_gray.width(), current_gray.height(), current_gray.stride());
    
    for (const auto& face : tracked_faces_) {
        // Get face center and corners for robust tracking
        std::vector<Point2f> prev_points;
        prev_points.push_back(Point2f(face.x + face.width/2.0f, face.y + face.height/2.0f));  // center
        prev_points.push_back(Point2f(face.x, face.y));  // top-left
        prev_points.push_back(Point2f(face.x + face.width, face.y + face.height));  // bottom-right
        
        // Track points using OpenCV-free pyramid Lucas-Kanade
        std::vector<Point2f> new_points;
        std::vector<bool> status;
        
        OpticalFlow::trackPoints(
            prev_gray,
            curr_gray,
            prev_points,
            new_points,
            status,
            15,  // window size (smaller than OpenCV's 21 for speed)
            3    // pyramid levels
        );
        
        // If all points tracked successfully, update face position
        if (status[0] && status[1] && status[2]) {
            // Calculate movement delta from center point
            float dx = new_points[0].x - prev_points[0].x;
            float dy = new_points[0].y - prev_points[0].y;
            
            // Update face rectangle
            Rect updated_face(
                face.x + static_cast<int>(dx),
                face.y + static_cast<int>(dy),
                face.width,
                face.height
            );
            
            // Ensure face is within frame bounds
            updated_face &= Rect(0, 0, current_frame.width(), current_frame.height());
            
            if (updated_face.width > 0 && updated_face.height > 0) {
                updated_faces.push_back(updated_face);
            }
        }
    }
    
    // Update tracking state
    tracked_faces_ = updated_faces;
    prev_gray_frame_ = std::move(current_gray);
    
    // If tracking lost all faces, force re-detection next frame
    if (updated_faces.empty()) {
        tracking_initialized_ = false;
    }
    
    return updated_faces;
}

void FaceDetector::resetTracking() {
    tracking_initialized_ = false;
    tracked_faces_.clear();
    prev_gray_frame_ = Image();  // Clear by creating empty Image
    frames_since_detection_ = 0;
}

Image FaceDetector::alignFace(const ImageView& frame, const Rect& face_rect) {
    // Extract face region using roi
    ImageView face_roi = frame.roi(face_rect);
    Image face_img = face_roi.clone();
    
    // SFace expects 112x112 aligned face
    return resizeImage(face_img.data(), face_img.width(), face_img.height(), face_img.stride(), 112, 112);
}

std::vector<FaceEncoding> FaceDetector::encodeFaces(
    const ImageView& frame,
    const std::vector<Rect>& face_locations) {
    
    if (!models_loaded_ || face_locations.empty()) {
        return {};
    }
    
    std::vector<FaceEncoding> encodings;
    
    for (const auto& face_rect : face_locations) {
        // Align face for SFace (112x112)
        Image aligned = alignFace(frame, face_rect);
        
        // Convert to NCNN format (no manual normalization - model has built-in preprocessing)
        ncnn::Mat in = ncnn::Mat::from_pixels(
            aligned.data(), 
            ncnn::Mat::PIXEL_BGR, 
            aligned.width(), 
            aligned.height()
        );
        
        // Create extractor and run inference
        ncnn::Extractor ex = ncnn_net_.create_extractor();
        ex.set_light_mode(true);  // Optimize for speed
        ex.input("in0", in);  // SFace model uses "in0" as input layer
        
        // Extract features
        ncnn::Mat out;
        int ret = ex.extract("out0", out);  // SFace model uses "out0" as output layer
        if (ret != 0) {
            // Inference failed - skip this face
            // This can happen with corrupted models or invalid input
            continue;
        }
        
        // Validate output dimensions (SFace produces 512D vector)
        if (out.w != static_cast<int>(FACE_ENCODING_DIM) || out.h != 1 || out.c != 1) {
            // Unexpected output dimensions - skip this face
            continue;
        }
        
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
    
    // Check size (should match FACE_ENCODING_DIM for current model)
    if (encoding1.size() != encoding2.size() || encoding1.size() != FACE_ENCODING_DIM) {
        return 999.0;  // Size mismatch
    }
    
    // Calculate dot product (encodings are already L2 normalized)
    double dot_product = 0.0;
    
    for (size_t i = 0; i < encoding1.size(); i++) {
        dot_product += encoding1[i] * encoding2[i];
    }
    
    // Clamp dot product to [-1, 1] to handle floating point precision errors
    // (normalized vectors should have dot product in [-1, 1], but fp math can exceed)
    if (dot_product > 1.0) dot_product = 1.0;
    if (dot_product < -1.0) dot_product = -1.0;
    
    // Since encodings are already normalized, dot product IS cosine similarity
    double cosine_sim = dot_product;
    
    // Convert to distance (lower = more similar)
    // Cosine similarity: 1 (identical) to -1 (opposite)
    // Distance: 0 (identical) to 2 (opposite)
    double distance = 1.0 - cosine_sim;
    
    return distance;
}

Image FaceDetector::preprocessFrame(const ImageView& frame) {
    // Handle RGBA (4 channels) - extract RGB first
    Image processed;
    if (frame.channels() == 4) {
        // Convert RGBA to RGB using libyuv
        // Note: ImageView data should be BGRA (OpenCV format), convert to BGR
        Image temp(frame.width(), frame.height(), 3);
        libyuv::ARGBToRGB24(frame.data(), frame.stride(), temp.data(), temp.stride(), frame.width(), frame.height());
        processed = std::move(temp);
    } else {
        processed = frame.clone();
    }
    
    // Enhance contrast for better detection using CLAHE on YUV color space
    // YUV is much faster than Lab and gives similar results for luminance-based CLAHE
    int width = processed.width();
    int height = processed.height();
    
    // First convert BGR to ARGB (libyuv intermediate format)
    Image argb_temp(width, height, 4);
    libyuv::RGB24ToARGB(processed.data(), processed.stride(), argb_temp.data(), argb_temp.stride(), width, height);
    
    // Allocate YUV I444 planes (full resolution, no chroma subsampling)
    Image y_plane(width, height, 1);
    Image u_plane(width, height, 1);
    Image v_plane(width, height, 1);
    
    // Convert ARGB to I444 (YUV 4:4:4) using libyuv
    libyuv::ARGBToI444(
        argb_temp.data(), argb_temp.stride(),
        y_plane.data(), y_plane.stride(),
        u_plane.data(), u_plane.stride(),
        v_plane.data(), v_plane.stride(),
        width, height
    );
    
    // Apply CLAHE to Y (luminance) channel only using standalone implementation
    faceid::CLAHE clahe(2.0, 8, 8);
    Image y_enhanced(width, height, 1);
    clahe.apply(y_plane.data(), y_enhanced.data(), width, height, width, width);
    
    // Convert I444 back to RGB24 (BGR) using libyuv
    Image result(width, height, 3);
    libyuv::I444ToRGB24(
        y_enhanced.data(), y_enhanced.stride(),
        u_plane.data(), u_plane.stride(),
        v_plane.data(), v_plane.stride(),
        result.data(), result.stride(),
        width, height
    );
    
    return result;
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

uint64_t FaceDetector::hashFrame(const ImageView& frame) {
    // Simple hash based on frame dimensions and checksum of subset of pixels
    uint64_t hash = frame.height() * 10000ULL + frame.width();
    
    // Sample some pixels for hash
    int step = std::max(1, frame.height() / 8);
    for (int i = 0; i < frame.height(); i += step) {
        for (int j = 0; j < frame.width(); j += step) {
            if (frame.channels() == 3) {
                // Access pixel data directly
                const uint8_t* pixel = frame.data() + i * frame.stride() + j * 3;
                hash = hash * 31 + pixel[0] + pixel[1] + pixel[2];
            }
        }
    }
    
    return hash;
}

// Multi-face detection helpers for "no peek" feature
double FaceDetector::faceDistance(const Rect& face1, const Rect& face2) {
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

bool FaceDetector::areDistinctFaces(const Rect& face1, const Rect& face2, int min_distance) {
    double distance = faceDistance(face1, face2);
    return distance >= min_distance;
}

double FaceDetector::getFaceSizePercent(const Rect& face, int frame_width) {
    if (frame_width <= 0) return 0.0;
    return static_cast<double>(face.width) / frame_width;
}

int FaceDetector::countDistinctFaces(const std::vector<Rect>& faces, int min_distance) {
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
