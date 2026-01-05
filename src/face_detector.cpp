#include "face_detector.h"
#include "clahe.h"
#include "optical_flow.h"
#include "config_paths.h"
#include "config.h"
#include "logger.h"
#include "detectors/common.h"
#include "detectors/detectors.h"
#include "detectors/yunet_model_data.h"
#include "detectors/retinaface_model_data.h"
#include <ncnn/datareader.h>
#include <libyuv.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <regex>
#include <dirent.h>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <chrono>

namespace faceid {

// Global model cache to avoid reloading models from disk
// Key: "param_path|bin_path", Value: shared_ptr to ncnn::Net
static std::unordered_map<std::string, std::shared_ptr<ncnn::Net>> g_model_cache;
static std::mutex g_cache_mutex;


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

// Helper function: Load NCNN model with caching
// This tracks which model files have been loaded and can skip redundant disk I/O
static bool isModelCached(const std::string& param_path, const std::string& bin_path) {
    std::string cache_key = param_path + "|" + bin_path;
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    return g_model_cache.find(cache_key) != g_model_cache.end();
}

static void markModelCached(const std::string& param_path, const std::string& bin_path) {
    std::string cache_key = param_path + "|" + bin_path;
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    if (g_model_cache.find(cache_key) == g_model_cache.end()) {
        // We don't actually store the model (ncnn::Net can't be copied), 
        // just mark it as "seen" to track which models we've already loaded
        g_model_cache[cache_key] = nullptr;
    }
}

// Helper: Parse NCNN param file to extract output dimension
size_t FaceDetector::parseModelOutputDim(const std::string& param_path) {
    // Try the provided path first
    std::ifstream file(param_path);
    std::string actual_path = param_path;
    
    // If it doesn't exist and ends with .param, try .ncnn.param instead
    if (!file.is_open() && param_path.size() >= 6 && 
        param_path.substr(param_path.size() - 6) == ".param") {
        std::string base = param_path.substr(0, param_path.size() - 6);
        std::string ncnn_path = base + ".ncnn.param";
        file.open(ncnn_path);
        if (file.is_open()) {
            actual_path = ncnn_path;
            Logger::getInstance().debug("Trying .ncnn.param extension: " + ncnn_path);
        }
    }
    
    if (!file.is_open()) {
        Logger::getInstance().debug("Failed to open param file: " + param_path);
        return 0;
    }
    
    std::string line;
    // Match: InnerProduct ... out0 0=<dimension>
    std::regex output_pattern("InnerProduct\\s+\\S+\\s+\\d+\\s+\\d+\\s+\\S+\\s+out0\\s+0=(\\d+)");
    
    while (std::getline(file, line)) {
        std::smatch match;
        if (std::regex_search(line, match, output_pattern)) {
            size_t dim = std::stoull(match[1].str());
            Logger::getInstance().debug("Detected output dimension: " + std::to_string(dim) + "D from " + actual_path);
            return dim;
        }
    }
    
    Logger::getInstance().debug("Could not detect output dimension from " + actual_path);
    return 0;
}

// Helper: Find first available recognition model in models directory
std::pair<std::string, size_t> FaceDetector::findAvailableModel(const std::string& models_dir) {
    Logger::getInstance().debug("Scanning for models in: " + models_dir);
    
    DIR* dir = opendir(models_dir.c_str());
    if (!dir) {
        Logger::getInstance().debug("Failed to open models directory: " + models_dir);
        return {"", 0};
    }
    
    std::vector<std::string> param_files;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (filename.length() > 6 && filename.substr(filename.length() - 6) == ".param") {
            param_files.push_back(filename);
        }
    }
    closedir(dir);
    
    Logger::getInstance().debug("Found " + std::to_string(param_files.size()) + " param file(s)");
    
    // Try each param file
    for (const auto& param_file : param_files) {
        std::string base_name = param_file.substr(0, param_file.length() - 6);  // Remove .param
        std::string base_path = models_dir + "/" + base_name;
        std::string param_path = base_path + ".param";
        std::string bin_path = base_path + ".bin";
        
        Logger::getInstance().debug("Checking model: " + base_name);
        
        // Check if .bin file exists
        std::ifstream bin_check(bin_path);
        if (!bin_check.good()) {
            Logger::getInstance().debug("  Missing .bin file, skipping");
            continue;
        }
        
        // Parse output dimension
        size_t output_dim = parseModelOutputDim(param_path);
        if (output_dim == 0) {
            Logger::getInstance().debug("  Could not detect output dimension, skipping");
            continue;
        }
        
        // Validate dimension - face recognition models typically have 64D-2048D embeddings
        // Filter out non-recognition models:
        //   - Expression recognition: 7D (7 emotions)
        //   - Age/gender classification: 2-10D
        //   - Face recognition: 64D+ (ArcFace 128D, MobileFaceNet 128D, SFace 512D, InsightFace 1024D)
        if (output_dim < 64) {
            Logger::getInstance().debug("  ✗ Invalid dimension " + std::to_string(output_dim) + "D (expected ≥64D for face recognition), skipping");
            Logger::getInstance().debug("    This appears to be a classification model (expression/age/gender), not face recognition");
            continue;
        }
        
        if (output_dim > 2048) {
            Logger::getInstance().debug("  ✗ Dimension " + std::to_string(output_dim) + "D too large (expected ≤2048D), skipping");
            continue;
        }
        
        // Valid recognition model found
        Logger::getInstance().debug("  ✓ Valid face recognition model: " + base_name + " (" + std::to_string(output_dim) + "D)");
        return {base_path, output_dim};
    }
    
    Logger::getInstance().debug("No valid recognition models found in " + models_dir);
    return {"", 0};
}

FaceDetector::FaceDetector() {
    // Models (recognition + embedded detection) loaded separately via loadModels()
}

// Helper: Load embedded YuNet model from memory (primary detection)
bool FaceDetector::loadEmbeddedYuNet() {
    Logger::getInstance().debug("Loading embedded YuNet detection model...");
    
    // Get num_threads from config (default: 4)
    int num_threads = Config::getInstance().getInt("recognition", "num_threads").value_or(4);
    
    yunet_net_.opt.use_vulkan_compute = false;
    yunet_net_.opt.num_threads = num_threads;
    yunet_net_.opt.use_fp16_packed = false;
    yunet_net_.opt.use_fp16_storage = false;
    
    // Load param from memory (null-terminated string)
    int ret = yunet_net_.load_param_mem(embedded::yunet_model_param);
    if (ret != 0) {
        Logger::getInstance().error("Failed to load YuNet param from memory, ret=" + std::to_string(ret));
        return false;
    }
    
    // Load binary model from memory using DataReaderFromMemory
    const unsigned char* model_data = embedded::yunet_model_bin;
    ncnn::DataReaderFromMemory mem_reader(model_data);
    ret = yunet_net_.load_model(mem_reader);
    if (ret != 0) {
        Logger::getInstance().error("Failed to load YuNet model from memory, ret=" + std::to_string(ret));
        return false;
    }
    
    Logger::getInstance().info("✓ Embedded YuNet model loaded successfully (" + 
                               std::to_string(embedded::yunet_model_bin_size / 1024) + " KB)");
    return true;
}

// Helper: Load embedded RetinaFace model from memory (fallback detection)
bool FaceDetector::loadEmbeddedRetinaFace() {
    Logger::getInstance().debug("Loading embedded RetinaFace detection model...");
    
    // Get num_threads from config (default: 4)
    int num_threads = Config::getInstance().getInt("recognition", "num_threads").value_or(4);
    
    retinaface_net_.opt.use_vulkan_compute = false;
    retinaface_net_.opt.num_threads = num_threads;
    retinaface_net_.opt.use_fp16_packed = false;
    retinaface_net_.opt.use_fp16_storage = false;
    
    // Load param from memory (null-terminated string)
    int ret = retinaface_net_.load_param_mem(embedded::retinaface_model_param);
    if (ret != 0) {
        Logger::getInstance().error("Failed to load RetinaFace param from memory, ret=" + std::to_string(ret));
        return false;
    }
    
    // Load binary model from memory using DataReaderFromMemory
    const unsigned char* model_data = embedded::retinaface_model_bin;
    ncnn::DataReaderFromMemory mem_reader(model_data);
    ret = retinaface_net_.load_model(mem_reader);
    if (ret != 0) {
        Logger::getInstance().error("Failed to load RetinaFace model from memory, ret=" + std::to_string(ret));
        return false;
    }
    
    Logger::getInstance().info("✓ Embedded RetinaFace model loaded successfully (" + 
                               std::to_string(embedded::retinaface_model_bin_size / 1024) + " KB)");
    return true;
}

bool FaceDetector::loadModels(const std::string& model_base_path) {
    try {
        // Load detection confidence threshold from config
        auto confidence_opt = Config::getInstance().getDouble("recognition", "confidence");
        bool user_specified_confidence = confidence_opt.has_value();
        
        if (user_specified_confidence) {
            detection_confidence_threshold_ = static_cast<float>(confidence_opt.value());
            Logger::getInstance().debug("Detection confidence threshold from config: " + 
                std::to_string(detection_confidence_threshold_));
        } else {
            // Use sensible defaults based on model type (will be adjusted after model detection)
            detection_confidence_threshold_ = 0.8f;
            Logger::getInstance().debug("Using default detection confidence threshold: 0.8 (will adjust based on model type)");
        }
        
        std::string base_path;
        size_t output_dim = 0;
        
        // If explicit path provided, use it
        if (!model_base_path.empty()) {
            base_path = model_base_path;
            Logger::getInstance().debug("Using explicit model path: " + base_path);
            
            // Try to detect output dimension
            std::string param_path = base_path + ".param";
            output_dim = parseModelOutputDim(param_path);
            if (output_dim == 0) {
                Logger::getInstance().debug("Warning: Could not auto-detect output dimension, using default " + 
                    std::to_string(FACE_ENCODING_DIM) + "D");
                output_dim = FACE_ENCODING_DIM;
            }
        } else {
            // Priority 1: Try standard name "recognition.{param,bin}"
            std::string standard_path = std::string(MODELS_DIR) + "/recognition";
            std::string standard_param = standard_path + ".param";
            std::string standard_bin = standard_path + ".bin";
            
            std::ifstream param_check(standard_param);
            std::ifstream bin_check(standard_bin);
            
            if (param_check.good() && bin_check.good()) {
                Logger::getInstance().debug("Found standard recognition model: recognition.{param,bin}");
                base_path = standard_path;
                output_dim = parseModelOutputDim(standard_param);
                if (output_dim == 0) {
                    Logger::getInstance().debug("Warning: Could not detect dimension, using default");
                    output_dim = FACE_ENCODING_DIM;
                }
            } else {
                // Priority 2: Auto-detect from available models
                Logger::getInstance().debug("Standard name not found, auto-detecting recognition model...");
                auto model_info = findAvailableModel(std::string(MODELS_DIR));
                base_path = model_info.first;
                output_dim = model_info.second;
                
                if (base_path.empty() || output_dim == 0) {
                    // Priority 3: Fall back to legacy "sface"
                    Logger::getInstance().debug("No valid models found, falling back to legacy sface");
                    base_path = std::string(MODELS_DIR) + "/sface";
                    output_dim = FACE_ENCODING_DIM;
                }
            }
        }
        
        std::string param_path = base_path + ".param";
        std::string bin_path = base_path + ".bin";
        
        // Check if .param exists, if not try .ncnn.param
        std::ifstream param_check(param_path);
        if (!param_check.good()) {
            param_path = base_path + ".ncnn.param";
            bin_path = base_path + ".ncnn.bin";
        }
        
        // Extract model name from path
        size_t last_slash = base_path.find_last_of("/\\");
        current_model_name_ = (last_slash != std::string::npos) ? 
            base_path.substr(last_slash + 1) : base_path;
        
        // Try to read original model name from .use file
        std::string use_file = std::string(MODELS_DIR) + "/.use";
        std::ifstream use_stream(use_file);
        if (use_stream.good()) {
            std::string line;
            while (std::getline(use_stream, line)) {
                if (line.empty() || line[0] == '#') continue;
                
                size_t eq_pos = line.find('=');
                if (eq_pos != std::string::npos) {
                    std::string key = line.substr(0, eq_pos);
                    std::string value = line.substr(eq_pos + 1);
                    if (key == "recognition") {
                        current_model_name_ = value;
                        break;
                    }
                }
            }
        }
        
        current_encoding_dim_ = output_dim;
        
        Logger::getInstance().debug("Loading recognition model: " + current_model_name_ + 
            " (" + std::to_string(current_encoding_dim_) + "D)");
        Logger::getInstance().debug("  param: " + param_path);
        Logger::getInstance().debug("  bin:   " + bin_path);
        
        // Check if this model was already loaded (file system cache helps)
        bool was_cached = isModelCached(param_path, bin_path);
        if (was_cached) {
            Logger::getInstance().debug("Model cache HIT: This model was loaded before (faster due to FS cache)");
        }
        
        // Get num_threads from config (default: 4)
        int num_threads = Config::getInstance().getInt("recognition", "num_threads").value_or(4);
        Logger::getInstance().debug("NCNN num_threads: " + std::to_string(num_threads));
        
        // Configure NCNN options for optimal CPU performance
        ncnn_net_.opt.use_vulkan_compute = false;
        ncnn_net_.opt.num_threads = num_threads;
        ncnn_net_.opt.use_fp16_packed = false;
        ncnn_net_.opt.use_fp16_storage = false;
        
        Logger::getInstance().debug("Loading param file...");
        int ret = ncnn_net_.load_param(param_path.c_str());
        if (ret != 0) {
            Logger::getInstance().debug("Failed to load param file, ret=" + std::to_string(ret));
            return false;
        }
        Logger::getInstance().debug("Param file loaded successfully");
        
        Logger::getInstance().debug("Loading model file...");
        ret = ncnn_net_.load_model(bin_path.c_str());
        if (ret != 0) {
            Logger::getInstance().debug("Failed to load model file, ret=" + std::to_string(ret));
            return false;
        }
        Logger::getInstance().debug("Model file loaded successfully");
        
        try {
            ncnn::Extractor ex = ncnn_net_.create_extractor();
            Logger::getInstance().debug("NCNN extractor created successfully");
        } catch (...) {
            Logger::getInstance().debug("Failed to create NCNN extractor");
            return false;
        }
        
        // Mark this model as cached for future reference
        if (!was_cached) {
            markModelCached(param_path, bin_path);
        }
        
        models_loaded_ = true;
        Logger::getInstance().debug("✓ Recognition model loaded: " + current_model_name_ + 
            " (" + std::to_string(current_encoding_dim_) + "D)");
        
        // Load embedded detection models
        if (!loadEmbeddedYuNet()) {
            Logger::getInstance().error("Failed to load embedded YuNet model");
            return false;
        }
        detection_model_loaded_ = true;
        detection_model_type_ = DetectionModelType::YUNET;
        
        if (!loadEmbeddedRetinaFace()) {
            Logger::getInstance().error("Failed to load embedded RetinaFace model");
            return false;
        }
        detection2_model_loaded_ = true;
        detection2_model_type_ = DetectionModelType::RETINAFACE;
        
        return true;
    } catch (const std::exception& e) {
        Logger::getInstance().error("Exception in loadModels: " + std::string(e.what()));
        return false;
    }
}

std::vector<Rect> FaceDetector::detectFaces(const ImageView& frame, bool downscale, float confidence_threshold) {
    (void)downscale;  // Parameter kept for API compatibility
    
    // Check if detection model is loaded
    if (!detection_model_loaded_) {
        return {};
    }
    
    // Use default threshold from config if not specified
    if (confidence_threshold <= 0.0f) {
        confidence_threshold = detection_confidence_threshold_;
    }
    
    // Check cache first
    if (use_cache_) {
        uint64_t frame_hash = hashFrame(frame);
        auto it = detection_cache_.find(frame_hash);
        if (it != detection_cache_.end()) {
            return it->second;
        }
    }
    
    int img_w = frame.width();
    int img_h = frame.height();
    
    // Route to appropriate detector based on model type
    std::vector<Rect> faces;
    switch (detection_model_type_) {
        case DetectionModelType::RETINAFACE:
            {
                // Convert BGR to RGB (all models expect RGB)
                ncnn::Mat in = ncnn::Mat::from_pixels(frame.data(), ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h);
                faces = detectWithRetinaFace(in, img_w, img_h, confidence_threshold);
            }
            break;
        case DetectionModelType::YUNET:
            {
                // Convert BGR to RGB (all models expect RGB)
                ncnn::Mat in = ncnn::Mat::from_pixels(frame.data(), ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h);
                faces = detectWithYuNet(in, img_w, img_h, confidence_threshold);
            }
            break;
        default:
            Logger::getInstance().error("Unknown detection model type");
            return {};
    }
    
    // Cache results
    if (use_cache_) {
        uint64_t frame_hash = hashFrame(frame);
        detection_cache_[frame_hash] = faces;
    }
    
    return faces;
}

// RetinaFace detection implementation
std::vector<Rect> FaceDetector::detectWithRetinaFace(const ncnn::Mat& in, int img_w, int img_h, float confidence_threshold) {
    if (confidence_threshold <= 0.0f) {
        confidence_threshold = detection_confidence_threshold_;
    }
    return ::faceid::detectWithRetinaFace(retinaface_net_, in, img_w, img_h, confidence_threshold);
}

// YuNet detection implementation
std::vector<Rect> FaceDetector::detectWithYuNet(const ncnn::Mat& in, int img_w, int img_h, float confidence_threshold) {
    if (confidence_threshold <= 0.0f) {
        confidence_threshold = detection_confidence_threshold_;
    }
    return ::faceid::detectWithYuNet(yunet_net_, in, img_w, img_h, confidence_threshold);
}

std::vector<Rect> FaceDetector::detectOrTrackFaces(const ImageView& frame, int track_interval, float confidence_threshold) {
    // Always detect if tracking disabled (track_interval == 0)
    if (track_interval == 0) {
        return detectFaces(frame, false, confidence_threshold);
    }
    
    // Detect if we haven't initialized tracking yet or interval reached
    if (!tracking_initialized_ || frames_since_detection_ >= track_interval) {
        // Run full detection
        std::vector<Rect> faces = detectFaces(frame, false, confidence_threshold);
        
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
            
            // Preserve and update landmarks if available
            if (face.hasLandmarks()) {
                for (const auto& pt : face.landmarks) {
                    Point updated_pt;
                    updated_pt.x = pt.x + dx;
                    updated_pt.y = pt.y + dy;
                    updated_face.landmarks.push_back(updated_pt);
                }
            }
            
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
    const int OUTPUT_SIZE = 112;  // SFace expects 112x112
    
    // Check if landmarks are available for proper alignment
    if (face_rect.hasLandmarks() && face_rect.landmarks.size() >= 5) {
        // Standard 5-point landmark positions for 112x112 output
        // These are empirically determined reference positions that work well for face recognition
        // Format: [left_eye, right_eye, nose, left_mouth_corner, right_mouth_corner]
        const Point reference_landmarks[5] = {
            Point(38.2946f, 51.6963f),  // left eye
            Point(73.5318f, 51.5014f),  // right eye
            Point(56.0252f, 71.7366f),  // nose tip
            Point(41.5493f, 92.3655f),  // left mouth corner
            Point(70.7299f, 92.2041f)   // right mouth corner
        };
        
        // Get detected landmarks (already in absolute image coordinates)
        const auto& src_landmarks = face_rect.landmarks;
        
        // Compute similarity transform (rotation, scale, translation) using eyes and nose
        // We use a least-squares approach to find the best affine transformation
        
        // Calculate centroid of source and destination points
        float src_cx = 0.0f, src_cy = 0.0f;
        float dst_cx = 0.0f, dst_cy = 0.0f;
        
        for (int i = 0; i < 5; i++) {
            src_cx += src_landmarks[i].x;
            src_cy += src_landmarks[i].y;
            dst_cx += reference_landmarks[i].x;
            dst_cy += reference_landmarks[i].y;
        }
        src_cx /= 5.0f; src_cy /= 5.0f;
        dst_cx /= 5.0f; dst_cy /= 5.0f;
        
        // Compute scale and rotation using the eye positions
        float src_eye_dx = src_landmarks[1].x - src_landmarks[0].x;  // right_eye - left_eye
        float src_eye_dy = src_landmarks[1].y - src_landmarks[0].y;
        float dst_eye_dx = reference_landmarks[1].x - reference_landmarks[0].x;
        float dst_eye_dy = reference_landmarks[1].y - reference_landmarks[0].y;
        
        float src_eye_dist = std::sqrt(src_eye_dx * src_eye_dx + src_eye_dy * src_eye_dy);
        float dst_eye_dist = std::sqrt(dst_eye_dx * dst_eye_dx + dst_eye_dy * dst_eye_dy);
        
        if (src_eye_dist < 1.0f) {
            // Landmarks too close, fall back to bbox cropping
            Logger::getInstance().debug("Landmarks too close, falling back to bbox alignment");
            goto bbox_fallback;
        }
        
        float scale = dst_eye_dist / src_eye_dist;
        
        // Calculate rotation angle from eye line
        float src_angle = std::atan2(src_eye_dy, src_eye_dx);
        float dst_angle = std::atan2(dst_eye_dy, dst_eye_dx);
        float angle = dst_angle - src_angle;
        
        float cos_a = std::cos(angle);
        float sin_a = std::sin(angle);
        
        // Build 2x3 affine transformation matrix
        // [a b tx]   [scale*cos  -scale*sin  tx]
        // [c d ty] = [scale*sin   scale*cos  ty]
        float a = scale * cos_a;
        float b = -scale * sin_a;
        float c = scale * sin_a;
        float d = scale * cos_a;
        
        // Translation: dst_center = M * src_center
        float tx = dst_cx - (a * src_cx + b * src_cy);
        float ty = dst_cy - (c * src_cx + d * src_cy);
        
        // Apply affine transformation using libyuv's warp
        // libyuv doesn't have affine warp, so we'll do manual bilinear sampling
        Image aligned(OUTPUT_SIZE, OUTPUT_SIZE, 3);
        uint8_t* dst_data = aligned.data();
        
        const uint8_t* src_data = frame.data();
        int src_width = frame.width();
        int src_height = frame.height();
        int src_stride = frame.stride();
        
        // Inverse transformation for backward mapping
        float det = a * d - b * c;
        if (std::abs(det) < 1e-6f) {
            Logger::getInstance().debug("Singular transformation matrix, falling back to bbox alignment");
            goto bbox_fallback;
        }
        
        float inv_a = d / det;
        float inv_b = -b / det;
        float inv_c = -c / det;
        float inv_d = a / det;
        float inv_tx = -(inv_a * tx + inv_b * ty);
        float inv_ty = -(inv_c * tx + inv_d * ty);
        
        // Apply transformation with bilinear interpolation
        for (int y = 0; y < OUTPUT_SIZE; y++) {
            for (int x = 0; x < OUTPUT_SIZE; x++) {
                // Map destination pixel to source
                float src_x = inv_a * x + inv_b * y + inv_tx;
                float src_y = inv_c * x + inv_d * y + inv_ty;
                
                // Bilinear interpolation
                int x0 = static_cast<int>(std::floor(src_x));
                int y0 = static_cast<int>(std::floor(src_y));
                int x1 = x0 + 1;
                int y1 = y0 + 1;
                
                // Check bounds
                if (x0 < 0 || y0 < 0 || x1 >= src_width || y1 >= src_height) {
                    // Out of bounds - use black
                    dst_data[(y * OUTPUT_SIZE + x) * 3 + 0] = 0;
                    dst_data[(y * OUTPUT_SIZE + x) * 3 + 1] = 0;
                    dst_data[(y * OUTPUT_SIZE + x) * 3 + 2] = 0;
                    continue;
                }
                
                float fx = src_x - x0;
                float fy = src_y - y0;
                
                // Sample 4 corners (BGR format)
                for (int c = 0; c < 3; c++) {
                    float p00 = src_data[y0 * src_stride + x0 * 3 + c];
                    float p10 = src_data[y0 * src_stride + x1 * 3 + c];
                    float p01 = src_data[y1 * src_stride + x0 * 3 + c];
                    float p11 = src_data[y1 * src_stride + x1 * 3 + c];
                    
                    float val = p00 * (1 - fx) * (1 - fy) +
                               p10 * fx * (1 - fy) +
                               p01 * (1 - fx) * fy +
                               p11 * fx * fy;
                    
                    dst_data[(y * OUTPUT_SIZE + x) * 3 + c] = static_cast<uint8_t>(val);
                }
            }
        }
        
        Logger::getInstance().debug("Face aligned using 5-point landmarks with affine transformation");
        return aligned;
    }
    
bbox_fallback:
    // No landmarks available - fall back to simple bounding box crop and resize
    Logger::getInstance().debug("No landmarks available, using bbox-based alignment");
    ImageView face_roi = frame.roi(face_rect);
    Image face_img = face_roi.clone();
    
    // SFace expects 112x112 aligned face
    return resizeImage(face_img.data(), face_img.width(), face_img.height(), face_img.stride(), OUTPUT_SIZE, OUTPUT_SIZE);
}

std::vector<FaceEncoding> FaceDetector::encodeFaces(
    const ImageView& frame,
    const std::vector<Rect>& face_locations,
    double min_quality_override,
    std::vector<float>* out_quality_scores) {
    
    if (!models_loaded_ || face_locations.empty()) {
        if (!models_loaded_) {
            Logger::getInstance().debug("encodeFaces() called but models_loaded_=false");
        }
        if (face_locations.empty()) {
            Logger::getInstance().debug("encodeFaces() called but face_locations is empty");
        }
        return {};
    }
    
    Logger::getInstance().debug("encodeFaces() processing " + std::to_string(face_locations.size()) + " face(s)");
    
    std::vector<FaceEncoding> encodings;
    
    for (size_t idx = 0; idx < face_locations.size(); idx++) {
        const auto& face_rect = face_locations[idx];
        Logger::getInstance().debug("Processing face " + std::to_string(idx) + ": rect(" + 
            std::to_string(face_rect.x) + "," + std::to_string(face_rect.y) + "," +
            std::to_string(face_rect.width) + "x" + std::to_string(face_rect.height) + ")");
        
        // QUICK WIN #4: Apply head pose correction if enabled and landmarks available
        Image corrected_frame;
        bool pose_corrected = false;
        if (Config::getInstance().getBool("recognition", "enable_head_pose_correction").value_or(true) &&
            face_rect.hasLandmarks()) {
            HeadPose pose = estimateHeadPose(face_rect.landmarks);
            Logger::getInstance().debug("Face " + std::to_string(idx) + " head pose: yaw=" + 
                std::to_string(pose.yaw) + " pitch=" + std::to_string(pose.pitch) + 
                " roll=" + std::to_string(pose.roll));
            
            corrected_frame = applyHeadPoseCorrection(frame, face_rect, pose);
            pose_corrected = true;
            Logger::getInstance().debug("Head pose correction applied to face " + std::to_string(idx));
        }
        
        // Align face for SFace (112x112) - use corrected frame if available
        Image aligned;
        if (pose_corrected) {
            // Create a temporary Rect for the corrected frame (face is centered)
            Rect temp_rect;
            temp_rect.x = 0;
            temp_rect.y = 0;
            temp_rect.width = corrected_frame.width();
            temp_rect.height = corrected_frame.height();
            temp_rect.landmarks = face_rect.landmarks; // Preserve landmarks
            aligned = alignFace(corrected_frame.view(), temp_rect);
        } else {
            aligned = alignFace(frame, face_rect);
        }

        // PHASE 2: Adaptive Preprocessing Pipeline
        // Order: Gamma → Brightness → CLAHE → Histogram EQ (legacy)
        
        // Step 1: Adaptive gamma correction (for extreme lighting)
        if (Config::getInstance().getBool("recognition", "enable_gamma_correction").value_or(true)) {
            aligned = applyAdaptiveGammaCorrection(aligned);
            Logger::getInstance().debug("Gamma correction applied to face " + std::to_string(idx));
        }
        
        // Step 2: Brightness normalization (linear scaling)
        if (Config::getInstance().getBool("recognition", "enable_brightness_normalization").value_or(true)) {
            aligned = normalizeBrightness(aligned);
            Logger::getInstance().debug("Brightness normalization applied to face " + std::to_string(idx));
        }
        
        // Step 3: Adaptive CLAHE (contrast enhancement)
        if (Config::getInstance().getBool("recognition", "enable_adaptive_clahe").value_or(true)) {
            aligned = applyAdaptiveCLAHE(aligned);
            Logger::getInstance().debug("Adaptive CLAHE applied to face " + std::to_string(idx));
        }
        
        // QUICK WIN #2: Assess face quality and skip low-quality faces
        // Use override if provided (for enrollment auto-detection), otherwise use config
        // Note: -1.0 means "use config", 0.0 means "accept all", >0 means specific threshold
        double min_quality = (min_quality_override >= 0.0) ? min_quality_override :
                             Config::getInstance().getDouble("recognition", "min_face_quality")
                             .value_or(0.70);
        bool debug_quality = Config::getInstance().getBool("recognition", "debug_face_quality")
                             .value_or(false);
        
        FaceQuality quality = assessFaceQuality(aligned, face_rect, frame.width(), 1.0f);
        
        // Store quality score if requested (even if face will be rejected)
        if (out_quality_scores) {
            out_quality_scores->push_back(quality.overall_score);
        }
        
        if (debug_quality) {
            char buf[256];
            snprintf(buf, sizeof(buf), 
                     "Face %zu quality: overall=%.3f blur=%.3f size=%.3f brightness=%.3f confidence=%.3f",
                     idx, quality.overall_score, quality.blur_score, quality.size_score,
                     quality.brightness_score, quality.confidence_score);
            Logger::getInstance().debug(buf);
        }
        
        if (quality.overall_score < min_quality) {
            std::string msg = "Face " + std::to_string(idx) + " skipped - low quality: " + 
                             std::to_string(quality.overall_score) + " < " +
                             std::to_string(min_quality);
            Logger::getInstance().debug(msg);
            // Debug output disabled - check /var/log/faceid.log if needed
            // std::cerr << "[ENROLLMENT DEBUG] " << msg << std::endl;
            continue;
        }
        Logger::getInstance().debug("Aligned face to " + std::to_string(aligned.width()) + "x" + std::to_string(aligned.height()));
        
        // Convert to NCNN format (no manual normalization - model has built-in preprocessing)
        ncnn::Mat in = ncnn::Mat::from_pixels(
            aligned.data(), 
            ncnn::Mat::PIXEL_BGR, 
            aligned.width(), 
            aligned.height()
        );
        
        Logger::getInstance().debug("Created NCNN input mat: " + std::to_string(in.w) + "x" + 
            std::to_string(in.h) + "x" + std::to_string(in.c));
        
        // Create extractor and run inference
        ncnn::Extractor ex = ncnn_net_.create_extractor();
        ex.set_light_mode(true);  // Optimize for speed
        ex.input("in0", in);  // SFace model uses "in0" as input layer
        
        Logger::getInstance().debug("Running NCNN inference...");
        
        // Extract features
        ncnn::Mat out;
        int ret = ex.extract("out0", out);  // SFace model uses "out0" as output layer
        if (ret != 0) {
            // Inference failed - skip this face
            // This can happen with corrupted models or invalid input
            std::string msg = "NCNN inference FAILED with ret=" + std::to_string(ret);
            Logger::getInstance().debug(msg);
            // std::cerr << "[ENROLLMENT DEBUG] " << msg << std::endl;
            continue;
        }
        
        Logger::getInstance().debug("NCNN inference SUCCESS, output dims: w=" + std::to_string(out.w) + 
            " h=" + std::to_string(out.h) + " c=" + std::to_string(out.c));
        
        // Validate output dimensions (check against detected model dimension)
        if (out.w != static_cast<int>(current_encoding_dim_) || out.h != 1 || out.c != 1) {
            // Unexpected output dimensions - skip this face
            std::string msg = "Output dimensions INVALID: expected w=" + std::to_string(current_encoding_dim_) + 
                " h=1 c=1, got w=" + std::to_string(out.w) + " h=" + std::to_string(out.h) + " c=" + std::to_string(out.c);
            Logger::getInstance().debug(msg);
            // std::cerr << "[ENROLLMENT DEBUG] " << msg << std::endl;
            continue;
        }
        
        Logger::getInstance().debug("Output dimensions valid, converting to encoding vector");
        
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
        
        Logger::getInstance().debug("L2 norm before normalization: " + std::to_string(norm));
        
        if (norm > 0) {
            for (float& val : encoding) {
                val /= norm;
            }
            Logger::getInstance().debug("L2 normalization applied successfully");
        } else {
            Logger::getInstance().debug("WARNING: L2 norm is zero, skipping normalization");
        }
        
        encodings.push_back(encoding);
        Logger::getInstance().debug("Face " + std::to_string(idx) + " encoded successfully");
    }
    
    Logger::getInstance().debug("encodeFaces() returning " + std::to_string(encodings.size()) + " encoding(s)");
    
    return encodings;
}

double FaceDetector::compareFaces(const FaceEncoding& encoding1, const FaceEncoding& encoding2) {
    // Manual cosine similarity calculation for std::vector<float>
    
    // Ensure encodings are valid
    if (encoding1.empty() || encoding2.empty()) {
        return 999.0;  // Return large distance for invalid comparison
    }
    
    // Check size compatibility (both encodings must have same size)
    if (encoding1.size() != encoding2.size()) {
        Logger::getInstance().debug("Encoding size mismatch: " + std::to_string(encoding1.size()) + 
            " vs " + std::to_string(encoding2.size()));
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

// ============================================================================
// PHASE 3: Quality-Weighted Matching
// ============================================================================

double FaceDetector::compareFacesWeighted(const FaceEncoding& test_encoding,
                                          const FaceEncoding& stored_encoding,
                                          float quality_score) {
    // Calculate raw cosine distance
    double raw_distance = compareFaces(test_encoding, stored_encoding);
    
    // If comparison failed (invalid encodings), return as-is
    if (raw_distance >= 999.0) {
        return raw_distance;
    }
    
    // Apply quality weighting:
    // Higher quality → lower effective distance (more weight)
    // Lower quality → higher effective distance (less weight)
    //
    // Quality factor formula: 1.0 + quality_score
    // - quality_score = 1.0 (perfect): factor = 2.0, distance halved (2x weight)
    // - quality_score = 0.5 (medium): factor = 1.5, distance reduced by 33%
    // - quality_score = 0.0 (poor): factor = 1.0, distance unchanged
    //
    // This gives high-quality encodings up to 2x advantage without extreme weighting
    float quality_factor = 1.0f + quality_score;
    double weighted_distance = raw_distance / quality_factor;
    
    return weighted_distance;
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
    
    // Calculate average brightness for adaptive preprocessing
    int width = processed.width();
    int height = processed.height();
    
    // Quick brightness estimation from first channel (B in BGR)
    uint64_t sum = 0;
    const uint8_t* data = processed.data();
    int channels = processed.channels();
    int total_pixels = width * height;
    
    if (channels >= 3) {
        // Sample every 4th pixel for speed (statistically sufficient)
        for (int i = 0; i < total_pixels; i += 4) {
            sum += data[i * channels];  // B channel
        }
        sum *= 4;  // Compensate for sampling
    } else {
        for (int i = 0; i < total_pixels; i++) {
            sum += data[i];
        }
    }
    float avg_brightness = static_cast<float>(sum) / (total_pixels * 255.0f);
    
    // OPTIMIZATION: Skip CLAHE entirely in good lighting conditions
    // CLAHE is for contrast enhancement in low-light/IR cameras
    // In normal/bright lighting, it's unnecessary CPU work
    if (avg_brightness >= 0.5f) {
        // Optional debug logging
        if (Config::getInstance().getBool("debug", "log_brightness").value_or(false)) {
            char buf[128];
            snprintf(buf, sizeof(buf), "Frame brightness: %.2f - SKIPPING CLAHE (good lighting)", 
                     avg_brightness);
            Logger::getInstance().debug(buf);
        }
        return processed;  // Return original without CLAHE
    }
    
    // Below 0.5 brightness: apply CLAHE for contrast enhancement
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
    
    // Adaptive CLAHE parameters based on brightness
    // For IR cameras in low-light conditions, we need more aggressive enhancement
    double clip_limit;
    if (avg_brightness < 0.15f) {        // Very dark (e.g., IR in low-light)
        clip_limit = 4.0;                 // Aggressive enhancement
    } else if (avg_brightness < 0.30f) { // Dark
        clip_limit = 3.0;
    } else {                              // Dim (0.3-0.5)
        clip_limit = 2.0;                 // Moderate enhancement
    }
    
    // Optional debug logging (controlled by config: [debug] log_brightness = true)
    if (Config::getInstance().getBool("debug", "log_brightness").value_or(false)) {
        char buf[128];
        snprintf(buf, sizeof(buf), "Frame brightness: %.2f, CLAHE clip: %.1f", 
                 avg_brightness, clip_limit);
        Logger::getInstance().debug(buf);
    }
    
    // Apply CLAHE to Y (luminance) channel only using standalone implementation
    faceid::CLAHE clahe(clip_limit, 8, 8);
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

// Enhanced preprocessing with more aggressive CLAHE (for very dark/difficult images)
Image FaceDetector::preprocessFrameAggressive(const ImageView& frame) {
    // Handle RGBA (4 channels) - extract RGB first
    Image processed;
    if (frame.channels() == 4) {
        Image temp(frame.width(), frame.height(), 3);
        libyuv::ARGBToRGB24(frame.data(), frame.stride(), temp.data(), temp.stride(), frame.width(), frame.height());
        processed = std::move(temp);
    } else {
        processed = frame.clone();
    }
    
    // Enhance contrast using VERY aggressive CLAHE on YUV color space
    int width = processed.width();
    int height = processed.height();
    
    // Convert BGR to ARGB (libyuv intermediate format)
    Image argb_temp(width, height, 4);
    libyuv::RGB24ToARGB(processed.data(), processed.stride(), argb_temp.data(), argb_temp.stride(), width, height);
    
    // Allocate YUV I444 planes
    Image y_plane(width, height, 1);
    Image u_plane(width, height, 1);
    Image v_plane(width, height, 1);
    
    // Convert ARGB to I444 (YUV 4:4:4)
    libyuv::ARGBToI444(
        argb_temp.data(), argb_temp.stride(),
        y_plane.data(), y_plane.stride(),
        u_plane.data(), u_plane.stride(),
        v_plane.data(), v_plane.stride(),
        width, height
    );
    
    // Calculate average brightness for adaptive parameters
    uint64_t sum = 0;
    const uint8_t* y_data = y_plane.data();
    int total_pixels = width * height;
    for (int i = 0; i < total_pixels; i++) {
        sum += y_data[i];
    }
    float avg_brightness = static_cast<float>(sum) / (total_pixels * 255.0f);
    
    // MORE aggressive CLAHE parameters for cascade fallback
    // Using smaller tiles (4×4 vs 8×8) for more localized enhancement
    double clip_limit;
    int tile_size;
    if (avg_brightness < 0.15f) {        // Very dark
        clip_limit = 6.0;                 // Very aggressive enhancement
        tile_size = 4;
    } else if (avg_brightness < 0.30f) { // Dark
        clip_limit = 4.5;
        tile_size = 4;
    } else {                              // Moderate
        clip_limit = 3.0;
        tile_size = 6;
    }
    
    // Optional debug logging
    if (Config::getInstance().getBool("debug", "log_brightness").value_or(false)) {
        char buf[128];
        snprintf(buf, sizeof(buf), "Aggressive preprocessing: brightness=%.2f, CLAHE clip=%.1f, tile=%dx%d", 
                 avg_brightness, clip_limit, tile_size, tile_size);
        Logger::getInstance().debug(buf);
    }
    
    // Apply aggressive CLAHE to Y (luminance) channel
    faceid::CLAHE clahe(clip_limit, tile_size, tile_size);
    Image y_enhanced(width, height, 1);
    clahe.apply(y_plane.data(), y_enhanced.data(), width, height, width, width);
    
    // Convert I444 back to RGB24 (BGR)
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

// Motion detection using simple frame differencing
bool FaceDetector::detectMotion(const ImageView& current_frame, double threshold) {
    // Convert current frame to grayscale
    Image current_gray = toGrayscale(
        current_frame.data(),
        current_frame.width(),
        current_frame.height(),
        current_frame.stride()
    );
    
    // Initialize on first call
    if (!motion_initialized_) {
        motion_prev_frame_ = std::move(current_gray);
        motion_initialized_ = true;
        return true;  // Assume motion on first frame
    }
    
    // Check if frame sizes match
    if (motion_prev_frame_.width() != current_gray.width() ||
        motion_prev_frame_.height() != current_gray.height()) {
        motion_prev_frame_ = std::move(current_gray);
        return true;  // Size changed, assume motion
    }
    
    // Calculate frame difference
    const uint8_t* prev_data = motion_prev_frame_.data();
    const uint8_t* curr_data = current_gray.data();
    int total_pixels = current_gray.width() * current_gray.height();
    
    uint64_t diff_sum = 0;
    for (int i = 0; i < total_pixels; i++) {
        int diff = std::abs(static_cast<int>(curr_data[i]) - static_cast<int>(prev_data[i]));
        diff_sum += diff;
    }
    
    // Calculate average difference (normalized 0.0-1.0)
    double avg_diff = static_cast<double>(diff_sum) / (total_pixels * 255.0);
    
    // Update previous frame
    motion_prev_frame_ = std::move(current_gray);
    
    // Motion detected if average difference exceeds threshold
    bool has_motion = avg_diff > threshold;
    
    Logger::getInstance().debug("Motion detection: avg_diff=" + 
                               std::to_string(avg_diff) + 
                               " threshold=" + std::to_string(threshold) +
                               " result=" + (has_motion ? "MOTION" : "STATIC"));
    
    return has_motion;
}

FaceDetector::CascadeResult FaceDetector::detectFacesCascade(
    const ImageView& frame,
    bool enable_motion_check,
    float confidence_threshold) {
    
    CascadeResult result;
    result.stage_used = 0;
    result.has_motion = true;  // Assume motion by default
    result.avg_brightness = 0.0;
    result.stage1_time_ms = 0.0;
    result.stage2_time_ms = 0.0;
    result.stage3_time_ms = 0.0;
    
    // Use default threshold from config if not specified
    if (confidence_threshold <= 0.0f) {
        confidence_threshold = detection_confidence_threshold_;
    }
    
    // Motion detection pre-check (optional optimization)
    if (enable_motion_check) {
        result.has_motion = detectMotion(frame, 0.02);  // 2% threshold
        if (!result.has_motion) {
            Logger::getInstance().debug("Cascade: No motion detected, skipping detection");
            return result;
        }
    }
    
    // Calculate brightness for decision making
    const uint8_t* data = frame.data();
    int channels = frame.channels();
    
    // Quick brightness estimation from first channel (B in BGR)
    uint64_t sum = 0;
    int total_pixels = frame.width() * frame.height();
    if (channels >= 3) {
        for (int i = 0; i < total_pixels; i++) {
            sum += data[i * channels];
        }
    } else {
        for (int i = 0; i < total_pixels; i++) {
            sum += data[i];
        }
    }
    result.avg_brightness = static_cast<double>(sum) / (total_pixels * 255.0);
    
    // Stage 1: Standard preprocessing + primary detector
    Logger::getInstance().debug("Cascade Stage 1: Standard CLAHE + primary detector");
    auto stage1_start = std::chrono::high_resolution_clock::now();
    
    result.processed_frame = preprocessFrame(frame);
    result.faces = detectFaces(result.processed_frame.view(), false, confidence_threshold);
    result.stage_used = 1;
    
    auto stage1_end = std::chrono::high_resolution_clock::now();
    result.stage1_time_ms = std::chrono::duration<double, std::milli>(stage1_end - stage1_start).count();
    
    if (!result.faces.empty()) {
        Logger::getInstance().debug("Cascade Stage 1: SUCCESS - detected " + 
                                   std::to_string(result.faces.size()) + " face(s) in " +
                                   std::to_string(result.stage1_time_ms) + "ms");
        return result;
    }
    
    // Check if we should skip cascade for good lighting + no faces
    // (clearly nobody is there, don't waste CPU)
    if (result.avg_brightness > 0.40 && !enable_motion_check) {
        Logger::getInstance().debug("Cascade Stage 1: No faces in good lighting (brightness=" + 
                                   std::to_string(result.avg_brightness) + ") - skipping cascade");
        return result;
    }
    
    // Stage 2: Aggressive preprocessing + primary detector
    Logger::getInstance().debug("Cascade Stage 2: Aggressive CLAHE (4x4 tiles) + primary detector");
    auto stage2_start = std::chrono::high_resolution_clock::now();
    
    result.processed_frame = preprocessFrameAggressive(frame);
    result.faces = detectFaces(result.processed_frame.view(), false, confidence_threshold);
    result.stage_used = 2;
    
    auto stage2_end = std::chrono::high_resolution_clock::now();
    result.stage2_time_ms = std::chrono::duration<double, std::milli>(stage2_end - stage2_start).count();
    
    if (!result.faces.empty()) {
        Logger::getInstance().info("Cascade Stage 2: SUCCESS - detected " + 
                                  std::to_string(result.faces.size()) + " face(s) with aggressive preprocessing in " +
                                  std::to_string(result.stage2_time_ms) + "ms (total: " +
                                  std::to_string(result.stage1_time_ms + result.stage2_time_ms) + "ms)");
        return result;
    }
    
    // Stage 3: Aggressive preprocessing + RetinaFace fallback (if available)
    if (detection2_model_loaded_) {
        Logger::getInstance().debug("Cascade Stage 3: Trying RetinaFace fallback");
        auto stage3_start = std::chrono::high_resolution_clock::now();
        
        // Use RetinaFace model directly
        int img_w = result.processed_frame.width();
        int img_h = result.processed_frame.height();
        
        ncnn::Mat in = ncnn::Mat::from_pixels(result.processed_frame.data(), 
                                              ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h);
        
        std::vector<Rect> stage3_faces = faceid::detectWithRetinaFace(retinaface_net_, in, img_w, img_h, confidence_threshold);
        
        auto stage3_end = std::chrono::high_resolution_clock::now();
        result.stage3_time_ms = std::chrono::duration<double, std::milli>(stage3_end - stage3_start).count();
        
        if (!stage3_faces.empty()) {
            result.faces = stage3_faces;
            result.stage_used = 3;
            Logger::getInstance().info("Cascade Stage 3: SUCCESS - detected " + 
                                      std::to_string(result.faces.size()) + " face(s) with " + 
                                      detection2_model_name_ + " in " +
                                      std::to_string(result.stage3_time_ms) + "ms (total: " +
                                      std::to_string(result.stage1_time_ms + result.stage2_time_ms + result.stage3_time_ms) + "ms)");
            return result;
        }
    } else {
        Logger::getInstance().debug("Cascade Stage 3: detection2 model not available (install with 'faceid use --detection2')");
    }
    
    // All stages failed
    double total_time = result.stage1_time_ms + result.stage2_time_ms + result.stage3_time_ms;
    Logger::getInstance().warning("Cascade detection: All stages failed (total time: " + 
                                 std::to_string(total_time) + "ms, brightness: " +
                                 std::to_string(result.avg_brightness) + ")");
    result.stage_used = detection2_model_loaded_ ? 3 : 2;  // Indicate how many stages were tried
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

// Deduplicate faces based on encoding similarity
// This prevents the same person detected at multiple angles/positions from being counted multiple times
std::vector<size_t> FaceDetector::deduplicateFaces(
    const std::vector<Rect>& faces,
    const std::vector<FaceEncoding>& encodings,
    double similarity_threshold) {
    
    if (faces.empty() || encodings.empty() || faces.size() != encodings.size()) {
        return {};
    }
    
    std::vector<size_t> unique_indices;
    std::vector<bool> is_duplicate(faces.size(), false);
    
    // Strategy: Keep the largest face from each group of similar faces
    // Sort by face size (area) in descending order
    std::vector<size_t> sorted_indices(faces.size());
    for (size_t i = 0; i < faces.size(); i++) {
        sorted_indices[i] = i;
    }
    
    std::sort(sorted_indices.begin(), sorted_indices.end(),
        [&faces](size_t a, size_t b) {
            return faces[a].area() > faces[b].area();
        });
    
    // For each face (starting with largest), check if it's similar to any already-kept face
    for (size_t i : sorted_indices) {
        if (is_duplicate[i]) continue;
        
        bool is_similar_to_kept = false;
        
        // Compare with all previously kept faces
        for (size_t kept_idx : unique_indices) {
            // Calculate cosine distance between encodings
            double distance = 0.0;
            double dot = 0.0;
            double norm1 = 0.0;
            double norm2 = 0.0;
            
            const auto& enc1 = encodings[i];
            const auto& enc2 = encodings[kept_idx];
            
            if (enc1.size() != enc2.size()) continue;
            
            for (size_t j = 0; j < enc1.size(); j++) {
                dot += enc1[j] * enc2[j];
                norm1 += enc1[j] * enc1[j];
                norm2 += enc2[j] * enc2[j];
            }
            
            norm1 = std::sqrt(norm1);
            norm2 = std::sqrt(norm2);
            
            if (norm1 > 0 && norm2 > 0) {
                distance = 1.0 - (dot / (norm1 * norm2));
                
                // If distance is below threshold, they're the same person
                if (distance < similarity_threshold) {
                    is_similar_to_kept = true;
                    is_duplicate[i] = true;
                    Logger::getInstance().debug(
                        "Face " + std::to_string(i) + " is duplicate of face " + 
                        std::to_string(kept_idx) + " (distance: " + std::to_string(distance) + ")"
                    );
                    break;
                }
            }
        }
        
        // If not similar to any kept face, keep this one
        if (!is_similar_to_kept) {
            unique_indices.push_back(i);
            Logger::getInstance().debug(
                "Face " + std::to_string(i) + " kept as unique (area: " + 
                std::to_string(faces[i].area()) + ")"
            );
        }
    }
    
    // Sort unique indices by original order (for consistent display)
    std::sort(unique_indices.begin(), unique_indices.end());
    
    Logger::getInstance().debug(
        "Deduplicated " + std::to_string(faces.size()) + " faces to " + 
        std::to_string(unique_indices.size()) + " unique faces"
    );
    
    return unique_indices;
}


// ============================================================================
// QUICK WIN #1: Histogram Equalization
// ============================================================================
// ============================================================================
// PHASE 2: Adaptive Preprocessing Pipeline for Variable Lighting
// ============================================================================

/**
 * Calculate mean brightness of an image (0-255 scale)
 */
static float calculateMeanBrightness(const uint8_t* data, int width, int height, int stride, int channels) {
    double sum = 0.0;
    int pixel_count = 0;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (channels == 3) {
                // Use standard luminance formula for RGB
                uint8_t b = data[y * stride + x * 3 + 0];
                uint8_t g = data[y * stride + x * 3 + 1];
                uint8_t r = data[y * stride + x * 3 + 2];
                sum += 0.299 * r + 0.587 * g + 0.114 * b;
            } else {
                sum += data[y * stride + x];
            }
            pixel_count++;
        }
    }
    
    return static_cast<float>(sum / pixel_count);
}

/**
 * Apply gamma correction to adjust brightness
 * gamma < 1.0: brighten image (for dark conditions)
 * gamma > 1.0: darken image (for bright conditions)
 */
Image FaceDetector::applyGammaCorrection(const Image& image, float gamma) {
    // Build lookup table for gamma correction
    uint8_t lut[256];
    for (int i = 0; i < 256; i++) {
        float normalized = i / 255.0f;
        float corrected = std::pow(normalized, gamma);
        lut[i] = static_cast<uint8_t>(std::min(255.0f, corrected * 255.0f + 0.5f));
    }
    
    // Apply LUT to image
    Image result = image.clone();
    uint8_t* data = result.data();
    int total_pixels = result.width() * result.height() * result.channels();
    
    for (int i = 0; i < total_pixels; i++) {
        data[i] = lut[data[i]];
    }
    
    return result;
}

/**
 * Adaptive gamma correction based on image brightness
 * Automatically brightens dark images and darkens bright images
 */
Image FaceDetector::applyAdaptiveGammaCorrection(const Image& image) {
    float mean_brightness = calculateMeanBrightness(
        image.data(), image.width(), image.height(), 
        image.stride(), image.channels()
    );
    
    const float target_brightness = 127.5f;  // Middle of 0-255 range
    const float brightness_tolerance = 20.0f;  // Don't adjust if within ±20 of target
    
    // Check if adjustment is needed
    if (std::abs(mean_brightness - target_brightness) < brightness_tolerance) {
        Logger::getInstance().debug("Brightness OK (" + std::to_string(mean_brightness) + 
                                   "), skipping gamma correction");
        return image.clone();
    }
    
    // Calculate gamma: lower gamma brightens, higher gamma darkens
    // For dark images (brightness < target): gamma < 1.0
    // For bright images (brightness > target): gamma > 1.0
    float gamma = target_brightness / std::max(1.0f, mean_brightness);
    
    // Clamp gamma to reasonable range to avoid over-correction
    gamma = std::max(0.5f, std::min(2.0f, gamma));
    
    Logger::getInstance().debug("Applying gamma correction: brightness=" + 
                               std::to_string(mean_brightness) + ", gamma=" + 
                               std::to_string(gamma));
    
    return applyGammaCorrection(image, gamma);
}

/**
 * Normalize brightness toward target level (127.5 on 0-255 scale)
 * Uses simple linear scaling
 */
Image FaceDetector::normalizeBrightness(const Image& image) {
    float mean_brightness = calculateMeanBrightness(
        image.data(), image.width(), image.height(),
        image.stride(), image.channels()
    );
    
    const float target_brightness = 127.5f;
    const float brightness_tolerance = 15.0f;
    
    // Check if adjustment needed
    if (std::abs(mean_brightness - target_brightness) < brightness_tolerance) {
        Logger::getInstance().debug("Brightness normalized (within tolerance): " + 
                                   std::to_string(mean_brightness));
        return image.clone();
    }
    
    // Calculate linear scaling factor
    float scale = target_brightness / std::max(1.0f, mean_brightness);
    
    // Clamp scale to avoid extreme adjustments
    scale = std::max(0.5f, std::min(2.0f, scale));
    
    Logger::getInstance().debug("Normalizing brightness: " + std::to_string(mean_brightness) + 
                               " -> " + std::to_string(target_brightness) + 
                               ", scale=" + std::to_string(scale));
    
    // Apply linear scaling
    Image result = image.clone();
    uint8_t* data = result.data();
    int total_pixels = result.width() * result.height() * result.channels();
    
    for (int i = 0; i < total_pixels; i++) {
        float val = data[i] * scale;
        data[i] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, val)));
    }
    
    return result;
}

/**
 * Apply adaptive CLAHE based on image histogram characteristics
 * Adjusts clip limit based on contrast level
 */
Image FaceDetector::applyAdaptiveCLAHE(const Image& image) {
    // Convert to grayscale if needed
    Image work_img;
    if (image.channels() == 3) {
        work_img = toGrayscale(image.data(), image.width(), 
                              image.height(), image.stride());
    } else {
        work_img = image.clone();
    }
    
    // Calculate histogram statistics
    std::vector<int> histogram(256, 0);
    const uint8_t* data = work_img.data();
    int total_pixels = work_img.width() * work_img.height();
    
    for (int i = 0; i < total_pixels; i++) {
        histogram[data[i]]++;
    }
    
    // Calculate histogram spread (standard deviation)
    double mean = 0.0;
    for (int i = 0; i < 256; i++) {
        mean += i * histogram[i];
    }
    mean /= total_pixels;
    
    double variance = 0.0;
    for (int i = 0; i < 256; i++) {
        double diff = i - mean;
        variance += diff * diff * histogram[i];
    }
    variance /= total_pixels;
    double std_dev = std::sqrt(variance);
    
    // Adaptive clip limit based on contrast
    // Low contrast (low std_dev): higher clip limit for more enhancement
    // High contrast (high std_dev): lower clip limit to avoid over-enhancement
    double base_clip_limit = 2.0;
    double clip_limit;
    
    if (std_dev < 30.0) {
        // Very low contrast - apply strong enhancement
        clip_limit = base_clip_limit * 2.5;
        Logger::getInstance().debug("Low contrast detected (std=" + std::to_string(std_dev) + 
                                   "), using high CLAHE clip limit: " + std::to_string(clip_limit));
    } else if (std_dev < 60.0) {
        // Medium contrast - moderate enhancement
        clip_limit = base_clip_limit * 1.5;
        Logger::getInstance().debug("Medium contrast (std=" + std::to_string(std_dev) + 
                                   "), using moderate CLAHE clip limit: " + std::to_string(clip_limit));
    } else {
        // High contrast - gentle enhancement
        clip_limit = base_clip_limit;
        Logger::getInstance().debug("High contrast (std=" + std::to_string(std_dev) + 
                                   "), using base CLAHE clip limit: " + std::to_string(clip_limit));
    }
    
    // Apply CLAHE
    CLAHE clahe(clip_limit, 8, 8);
    Image result(work_img.width(), work_img.height(), 1);
    
    clahe.apply(work_img.data(), result.data(), 
               work_img.width(), work_img.height(),
               work_img.width(), result.width());
    
    // Convert back to BGR if original was color
    if (image.channels() == 3) {
        Image bgr_result(result.width(), result.height(), 3);
        uint8_t* bgr_data = bgr_result.data();
        const uint8_t* gray_data = result.data();
        
        for (int i = 0; i < total_pixels; i++) {
            bgr_data[i * 3 + 0] = gray_data[i];  // B
            bgr_data[i * 3 + 1] = gray_data[i];  // G
            bgr_data[i * 3 + 2] = gray_data[i];  // R
        }
        return bgr_result;
    }
    
    return result;
}

// ============================================================================
// PHASE 5: Synthetic Lighting Augmentation for Enrollment
// ============================================================================

Image FaceDetector::simulateBrightLighting(const Image& image) {
    // Simulate bright lighting: increase brightness by 1.7x with gamma adjustment
    // This mimics well-lit office environments or outdoor sunny conditions
    
    Image result = image.clone();
    uint8_t* data = result.data();
    int total_pixels = result.width() * result.height() * result.channels();
    
    // Apply brightness multiplier with saturation
    const float brightness_factor = 1.7f;
    
    for (int i = 0; i < total_pixels; i++) {
        float pixel = data[i] * brightness_factor;
        // Clamp to 0-255 range
        data[i] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, pixel)));
    }
    
    // Apply gamma correction to balance the brightness boost
    // Gamma < 1.0 brightens dark areas more than bright areas
    result = applyGammaCorrection(result, 0.8f);
    
    Logger::getInstance().debug("Applied bright lighting simulation (1.7x brightness, gamma=0.8)");
    
    return result;
}

Image FaceDetector::simulateDimLighting(const Image& image) {
    // Simulate dim/shadow lighting: reduce brightness to 0.5x with contrast reduction
    // This mimics low-light indoor environments or shadowed conditions
    
    Image result = image.clone();
    uint8_t* data = result.data();
    int total_pixels = result.width() * result.height() * result.channels();
    
    // Apply brightness reduction
    const float brightness_factor = 0.5f;
    
    for (int i = 0; i < total_pixels; i++) {
        float pixel = data[i] * brightness_factor;
        data[i] = static_cast<uint8_t>(pixel);
    }
    
    // Apply gamma correction to simulate shadow characteristics
    // Gamma > 1.0 darkens the image with more emphasis on mid-tones
    result = applyGammaCorrection(result, 1.3f);
    
    // Reduce contrast slightly to simulate diffuse lighting in shadows
    // Apply subtle CLAHE to maintain some detail
    result = applyAdaptiveCLAHE(result);
    
    Logger::getInstance().debug("Applied dim lighting simulation (0.5x brightness, gamma=1.3, contrast reduction)");
    
    return result;
}

// ============================================================================
// QUICK WIN #2: Face Quality Assessment
// ============================================================================
faceid::FaceQuality FaceDetector::assessFaceQuality(
    const Image& aligned_face,
    const Rect& original_face,
    int frame_width,
    float detector_confidence) {
    
    FaceQuality quality;
    
    // Convert to grayscale for analysis
    Image gray_aligned = aligned_face.channels() == 1 ? 
        aligned_face.clone() : 
        toGrayscale(aligned_face.data(), aligned_face.width(), 
                   aligned_face.height(), aligned_face.stride());
    
    int width = gray_aligned.width();
    int height = gray_aligned.height();
    const uint8_t* data = gray_aligned.data();
    int total_pixels = width * height;
    
    // 1. Blur detection using Laplacian operator
    {
        double laplacian_sum = 0.0;
        int sample_count = 0;
        
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x += 2) {
                uint8_t center = data[y * width + x];
                uint8_t top = data[(y - 1) * width + x];
                uint8_t bottom = data[(y + 1) * width + x];
                uint8_t left = data[y * width + (x - 1)];
                uint8_t right = data[y * width + (x + 1)];
                
                int laplacian = 4 * center - top - bottom - left - right;
                laplacian_sum += std::abs(laplacian);
                sample_count++;
            }
        }
        
        if (sample_count > 0) {
            double mean_laplacian = laplacian_sum / sample_count;
            quality.blur_score = std::min(1.0, mean_laplacian / 80.0);
        } else {
            quality.blur_score = 0.5;
        }
    }
    
    // 2. Face size relative to frame
    {
        double face_size_percent = static_cast<double>(original_face.width) / frame_width;
        if (face_size_percent < 0.05) {
            quality.size_score = 0.2;  // Too small
        } else if (face_size_percent > 0.3) {
            quality.size_score = 0.7;  // Very large
        } else {
            quality.size_score = std::min(1.0, face_size_percent / 0.15);
        }
    }
    
    // 3. Brightness uniformity
    {
        uint64_t sum = 0;
        for (int i = 0; i < total_pixels; i++) {
            sum += data[i];
        }
        double mean = static_cast<double>(sum) / total_pixels;
        
        uint64_t var_sum = 0;
        for (int i = 0; i < total_pixels; i++) {
            double diff = data[i] - mean;
            var_sum += static_cast<uint64_t>(diff * diff);
        }
        double variance = static_cast<double>(var_sum) / total_pixels;
        double stddev = std::sqrt(variance);
        
        double uniformity_score = std::max(0.0, 1.0 - std::abs(stddev - 35.0) / 100.0);
        double brightness_in_range = (mean > 20 && mean < 235) ? 1.0 : 0.5;
        
        quality.brightness_score = uniformity_score * 0.7 + brightness_in_range * 0.3;
    }
    
    // 4. Detector confidence weight
    quality.confidence_score = detector_confidence;
    
    // 5. Combined score
    quality.overall_score = 
        quality.blur_score * 0.35 +
        quality.size_score * 0.25 +
        quality.brightness_score * 0.25 +
        quality.confidence_score * 0.15;
    
    return quality;
}

// ============================================================================
// QUICK WIN #4: Head Pose Estimation
// ============================================================================
FaceDetector::HeadPose FaceDetector::estimateHeadPose(const std::vector<Point>& landmarks) {
    HeadPose pose;
    pose.yaw = 0.0f;
    pose.pitch = 0.0f;
    pose.roll = 0.0f;
    
    if (landmarks.size() < 5) {
        return pose;
    }
    
    Point left_eye = landmarks[0];
    Point right_eye = landmarks[1];
    Point nose = landmarks[2];
    Point left_mouth = landmarks[3];
    Point right_mouth = landmarks[4];
    
    Point face_center(
        (left_eye.x + right_eye.x) / 2.0f,
        (left_eye.y + right_eye.y) / 2.0f
    );
    
    // Roll: Eye line alignment
    float eye_dx = right_eye.x - left_eye.x;
    float eye_dy = right_eye.y - left_eye.y;
    pose.roll = std::atan2(eye_dy, eye_dx) * (180.0f / M_PI);
    
    // Yaw: Nose offset from face center
    float nose_dx = nose.x - face_center.x;
    if (eye_dx > 1.0f) {
        pose.yaw = (nose_dx / eye_dx) * 30.0f;
        pose.yaw = std::max(-45.0f, std::min(45.0f, pose.yaw));
    }
    
    // Pitch: Nose vertical position
    float eye_y = (left_eye.y + right_eye.y) / 2.0f;
    float mouth_y = (left_mouth.y + right_mouth.y) / 2.0f;
    float expected_nose_y = (eye_y + mouth_y) / 2.0f;
    float nose_dy = nose.y - expected_nose_y;
    float face_height = mouth_y - eye_y;
    
    if (face_height > 1.0f) {
        pose.pitch = -(nose_dy / face_height) * 25.0f;
        pose.pitch = std::max(-45.0f, std::min(45.0f, pose.pitch));
    }
    
    return pose;
}

// ========================================================================
// QUICK WIN #4: Head Pose Correction - Complete Implementation
// ========================================================================
// Applies perspective correction to compensate for head rotation
// This function is COMPLETE but NOT yet integrated into the recognition pipeline
// To use: Call before alignFace() in encodeFaces() when enable_head_pose_correction=true

/**
 * Apply perspective correction to face region based on head pose
 * Compensates for yaw/pitch/roll to normalize head orientation
 * @param frame Source image containing the face
 * @param face_rect Face bounding box with landmarks
 * @param pose Estimated head rotation angles
 * @return Corrected image with normalized head orientation
 */
Image FaceDetector::applyHeadPoseCorrection(const ImageView& frame, const Rect& face_rect, const FaceDetector::HeadPose& pose) {
    // If rotation is minimal, skip correction (performance optimization)
    if (std::abs(pose.yaw) < 5.0f && std::abs(pose.pitch) < 5.0f && std::abs(pose.roll) < 5.0f) {
        // Extract and return face region without correction
        int x = std::max(0, face_rect.x);
        int y = std::max(0, face_rect.y);
        int w = std::min(face_rect.width, frame.width() - x);
        int h = std::min(face_rect.height, frame.height() - y);
        
        Image face_region(w, h, frame.channels());
        const uint8_t* src = frame.data() + y * frame.stride() + x * frame.channels();
        uint8_t* dst = face_region.data();
        
        for (int row = 0; row < h; row++) {
            std::memcpy(dst, src, w * frame.channels());
            src += frame.stride();
            dst += face_region.stride();
        }
        
        return face_region;
    }
    
    // Get face center and landmarks
    if (!face_rect.hasLandmarks() || face_rect.landmarks.size() < 5) {
        // Fallback: return uncorrected face region
        int x = std::max(0, face_rect.x);
        int y = std::max(0, face_rect.y);
        int w = std::min(face_rect.width, frame.width() - x);
        int h = std::min(face_rect.height, frame.height() - y);
        
        Image face_region(w, h, frame.channels());
        const uint8_t* src = frame.data() + y * frame.stride() + x * frame.channels();
        uint8_t* dst = face_region.data();
        
        for (int row = 0; row < h; row++) {
            std::memcpy(dst, src, w * frame.channels());
            src += frame.stride();
            dst += face_region.stride();
        }
        
        return face_region;
    }
    
    const auto& landmarks = face_rect.landmarks;
    Point face_center(
        (landmarks[0].x + landmarks[1].x) / 2.0f,  // Eye center X
        (landmarks[0].y + landmarks[1].y) / 2.0f   // Eye center Y
    );
    
    // Build 3D rotation matrix from Euler angles (yaw, pitch, roll)
    // Convert degrees to radians
    float yaw_rad = pose.yaw * M_PI / 180.0f;
    float pitch_rad = pose.pitch * M_PI / 180.0f;
    float roll_rad = pose.roll * M_PI / 180.0f;
    
    // Rotation matrices (right-handed coordinate system)
    // Yaw (Y-axis rotation)
    float cos_yaw = std::cos(yaw_rad);
    // Note: sin_yaw not needed for simplified 2D affine transform
    
    // Pitch (X-axis rotation)
    // Note: cos_pitch not needed for simplified 2D affine transform
    float sin_pitch = std::sin(pitch_rad);
    
    // Roll (Z-axis rotation)
    float cos_roll = std::cos(roll_rad);
    float sin_roll = std::sin(roll_rad);
    
    // Combined rotation matrix (ZYX order: roll * pitch * yaw)
    // This is a simplified 2D projection of the 3D rotation
    // For full perspective correction, we'd need a proper 3D→2D projection
    // Here we approximate with an affine transform
    
    // For yaw correction: scale X based on cosine (simulate foreshortening)
    float yaw_scale = std::max(0.7f, cos_yaw);  // Don't over-compress
    
    // For pitch: shift Y position
    float pitch_shift = sin_pitch * face_rect.height * 0.2f;
    
    // Build 2D affine transformation matrix
    // [a b tx]
    // [c d ty]
    float a = yaw_scale * cos_roll;
    float b = -yaw_scale * sin_roll;
    float c = sin_roll;
    float d = cos_roll;
    
    // Translation to keep face centered
    float tx = face_center.x - (a * face_center.x + b * face_center.y);
    float ty = face_center.y + pitch_shift - (c * face_center.x + d * face_center.y);
    
    // Output size (use original face rect size)
    int out_w = face_rect.width;
    int out_h = face_rect.height;
    
    // Apply transformation
    Image corrected(out_w, out_h, frame.channels());
    uint8_t* dst_data = corrected.data();
    const uint8_t* src_data = frame.data();
    int src_width = frame.width();
    int src_height = frame.height();
    int src_stride = frame.stride();
    int channels = frame.channels();
    
    // Inverse transformation for backward mapping
    float det = a * d - b * c;
    if (std::abs(det) < 1e-6f) {
        // Singular matrix, return uncorrected
        Logger::getInstance().debug("Singular transformation matrix in head pose correction");
        
        int x = std::max(0, face_rect.x);
        int y = std::max(0, face_rect.y);
        int w = std::min(face_rect.width, frame.width() - x);
        int h = std::min(face_rect.height, frame.height() - y);
        
        Image face_region(w, h, channels);
        const uint8_t* src = frame.data() + y * frame.stride() + x * channels;
        uint8_t* dst = face_region.data();
        
        for (int row = 0; row < h; row++) {
            std::memcpy(dst, src, w * channels);
            src += frame.stride();
            dst += face_region.stride();
        }
        
        return face_region;
    }
    
    float inv_a = d / det;
    float inv_b = -b / det;
    float inv_c = -c / det;
    float inv_d = a / det;
    float inv_tx = -(inv_a * tx + inv_b * ty);
    float inv_ty = -(inv_c * tx + inv_d * ty);
    
    // Apply transformation with bilinear interpolation
    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            // Map destination pixel to source (relative to face_rect origin)
            float rel_x = static_cast<float>(x);
            float rel_y = static_cast<float>(y);
            float src_x = inv_a * rel_x + inv_b * rel_y + inv_tx + face_rect.x;
            float src_y = inv_c * rel_x + inv_d * rel_y + inv_ty + face_rect.y;
            
            // Bounds check
            if (src_x < 0 || src_x >= src_width - 1 || src_y < 0 || src_y >= src_height - 1) {
                // Out of bounds: fill with black
                for (int c = 0; c < channels; c++) {
                    dst_data[y * corrected.stride() + x * channels + c] = 0;
                }
                continue;
            }
            
            // Bilinear interpolation
            int x0 = static_cast<int>(src_x);
            int y0 = static_cast<int>(src_y);
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            float fx = src_x - x0;
            float fy = src_y - y0;
            
            for (int c = 0; c < channels; c++) {
                float v00 = src_data[y0 * src_stride + x0 * channels + c];
                float v10 = src_data[y0 * src_stride + x1 * channels + c];
                float v01 = src_data[y1 * src_stride + x0 * channels + c];
                float v11 = src_data[y1 * src_stride + x1 * channels + c];
                
                float v0 = v00 * (1.0f - fx) + v10 * fx;
                float v1 = v01 * (1.0f - fx) + v11 * fx;
                float v = v0 * (1.0f - fy) + v1 * fy;
                
                dst_data[y * corrected.stride() + x * channels + c] = static_cast<uint8_t>(v);
            }
        }
    }
    
    Logger::getInstance().debug("Applied head pose correction: yaw=" + std::to_string(pose.yaw) + 
                               "°, pitch=" + std::to_string(pose.pitch) + 
                               "°, roll=" + std::to_string(pose.roll) + "°");
    
    return corrected;
}


} // namespace faceid
