#include "face_detector.h"
#include "clahe.h"
#include "optical_flow.h"
#include "config_paths.h"
#include "config.h"
#include "logger.h"
#include "detectors/common.h"
#include "detectors/detectors.h"
#include <libyuv.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <regex>
#include <dirent.h>
#include <memory>
#include <unordered_map>
#include <mutex>

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

// Helper: Auto-detect detection model type from param file structure
DetectionModelType FaceDetector::detectModelType(const std::string& param_path) {
    std::ifstream file(param_path);
    if (!file.is_open()) {
        Logger::getInstance().debug("Failed to open detection param file: " + param_path);
        return DetectionModelType::UNKNOWN;
    }
    
    std::string line;
    bool has_data_input = false;          // RetinaFace uses "data" as input
    bool has_in0_input = false;           // YuNet uses "in0" as input
    bool has_face_rpn_outputs = false;    // RetinaFace has face_rpn_* outputs
    int out_count = 0;                    // Count out0, out1, out2, ... outputs
    
    while (std::getline(file, line)) {
        // Check for input layer names
        if (line.find("Input") != std::string::npos) {
            if (line.find(" data ") != std::string::npos) {
                has_data_input = true;
            } else if (line.find(" in0 ") != std::string::npos) {
                has_in0_input = true;
            }
        }
        
        // Check for RetinaFace-specific output blobs
        if (line.find("face_rpn") != std::string::npos) {
            has_face_rpn_outputs = true;
        }
        
        // Count generic outputs (out0, out1, out2, ...)
        for (int i = 0; i < 20; i++) {
            std::string out_name = " out" + std::to_string(i);
            if (line.find(out_name) != std::string::npos) {
                // Make sure it's actually an output (appears after layer name)
                size_t pos = line.find(out_name);
                if (pos != std::string::npos && pos > 20) {  // Not at the beginning of line
                    out_count = std::max(out_count, i + 1);
                }
            }
        }
    }
    
    // Determine model type based on structure
    if (has_data_input && has_face_rpn_outputs) {
        Logger::getInstance().debug("Detected RetinaFace model (input='data', outputs=face_rpn_*)");
        return DetectionModelType::RETINAFACE;
    } else if (has_in0_input && out_count >= 12) {
        Logger::getInstance().debug("Detected YuNet model (input='in0', " + std::to_string(out_count) + " outputs)");
        return DetectionModelType::YUNET;
    }
    
    // Check for YOLO-specific patterns
    bool has_yolov5_outputs = false;  // YOLOv5: outputs "981", "983", "985"
    bool has_yolov7_outputs = false;  // YOLOv7: outputs "stride_8", "stride_16", "stride_32"
    bool has_yolov8_outputs = false;  // YOLOv8: outputs "output0", "1076", "1084"
    bool has_images_input = false;    // YOLOv7/v8 use "images" as input
    
    // Rewind file to check again
    file.clear();
    file.seekg(0);
    while (std::getline(file, line)) {
        // Check for YOLOv5 output layers (appear as outputs at end of line)
        if (line.find(" 981") != std::string::npos || 
            line.find(" 983") != std::string::npos || 
            line.find(" 985") != std::string::npos) {
            has_yolov5_outputs = true;
        }
        // Check for YOLOv7 output layers
        if (line.find("stride_8") != std::string::npos || 
            line.find("stride_16") != std::string::npos || 
            line.find("stride_32") != std::string::npos) {
            has_yolov7_outputs = true;
        }
        // Check for YOLOv8 output layers (appear as outputs at end of line)
        if (line.find(" output0") != std::string::npos || 
            line.find(" 1076") != std::string::npos || 
            line.find(" 1084") != std::string::npos) {
            has_yolov8_outputs = true;
        }
        // Check for "images" input (YOLOv7/v8)
        if (line.find("Input") != std::string::npos && line.find(" images ") != std::string::npos) {
            has_images_input = true;
        }
    }
    
    // Determine YOLO version
    if (has_data_input && has_yolov5_outputs) {
        Logger::getInstance().debug("Detected YOLOv5-Face model (input='data', outputs='981', '983', '985')");
        return DetectionModelType::YOLOV5;
    } else if (has_images_input && has_yolov7_outputs) {
        Logger::getInstance().debug("Detected YOLOv7-Face model (input='images', outputs='stride_8', 'stride_16', 'stride_32')");
        return DetectionModelType::YOLOV7;
    } else if (has_images_input && has_yolov8_outputs) {
        Logger::getInstance().debug("Detected YOLOv8-Face model (input='images', outputs='output0', '1076', '1084')");
        return DetectionModelType::YOLOV8;
    }
    
    Logger::getInstance().debug("Unknown detection model type (data=" + std::to_string(has_data_input) + 
        ", in0=" + std::to_string(has_in0_input) + ", face_rpn=" + std::to_string(has_face_rpn_outputs) + 
        ", out_count=" + std::to_string(out_count) + ")");
    return DetectionModelType::UNKNOWN;
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
    // RetinaFace and SFace models loaded separately via loadModels()
}

bool FaceDetector::loadModels(const std::string& model_base_path, const std::string& detection_model_path) {
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
        
        // Configure NCNN options for optimal CPU performance
        ncnn_net_.opt.use_vulkan_compute = false;
        ncnn_net_.opt.num_threads = 4;
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
        
        // Load detection model with priority system
        std::string detection_base;
        
        // If explicit detection path provided, use it
        if (!detection_model_path.empty()) {
            detection_base = detection_model_path;
            Logger::getInstance().debug("Using explicit detection model path: " + detection_base);
        } else {
            // Priority 1: Try standard name "detection.{param,bin}"
            std::string standard_detection = std::string(MODELS_DIR) + "/detection";
            std::string standard_detection_param = standard_detection + ".param";
            std::string standard_detection_bin = standard_detection + ".bin";
            
            std::ifstream det_param_check(standard_detection_param);
            std::ifstream det_bin_check(standard_detection_bin);
            
            if (det_param_check.good() && det_bin_check.good()) {
                Logger::getInstance().debug("Found standard detection model: detection.{param,bin}");
                detection_base = standard_detection;
            } else {
                // Priority 2: Try legacy mnet.25-opt (RetinaFace)
                Logger::getInstance().debug("Standard detection name not found, trying mnet.25-opt");
                detection_base = std::string(MODELS_DIR) + "/mnet.25-opt";
                
                std::string legacy_param = detection_base + ".param";
                std::string legacy_bin = detection_base + ".bin";
                std::ifstream legacy_param_check(legacy_param);
                std::ifstream legacy_bin_check(legacy_bin);
                
                if (!legacy_param_check.good() || !legacy_bin_check.good()) {
                    // Priority 3: Try RFB-320
                    Logger::getInstance().debug("mnet.25-opt not found, trying RFB-320");
                    detection_base = std::string(MODELS_DIR) + "/RFB-320";
                }
            }
        }
        
        std::string retinaface_param = detection_base + ".param";
        std::string retinaface_bin = detection_base + ".bin";
        
        // Check if .param exists, if not try .ncnn.param
        std::ifstream det_param_check(retinaface_param);
        if (!det_param_check.good()) {
            retinaface_param = detection_base + ".ncnn.param";
            retinaface_bin = detection_base + ".ncnn.bin";
        }
        
        Logger::getInstance().debug("Loading detection model from: " + detection_base);
        
        // Check if this model was already loaded
        bool det_was_cached = isModelCached(retinaface_param, retinaface_bin);
        if (det_was_cached) {
            Logger::getInstance().debug("Detection model cache HIT (faster due to FS cache)");
        }
        
        retinaface_net_.opt.use_vulkan_compute = false;
        retinaface_net_.opt.num_threads = 4;
        retinaface_net_.opt.use_fp16_packed = false;
        retinaface_net_.opt.use_fp16_storage = false;
        
        ret = retinaface_net_.load_param(retinaface_param.c_str());
        if (ret != 0) {
            // Detection model not found - this is OK, will fall back if needed
            detection_model_loaded_ = false;
            Logger::getInstance().debug("Detection model param not found (ret=" + std::to_string(ret) + "), detection_model_loaded_=false");
        } else {
            ret = retinaface_net_.load_model(retinaface_bin.c_str());
            if (ret != 0) {
                detection_model_loaded_ = false;
                Logger::getInstance().debug("Detection model bin not found (ret=" + std::to_string(ret) + "), detection_model_loaded_=false");
            } else {
                detection_model_loaded_ = true;
                
                // Mark as cached
                if (!det_was_cached) {
                    markModelCached(retinaface_param, retinaface_bin);
                }
                // Auto-detect detection model type
                detection_model_type_ = detectModelType(retinaface_param);
                detection_model_name_ = detection_base.substr(detection_base.find_last_of("/\\") + 1);
                
                // Adjust default confidence threshold based on model type (if not set by user)
                auto confidence_opt = Config::getInstance().getDouble("recognition", "confidence");
                if (!confidence_opt.has_value()) {
                    // User didn't specify confidence in config, use default
                    detection_confidence_threshold_ = 0.8f;
                    Logger::getInstance().debug("Using default confidence: 0.8");
                }
                
                // Try to read original detection model name from .use file
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
                            if (key == "detection") {
                                detection_model_name_ = value;
                                break;
                            }
                        }
                    }
                }
                
                std::string model_type_str;
                switch (detection_model_type_) {
                    case DetectionModelType::RETINAFACE: model_type_str = "RetinaFace"; break;
                    case DetectionModelType::YUNET: model_type_str = "YuNet"; break;
                    case DetectionModelType::YOLOV5: model_type_str = "YOLOv5-Face"; break;
                    case DetectionModelType::YOLOV7: model_type_str = "YOLOv7-Face"; break;
                    case DetectionModelType::YOLOV8: model_type_str = "YOLOv8-Face"; break;
                    default: model_type_str = "Unknown"; break;
                }
                
                Logger::getInstance().debug("Detection model loaded successfully: " + detection_model_name_ + " (type: " + model_type_str + ")");
            }
        }
        
        return true;
    } catch (const std::exception& e) {
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
        case DetectionModelType::YUNET:
            {
                // Convert BGR to RGB (all models expect RGB)
                ncnn::Mat in = ncnn::Mat::from_pixels(frame.data(), ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h);
                
                if (detection_model_type_ == DetectionModelType::RETINAFACE) {
                    faces = detectWithRetinaFace(in, img_w, img_h, confidence_threshold);
                } else if (detection_model_type_ == DetectionModelType::YUNET) {
                    faces = detectWithYuNet(in, img_w, img_h, confidence_threshold);
                }
            }
            break;
        case DetectionModelType::YOLOV5:
        case DetectionModelType::YOLOV7:
        case DetectionModelType::YOLOV8:
            {
                // Convert BGR to RGB (all YOLO models expect RGB)
                ncnn::Mat in = ncnn::Mat::from_pixels(frame.data(), ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h);
                
                // Apply model-specific default thresholds if not specified
                float yolo_threshold = confidence_threshold;
                if (confidence_threshold == detection_confidence_threshold_) {
                    // User didn't specify threshold, use model-specific default
                    if (detection_model_type_ == DetectionModelType::YOLOV7) {
                        yolo_threshold = 0.65f;  // YOLOv7 needs higher threshold
                    } else {
                        yolo_threshold = 0.5f;   // YOLOv5 and YOLOv8 use 0.5
                    }
                }
                
                if (detection_model_type_ == DetectionModelType::YOLOV5) {
                    faces = detectWithYOLOv5(retinaface_net_, in, img_w, img_h, yolo_threshold);
                } else if (detection_model_type_ == DetectionModelType::YOLOV7) {
                    faces = detectWithYOLOv7(retinaface_net_, in, img_w, img_h, yolo_threshold);
                } else {
                    faces = detectWithYOLOv8(retinaface_net_, in, img_w, img_h, yolo_threshold);
                }
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
    return ::faceid::detectWithYuNet(retinaface_net_, in, img_w, img_h, confidence_threshold);
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
    const std::vector<Rect>& face_locations) {
    
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
        
        // Align face for SFace (112x112)
        Image aligned = alignFace(frame, face_rect);
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
            Logger::getInstance().debug("NCNN inference FAILED with ret=" + std::to_string(ret));
            continue;
        }
        
        Logger::getInstance().debug("NCNN inference SUCCESS, output dims: w=" + std::to_string(out.w) + 
            " h=" + std::to_string(out.h) + " c=" + std::to_string(out.c));
        
        // Validate output dimensions (check against detected model dimension)
        if (out.w != static_cast<int>(current_encoding_dim_) || out.h != 1 || out.c != 1) {
            // Unexpected output dimensions - skip this face
            Logger::getInstance().debug("Output dimensions INVALID: expected w=" + std::to_string(current_encoding_dim_) + 
                " h=1 c=1, got w=" + std::to_string(out.w) + " h=" + std::to_string(out.h) + " c=" + std::to_string(out.c));
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

} // namespace faceid
