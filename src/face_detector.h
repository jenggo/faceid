#ifndef FACEID_FACE_DETECTOR_H
#define FACEID_FACE_DETECTOR_H

#include "encoding_config.h"  // FACE_ENCODING_DIM constant
#include "image.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <ncnn/net.h>             // NCNN for face recognition and detection

namespace faceid {

// Face encodings are stored as std::vector<float> for NCNN compatibility
using FaceEncoding = std::vector<float>;

// Detection model types (auto-detected from .param file structure)
enum class DetectionModelType {
    RETINAFACE,   // RetinaFace (mnet.25-opt): input="data", outputs=face_rpn_*
    YUNET,        // YuNet: input="in0", 12 outputs (out0-out11)
    YOLOV5,       // YOLOv5-Face: input="data", outputs="981", "983", "985"
    YOLOV7,       // YOLOv7-Face: input="images", outputs="stride_8", "stride_16", "stride_32"
    YOLOV8,       // YOLOv8-Face: input="images", outputs="output0", "1076", "1084"
    UNKNOWN
};

class FaceDetector {
public:
    FaceDetector();
    
    // Load models from base path
    // - Recognition model (SFace): defaults to CONFIG_DIR/models/sface
    // - Detection model (RetinaFace): defaults to CONFIG_DIR/models/mnet.25-opt
    // Pass empty string to use defaults, or full path without extension
    // Will load .param and .bin files automatically
    // detection_model_path: optional separate path for detection model (if empty, uses standard location)
    bool loadModels(const std::string& model_base_path = "", const std::string& detection_model_path = "");
    
    // Detect faces in frame using RetinaFace
    // confidence_threshold: minimum detection confidence (0.0-1.0, 0 = use config default)
    std::vector<Rect> detectFaces(const ImageView& frame, bool downscale = false, float confidence_threshold = 0.0f);
    
    // Detect or track faces (automatically uses tracking when possible)
    // track_interval: how many frames to track before re-detecting (0 = always detect)
    // confidence_threshold: minimum detection confidence (0.0-1.0, 0 = use config default)
    std::vector<Rect> detectOrTrackFaces(const ImageView& frame, int track_interval = 5, float confidence_threshold = 0.0f);
    
    // Force re-detection on next frame (useful after scene change)
    void resetTracking();
    
    // Encode faces using SFace
    std::vector<FaceEncoding> encodeFaces(const ImageView& frame,
                                          const std::vector<Rect>& face_locations);
    
    // Compare two face encodings (cosine similarity)
    double compareFaces(const FaceEncoding& encoding1, const FaceEncoding& encoding2);
    
    // Performance: Pre-process frame for faster detection (CLAHE enhancement)
    Image preprocessFrame(const ImageView& frame);
    
    // Enhanced preprocessing with more aggressive CLAHE (for very dark/difficult images)
    // Uses 4x4 tiles and higher clip limit for stronger local enhancement
    Image preprocessFrameAggressive(const ImageView& frame);
    
    // Cascading detection with automatic fallback (for PAM/Presence/CLI)
    // Stage 1: Standard preprocessing + primary detector
    // Stage 2: Aggressive preprocessing + primary detector
    // Stage 3: Aggressive preprocessing + detection2 fallback (if available)
    // Returns: detected faces and cascade stage used (1, 2, or 3)
    struct CascadeResult {
        std::vector<Rect> faces;
        int stage_used;              // Which cascade stage succeeded (1, 2, or 3)
        Image processed_frame;       // Cached preprocessed frame
        bool has_motion;             // Motion detected (if pre-check enabled)
        double avg_brightness;       // Frame brightness (0.0-1.0)
        double stage1_time_ms;       // Stage 1 execution time
        double stage2_time_ms;       // Stage 2 execution time
        double stage3_time_ms;       // Stage 3 execution time
    };
    CascadeResult detectFacesCascade(const ImageView& frame, 
                                     bool enable_motion_check = false,
                                     float confidence_threshold = 0.0f);
    
    // Enable/disable caching for repeated detections
    void enableCache(bool enable);
    
    // Clear encoding cache
    void clearCache();
    
    // Multi-face detection helpers for "no peek" feature
    // Calculate distance between two face rectangles (center-to-center)
    static double faceDistance(const Rect& face1, const Rect& face2);
    
    // Check if two faces are distinct persons (not same person at different position)
    static bool areDistinctFaces(const Rect& face1, const Rect& face2, int min_distance);
    
    // Get face size as percentage of frame width
    static double getFaceSizePercent(const Rect& face, int frame_width);
    
    // Count distinct faces (filtering out duplicates/reflections)
    static int countDistinctFaces(const std::vector<Rect>& faces, int min_distance);
    
    // Get current encoding dimension (from loaded model)
    size_t getEncodingDimension() const { return current_encoding_dim_; }
    
    // Get current recognition model name
    const std::string& getModelName() const { return current_model_name_; }
    
    // Get current detection model name
    const std::string& getDetectionModelName() const { return detection_model_name_; }
    
    // Get current detection2 model name (fallback)
    const std::string& getDetection2ModelName() const { return detection2_model_name_; }
    
    // Check if detection2 model is loaded
    bool hasDetection2Model() const { return detection2_model_loaded_; }
    
    // Get current detection model type as string
    std::string getDetectionModelType() const {
        switch (detection_model_type_) {
            case DetectionModelType::RETINAFACE: return "RetinaFace";
            case DetectionModelType::YUNET: return "YuNet";
            case DetectionModelType::YOLOV5: return "YOLOv5-Face";
            case DetectionModelType::YOLOV7: return "YOLOv7-Face";
            case DetectionModelType::YOLOV8: return "YOLOv8-Face";
            default: return "Unknown";
        }
    }
    
    // Get current recognition model type as string
    std::string getRecognitionModelType() const {
        // Return actual model name if available
        if (!current_model_name_.empty() && current_model_name_ != "recognition") {
            // Extract model type from filename
            std::string model_lower = current_model_name_;
            std::transform(model_lower.begin(), model_lower.end(), model_lower.begin(), ::tolower);
            
            if (model_lower.find("sface") != std::string::npos) {
                return "SFace";
            } else if (model_lower.find("mobilefacenet") != std::string::npos || 
                       model_lower.find("mobile_face") != std::string::npos) {
                return "MobileFaceNet";
            } else if (model_lower.find("arcface") != std::string::npos) {
                if (current_encoding_dim_ == 256) {
                    return "ArcFace-R34";
                } else {
                    return "ArcFace-R50";
                }
            } else if (model_lower.find("webface") != std::string::npos) {
                return "WebFace-R50";
            } else if (model_lower.find("glint360k") != std::string::npos) {
                return "Glint360K-R50";
            } else if (model_lower.find("ms1m") != std::string::npos || 
                       model_lower.find("ms1mv2") != std::string::npos) {
                return "MS1M-R50";
            }
        }
        
        // Fallback: For generic names like "recognition", show dimension
        // This happens when models are installed via "faceid use" command
        if (current_encoding_dim_ == 128) {
            return "Unknown (128D - likely SFace)";
        } else if (current_encoding_dim_ == 192) {
            return "Unknown (192D - likely MobileFaceNet)";
        } else if (current_encoding_dim_ == 256) {
            return "Unknown (256D - likely ArcFace-R34)";
        } else if (current_encoding_dim_ == 512) {
            return "Unknown (512D - ArcFace/WebFace/MS1M)";
        } else {
            return "Unknown (" + std::to_string(current_encoding_dim_) + "D)";
        }
    }
    
    // Helper: Parse param file to extract output dimensions
    size_t parseModelOutputDim(const std::string& param_path);
    
    // Helper: Auto-detect detection model type from param file
    DetectionModelType detectModelType(const std::string& param_path);
    
    // Helper: Deduplicate faces based on encoding similarity
    // Returns indices of unique faces (filters out duplicates of the same person)
    static std::vector<size_t> deduplicateFaces(
        const std::vector<Rect>& faces,
        const std::vector<FaceEncoding>& encodings,
        double similarity_threshold = 0.15  // Faces with distance < 0.15 are considered same person
    );

private:
    // NCNN networks
    ncnn::Net ncnn_net_;          // Face recognition model (auto-detected)
    ncnn::Net retinaface_net_;    // Face detection model (auto-detected type)
    ncnn::Net detection2_net_;    // Face detection2 model (cascade fallback)
    
    bool models_loaded_ = false;
    bool detection_model_loaded_ = false;
    bool detection2_model_loaded_ = false;
    
    // Detection model information (auto-detected from param file)
    DetectionModelType detection_model_type_ = DetectionModelType::UNKNOWN;
    DetectionModelType detection2_model_type_ = DetectionModelType::UNKNOWN;
    std::string detection_model_name_;
    std::string detection2_model_name_;
    
    // Recognition model information (auto-detected from param file)
    size_t current_encoding_dim_ = FACE_ENCODING_DIM;  // Default to 512D
    std::string current_model_name_;
    
    // Detection configuration
    float detection_confidence_threshold_ = 0.8f;  // Default confidence threshold
    
    // Performance optimizations
    bool use_cache_ = true;
    std::unordered_map<uint64_t, std::vector<Rect>> detection_cache_;
    
    // Face tracking state (to reduce detection frequency)
    std::vector<Rect> tracked_faces_;
    Image prev_gray_frame_;
    int frames_since_detection_ = 0;
    bool tracking_initialized_ = false;
    
    // Motion detection state (for cascade pre-check)
    Image motion_prev_frame_;
    bool motion_initialized_ = false;
    
    // Hash function for frame caching
    uint64_t hashFrame(const ImageView& frame);
    
    // Helper: Detect motion using frame differencing
    bool detectMotion(const ImageView& current_frame, double threshold = 0.02);
    
    // Helper: Track faces using optical flow
    std::vector<Rect> trackFaces(const ImageView& current_frame);
    
    // Helper: Align face for recognition model (expects 112x112)
    Image alignFace(const ImageView& frame, const Rect& face_rect);
    
    // Helper: Find first available recognition model in directory
    std::pair<std::string, size_t> findAvailableModel(const std::string& models_dir);
    
    // Detection decoders for different model types
    std::vector<Rect> detectWithRetinaFace(const ncnn::Mat& in, int img_w, int img_h, float confidence_threshold = 0.0f);
    std::vector<Rect> detectWithYuNet(const ncnn::Mat& in, int img_w, int img_h, float confidence_threshold = 0.0f);
};

} // namespace faceid

#endif // FACEID_FACE_DETECTOR_H
