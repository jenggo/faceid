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
    ULTRAFACE,    // UltraFace/RFB-320: input="in0", 2 outputs (out0, out1)
    SCRFD,        // SCRFD: input="input.1", 6-9 outputs (score_*/bbox_*/kps_*)
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
    std::vector<Rect> detectFaces(const ImageView& frame, bool downscale = false);
    
    // Detect or track faces (automatically uses tracking when possible)
    // track_interval: how many frames to track before re-detecting (0 = always detect)
    std::vector<Rect> detectOrTrackFaces(const ImageView& frame, int track_interval = 5);
    
    // Force re-detection on next frame (useful after scene change)
    void resetTracking();
    
    // Encode faces using SFace
    std::vector<FaceEncoding> encodeFaces(const ImageView& frame,
                                          const std::vector<Rect>& face_locations);
    
    // Compare two face encodings (cosine similarity)
    double compareFaces(const FaceEncoding& encoding1, const FaceEncoding& encoding2);
    
    // Performance: Pre-process frame for faster detection (CLAHE enhancement)
    Image preprocessFrame(const ImageView& frame);
    
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
    
    // Get current model name
    const std::string& getModelName() const { return current_model_name_; }

private:
    // NCNN networks
    ncnn::Net ncnn_net_;          // Face recognition model (auto-detected)
    ncnn::Net retinaface_net_;    // Face detection model (auto-detected type)
    
    bool models_loaded_ = false;
    bool detection_model_loaded_ = false;
    
    // Detection model information (auto-detected from param file)
    DetectionModelType detection_model_type_ = DetectionModelType::UNKNOWN;
    std::string detection_model_name_;
    
    // Recognition model information (auto-detected from param file)
    size_t current_encoding_dim_ = FACE_ENCODING_DIM;  // Default to 512D
    std::string current_model_name_;
    
    // Performance optimizations
    bool use_cache_ = true;
    std::unordered_map<uint64_t, std::vector<Rect>> detection_cache_;
    
    // Face tracking state (to reduce detection frequency)
    std::vector<Rect> tracked_faces_;
    Image prev_gray_frame_;
    int frames_since_detection_ = 0;
    bool tracking_initialized_ = false;
    
    // Hash function for frame caching
    uint64_t hashFrame(const ImageView& frame);
    
    // Helper: Track faces using optical flow
    std::vector<Rect> trackFaces(const ImageView& current_frame);
    
    // Helper: Align face for recognition model (expects 112x112)
    Image alignFace(const ImageView& frame, const Rect& face_rect);
    
    // Helper: Parse param file to extract output dimensions
    size_t parseModelOutputDim(const std::string& param_path);
    
    // Helper: Find first available recognition model in directory
    std::pair<std::string, size_t> findAvailableModel(const std::string& models_dir);
    
    // Helper: Auto-detect detection model type from param file
    DetectionModelType detectModelType(const std::string& param_path);
    
    // Detection decoders for different model types
    std::vector<Rect> detectWithRetinaFace(const ncnn::Mat& in, int img_w, int img_h);
    std::vector<Rect> detectWithYuNet(const ncnn::Mat& in, int img_w, int img_h);
    std::vector<Rect> detectWithUltraFace(const ncnn::Mat& in, int img_w, int img_h);
    std::vector<Rect> detectWithSCRFD(const ncnn::Mat& in, int img_w, int img_h);
};

} // namespace faceid

#endif // FACEID_FACE_DETECTOR_H
