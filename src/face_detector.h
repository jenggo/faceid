#ifndef FACEID_FACE_DETECTOR_H
#define FACEID_FACE_DETECTOR_H

#include <string>
#include <vector>
#include <unordered_map>
#include <opencv2/opencv.hpp>     // For cv::Mat and other OpenCV types
#include <opencv2/objdetect.hpp>  // For FaceDetectorYN
#include <opencv2/objdetect/face.hpp>  // For FaceRecognizerSF

namespace faceid {

// SFace uses cv::Mat for encodings (128D float vector)
using FaceEncoding = cv::Mat;

class FaceDetector {
public:
    FaceDetector();
    
    // Simplified: Only need recognition model path (no shape predictor needed)
    bool loadModels(const std::string& face_recognition_model_path);
    
    // YuNet returns cv::Rect instead of dlib::rectangle
    std::vector<cv::Rect> detectFaces(const cv::Mat& frame, bool downscale = true);
    
    // Encode faces using SFace
    std::vector<FaceEncoding> encodeFaces(const cv::Mat& frame,
                                          const std::vector<cv::Rect>& face_locations);
    
    // Compare two face encodings (cosine similarity)
    double compareFaces(const FaceEncoding& encoding1, const FaceEncoding& encoding2);
    
    // Performance: Pre-process frame for faster detection
    cv::Mat preprocessFrame(const cv::Mat& frame);
    
    // Enable/disable caching for repeated detections
    void enableCache(bool enable);
    
    // Clear encoding cache
    void clearCache();
    
    // Multi-face detection helpers for "no peek" feature
    // Calculate distance between two face rectangles (center-to-center)
    static double faceDistance(const cv::Rect& face1, const cv::Rect& face2);
    
    // Check if two faces are distinct persons (not same person at different position)
    static bool areDistinctFaces(const cv::Rect& face1, const cv::Rect& face2, int min_distance);
    
    // Get face size as percentage of frame width
    static double getFaceSizePercent(const cv::Rect& face, int frame_width);
    
    // Count distinct faces (filtering out duplicates/reflections)
    static int countDistinctFaces(const std::vector<cv::Rect>& faces, int min_distance);

private:
    // YuNet face detector
    cv::Ptr<cv::FaceDetectorYN> yunet_detector_;
    
    // SFace face recognizer (replaces dlib)
    cv::Ptr<cv::FaceRecognizerSF> sface_recognizer_;
    
    bool models_loaded_ = false;
    
    // Performance optimizations
    bool use_cache_ = true;
    std::unordered_map<uint64_t, std::vector<cv::Rect>> detection_cache_;
    
    // Hash function for frame caching
    uint64_t hashFrame(const cv::Mat& frame);
    
    // Helper: Align face for SFace (expects 112x112)
    cv::Mat alignFace(const cv::Mat& frame, const cv::Rect& face_rect);
};

} // namespace faceid

#endif // FACEID_FACE_DETECTOR_H
