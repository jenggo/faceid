#ifndef FACEID_FACE_DETECTOR_H
#define FACEID_FACE_DETECTOR_H

#include <string>
#include <vector>
#include <unordered_map>
#include <opencv2/opencv.hpp>     // For cv::Mat and other OpenCV types
#include <ncnn/net.h>             // NCNN for face recognition
#include "libfacedetection/facedetectcnn.h"  // LibFaceDetection

namespace faceid {

// Face encodings are 128D float vectors stored in std::vector for NCNN
using FaceEncoding = std::vector<float>;

class FaceDetector {
public:
    FaceDetector();
    
    // Only need recognition model path (no detection model needed - embedded in LibFaceDetection)
    bool loadModels(const std::string& face_recognition_model_path);
    
    // LibFaceDetection returns cv::Rect directly
    std::vector<cv::Rect> detectFaces(const cv::Mat& frame, bool downscale = false);
    
    // Detect or track faces (automatically uses tracking when possible)
    // track_interval: how many frames to track before re-detecting (0 = always detect)
    std::vector<cv::Rect> detectOrTrackFaces(const cv::Mat& frame, int track_interval = 5);
    
    // Force re-detection on next frame (useful after scene change)
    void resetTracking();
    
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
    // NCNN face recognition network (SFace model)
    ncnn::Net ncnn_net_;
    
    bool models_loaded_ = false;
    
    // Performance optimizations
    bool use_cache_ = true;
    std::unordered_map<uint64_t, std::vector<cv::Rect>> detection_cache_;
    
    // Face tracking state (to reduce detection frequency)
    std::vector<cv::Rect> tracked_faces_;
    cv::Mat prev_gray_frame_;
    int frames_since_detection_ = 0;
    bool tracking_initialized_ = false;
    
    // Hash function for frame caching
    uint64_t hashFrame(const cv::Mat& frame);
    
    // Helper: Track faces using optical flow
    std::vector<cv::Rect> trackFaces(const cv::Mat& current_frame);
    
    // Helper: Align face for SFace (expects 112x112)
    cv::Mat alignFace(const cv::Mat& frame, const cv::Rect& face_rect);
};

} // namespace faceid

#endif // FACEID_FACE_DETECTOR_H
