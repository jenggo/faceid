#pragma once

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include "../face_detector.h"
#include "../image.h"

/**
 * Face Detector Manager - Wrapper for face detection ML models
 * 
 * Integrates YuNet (face detection) and SFace (face recognition) models.
 * Thread-safe for concurrent authentication requests.
 * 
 * Responsibilities:
 * - Load face detection and recognition models (YuNet + SFace)
 * - Provide face detection interface (detect bounding boxes)
 * - Provide face recognition interface (extract features, compare)
 * - Cache models in memory for reuse
 * - Thread safety for multi-user parallel authentication
 */
class FaceDetectorManager {
public:
    static FaceDetectorManager& instance();
    
    /**
     * Initialize face detection models (YuNet + SFace)
     * Returns true if models loaded successfully
     */
    bool initialize();
    
    /**
     * Check if models are ready
     */
    bool is_ready() const { return models_ready_; }
    
    /**
     * Detect faces in a frame
     * Returns bounding boxes for detected faces
     */
    std::vector<faceid::Rect> detect_faces(const faceid::Image& frame);
    
    /**
     * Verify a face against stored user models
     * Returns confidence score (0.0 to 1.0), or -1.0 on error
     */
    float verify_face(const faceid::Image& frame,
                      const std::vector<faceid::Rect>& faces,
                      const std::vector<faceid::FaceEncoding>& stored_models);
    
    /**
     * Extract face encoding from a single detected face (for enrollment)
     * frame: input image
     * face_box: bounding box of the face to encode
     * Returns: FaceEncoding (std::vector<float>) with 512 dimensions, or empty vector on error
     */
    faceid::FaceEncoding encode_face(const faceid::Image& frame,
                                     const faceid::Rect& face_box);
    
    /**
     * Get status string for logging
     */
    std::string get_status() const;
    
    ~FaceDetectorManager();
    
private:
    FaceDetectorManager();
    
    bool models_ready_ = false;
    std::unique_ptr<faceid::FaceDetector> detector_;
    std::mutex detector_mutex_;  // Thread safety for NCNN models
};
