#pragma once

#include <memory>
#include <string>

/**
 * Face Detector Manager - Wrapper for face detection ML models
 * 
 * This is a facade that will be implemented with actual ML models
 * (YuNet for detection, SFace for recognition) when the full implementation
 * is integrated. For now, this is a placeholder.
 * 
 * Responsibilities:
 * - Load face detection and recognition models
 * - Provide face verification interface
 * - Cache models in memory for reuse
 * - Report model availability status
 */
class FaceDetectorManager {
public:
    static FaceDetectorManager& instance();
    
    /**
     * Initialize face detection models
     * Returns true if models loaded successfully
     */
    bool initialize();
    
    /**
     * Check if models are ready
     */
    bool is_ready() const { return models_ready_; }
    
    /**
     * Verify a face against a user's model
     * Returns confidence score (0.0 to 1.0), or -1.0 on error
     */
    float verify_face(const std::string& username);
    
    /**
     * Get status string for logging
     */
    std::string get_status() const;
    
    ~FaceDetectorManager();
    
private:
    FaceDetectorManager() = default;
    
    bool models_ready_ = false;
    // TODO: Add actual model objects when integrating YuNet and SFace
};
