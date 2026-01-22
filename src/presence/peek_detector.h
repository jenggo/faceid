#ifndef FACEID_PRESENCE_PEEK_DETECTOR_H
#define FACEID_PRESENCE_PEEK_DETECTOR_H

#include "../face_detector.h"
#include "../image.h"
#include <vector>

namespace faceid {

/**
 * Peek Analysis Result
 * 
 * Contains detailed information about detected faces and their gaze directions
 */
struct PeekAnalysis {
    int total_faces;           // Total number of faces detected
    int authorized_faces;      // Number of faces that are authorized (recognized)
    int watchers;              // Number of people looking at screen
    int passing_by;            // Number of people not looking at screen
    bool should_alert;         // True if security alert should be raised
    
    PeekAnalysis()
        : total_faces(0)
        , authorized_faces(0)
        , watchers(0)
        , passing_by(0)
        , should_alert(false) {}
};

/**
 * Peek Detector with Gaze Analysis
 * 
 * Detects shoulder surfing by:
 * 1. Detecting multiple faces in frame
 * 2. Estimating head pose / gaze direction for each face
 * 3. Classifying faces as "watching screen" or "passing by"
 * 4. Raising alerts only for people actively looking at screen
 * 
 * Uses existing FaceDetector head pose estimation (yaw, pitch, roll)
 */
class PeekDetector {
public:
    /**
     * Constructor
     * 
     * @param detector Pointer to existing FaceDetector instance
     */
    explicit PeekDetector(FaceDetector* detector);
    
    /**
     * Analyze frame for peek detection with gaze awareness
     * 
     * @param frame Input frame (BGR format)
     * @param face_rects Pre-detected face rectangles (optional, if already detected)
     * @return Analysis result with watcher count and alert status
     */
    PeekAnalysis analyze(const ImageView& frame, 
                         const std::vector<Rect>& face_rects = std::vector<Rect>());
    
    /**
     * Check if a person is looking at the screen based on head pose
     * 
     * @param pose Head pose estimation (yaw, pitch, roll in degrees)
     * @return true if person is looking at screen
     */
    bool isLookingAtScreen(const FaceDetector::HeadPose& pose) const;
    
    /**
     * Configure gaze detection thresholds
     * 
     * @param yaw_threshold Maximum yaw angle (side-to-side) in degrees
     * @param pitch_threshold Maximum pitch angle (up-down) in degrees
     */
    void setGazeThresholds(float yaw_threshold, float pitch_threshold);
    
    /**
     * Configure minimum face size and distance filters
     * 
     * @param min_face_size_percent Minimum face size as percentage of frame width
     * @param min_face_distance_pixels Minimum distance between faces to be considered distinct
     */
    void setFaceFilters(double min_face_size_percent, int min_face_distance_pixels);
    
private:
    FaceDetector* face_detector_;
    
    // Gaze detection thresholds (degrees)
    float gaze_yaw_threshold_;      // Default: 30° (side-to-side)
    float gaze_pitch_threshold_;    // Default: 30° (up-down)
    
    // Face filtering thresholds
    double min_face_size_percent_;  // Default: 0.08 (8% of frame width)
    int min_face_distance_pixels_;  // Default: 80 pixels
};

} // namespace faceid

#endif // FACEID_PRESENCE_PEEK_DETECTOR_H
