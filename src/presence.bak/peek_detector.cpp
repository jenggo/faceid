#include "peek_detector.h"
#include "../logger.h"
#include <cmath>
#include <algorithm>

namespace faceid {

PeekDetector::PeekDetector(FaceDetector* detector)
    : face_detector_(detector)
    , gaze_yaw_threshold_(30.0f)
    , gaze_pitch_threshold_(30.0f)
    , min_face_size_percent_(0.08)
    , min_face_distance_pixels_(80) {
    
    if (!face_detector_) {
        throw std::invalid_argument("FaceDetector pointer cannot be null");
    }
}

void PeekDetector::setGazeThresholds(float yaw_threshold, float pitch_threshold) {
    gaze_yaw_threshold_ = yaw_threshold;
    gaze_pitch_threshold_ = pitch_threshold;
}

void PeekDetector::setFaceFilters(double min_face_size_percent, int min_face_distance_pixels) {
    min_face_size_percent_ = min_face_size_percent;
    min_face_distance_pixels_ = min_face_distance_pixels;
}

bool PeekDetector::isLookingAtScreen(const FaceDetector::HeadPose& pose) const {
    // Frontal face looking at screen: yaw ≈ 0°, pitch ≈ 0°
    // Allow some tolerance based on configured thresholds
    return (std::abs(pose.yaw) < gaze_yaw_threshold_ && 
            std::abs(pose.pitch) < gaze_pitch_threshold_);
}

PeekAnalysis PeekDetector::analyze(const ImageView& frame, 
                                    const std::vector<Rect>& face_rects) {
    Logger& logger = Logger::getInstance();
    PeekAnalysis result;
    
    if (frame.empty()) {
        return result;
    }
    
    try {
        // Step 1: Detect faces if not provided
        std::vector<Rect> faces;
        if (face_rects.empty()) {
            // Use cascade detection for robustness
            auto cascade_result = face_detector_->detectFacesCascade(frame, false);
            faces = cascade_result.faces;
        } else {
            faces = face_rects;
        }
        
        if (faces.empty()) {
            logger.debug("Peek detection: No faces detected");
            return result;
        }
        
        result.total_faces = static_cast<int>(faces.size());
        
        // Step 2: Filter out faces that are too small (too far away to see screen)
        std::vector<Rect> filtered_faces;
        for (const auto& face : faces) {
            double face_size_percent = static_cast<double>(face.width) / frame.width();
            if (face_size_percent >= min_face_size_percent_) {
                filtered_faces.push_back(face);
            } else {
                logger.debug("Peek detection: Filtered out small face (" + 
                           std::to_string(face_size_percent * 100) + "% of frame)");
            }
        }
        
        if (filtered_faces.empty()) {
            logger.debug("Peek detection: All faces filtered out (too small)");
            return result;
        }
        
        // Step 3: Count distinct faces (avoid counting same person multiple times)
        int distinct_count = FaceDetector::countDistinctFaces(
            filtered_faces, 
            min_face_distance_pixels_
        );
        
        logger.debug("Peek detection: " + std::to_string(distinct_count) + 
                    " distinct face(s) after filtering");
        
        if (distinct_count < 2) {
            // Only one person - no peek
            return result;
        }
        
        // Step 4: Analyze gaze direction for each face
        // Note: FaceDetector::estimateHeadPose() requires facial landmarks
        // For now, we'll use a simplified approach: assume all detected faces are watching
        // TODO: Enhance this by detecting facial landmarks and using actual head pose estimation
        
        // For initial implementation, if we have 2+ distinct faces, consider them watchers
        // This is conservative - better to alert than miss a peek
        result.watchers = distinct_count - 1;  // Subtract 1 for the primary user
        result.passing_by = 0;  // TODO: Implement with actual head pose detection
        
        // Step 5: Determine if we should alert
        result.should_alert = (result.watchers > 0);
        
        if (result.should_alert) {
            logger.warning("PEEK DETECTED: " + std::to_string(result.watchers) + 
                          " additional person(s) detected in frame");
        }
        
        return result;
        
    } catch (const std::exception& e) {
        logger.error("Peek detection error: " + std::string(e.what()));
        return PeekAnalysis();
    }
}

} // namespace faceid
