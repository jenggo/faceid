#include "face_detector.h"
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace faceid {

FaceDetector::FaceDetector() {
    detector_ = dlib::get_frontal_face_detector();
}

bool FaceDetector::loadModels(const std::string& shape_predictor_path,
                              const std::string& face_recognition_model_path) {
    try {
        dlib::deserialize(shape_predictor_path) >> shape_predictor_;
        dlib::deserialize(face_recognition_model_path) >> net_;
        models_loaded_ = true;
        return true;
    } catch (...) {
        return false;
    }
}

std::vector<dlib::rectangle> FaceDetector::detectFaces(const cv::Mat& frame, bool downscale) {
    if (!models_loaded_) {
        return {};
    }
    
    // Check cache first
    if (use_cache_) {
        uint64_t frame_hash = hashFrame(frame);
        auto it = detection_cache_.find(frame_hash);
        if (it != detection_cache_.end()) {
            return it->second;
        }
    }
    
    cv::Mat processed_frame = frame;
    double scale = 1.0;
    
    // Downscale for faster detection if enabled
    if (downscale && frame.cols > 640) {
        scale = 640.0 / frame.cols;
        cv::resize(frame, processed_frame, cv::Size(), scale, scale, cv::INTER_LINEAR);
    }
    
    dlib::cv_image<dlib::bgr_pixel> dlib_img(processed_frame);
    auto faces = detector_(dlib_img);
    
    // Scale face locations back to original size
    if (scale != 1.0) {
        for (auto& face : faces) {
            face = dlib::rectangle(
                static_cast<long>(face.left() / scale),
                static_cast<long>(face.top() / scale),
                static_cast<long>(face.right() / scale),
                static_cast<long>(face.bottom() / scale)
            );
        }
    }
    
    // Cache result
    if (use_cache_) {
        uint64_t frame_hash = hashFrame(frame);
        detection_cache_[frame_hash] = faces;
        
        // Limit cache size
        if (detection_cache_.size() > 10) {
            detection_cache_.clear();
        }
    }
    
    return faces;
}

std::vector<FaceEncoding> FaceDetector::encodeFaces(
    const cv::Mat& frame,
    const std::vector<dlib::rectangle>& face_locations) {
    
    if (!models_loaded_ || face_locations.empty()) {
        return {};
    }
    
    dlib::cv_image<dlib::bgr_pixel> dlib_img(frame);
    std::vector<FaceEncoding> encodings;
    
    for (const auto& face_loc : face_locations) {
        auto shape = shape_predictor_(dlib_img, face_loc);
        dlib::matrix<dlib::rgb_pixel> face_chip;
        dlib::extract_image_chip(dlib_img, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);
        encodings.push_back(net_(face_chip));
    }
    
    return encodings;
}

double FaceDetector::compareFaces(const FaceEncoding& encoding1, const FaceEncoding& encoding2) {
    return dlib::length(encoding1 - encoding2);
}

cv::Mat FaceDetector::preprocessFrame(const cv::Mat& frame) {
    cv::Mat processed;
    
    // Convert to RGB if needed
    if (frame.channels() == 4) {
        cv::cvtColor(frame, processed, cv::COLOR_BGRA2BGR);
    } else {
        processed = frame;
    }
    
    // Enhance contrast for better detection
    cv::Mat lab;
    cv::cvtColor(processed, lab, cv::COLOR_BGR2Lab);
    
    std::vector<cv::Mat> lab_planes;
    cv::split(lab, lab_planes);
    
    // Apply CLAHE to L channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(lab_planes[0], lab_planes[0]);
    
    cv::merge(lab_planes, lab);
    cv::cvtColor(lab, processed, cv::COLOR_Lab2BGR);
    
    return processed;
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

uint64_t FaceDetector::hashFrame(const cv::Mat& frame) {
    // Simple hash based on frame dimensions and checksum of subset of pixels
    uint64_t hash = frame.rows * 10000ULL + frame.cols;
    
    // Sample some pixels for hash
    int step = std::max(1, frame.rows / 8);
    for (int i = 0; i < frame.rows; i += step) {
        for (int j = 0; j < frame.cols; j += step) {
            if (frame.channels() == 3) {
                cv::Vec3b pixel = frame.at<cv::Vec3b>(i, j);
                hash = hash * 31 + pixel[0] + pixel[1] + pixel[2];
            }
        }
    }
    
    return hash;
}

} // namespace faceid
