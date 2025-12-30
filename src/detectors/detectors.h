#ifndef FACEID_DETECTORS_H
#define FACEID_DETECTORS_H

#include "../image.h"
#include <ncnn/net.h>
#include <vector>

namespace faceid {

// RetinaFace detector
// Model: RetinaFace (mnet.25-opt)
// Input: RGB image (converted from BGR), variable size
// Output: Vector of face rectangles
// confidence_threshold: minimum confidence score (0.0-1.0, default 0.8)
std::vector<Rect> detectWithRetinaFace(ncnn::Net& net, const ncnn::Mat& in, int img_w, int img_h, 
                                       float confidence_threshold = 0.8f);

// YuNet detector  
// Model: YuNet (libfacedetection)
// Input: RGB image (converted from BGR), variable size
// Output: Vector of face rectangles (typically single largest face)
// confidence_threshold: minimum confidence score (0.0-1.0, default 0.8)
std::vector<Rect> detectWithYuNet(ncnn::Net& net, const ncnn::Mat& in, int img_w, int img_h,
                                  float confidence_threshold = 0.8f);

// UltraFace/RFB-320 detector
// Model: UltraFace/RFB-320
// Input: RGB image (converted from BGR), variable size (resized internally)
// Output: Vector of face rectangles (typically single largest face)
// confidence_threshold: minimum confidence score (0.0-1.0, default 0.5)
std::vector<Rect> detectWithUltraFace(ncnn::Net& net, const ncnn::Mat& in, int img_w, int img_h,
                                      float confidence_threshold = 0.5f);

// SCRFD detector
// Model: SCRFD (Scaled-RoI Face Detector)
// Input: RGB image (converted from BGR), preprocessed with aspect-ratio preserving resize + padding
// Output: Vector of face rectangles (coordinates adjusted back to original image space)
// confidence_threshold: minimum confidence score (0.0-1.0, default 0.5)
// scale: resize scale factor used in preprocessing
// wpad/hpad: padding added in preprocessing
// orig_w/orig_h: original image dimensions before preprocessing
std::vector<Rect> detectWithSCRFD(ncnn::Net& net, const ncnn::Mat& in, int img_w, int img_h,
                                  float confidence_threshold, float scale, int wpad, int hpad,
                                  int orig_w, int orig_h);

// YOLOv5-Face detector
// Model: YOLOv5-Face (yolov5n)
// Input: RGB image (converted from BGR), 640x640 with letterbox padding
// Output: Vector of face rectangles with facial keypoints
// confidence_threshold: minimum confidence score (0.0-1.0, default 0.5)
std::vector<Rect> detectWithYOLOv5(ncnn::Net& net, const ncnn::Mat& in, int img_w, int img_h,
                                   float confidence_threshold = 0.5f);

// YOLOv7-Face detector
// Model: YOLOv7-Face (yolov7-tiny)
// Input: RGB image (converted from BGR), 640x640 with letterbox padding
// Output: Vector of face rectangles with facial keypoints
// confidence_threshold: minimum confidence score (0.0-1.0, default 0.65)
std::vector<Rect> detectWithYOLOv7(ncnn::Net& net, const ncnn::Mat& in, int img_w, int img_h,
                                   float confidence_threshold = 0.65f);

// YOLOv8-Face detector
// Model: YOLOv8-Face (yolov8-lite-s)
// Input: RGB image (converted from BGR), 640x640 with letterbox padding
// Output: Vector of face rectangles with facial keypoints
// confidence_threshold: minimum confidence score (0.0-1.0, default 0.5)
std::vector<Rect> detectWithYOLOv8(ncnn::Net& net, const ncnn::Mat& in, int img_w, int img_h,
                                   float confidence_threshold = 0.5f);

} // namespace faceid

#endif // FACEID_DETECTORS_H
