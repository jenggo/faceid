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
// Output: Vector of face rectangles with 5-point facial landmarks
// confidence_threshold: minimum confidence score (0.0-1.0, default 0.8)
std::vector<Rect> detectWithYuNet(ncnn::Net& net, const ncnn::Mat& in, int img_w, int img_h,
                                  float confidence_threshold = 0.8f);

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
