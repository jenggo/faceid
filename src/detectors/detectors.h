#ifndef FACEID_DETECTORS_H
#define FACEID_DETECTORS_H

#include "../image.h"
#include <ncnn/net.h>
#include <vector>

namespace faceid {

// YuNet detector (primary - embedded in binary)
// Model: YuNet 2023mar dynamic (libfacedetection)
// Input: RGB image (converted from BGR), variable size
// Output: Vector of face rectangles with 5-point facial landmarks
// confidence_threshold: minimum confidence score (0.0-1.0, default 0.5)
std::vector<Rect> detectWithYuNet(ncnn::Net& net, const ncnn::Mat& in, int img_w, int img_h,
                                  float confidence_threshold = 0.5f);

// RetinaFace detector (fallback - embedded in binary)
// Model: RetinaFace mnet (MobileNet backbone)
// Input: RGB image (converted from BGR), variable size
// Output: Vector of face rectangles with landmarks
// confidence_threshold: minimum confidence score (0.0-1.0, default 0.8)
std::vector<Rect> detectWithRetinaFace(ncnn::Net& net, const ncnn::Mat& in, int img_w, int img_h, 
                                       float confidence_threshold = 0.8f);

} // namespace faceid

#endif // FACEID_DETECTORS_H
