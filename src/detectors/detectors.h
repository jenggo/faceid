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
std::vector<Rect> detectWithRetinaFace(ncnn::Net& net, const ncnn::Mat& in, int img_w, int img_h);

// YuNet detector  
// Model: YuNet (libfacedetection)
// Input: RGB image (converted from BGR), variable size
// Output: Vector of face rectangles (typically single largest face)
std::vector<Rect> detectWithYuNet(ncnn::Net& net, const ncnn::Mat& in, int img_w, int img_h);

// UltraFace/RFB-320 detector
// Model: UltraFace/RFB-320
// Input: RGB image (converted from BGR), variable size (resized internally)
// Output: Vector of face rectangles (typically single largest face)
std::vector<Rect> detectWithUltraFace(ncnn::Net& net, const ncnn::Mat& in, int img_w, int img_h);

// SCRFD detector
// Model: SCRFD (Scaled-RoI Face Detector)
// Input: RGB image (converted from BGR), variable size
// Output: Vector of face rectangles
std::vector<Rect> detectWithSCRFD(ncnn::Net& net, const ncnn::Mat& in, int img_w, int img_h);

} // namespace faceid

#endif // FACEID_DETECTORS_H
