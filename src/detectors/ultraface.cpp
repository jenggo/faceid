// UltraFace/RFB-320 Face Detector Implementation
// Model: UltraFace/RFB-320 (Ultra-Light-Fast-Generic-Face-Detector-1MB)
// Input: RGB image (converted from BGR), variable size (resized to 320x240 internally), "in0" layer
// Output: Bounding boxes with confidence scores
//         - out0: Classification scores (2, 4420) - [background, face] probabilities
//         - out1: Bounding box offsets (4, 4420) - [cx, cy, w, h] offsets
// Reference: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB

#include "common.h"
#include "../logger.h"
#include <ncnn/net.h>
#include <vector>
#include <cmath>

namespace faceid {

std::vector<Rect> detectWithUltraFace(ncnn::Net& net, const ncnn::Mat& in, int img_w, int img_h,
                                      float confidence_threshold) {
    // UltraFace/RFB-320: input 320x240, outputs raw scores and bbox offsets
    
    // Resize input to 320x240 (model's expected input size)
    const int model_w = 320;
    const int model_h = 240;
    ncnn::Mat in_resized;
    ncnn::resize_bilinear(in, in_resized, model_w, model_h);
    
    // Normalize: mean=[127,127,127], norm=[1/128,1/128,1/128]
    const float mean_vals[3] = {127.f, 127.f, 127.f};
    const float norm_vals[3] = {1.f/128.f, 1.f/128.f, 1.f/128.f};
    in_resized.substract_mean_normalize(mean_vals, norm_vals);
    
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    ex.input("in0", in_resized);
    
    ncnn::Mat scores, boxes;
    ex.extract("out0", scores);  // (2, 4420)
    ex.extract("out1", boxes);   // (4, 4420)
    
    const float prob_threshold = confidence_threshold;
    const float nms_threshold = 0.3f;
    std::vector<FaceObject> proposals;
    
    // UltraFace constants (from reference implementation)
    const float center_variance = 0.1f;
    const float size_variance = 0.2f;
    const std::vector<std::vector<float>> min_boxes = {
        {10.0f, 16.0f, 24.0f},
        {32.0f, 48.0f},
        {64.0f, 96.0f},
        {128.0f, 192.0f, 256.0f}
    };
    const std::vector<float> strides = {8.0f, 16.0f, 32.0f, 64.0f};
    const int in_w = 320;
    const int in_h = 240;
    
    // Generate priors/anchors
    std::vector<std::vector<float>> priors;
    for (int scale_idx = 0; scale_idx < 4; scale_idx++) {
        float stride = strides[scale_idx];
        int feat_w = (int)std::ceil(in_w / stride);
        int feat_h = (int)std::ceil(in_h / stride);
        
        for (int j = 0; j < feat_h; j++) {
            for (int i = 0; i < feat_w; i++) {
                float x_center = (i + 0.5f) / (in_w / stride);
                float y_center = (j + 0.5f) / (in_h / stride);
                
                for (float min_box : min_boxes[scale_idx]) {
                    float w = min_box / in_w;
                    float h = min_box / in_h;
                    // Clip to [0, 1]
                    priors.push_back({
                        std::min(std::max(x_center, 0.0f), 1.0f),
                        std::min(std::max(y_center, 0.0f), 1.0f),
                        std::min(std::max(w, 0.0f), 1.0f),
                        std::min(std::max(h, 0.0f), 1.0f)
                    });
                }
            }
        }
    }
    
    int num_anchors = priors.size();
    const float* scores_ptr = (const float*)scores.data;
    const float* boxes_ptr = (const float*)boxes.data;
    
    // Decode bboxes
    for (int i = 0; i < num_anchors; i++) {
        // Scores: interleaved [bg_0, face_0, bg_1, face_1, ...]
        float face_score = scores_ptr[i * 2 + 1];
        if (face_score < prob_threshold) continue;
        
        // Boxes: interleaved [cx_0, cy_0, w_0, h_0, cx_1, cy_1, w_1, h_1, ...]
        float cx_offset = boxes_ptr[i * 4 + 0];
        float cy_offset = boxes_ptr[i * 4 + 1];
        float w_log = boxes_ptr[i * 4 + 2];
        float h_log = boxes_ptr[i * 4 + 3];
        
        // Decode using priors (normalized coordinates [0, 1])
        float cx = cx_offset * center_variance * priors[i][2] + priors[i][0];
        float cy = cy_offset * center_variance * priors[i][3] + priors[i][1];
        float w = std::exp(w_log * size_variance) * priors[i][2];
        float h = std::exp(h_log * size_variance) * priors[i][3];
        
        // Convert to corners and clip to [0, 1]
        float x1 = std::min(std::max(cx - w / 2.0f, 0.0f), 1.0f);
        float y1 = std::min(std::max(cy - h / 2.0f, 0.0f), 1.0f);
        float x2 = std::min(std::max(cx + w / 2.0f, 0.0f), 1.0f);
        float y2 = std::min(std::max(cy + h / 2.0f, 0.0f), 1.0f);
        
        // Scale to original image size
        float box_x1 = x1 * img_w;
        float box_y1 = y1 * img_h;
        float box_x2 = x2 * img_w;
        float box_y2 = y2 * img_h;
        float box_w = box_x2 - box_x1;
        float box_h = box_y2 - box_y1;
        
        // Filter by minimum size
        if (box_w > 0 && box_h > 0) {
            FaceObject faceobj;
            faceobj.rect.x = box_x1;
            faceobj.rect.y = box_y1;
            faceobj.rect.width = box_w;
            faceobj.rect.height = box_h;
            faceobj.prob = face_score;
            
            proposals.push_back(faceobj);
        }
    }
    
    // Apply NMS
    qsort_descent_inplace(proposals);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);
    
    // Convert to Rect - keep only largest box if multiple detections
    std::vector<Rect> faces;
    FaceObject* best = nullptr;
    float max_area = 0;
    
    for (int idx : picked) {
        FaceObject& faceobj = proposals[idx];
        float area = faceobj.rect.width * faceobj.rect.height;
        if (area > max_area) {
            max_area = area;
            best = &faceobj;
        }
    }
    
    if (best) {
        Rect r;
        r.x = (int)best->rect.x;
        r.y = (int)best->rect.y;
        r.width = (int)best->rect.width;
        r.height = (int)best->rect.height;
        faces.push_back(r);
    }
    
    return faces;
}

} // namespace faceid
