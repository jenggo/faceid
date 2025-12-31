#ifndef FACEID_DETECTORS_COMMON_H
#define FACEID_DETECTORS_COMMON_H

#include "../image.h"
#include <ncnn/net.h>
#include <vector>
#include <algorithm>
#include <cmath>

namespace faceid {

// Shared data structure for face detection results
struct FaceObject {
    Rect rect;
    float prob;
};

// Calculate intersection area between two face objects
inline float intersection_area(const FaceObject& a, const FaceObject& b) {
    int x1 = std::max(a.rect.x, b.rect.x);
    int y1 = std::max(a.rect.y, b.rect.y);
    int x2 = std::min(a.rect.x + a.rect.width, b.rect.x + b.rect.width);
    int y2 = std::min(a.rect.y + a.rect.height, b.rect.y + b.rect.height);
    
    if (x2 < x1 || y2 < y1) return 0.0f;
    return (x2 - x1) * (y2 - y1);
}

// Quicksort helper for descending probability sort
inline void qsort_descent_inplace(std::vector<FaceObject>& faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;
    
    while (i <= j) {
        while (faceobjects[i].prob > p) i++;
        while (faceobjects[j].prob < p) j--;
        
        if (i <= j) {
            std::swap(faceobjects[i], faceobjects[j]);
            i++;
            j--;
        }
    }
    
    if (left < j) qsort_descent_inplace(faceobjects, left, j);
    if (i < right) qsort_descent_inplace(faceobjects, i, right);
}

// Quicksort entry point (main vector)
inline void qsort_descent_inplace(std::vector<FaceObject>& faceobjects) {
    if (faceobjects.empty()) return;
    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

// Non-Maximum Suppression for sorted bboxes
inline void nms_sorted_bboxes(const std::vector<FaceObject>& faceobjects, std::vector<int>& picked, float nms_threshold) {
    picked.clear();
    const int n = faceobjects.size();
    
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = faceobjects[i].rect.width * faceobjects[i].rect.height;
    }
    
    for (int i = 0; i < n; i++) {
        const FaceObject& a = faceobjects[i];
        
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const FaceObject& b = faceobjects[picked[j]];
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }
        
        if (keep) picked.push_back(i);
    }
}

// Generate anchor boxes for RetinaFace-style detection
inline ncnn::Mat generate_anchors(int base_size, const ncnn::Mat& ratios, const ncnn::Mat& scales) {
    int num_ratio = ratios.w;
    int num_scale = scales.w;
    
    ncnn::Mat anchors;
    anchors.create(4, num_ratio * num_scale);
    
    const float cx = base_size * 0.5f;
    const float cy = base_size * 0.5f;
    
    for (int i = 0; i < num_ratio; i++) {
        float ar = ratios[i];
        int r_w = round(base_size / sqrt(ar));
        int r_h = round(r_w * ar);
        
        for (int j = 0; j < num_scale; j++) {
            float scale = scales[j];
            float rs_w = r_w * scale;
            float rs_h = r_h * scale;
            
            float* anchor = anchors.row(i * num_scale + j);
            anchor[0] = cx - rs_w * 0.5f;
            anchor[1] = cy - rs_h * 0.5f;
            anchor[2] = cx + rs_w * 0.5f;
            anchor[3] = cy + rs_h * 0.5f;
        }
    }
    
    return anchors;
}

// Generate detection proposals from anchors and model outputs (RetinaFace-style)
inline void generate_proposals(const ncnn::Mat& anchors, int feat_stride, const ncnn::Mat& score_blob, 
                               const ncnn::Mat& bbox_blob, float prob_threshold, std::vector<FaceObject>& faceobjects,
                               const ncnn::Mat& landmark_blob = ncnn::Mat()) {
    int w = score_blob.w;
    int h = score_blob.h;
    const int num_anchors = anchors.h;
    const bool has_landmarks = !landmark_blob.empty();
    
    for (int q = 0; q < num_anchors; q++) {
        const float* anchor = anchors.row(q);
        const ncnn::Mat score = score_blob.channel(q + num_anchors);
        const ncnn::Mat bbox = bbox_blob.channel_range(q * 4, 4);
        
        // Landmark blob: 10 channels per anchor (5 points × 2 coords)
        ncnn::Mat landmark;
        if (has_landmarks) {
            landmark = landmark_blob.channel_range(q * 10, 10);
        }
        
        float anchor_y = anchor[1];
        float anchor_w = anchor[2] - anchor[0];
        float anchor_h = anchor[3] - anchor[1];
        
        for (int i = 0; i < h; i++) {
            float anchor_x = anchor[0];
            
            for (int j = 0; j < w; j++) {
                int index = i * w + j;
                float prob = score[index];
                
                if (prob >= prob_threshold) {
                    float dx = bbox.channel(0)[index];
                    float dy = bbox.channel(1)[index];
                    float dw = bbox.channel(2)[index];
                    float dh = bbox.channel(3)[index];
                    
                    float cx = anchor_x + anchor_w * 0.5f;
                    float cy = anchor_y + anchor_h * 0.5f;
                    
                    float pb_cx = cx + anchor_w * dx;
                    float pb_cy = cy + anchor_h * dy;
                    float pb_w = anchor_w * exp(dw);
                    float pb_h = anchor_h * exp(dh);
                    
                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;
                    
                    FaceObject obj;
                    obj.rect.x = (int)x0;
                    obj.rect.y = (int)y0;
                    obj.rect.width = (int)(x1 - x0 + 1);
                    obj.rect.height = (int)(y1 - y0 + 1);
                    obj.prob = prob;
                    
                    // Extract landmarks if available (5 points)
                    if (has_landmarks) {
                        for (int k = 0; k < 5; k++) {
                            float lm_x = landmark.channel(k * 2)[index];
                            float lm_y = landmark.channel(k * 2 + 1)[index];
                            
                            Point pt;
                            pt.x = cx + anchor_w * lm_x;
                            pt.y = cy + anchor_h * lm_y;
                            obj.rect.landmarks.push_back(pt);
                        }
                    }
                    
                    faceobjects.push_back(obj);
                }
                
                anchor_x += feat_stride;
            }
            
            anchor_y += feat_stride;
        }
    }
}

} // namespace faceid

#endif // FACEID_DETECTORS_COMMON_H
