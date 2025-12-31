// YuNet Face Detector Implementation
// Model: YuNet (libfacedetection style)
// Input: RGB image (converted from BGR), variable size, "in0" layer
// Output: Multiple detections across 3 scales with keypoints
//         - out0-out2: Classification scores (3 scales)
//         - out3-out5: Object scores (3 scales)
//         - out6-out8: Bounding boxes (3 scales)
//         - out9-out11: Keypoints (3 scales)
// Reference: https://github.com/ShiqiYu/libfacedetection

#include "common.h"
#include "../logger.h"
#include <ncnn/net.h>
#include <vector>
#include <cmath>

namespace faceid {

std::vector<Rect> detectWithYuNet(ncnn::Net& net, const ncnn::Mat& in, int img_w, int img_h,
                                  float confidence_threshold) {
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    ex.input("in0", in);
    
    const float conf_threshold = confidence_threshold;
    const float nms_threshold = 0.3f;
    std::vector<FaceObject> proposals;
    
    // Extract all 12 outputs (3 scales × 4 types)
    // kpss = keypoints (5 points: 2 eyes, nose, 2 mouth corners)
    std::vector<ncnn::Mat> cls_scores(3), obj_scores(3), bboxes(3), kpss(3);
    for (int i = 0; i < 3; i++) {
        ex.extract(("out" + std::to_string(i)).c_str(), cls_scores[i]);
        ex.extract(("out" + std::to_string(i + 3)).c_str(), obj_scores[i]);
        ex.extract(("out" + std::to_string(i + 6)).c_str(), bboxes[i]);
        ex.extract(("out" + std::to_string(i + 9)).c_str(), kpss[i]);
    }
    
    // Process 3 scales
    int strides[] = {8, 16, 32};
    for (int scale_idx = 0; scale_idx < 3; scale_idx++) {
        int stride = strides[scale_idx];
        const ncnn::Mat& cls = cls_scores[scale_idx];
        const ncnn::Mat& obj = obj_scores[scale_idx];
        const ncnn::Mat& bbox = bboxes[scale_idx];
        const ncnn::Mat& kps = kpss[scale_idx];
        
        // YuNet outputs are flattened - reshape to proper grid
        int total_elements = cls.w * cls.h;
        int feat_w = (int)std::sqrt(total_elements);
        int feat_h = feat_w;
        
        for (int i = 0; i < feat_h; i++) {
            for (int j = 0; j < feat_w; j++) {
                int idx = i * feat_w + j;
                
                // Score = cls * obj (both already sigmoid activated)
                float cls_score = cls.channel(0)[idx];
                float obj_score = obj.channel(0)[idx];
                float score = cls_score * obj_score;
                
                if (score < conf_threshold) continue;
                
                // Decode bbox - YuNet uses center+size format (libfacedetection compatible)
                // NCNN bbox shape (4, H*W) is stored row-major: [cx, cy, w, h] interleaved
                const float* bbox_ptr = (const float*)bbox.data;
                
                // Row-major access: each position has 4 consecutive values
                float cx_offset = bbox_ptr[idx * 4 + 0];
                float cy_offset = bbox_ptr[idx * 4 + 1];
                float w_log = bbox_ptr[idx * 4 + 2];
                float h_log = bbox_ptr[idx * 4 + 3];
                
                float anchor_x = (j + 0.5f) * stride;
                float anchor_y = (i + 0.5f) * stride;
                
                // Decode center and size (following libfacedetection)
                float cx = cx_offset * stride + anchor_x;
                float cy = cy_offset * stride + anchor_y;
                float w = std::exp(w_log) * stride;
                float h = std::exp(h_log) * stride;
                
                // Convert to corners
                float x1 = cx - w / 2.0f;
                float y1 = cy - h / 2.0f;
                float x2 = cx + w / 2.0f;
                float y2 = cy + h / 2.0f;
                
                // Clip and validate
                x1 = std::max(0.0f, std::min((float)img_w, x1));
                y1 = std::max(0.0f, std::min((float)img_h, y1));
                x2 = std::max(0.0f, std::min((float)img_w, x2));
                y2 = std::max(0.0f, std::min((float)img_h, y2));
                
                float box_w = x2 - x1;
                float box_h = y2 - y1;
                
                // Filter by minimum size
                if (box_w > 0 && box_h > 0) {
                    FaceObject faceobj;
                    faceobj.rect.x = x1;
                    faceobj.rect.y = y1;
                    faceobj.rect.width = box_w;
                    faceobj.rect.height = box_h;
                    faceobj.prob = score;
                    
                    // Decode 5-point landmarks (2 eyes, nose, 2 mouth corners)
                    // YuNet outputs landmarks in normalized format relative to the feature map
                    // Data layout: (w=10, h=grid_size, c=1) - row-major with 10 values per cell
                    const float* kps_ptr = (const float*)kps.data;
                    for (int k = 0; k < 5; k++) {
                        float kps_x_raw = kps_ptr[idx * 10 + k * 2];
                        float kps_y_raw = kps_ptr[idx * 10 + k * 2 + 1];
                        
                        // Use grid cell + offset interpretation
                        float kps_x = (j + kps_x_raw) * stride;
                        float kps_y = (i + kps_y_raw) * stride;
                        
                        // Clip to image bounds
                        kps_x = std::max(0.0f, std::min((float)img_w, kps_x));
                        kps_y = std::max(0.0f, std::min((float)img_h, kps_y));
                        
                        faceobj.rect.landmarks.push_back(Point(kps_x, kps_y));
                    }
                    
                    proposals.push_back(faceobj);
                }
            }
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
        r.landmarks = best->rect.landmarks;  // Copy landmarks
        faces.push_back(r);
    }
    
    return faces;
}

} // namespace faceid
