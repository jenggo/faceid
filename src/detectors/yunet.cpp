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
    const float nms_threshold = 0.5f;  // Increased from 0.3 to 0.5 for more aggressive merging
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
        
        // DEBUG: Log tensor shapes
        Logger::getInstance().debug("YuNet scale " + std::to_string(scale_idx) + " (stride=" + std::to_string(stride) + "):");
        Logger::getInstance().debug("  cls: dims=" + std::to_string(cls.dims) + " w=" + std::to_string(cls.w) + " h=" + std::to_string(cls.h) + " c=" + std::to_string(cls.c));
        Logger::getInstance().debug("  obj: dims=" + std::to_string(obj.dims) + " w=" + std::to_string(obj.w) + " h=" + std::to_string(obj.h) + " c=" + std::to_string(obj.c));
        Logger::getInstance().debug("  bbox: dims=" + std::to_string(bbox.dims) + " w=" + std::to_string(bbox.w) + " h=" + std::to_string(bbox.h) + " c=" + std::to_string(bbox.c));
        Logger::getInstance().debug("  kps: dims=" + std::to_string(kps.dims) + " w=" + std::to_string(kps.w) + " h=" + std::to_string(kps.h) + " c=" + std::to_string(kps.c));
        
        // Calculate feature map grid dimensions from input size and stride
        // This correctly handles both square (320x320, 640x640) and non-square (640x360) inputs
        // For dynamic models, the grid adapts to input size
        int feat_w = in.w / stride;  // e.g., 640/8 = 80
        int feat_h = in.h / stride;  // e.g., 360/8 = 45
        
        Logger::getInstance().debug("  feat_grid: " + std::to_string(feat_w) + "x" + std::to_string(feat_h) + 
                                   " = " + std::to_string(feat_w * feat_h) + " cells");
        
        for (int i = 0; i < feat_h; i++) {
            for (int j = 0; j < feat_w; j++) {
                int idx = i * feat_w + j;
                
                // Score = cls * obj (both already sigmoid activated)
                // Access pattern depends on tensor layout
                float cls_score, obj_score;
                
                if (cls.dims == 1) {
                    // 1D tensor: direct index
                    cls_score = cls[idx];
                    obj_score = obj[idx];
                } else if (cls.dims == 2) {
                    if (cls.w == 1) {
                        // Shape (1, H*W): stored as H*W rows with 1 column each
                        // Access: cls.row(idx)[0] - row idx, column 0
                        cls_score = cls.row(idx)[0];
                        obj_score = obj.row(idx)[0];
                    } else if (cls.h == 1) {
                        // Shape (W*H, 1): w-major, access as cls[idx]
                        cls_score = ((const float*)cls.data)[idx];
                        obj_score = ((const float*)obj.data)[idx];
                    } else {
                        // Shape (W, H): use row-col access
                        cls_score = cls.row(i)[j];
                        obj_score = obj.row(i)[j];
                    }
                } else {
                    // 3D: use row-col access
                    cls_score = cls.row(i)[j];
                    obj_score = obj.row(i)[j];
                }
                
                float score = cls_score * obj_score;
                
                if (score < conf_threshold) continue;
                
                // Decode bbox - YuNet uses center+size format
                // Bbox shapes: (4, H*W) or (1, 4*H*W) or (4*H*W, 1) or (H, W, 4)
                float cx_offset, cy_offset, w_log, h_log;
                
                if (bbox.dims == 1) {
                    // 1D: (4*H*W) flattened - [cx, cy, w, h] interleaved
                    const float* bbox_ptr = (const float*)bbox.data;
                    cx_offset = bbox_ptr[idx * 4 + 0];
                    cy_offset = bbox_ptr[idx * 4 + 1];
                    w_log = bbox_ptr[idx * 4 + 2];
                    h_log = bbox_ptr[idx * 4 + 3];
                } else if (bbox.dims == 2) {
                    if (bbox.w == 4) {
                        // Shape (4, H*W): stored as H*W rows with 4 columns each
                        // NCNN uses row-major: bbox.row(idx) gives row idx, then [0..3] for columns
                        cx_offset = bbox.row(idx)[0];
                        cy_offset = bbox.row(idx)[1];
                        w_log = bbox.row(idx)[2];
                        h_log = bbox.row(idx)[3];
                    } else if (bbox.h == 1) {
                        // Shape (4*H*W, 1): fully flattened
                        const float* bbox_ptr = (const float*)bbox.data;
                        cx_offset = bbox_ptr[idx * 4 + 0];
                        cy_offset = bbox_ptr[idx * 4 + 1];
                        w_log = bbox_ptr[idx * 4 + 2];
                        h_log = bbox_ptr[idx * 4 + 3];
                    } else {
                        // Shape (W, H) with implicit c=4
                        const float* bbox_row = bbox.row(i);
                        cx_offset = bbox_row[j * 4 + 0];
                        cy_offset = bbox_row[j * 4 + 1];
                        w_log = bbox_row[j * 4 + 2];
                        h_log = bbox_row[j * 4 + 3];
                    }
                } else {
                    // 3D: (W, H) with c=4 - row-col with channels
                    const float* bbox_row = bbox.row(i);
                    cx_offset = bbox_row[j * 4 + 0];
                    cy_offset = bbox_row[j * 4 + 1];
                    w_log = bbox_row[j * 4 + 2];
                    h_log = bbox_row[j * 4 + 3];
                }
                
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
                    // Kps shape: (10, H*W) for fixed, or (H, W, 10) for dynamic
                    for (int k = 0; k < 5; k++) {
                        float kps_x_raw, kps_y_raw;
                        
                        if (kps.dims == 1) {
                            // Fixed: (10*H*W) flattened - [x0,y0,x1,y1,...] interleaved
                            const float* kps_ptr = (const float*)kps.data;
                            kps_x_raw = kps_ptr[idx * 10 + k * 2];
                            kps_y_raw = kps_ptr[idx * 10 + k * 2 + 1];
                        } else if (kps.dims == 2) {
                            if (kps.w == 10) {
                                // Shape (10, H*W): stored as H*W rows with 10 columns each
                                // NCNN uses row-major: kps.row(idx) gives row idx, then [0..9] for columns
                                kps_x_raw = kps.row(idx)[k * 2];
                                kps_y_raw = kps.row(idx)[k * 2 + 1];
                            } else if (kps.h == 1) {
                                // Shape (10*H*W, 1): fully flattened
                                const float* kps_ptr = (const float*)kps.data;
                                kps_x_raw = kps_ptr[idx * 10 + k * 2];
                                kps_y_raw = kps_ptr[idx * 10 + k * 2 + 1];
                            } else {
                                // Shape (W, H) with implicit c=10
                                const float* kps_row = kps.row(i);
                                kps_x_raw = kps_row[j * 10 + k * 2];
                                kps_y_raw = kps_row[j * 10 + k * 2 + 1];
                            }
                        } else {
                            // 3D: (W, H) with c=10 - row-col with channels
                            const float* kps_row = kps.row(i);
                            kps_x_raw = kps_row[j * 10 + k * 2];
                            kps_y_raw = kps_row[j * 10 + k * 2 + 1];
                        }
                        
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
    
    // Convert to Rect - return all detected faces (not just largest)
    // Filter out invalid boxes (must have minimum size)
    std::vector<Rect> faces;
    const int min_face_size = 20;  // Minimum 20x20 pixels
    
    for (int idx : picked) {
        FaceObject& faceobj = proposals[idx];
        
        // Validate box dimensions
        int w = (int)faceobj.rect.width;
        int h = (int)faceobj.rect.height;
        
        if (w >= min_face_size && h >= min_face_size) {
            Rect r;
            r.x = (int)faceobj.rect.x;
            r.y = (int)faceobj.rect.y;
            r.width = w;
            r.height = h;
            r.landmarks = faceobj.rect.landmarks;  // Copy landmarks
            faces.push_back(r);
        }
    }
    
    return faces;
}

} // namespace faceid
