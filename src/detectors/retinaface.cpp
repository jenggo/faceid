// RetinaFace Face Detector Implementation
// Model: RetinaFace (mnet.25-opt)
// Input: BGR image (converted to RGB), variable size, "data" layer
// Output: Bounding boxes at 3 scales (stride 8, 16, 32)
//         - face_rpn_cls_prob_reshape_stride*: Classification scores
//         - face_rpn_bbox_pred_stride*: Bounding box offsets
// Reference: https://github.com/deepinsight/insightface/tree/master/detection/retinaface

#include "common.h"
#include "../logger.h"
#include <ncnn/net.h>
#include <vector>

namespace faceid {

std::vector<Rect> detectWithRetinaFace(ncnn::Net& net, const ncnn::Mat& in, int img_w, int img_h) {
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);  // Optimize for speed
    ex.input("data", in);
    
    std::vector<FaceObject> faceproposals;
    
    const float prob_threshold = 0.8f;
    const float nms_threshold = 0.4f;
    
    // stride 32
    {
        ncnn::Mat score_blob, bbox_blob;
        ex.extract("face_rpn_cls_prob_reshape_stride32", score_blob);
        ex.extract("face_rpn_bbox_pred_stride32", bbox_blob);
        
        const int base_size = 16;
        const int feat_stride = 32;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 32.f;
        scales[1] = 16.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);
        
        std::vector<FaceObject> faceobjects32;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, prob_threshold, faceobjects32);
        faceproposals.insert(faceproposals.end(), faceobjects32.begin(), faceobjects32.end());
    }
    
    // stride 16
    {
        ncnn::Mat score_blob, bbox_blob;
        ex.extract("face_rpn_cls_prob_reshape_stride16", score_blob);
        ex.extract("face_rpn_bbox_pred_stride16", bbox_blob);
        
        const int base_size = 16;
        const int feat_stride = 16;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 8.f;
        scales[1] = 4.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);
        
        std::vector<FaceObject> faceobjects16;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, prob_threshold, faceobjects16);
        faceproposals.insert(faceproposals.end(), faceobjects16.begin(), faceobjects16.end());
    }
    
    // stride 8
    {
        ncnn::Mat score_blob, bbox_blob;
        ex.extract("face_rpn_cls_prob_reshape_stride8", score_blob);
        ex.extract("face_rpn_bbox_pred_stride8", bbox_blob);
        
        const int base_size = 16;
        const int feat_stride = 8;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 2.f;
        scales[1] = 1.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);
        
        std::vector<FaceObject> faceobjects8;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, prob_threshold, faceobjects8);
        faceproposals.insert(faceproposals.end(), faceobjects8.begin(), faceobjects8.end());
    }
    
    // Sort and apply NMS
    qsort_descent_inplace(faceproposals);
    
    std::vector<int> picked;
    nms_sorted_bboxes(faceproposals, picked, nms_threshold);
    
    // Convert to Rect and clip to image bounds
    std::vector<Rect> faces;
    for (int idx : picked) {
        FaceObject& obj = faceproposals[idx];
        
        // Clip to image size
        float x0 = obj.rect.x;
        float y0 = obj.rect.y;
        float x1 = x0 + obj.rect.width;
        float y1 = y0 + obj.rect.height;
        
        x0 = std::max(std::min(x0, (float)img_w - 1), 0.f);
        y0 = std::max(std::min(y0, (float)img_h - 1), 0.f);
        x1 = std::max(std::min(x1, (float)img_w - 1), 0.f);
        y1 = std::max(std::min(y1, (float)img_h - 1), 0.f);
        
        obj.rect.x = (int)x0;
        obj.rect.y = (int)y0;
        obj.rect.width = (int)(x1 - x0);
        obj.rect.height = (int)(y1 - y0);
        
        // Filter out invalid boxes
        if (obj.rect.width > 0 && obj.rect.height > 0) {
            faces.push_back(obj.rect);
        }
    }
    
    return faces;
}

} // namespace faceid
