// SCRFD Face Detector Implementation
// Model: SCRFD (Scaled-RoI Face Detector)
// Input: RGB image (converted from BGR), variable size, "input.1" layer
// Output: Bounding boxes at 3 scales with confidence scores
//         - score_8/16/32: Classification scores (2 anchors per location)
//         - bbox_8/16/32: Bounding box offsets (distance transform format)
//         - kps_8/16/32: Keypoints (optional)
// Reference: https://github.com/nihui/ncnn-android-scrfd (official NCNN implementation)

#include "common.h"
#include "../logger.h"
#include <ncnn/net.h>
#include <vector>
#include <cmath>
#include <unistd.h>
#include <fcntl.h>

namespace faceid {

// Generate SCRFD-style anchors (center at origin)
static ncnn::Mat generate_scrfd_anchors(int base_size, const ncnn::Mat& ratios, const ncnn::Mat& scales) {
    int num_ratio = ratios.w;
    int num_scale = scales.w;
    
    ncnn::Mat anchors;
    anchors.create(4, num_ratio * num_scale);
    
    const float cx = 0;
    const float cy = 0;
    
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

// Generate proposals using SCRFD distance transform decoding
static void generate_scrfd_proposals(const ncnn::Mat& anchors, int feat_stride, 
                                     const ncnn::Mat& score_blob, const ncnn::Mat& bbox_blob,
                                     float prob_threshold, std::vector<FaceObject>& faceobjects) {
    int w = score_blob.w;
    int h = score_blob.h;
    
    const int num_anchors = anchors.h;
    
    for (int q = 0; q < num_anchors; q++) {
        const float* anchor = anchors.row(q);
        
        const ncnn::Mat score = score_blob.channel(q);
        const ncnn::Mat bbox = bbox_blob.channel_range(q * 4, 4);
        
        // Shifted anchor
        float anchor_y = anchor[1];
        float anchor_w = anchor[2] - anchor[0];
        float anchor_h = anchor[3] - anchor[1];
        
        for (int i = 0; i < h; i++) {
            float anchor_x = anchor[0];
            
            for (int j = 0; j < w; j++) {
                int index = i * w + j;
                
                float prob = score[index];
                
                if (prob >= prob_threshold) {
                    // SCRFD distance transform: multiply by stride
                    float dx = bbox.channel(0)[index] * feat_stride;
                    float dy = bbox.channel(1)[index] * feat_stride;
                    float dw = bbox.channel(2)[index] * feat_stride;
                    float dh = bbox.channel(3)[index] * feat_stride;
                    
                    // Anchor center
                    float cx = anchor_x + anchor_w * 0.5f;
                    float cy = anchor_y + anchor_h * 0.5f;
                    
                    // Distance2bbox: center +/- distance
                    float x0 = cx - dx;
                    float y0 = cy - dy;
                    float x1 = cx + dw;
                    float y1 = cy + dh;
                    
                    FaceObject obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0 + 1;
                    obj.rect.height = y1 - y0 + 1;
                    obj.prob = prob;
                    
                    faceobjects.push_back(obj);
                }
                
                anchor_x += feat_stride;
            }
            
            anchor_y += feat_stride;
        }
    }
}

std::vector<Rect> detectWithSCRFD(ncnn::Net& net, const ncnn::Mat& in, int img_w, int img_h,
                                  float confidence_threshold, float scale, int wpad, int hpad,
                                  int orig_w, int orig_h) {
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    ex.input("input.1", in);
    
    const float prob_threshold = confidence_threshold;
    const float nms_threshold = 0.45f;
    std::vector<FaceObject> proposals;
    
    // Anchor settings (ratio=1.0, scales=[1.0, 2.0])
    ncnn::Mat ratios(1);
    ratios[0] = 1.f;
    ncnn::Mat scales(2);
    scales[0] = 1.f;
    scales[1] = 2.f;
    
    // Redirect stderr to suppress NCNN's "find_blob_index_by_name failed" warnings
    // These are expected for optimized models that use numbered blob indices instead of names
    int stderr_backup = dup(STDERR_FILENO);
    int devnull = open("/dev/null", O_WRONLY);
    dup2(devnull, STDERR_FILENO);
    close(devnull);
    
    // Stride 8 (base_size=16, NOT stride!)
    // Try named blobs first (original NCNN models), fall back to numbered indices (optimized models)
    ncnn::Mat score_blob_8, bbox_blob_8;
    {
        int ret = ex.extract("score_8", score_blob_8);
        if (ret != 0) {
            // Named blob failed, try numbered index for optimized models
            ex.extract("412", score_blob_8);
        }
        
        ret = ex.extract("bbox_8", bbox_blob_8);
        if (ret != 0) {
            ex.extract("415", bbox_blob_8);
        }
    }
    
    // Stride 16 (base_size=64)
    ncnn::Mat score_blob_16, bbox_blob_16;
    {
        int ret = ex.extract("score_16", score_blob_16);
        if (ret != 0) {
            ret = ex.extract("474", score_blob_16);
            if (ret != 0) {
                fprintf(stderr, "ERROR: Failed to extract score blob for stride 16 (tried 'score_16' and '474')\n");
            }
        }
        
        ret = ex.extract("bbox_16", bbox_blob_16);
        if (ret != 0) {
            ret = ex.extract("477", bbox_blob_16);
            if (ret != 0) {
                fprintf(stderr, "ERROR: Failed to extract bbox blob for stride 16 (tried 'bbox_16' and '477')\n");
            }
        }
    }
    
    // Stride 32 (base_size=256)
    ncnn::Mat score_blob_32, bbox_blob_32;
    {
        int ret = ex.extract("score_32", score_blob_32);
        if (ret != 0) {
            ret = ex.extract("536", score_blob_32);
            if (ret != 0) {
                fprintf(stderr, "ERROR: Failed to extract score blob for stride 32 (tried 'score_32' and '536')\n");
            }
        }
        
        ret = ex.extract("bbox_32", bbox_blob_32);
        if (ret != 0) {
            ret = ex.extract("539", bbox_blob_32);
            if (ret != 0) {
                fprintf(stderr, "ERROR: Failed to extract bbox blob for stride 32 (tried 'bbox_32' and '539')\n");
            }
        }
    }
    
    // Restore stderr before proposal generation
    dup2(stderr_backup, STDERR_FILENO);
    close(stderr_backup);
    
    // Generate proposals for each stride
    {
        const int base_size = 16;
        const int feat_stride = 8;
        ncnn::Mat anchors = generate_scrfd_anchors(base_size, ratios, scales);
        generate_scrfd_proposals(anchors, feat_stride, score_blob_8, bbox_blob_8, 
                                prob_threshold, proposals);
    }
    
    {
        const int base_size = 64;
        const int feat_stride = 16;
        ncnn::Mat anchors = generate_scrfd_anchors(base_size, ratios, scales);
        generate_scrfd_proposals(anchors, feat_stride, score_blob_16, bbox_blob_16, 
                                prob_threshold, proposals);
    }
    
    {
        const int base_size = 256;
        const int feat_stride = 32;
        ncnn::Mat anchors = generate_scrfd_anchors(base_size, ratios, scales);
        generate_scrfd_proposals(anchors, feat_stride, score_blob_32, bbox_blob_32, 
                                prob_threshold, proposals);
    }
    
    // Apply NMS
    qsort_descent_inplace(proposals);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);
    
    // Convert to Rect and adjust coordinates back to original image space
    std::vector<Rect> faces;
    for (int idx : picked) {
        FaceObject& faceobj = proposals[idx];
        
        // Adjust offset to original unpadded image
        // Reference: https://github.com/nihui/ncnn-webassembly-scrfd/blob/master/scrfd.cpp#L347-359
        float x0 = (faceobj.rect.x - (wpad / 2.0f)) / scale;
        float y0 = (faceobj.rect.y - (hpad / 2.0f)) / scale;
        float x1 = (faceobj.rect.x + faceobj.rect.width - (wpad / 2.0f)) / scale;
        float y1 = (faceobj.rect.y + faceobj.rect.height - (hpad / 2.0f)) / scale;
        
        // Clip to original image bounds
        x0 = std::max(std::min(x0, (float)orig_w - 1), 0.f);
        y0 = std::max(std::min(y0, (float)orig_h - 1), 0.f);
        x1 = std::max(std::min(x1, (float)orig_w - 1), 0.f);
        y1 = std::max(std::min(y1, (float)orig_h - 1), 0.f);
        
        int width = (int)(x1 - x0);
        int height = (int)(y1 - y0);
        
        // Filter by minimum size
        if (x1 > x0 && y1 > y0) {
            Rect r;
            r.x = (int)x0;
            r.y = (int)y0;
            r.width = width;
            r.height = height;
            faces.push_back(r);
        }
    }
    
    return faces;
}

} // namespace faceid
