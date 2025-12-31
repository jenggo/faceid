// YOLO Face Detector Implementation
// Supports YOLOv5, YOLOv7, YOLOv8 face detection models
// Input: RGB image (converted from BGR), 640x640 with letterbox padding
// Output: Bounding boxes with facial keypoints at 3 scales (stride 8, 16, 32)
// Reference implementations:
//   - YOLOv5: https://github.com/deepcam-cn/yolov5-face
//   - YOLOv7: https://github.com/derronqi/yolov7-face
//   - YOLOv8: https://github.com/derronqi/yolov8-face

#include "common.h"
#include "../logger.h"
#include <ncnn/net.h>
#include <algorithm>
#include <cmath>

namespace faceid {

// YOLO model version detection
enum class YoloVersion {
    UNKNOWN,
    YOLOV5,
    YOLOV7,
    YOLOV8
};

// Sigmoid activation
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Softmax for DFL (YOLOv8)
static std::vector<float> softmax(const std::vector<float>& input) {
    if (input.empty()) return {};
    
    float max_val = *std::max_element(input.begin(), input.end());
    float sum = 0.0f;
    std::vector<float> result(input.size());
    
    for (size_t i = 0; i < input.size(); i++) {
        result[i] = expf(input[i] - max_val);
        sum += result[i];
    }
    
    for (size_t i = 0; i < input.size(); i++) {
        result[i] /= sum;
    }
    
    return result;
}

// Detect YOLO version from param file
static YoloVersion detectYoloVersion(ncnn::Net& net) {
    // Try to detect by checking which output layers exist
    // YOLOv5: "981", "983", "985" (numbered outputs)
    // YOLOv7: "stride_8", "stride_16", "stride_32"
    // YOLOv8: "output0", "1076", "1084"
    
    ncnn::Extractor ex = net.create_extractor();
    ncnn::Mat test_out;
    
    // Test for YOLOv8
    if (ex.extract("output0", test_out) == 0) {
        return YoloVersion::YOLOV8;
    }
    
    // Test for YOLOv7
    ex = net.create_extractor();
    if (ex.extract("stride_8", test_out) == 0) {
        return YoloVersion::YOLOV7;
    }
    
    // Test for YOLOv5
    ex = net.create_extractor();
    if (ex.extract("981", test_out) == 0) {
        return YoloVersion::YOLOV5;
    }
    
    return YoloVersion::UNKNOWN;
}

// YOLOv5 proposal generation
static void generate_proposals_yolov5(
    const std::vector<float>& anchors,
    int stride,
    const ncnn::Mat& in_pad,
    const ncnn::Mat& feat_blob,
    float prob_threshold,
    std::vector<FaceObject>& objects)
{
    const int num_grid = feat_blob.h;
    
    int num_grid_x, num_grid_y;
    if (in_pad.w > in_pad.h) {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    } else {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }
    
    const int num_anchors = anchors.size() / 2;
    
    for (int q = 0; q < num_anchors; q++) {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];
        
        const ncnn::Mat feat = feat_blob.channel(q);
        
        for (int i = 0; i < num_grid_y; i++) {
            for (int j = 0; j < num_grid_x; j++) {
                const float* featptr = feat.row(i * num_grid_x + j);
                
                float box_confidence = sigmoid(featptr[4]);
                
                if (box_confidence < prob_threshold) {
                    continue;
                }
                
                // Decode box
                float dx = sigmoid(featptr[0]);
                float dy = sigmoid(featptr[1]);
                float dw = sigmoid(featptr[2]);
                float dh = sigmoid(featptr[3]);
                
                float pb_cx = (dx * 2.0f - 0.5f + j) * stride;
                float pb_cy = (dy * 2.0f - 0.5f + i) * stride;
                float pb_w = powf(dw * 2.0f, 2) * anchor_w;
                float pb_h = powf(dh * 2.0f, 2) * anchor_h;
                
                float x0 = pb_cx - pb_w * 0.5f;
                float y0 = pb_cy - pb_h * 0.5f;
                
                FaceObject obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = pb_w;
                obj.rect.height = pb_h;
                obj.prob = box_confidence;
                
                // Extract landmarks if available (5 points after box + confidence = offset 5)
                // YOLOv5 format: 10 values (x1, y1, x2, y2, ..., x5, y5)
                // Landmarks are scaled by anchor size and offset by grid position
                const float* ptr_kps = featptr + 5;
                for (int k = 0; k < 5; k++) {
                    float kps_x = ptr_kps[k * 2];
                    float kps_y = ptr_kps[k * 2 + 1];
                    
                    Point pt;
                    pt.x = kps_x * anchor_w + j * stride;
                    pt.y = kps_y * anchor_h + i * stride;
                    obj.rect.landmarks.push_back(pt);
                }
                
                objects.push_back(obj);
            }
        }
    }
}

// YOLOv7 proposal generation
static void generate_proposals_yolov7(
    const std::vector<float>& anchors,
    int stride,
    const ncnn::Mat& in_pad,
    const ncnn::Mat& feat_blob,
    float prob_threshold,
    std::vector<FaceObject>& objects)
{
    const int num_grid = feat_blob.h;
    
    int num_grid_x, num_grid_y;
    if (in_pad.w > in_pad.h) {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    } else {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }
    
    const int num_anchors = anchors.size() / 2;
    
    for (int q = 0; q < num_anchors; q++) {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];
        
        const ncnn::Mat feat = feat_blob.channel(q);
        
        for (int i = 0; i < num_grid_y; i++) {
            for (int j = 0; j < num_grid_x; j++) {
                const float* featptr = feat.row(i * num_grid_x + j);
                
                float box_confidence = sigmoid(featptr[4]);
                
                if (box_confidence < prob_threshold) {
                    continue;
                }
                
                // YOLOv7: multiply objectness with class score
                float class_score = featptr[5];
                float confidence = box_confidence * sigmoid(class_score);
                
                if (confidence < prob_threshold) {
                    continue;
                }
                
                // Decode box (same as YOLOv5)
                float dx = sigmoid(featptr[0]);
                float dy = sigmoid(featptr[1]);
                float dw = sigmoid(featptr[2]);
                float dh = sigmoid(featptr[3]);
                
                float pb_cx = (dx * 2.0f - 0.5f + j) * stride;
                float pb_cy = (dy * 2.0f - 0.5f + i) * stride;
                float pb_w = powf(dw * 2.0f, 2) * anchor_w;
                float pb_h = powf(dh * 2.0f, 2) * anchor_h;
                
                // Filter invalid boxes
                if (pb_w <= 0 || pb_h <= 0) continue;
                if (pb_w < 10 || pb_h < 10 || pb_w > 500 || pb_h > 500) continue;
                
                float aspect = pb_w / pb_h;
                if (aspect < 0.3f || aspect > 3.0f) continue;
                
                float x0 = pb_cx - pb_w * 0.5f;
                float y0 = pb_cy - pb_h * 0.5f;
                
                FaceObject obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = pb_w;
                obj.rect.height = pb_h;
                obj.prob = confidence;
                
                // Extract landmarks if available (5 points after box + confidence + class = offset 6)
                // YOLOv7 format: 15 values (5 landmarks × 3: x, y, visibility)
                const float* ptr_kps = featptr + 6;
                for (int k = 0; k < 5; k++) {
                    float kps_x = ptr_kps[k * 3];
                    float kps_y = ptr_kps[k * 3 + 1];
                    // float kps_vis = ptr_kps[k * 3 + 2];  // visibility (unused)
                    
                    Point pt;
                    // YOLOv7: Use raw values directly (no sigmoid), same formula as reference
                    pt.x = (kps_x * 2.0f - 0.5f + j) * stride;
                    pt.y = (kps_y * 2.0f - 0.5f + i) * stride;
                    obj.rect.landmarks.push_back(pt);
                }
                
                objects.push_back(obj);
            }
        }
    }
}

// YOLOv8 proposal generation (DFL-based)
static void generate_proposals_yolov8(
    int stride,
    const ncnn::Mat& in_pad,
    const ncnn::Mat& feat_blob,
    float prob_threshold,
    std::vector<FaceObject>& objects)
{
    const int reg_max = 16;  // DFL bins
    int fea_h = feat_blob.h;
    int fea_w = feat_blob.w;
    int spacial_size = fea_w * fea_h;
    
    // YOLOv8-face output structure:
    // - Box predictions: 64 values (4 sides × 16 DFL bins)
    // - Class confidence: 1 value
    // - Landmarks: 10 values (5 keypoints × 2 coordinates)
    // Total: 75 values per position
    
    // Data pointers
    const float* ptr_b = (const float*)feat_blob.data;  // Box predictions (64 values)
    const float* ptr_c = ptr_b + spacial_size * reg_max * 4;  // Class confidence (1 value)
    const float* ptr_kps = ptr_c + spacial_size;  // Landmarks (10 values)
    
    for (int i = 0; i < fea_h; i++) {
        for (int j = 0; j < fea_w; j++) {
            int index = i * fea_w + j;
            
            float box_confidence = sigmoid(ptr_c[index]);
            
            if (box_confidence < prob_threshold) {
                continue;
            }
            
            // Decode box using DFL (Distribution Focal Loss)
            float pred_ltrb[4];  // left, top, right, bottom distances
            
            for (int k = 0; k < 4; k++) {
                std::vector<float> dfl_value(reg_max);
                for (int n = 0; n < reg_max; n++) {
                    dfl_value[n] = ptr_b[index + (reg_max * k + n) * spacial_size];
                }
                
                // Apply softmax and weighted sum
                std::vector<float> dfl_softmax = softmax(dfl_value);
                float dis = 0.0f;
                for (int n = 0; n < reg_max; n++) {
                    dis += n * dfl_softmax[n];
                }
                
                pred_ltrb[k] = dis * stride;
            }
            
            // Anchor-free center
            float pb_cx = (j + 0.5f) * stride;
            float pb_cy = (i + 0.5f) * stride;
            
            // Convert LTRB to coordinates
            float x1 = pb_cx - pred_ltrb[0];
            float y1 = pb_cy - pred_ltrb[1];
            float x2 = pb_cx + pred_ltrb[2];
            float y2 = pb_cy + pred_ltrb[3];
            
            float pb_w = x2 - x1;
            float pb_h = y2 - y1;
            
            // Filter invalid boxes
            if (pb_w <= 0 || pb_h <= 0) continue;
            if (pb_w < 10 || pb_h < 10 || pb_w > 500 || pb_h > 500) continue;
            
            float aspect = pb_w / pb_h;
            if (aspect < 0.3f || aspect > 3.0f) continue;
            
            FaceObject obj;
            obj.rect.x = x1;
            obj.rect.y = y1;
            obj.rect.width = pb_w;
            obj.rect.height = pb_h;
            obj.prob = box_confidence;
            
            // Decode 5-point landmarks (2 eyes, nose, 2 mouth corners)
            // YOLOv8-face format: 15 values (5 landmarks × 3 values: x, y, visibility)
            // Channel-first layout: [x1, x2, ..., x5, y1, y2, ..., y5, vis1, vis2, ..., vis5]
            for (int k = 0; k < 5; k++) {
                float kps_x = ptr_kps[(k * 3 + 0) * spacial_size + index];
                float kps_y = ptr_kps[(k * 3 + 1) * spacial_size + index];
                // float kps_vis = ptr_kps[(k * 3 + 2) * spacial_size + index];  // visibility (unused)
                
                Point pt;
                pt.x = (kps_x * 2.0f + j) * stride;
                pt.y = (kps_y * 2.0f + i) * stride;
                obj.rect.landmarks.push_back(pt);
            }
            
            objects.push_back(obj);
        }
    }
}

// Main detection function for all YOLO versions
std::vector<Rect> detectWithYOLO(ncnn::Net& net, const ncnn::Mat& in, int img_w, int img_h,
                                  float confidence_threshold, YoloVersion version)
{
    // If version is unknown, detect it
    if (version == YoloVersion::UNKNOWN) {
        version = detectYoloVersion(net);
        if (version == YoloVersion::UNKNOWN) {
            Logger::getInstance().error("Could not detect YOLO model version");
            return {};
        }
    }
    
    const int target_size = 640;
    const float nms_threshold = 0.45f;
    
    // Letterbox resize
    int w = img_w;
    int h = img_h;
    float scale = 1.0f;
    
    if (w > h) {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    } else {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }
    
    // Convert and resize
    ncnn::Mat in_resized;
    ncnn::resize_bilinear(in, in_resized, w, h);
    
    // Pad to target_size (32-pixel aligned)
    int wpad = (target_size - w) / 2;
    int hpad = (target_size - h) / 2;
    
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in_resized, in_pad, hpad, target_size - h - hpad, 
                          wpad, target_size - w - wpad, 
                          ncnn::BORDER_CONSTANT, 114.f);
    
    // Normalize
    const float norm_vals[3] = {1.0f/255.0f, 1.0f/255.0f, 1.0f/255.0f};
    in_pad.substract_mean_normalize(0, norm_vals);
    
    // Extract features
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    
    // Determine input layer name
    const char* input_name = (version == YoloVersion::YOLOV5) ? "data" : "images";
    ex.input(input_name, in_pad);
    
    std::vector<FaceObject> proposals;
    
    if (version == YoloVersion::YOLOV5) {
        // YOLOv5: layers "981", "983", "985"
        const char* output_names[] = {"981", "983", "985"};
        const int strides[] = {8, 16, 32};
        const std::vector<std::vector<float>> anchors = {
            {4, 5, 8, 10, 13, 16},       // stride 8
            {23, 29, 43, 55, 73, 105},   // stride 16
            {146, 217, 231, 300, 335, 433}  // stride 32
        };
        
        for (int i = 0; i < 3; i++) {
            ncnn::Mat out;
            if (ex.extract(output_names[i], out) != 0) continue;
            
            std::vector<FaceObject> layer_proposals;
            generate_proposals_yolov5(anchors[i], strides[i], in_pad, out,
                                     confidence_threshold, layer_proposals);
            proposals.insert(proposals.end(), layer_proposals.begin(), layer_proposals.end());
        }
    } else if (version == YoloVersion::YOLOV7) {
        // YOLOv7: layers "stride_8", "stride_16", "stride_32"
        const char* output_names[] = {"stride_8", "stride_16", "stride_32"};
        const int strides[] = {8, 16, 32};
        const std::vector<std::vector<float>> anchors = {
            {4, 5, 6, 8, 10, 12},        // stride 8
            {15, 19, 23, 30, 39, 52},    // stride 16
            {72, 97, 123, 164, 209, 297} // stride 32
        };
        
        for (int i = 0; i < 3; i++) {
            ncnn::Mat out;
            if (ex.extract(output_names[i], out) != 0) continue;
            
            std::vector<FaceObject> layer_proposals;
            generate_proposals_yolov7(anchors[i], strides[i], in_pad, out,
                                     confidence_threshold, layer_proposals);
            proposals.insert(proposals.end(), layer_proposals.begin(), layer_proposals.end());
        }
    } else if (version == YoloVersion::YOLOV8) {
        // YOLOv8: layers "output0", "1076", "1084"
        const char* output_names[] = {"output0", "1076", "1084"};
        const int strides[] = {8, 16, 32};
        
        for (int i = 0; i < 3; i++) {
            ncnn::Mat out;
            if (ex.extract(output_names[i], out) != 0) continue;
            
            std::vector<FaceObject> layer_proposals;
            generate_proposals_yolov8(strides[i], in_pad, out,
                                     confidence_threshold, layer_proposals);
            proposals.insert(proposals.end(), layer_proposals.begin(), layer_proposals.end());
        }
    }
    
    // Sort by confidence
    qsort_descent_inplace(proposals);
    
    // Apply NMS
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);
    
    // Map coordinates back to original image and filter
    std::vector<Rect> faces;
    for (int idx : picked) {
        FaceObject& obj = proposals[idx];
        
        // Undo padding and scaling
        float x = (obj.rect.x - wpad) / scale;
        float y = (obj.rect.y - hpad) / scale;
        float w = obj.rect.width / scale;
        float h = obj.rect.height / scale;
        
        // Clamp to image bounds
        x = std::max(0.0f, std::min(x, (float)img_w));
        y = std::max(0.0f, std::min(y, (float)img_h));
        w = std::min(w, (float)img_w - x);
        h = std::min(h, (float)img_h - y);
        
        // Final filtering: remove invalid boxes after coordinate mapping
        if (w <= 0 || h <= 0) continue;
        
        float max_size = std::max(img_w, img_h) * 0.6f;
        if (w < 20 || h < 20 || w > max_size || h > max_size) continue;
        
        float aspect = w / h;
        if (aspect < 0.5f || aspect > 2.0f) continue;
        
        Rect face;
        face.x = static_cast<int>(x);
        face.y = static_cast<int>(y);
        face.width = static_cast<int>(w);
        face.height = static_cast<int>(h);
        
        // Map landmarks back to original image coordinates
        for (const auto& lm : obj.rect.landmarks) {
            float lm_x = (lm.x - wpad) / scale;
            float lm_y = (lm.y - hpad) / scale;
            
            // Clamp to image bounds
            lm_x = std::max(0.0f, std::min(lm_x, (float)img_w));
            lm_y = std::max(0.0f, std::min(lm_y, (float)img_h));
            
            face.landmarks.push_back(Point(lm_x, lm_y));
        }
        
        faces.push_back(face);
    }
    
    return faces;
}

// Convenience wrappers for each version
std::vector<Rect> detectWithYOLOv5(ncnn::Net& net, const ncnn::Mat& in, int img_w, int img_h,
                                    float confidence_threshold)
{
    return detectWithYOLO(net, in, img_w, img_h, confidence_threshold, YoloVersion::YOLOV5);
}

std::vector<Rect> detectWithYOLOv7(ncnn::Net& net, const ncnn::Mat& in, int img_w, int img_h,
                                    float confidence_threshold)
{
    return detectWithYOLO(net, in, img_w, img_h, confidence_threshold, YoloVersion::YOLOV7);
}

std::vector<Rect> detectWithYOLOv8(ncnn::Net& net, const ncnn::Mat& in, int img_w, int img_h,
                                    float confidence_threshold)
{
    return detectWithYOLO(net, in, img_w, img_h, confidence_threshold, YoloVersion::YOLOV8);
}

} // namespace faceid
