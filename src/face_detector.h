#ifndef FACEID_FACE_DETECTOR_H
#define FACEID_FACE_DETECTOR_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/dnn.h>
#include <dlib/opencv.h>

namespace faceid {

using FaceEncoding = dlib::matrix<float, 0, 1>;

class FaceDetector {
public:
    FaceDetector();
    
    bool loadModels(const std::string& shape_predictor_path,
                   const std::string& face_recognition_model_path);
    
    std::vector<dlib::rectangle> detectFaces(const cv::Mat& frame, bool downscale = true);
    
    std::vector<FaceEncoding> encodeFaces(const cv::Mat& frame,
                                          const std::vector<dlib::rectangle>& face_locations);
    
    double compareFaces(const FaceEncoding& encoding1, const FaceEncoding& encoding2);
    
    // Performance: Pre-process frame for faster detection
    cv::Mat preprocessFrame(const cv::Mat& frame);
    
    // Enable/disable caching for repeated detections
    void enableCache(bool enable);
    
    // Clear encoding cache
    void clearCache();

private:
    dlib::frontal_face_detector detector_;
    dlib::shape_predictor shape_predictor_;
    
    // Face recognition model template (from dlib)
    template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
    using residual = dlib::add_prev1<block<N,BN,1,dlib::tag1<SUBNET>>>;

    template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
    using residual_down = dlib::add_prev2<dlib::avg_pool<2,2,2,2,dlib::skip1<dlib::tag2<block<N,BN,2,dlib::tag1<SUBNET>>>>>>;

    template <int N, template <typename> class BN, int stride, typename SUBNET> 
    using block  = BN<dlib::con<N,3,3,1,1,dlib::relu<BN<dlib::con<N,3,3,stride,stride,SUBNET>>>>>;

    template <int N, typename SUBNET> using ares      = dlib::relu<residual<block,N,dlib::affine,SUBNET>>;
    template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block,N,dlib::affine,SUBNET>>;

    template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
    template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
    template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
    template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
    template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

    using anet_type = dlib::loss_metric<dlib::fc_no_bias<128,dlib::avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            dlib::max_pool<3,3,2,2,dlib::relu<dlib::affine<dlib::con<32,7,7,2,2,
                            dlib::input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

    anet_type net_;
    bool models_loaded_ = false;
    
    // Performance optimizations
    bool use_cache_ = true;
    std::unordered_map<uint64_t, std::vector<dlib::rectangle>> detection_cache_;
    
    // Hash function for frame caching
    uint64_t hashFrame(const cv::Mat& frame);
};

} // namespace faceid

#endif // FACEID_FACE_DETECTOR_H
