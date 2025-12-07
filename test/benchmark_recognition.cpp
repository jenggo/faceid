// Benchmark tool to compare SFace vs dlib face recognition performance
// Compile: g++ -std=c++20 -o benchmark_recognition benchmark_recognition.cpp \
//          `pkg-config --cflags --libs opencv4 dlib-1` -lpthread

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <dlib/image_processing.h>
#include <dlib/dnn.h>
#include <dlib/opencv.h>
#include <chrono>
#include <iostream>
#include <iomanip>

// dlib face recognition network (same as in face_detector.h)
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

class RecognitionBenchmark {
public:
    RecognitionBenchmark() = default;
    
    bool loadDlibModels(const std::string& shape_path, const std::string& recog_path) {
        try {
            dlib::deserialize(shape_path) >> shape_predictor_;
            dlib::deserialize(recog_path) >> dlib_net_;
            dlib_loaded_ = true;
            std::cout << "✓ Loaded dlib models\n";
            return true;
        } catch (const std::exception& e) {
            std::cerr << "✗ Failed to load dlib models: " << e.what() << "\n";
            return false;
        }
    }
    
    bool loadSFaceModel(const std::string& model_path) {
        try {
            sface_net_ = cv::FaceRecognizerSF::create(model_path, "");
            sface_loaded_ = true;
            std::cout << "✓ Loaded SFace model\n";
            return true;
        } catch (const std::exception& e) {
            std::cerr << "✗ Failed to load SFace model: " << e.what() << "\n";
            return false;
        }
    }
    
    bool loadYuNetDetector(const std::string& model_path) {
        try {
            yunet_detector_ = cv::FaceDetectorYN::create(
                model_path, "", cv::Size(320, 240), 0.6f, 0.3f, 5000
            );
            yunet_loaded_ = true;
            std::cout << "✓ Loaded YuNet detector\n";
            return true;
        } catch (const std::exception& e) {
            std::cerr << "✗ Failed to load YuNet: " << e.what() << "\n";
            return false;
        }
    }
    
    void runBenchmark(const cv::Mat& image, int iterations = 10) {
        if (!yunet_loaded_) {
            std::cerr << "YuNet detector not loaded\n";
            return;
        }
        
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "FACE RECOGNITION BENCHMARK\n";
        std::cout << std::string(80, '=') << "\n";
        std::cout << "Image size: " << image.cols << "x" << image.rows << "\n";
        std::cout << "Iterations: " << iterations << "\n\n";
        
        // Detect faces first
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(320, 240));
        cv::Mat faces;
        yunet_detector_->detect(resized, faces);
        
        if (faces.rows == 0) {
            std::cerr << "No faces detected in image\n";
            return;
        }
        
        std::cout << "Detected " << faces.rows << " face(s)\n\n";
        
        // Extract first face region
        float x = faces.at<float>(0, 0);
        float y = faces.at<float>(0, 1);
        float w = faces.at<float>(0, 2);
        float h = faces.at<float>(0, 3);
        
        double scale_x = static_cast<double>(image.cols) / 320.0;
        double scale_y = static_cast<double>(image.rows) / 240.0;
        
        cv::Rect face_rect(
            static_cast<int>(x * scale_x),
            static_cast<int>(y * scale_y),
            static_cast<int>(w * scale_x),
            static_cast<int>(h * scale_y)
        );
        
        // Ensure rect is within image bounds
        face_rect &= cv::Rect(0, 0, image.cols, image.rows);
        
        if (face_rect.width <= 0 || face_rect.height <= 0) {
            std::cerr << "Invalid face region\n";
            return;
        }
        
        cv::Mat face_img = image(face_rect).clone();
        
        // Benchmark dlib
        if (dlib_loaded_) {
            benchmarkDlib(face_img, face_rect, iterations);
        }
        
        // Benchmark SFace
        if (sface_loaded_) {
            benchmarkSFace(face_img, iterations);
        }
        
        // Summary
        printSummary();
    }
    
private:
    void benchmarkDlib(const cv::Mat& face_img, const cv::Rect& face_rect, int iterations) {
        std::cout << "Testing dlib ResNet recognition...\n";
        
        dlib::cv_image<dlib::bgr_pixel> dlib_img(face_img);
        dlib::rectangle dlib_rect(0, 0, face_img.cols, face_img.rows);
        
        // Warmup
        auto shape = shape_predictor_(dlib_img, dlib_rect);
        dlib::matrix<dlib::rgb_pixel> face_chip;
        dlib::extract_image_chip(dlib_img, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);
        auto encoding = dlib_net_(face_chip);
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            shape = shape_predictor_(dlib_img, dlib_rect);
            dlib::extract_image_chip(dlib_img, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);
            encoding = dlib_net_(face_chip);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        dlib_time_ = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
        dlib_encoding_size_ = encoding.size();
        
        std::cout << "  Avg time: " << std::fixed << std::setprecision(2) << dlib_time_ << " ms\n";
        std::cout << "  Encoding size: " << dlib_encoding_size_ << "D\n\n";
    }
    
    void benchmarkSFace(const cv::Mat& face_img, int iterations) {
        std::cout << "Testing SFace (MobileFaceNet) recognition...\n";
        
        // SFace expects aligned face (112x112)
        cv::Mat aligned;
        cv::resize(face_img, aligned, cv::Size(112, 112));
        
        // Warmup
        cv::Mat encoding;
        sface_net_->feature(aligned, encoding);
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            sface_net_->feature(aligned, encoding);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        sface_time_ = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
        sface_encoding_size_ = encoding.total();
        
        std::cout << "  Avg time: " << std::fixed << std::setprecision(2) << sface_time_ << " ms\n";
        std::cout << "  Encoding size: " << sface_encoding_size_ << "D\n\n";
    }
    
    void printSummary() {
        std::cout << std::string(80, '=') << "\n";
        std::cout << "SUMMARY\n";
        std::cout << std::string(80, '=') << "\n\n";
        
        std::cout << std::setw(30) << "Metric" 
                  << std::setw(20) << "dlib ResNet" 
                  << std::setw(20) << "SFace\n";
        std::cout << std::string(70, '-') << "\n";
        
        if (dlib_loaded_) {
            std::cout << std::setw(30) << "Inference time (ms)" 
                      << std::setw(20) << std::fixed << std::setprecision(2) << dlib_time_;
        }
        
        if (sface_loaded_) {
            std::cout << std::setw(20) << std::fixed << std::setprecision(2) << sface_time_;
        }
        std::cout << "\n";
        
        if (dlib_loaded_) {
            std::cout << std::setw(30) << "Embedding size" 
                      << std::setw(20) << dlib_encoding_size_ << "D";
        }
        
        if (sface_loaded_) {
            std::cout << std::setw(20) << sface_encoding_size_ << "D";
        }
        std::cout << "\n";
        
        if (dlib_loaded_ && sface_loaded_) {
            double speedup = dlib_time_ / sface_time_;
            std::cout << std::setw(30) << "Speedup" 
                      << std::setw(20) << "-"
                      << std::setw(20) << std::fixed << std::setprecision(2) << speedup << "x\n";
        }
        
        std::cout << "\n" << std::string(80, '=') << "\n";
    }
    
    // dlib
    dlib::shape_predictor shape_predictor_;
    anet_type dlib_net_;
    bool dlib_loaded_ = false;
    double dlib_time_ = 0.0;
    int dlib_encoding_size_ = 0;
    
    // SFace
    cv::Ptr<cv::FaceRecognizerSF> sface_net_;
    bool sface_loaded_ = false;
    double sface_time_ = 0.0;
    int sface_encoding_size_ = 0;
    
    // YuNet detector
    cv::Ptr<cv::FaceDetectorYN> yunet_detector_;
    bool yunet_loaded_ = false;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path> [iterations]\n";
        std::cerr << "\nExample:\n";
        std::cerr << "  " << argv[0] << " test_face.jpg 20\n";
        return 1;
    }
    
    std::string image_path = argv[1];
    int iterations = (argc > 2) ? std::stoi(argv[2]) : 10;
    
    // Load image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << "\n";
        return 1;
    }
    
    RecognitionBenchmark benchmark;
    
    // Load models
    std::cout << "Loading models...\n";
    benchmark.loadYuNetDetector("face_detection_yunet_2023mar.onnx");
    benchmark.loadDlibModels(
        "/etc/faceid/shape_predictor_5_face_landmarks.dat",
        "/etc/faceid/dlib_face_recognition_resnet_model_v1.dat"
    );
    benchmark.loadSFaceModel("face_recognition_sface_2021dec.onnx");
    
    // Run benchmark
    benchmark.runBenchmark(image, iterations);
    
    return 0;
}
