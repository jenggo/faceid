// GPU Benchmark - CPU vs Vulkan Performance Comparison
#include "cli_common.h"
#include <ncnn/gpu.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>

// Forward declarations from embedded_test_image.h
namespace faceid {
    extern const unsigned char models_face_test_single_face_jpg[];
    extern const int models_face_test_single_face_jpg_len;
}

// STB_IMAGE declaration
extern "C" {
    unsigned char *stbi_load(char const *filename, int *x, int *y, int *comp, int req_comp);
    unsigned char *stbi_load_from_memory(unsigned char const *buffer, int len, int *x, int *y, int *comp, int req_comp);
    void stbi_image_free(void *retval_from_stbi_load);
}

namespace faceid {

struct BenchResult {
    std::string name;
    std::vector<double> timings_ms;
    
    double avg() const {
        if (timings_ms.empty()) return 0.0;
        return std::accumulate(timings_ms.begin(), timings_ms.end(), 0.0) / timings_ms.size();
    }
    
    double min() const {
        return timings_ms.empty() ? 0.0 : *std::min_element(timings_ms.begin(), timings_ms.end());
    }
    
    double max() const {
        return timings_ms.empty() ? 0.0 : *std::max_element(timings_ms.begin(), timings_ms.end());
    }
    
    double median() const {
        if (timings_ms.empty()) return 0.0;
        auto sorted = timings_ms;
        std::sort(sorted.begin(), sorted.end());
        size_t mid = sorted.size() / 2;
        return (sorted.size() % 2 == 0) ? (sorted[mid-1] + sorted[mid]) / 2.0 : sorted[mid];
    }
    
    double stddev() const {
        if (timings_ms.size() < 2) return 0.0;
        double mean = avg();
        double sq_sum = 0.0;
        for (double t : timings_ms) sq_sum += (t - mean) * (t - mean);
        return std::sqrt(sq_sum / timings_ms.size());
    }
    
    double fps() const { return avg() > 0 ? 1000.0 / avg() : 0.0; }
};

// Run benchmark
BenchResult runBenchmark(FaceDetector& detector, const ImageView& frame, 
                         const std::string& name, int warmup, int iterations) {
    BenchResult result;
    result.name = name;
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        Image proc = detector.preprocessFrame(frame);
        auto faces = detector.detectFaces(proc.view());
        if (!faces.empty()) detector.encodeFaces(proc.view(), faces);
    }
    
    // Benchmark
    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        Image proc = detector.preprocessFrame(frame);
        auto faces = detector.detectFaces(proc.view());
        if (!faces.empty()) detector.encodeFaces(proc.view(), faces);
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        result.timings_ms.push_back(elapsed);
    }
    
    return result;
}

int cmd_bench_gpu(const std::string& models_dir, bool verbose, const std::string& custom_image) {
    std::cout << "=== FaceID GPU Benchmark ===" << std::endl;
    std::cout << "CPU vs Vulkan Performance Comparison\n" << std::endl;
    
    // Check GPU
    int gpu_count = ncnn::get_gpu_count();
    if (gpu_count == 0) {
        std::cout << "No Vulkan GPU available. Cannot run GPU benchmark." << std::endl;
        return 1;
    }
    
    std::cout << "Found " << gpu_count << " Vulkan device(s):" << std::endl;
    ncnn::create_gpu_instance();
    for (int i = 0; i < gpu_count; i++) {
        const ncnn::GpuInfo& info = ncnn::get_gpu_info(i);
        std::cout << "  GPU " << i << ": " << info.device_name() << std::endl;
        std::cout << "    FP16: " << (info.support_fp16_packed() ? "Yes" : "No")
                  << ", INT8: " << (info.support_int8_packed() ? "Yes" : "No") << std::endl;
    }
    ncnn::destroy_gpu_instance();
    std::cout << std::endl;
    
    // Load test image
    std::cout << "Loading test image..." << std::endl;
    int img_w, img_h, channels;
    unsigned char* img_data = nullptr;
    
    if (!custom_image.empty()) {
        img_data = stbi_load(custom_image.c_str(), &img_w, &img_h, &channels, 3);
        if (img_data) std::cout << "Loaded: " << custom_image << " (" << img_w << "x" << img_h << ")" << std::endl;
    }
    
    if (!img_data) {
        std::string test_path = models_dir + "/face-test/single-face.jpg";
        img_data = stbi_load(test_path.c_str(), &img_w, &img_h, &channels, 3);
        if (img_data) std::cout << "Loaded: " << test_path << " (" << img_w << "x" << img_h << ")" << std::endl;
    }
    
    if (!img_data) {
        std::cout << "Using embedded test image" << std::endl;
        img_data = stbi_load_from_memory(models_face_test_single_face_jpg,
                                         models_face_test_single_face_jpg_len,
                                         &img_w, &img_h, &channels, 3);
    }
    
    if (!img_data) {
        std::cerr << "Failed to load test image" << std::endl;
        return 1;
    }
    
    Image test_frame(img_w, img_h, 3);
    memcpy(test_frame.data(), img_data, img_w * img_h * 3);
    stbi_image_free(img_data);
    std::cout << std::endl;
    
    // Find model
    std::string model_path;
    if (!models_dir.empty() && models_dir != ".") {
        DIR* dir = opendir(models_dir.c_str());
        if (dir) {
            struct dirent* entry;
            while ((entry = readdir(dir)) != nullptr) {
                std::string filename = entry->d_name;
                if (filename.find(".param") != std::string::npos) {
                    model_path = models_dir + "/" + filename;
                    size_t ext_pos = model_path.rfind(".param");
                    if (ext_pos != std::string::npos) {
                        model_path = model_path.substr(0, ext_pos);
                    }
                    break;
                }
            }
            closedir(dir);
        }
    }
    
    std::vector<BenchResult> results;
    int warmup = 10;
    int iterations = 50;
    
    // Initialize GPU for all Vulkan tests
    ncnn::create_gpu_instance();
    
    std::cout << "=== Benchmarking (Warmup: " << warmup << ", Iterations: " << iterations << ") ===" << std::endl;
    std::cout << std::endl;
    
    // CPU Benchmark
    std::cout << "1. CPU (4 threads)" << std::endl;
    {
        FaceDetector detector;
        detector.disableVulkan();  // Ensure CPU mode
        detector.loadModels(model_path);
        auto result = runBenchmark(detector, test_frame.view(), "CPU (4 threads)", warmup, iterations);
        std::cout << "   Average: " << std::fixed << std::setprecision(2) 
                  << result.avg() << " ms (" << result.fps() << " fps)" << std::endl;
        results.push_back(result);
    }
    std::cout << std::endl;
    
    // Vulkan FP16 Benchmark
    std::cout << "2. Vulkan GPU (FP16)" << std::endl;
    {
        FaceDetector detector;
        detector.enableVulkan(true);  // FP16 mode
        detector.loadModels(model_path);
        auto result = runBenchmark(detector, test_frame.view(), "Vulkan (FP16)", warmup, iterations);
        std::cout << "   Average: " << std::fixed << std::setprecision(2) 
                  << result.avg() << " ms (" << result.fps() << " fps)" << std::endl;
        results.push_back(result);
    }
    std::cout << std::endl;
    
    // Vulkan FP32 Benchmark
    std::cout << "3. Vulkan GPU (FP32)" << std::endl;
    {
        FaceDetector detector;
        detector.enableVulkan(false);  // FP32 mode
        detector.loadModels(model_path);
        auto result = runBenchmark(detector, test_frame.view(), "Vulkan (FP32)", warmup, iterations);
        std::cout << "   Average: " << std::fixed << std::setprecision(2) 
                  << result.avg() << " ms (" << result.fps() << " fps)" << std::endl;
        results.push_back(result);
    }
    std::cout << std::endl;
    
    ncnn::destroy_gpu_instance();
    
    // Results table
    std::cout << "=== RESULTS ===" << std::endl;
    std::cout << std::endl;
    std::cout << std::setw(20) << std::left << "Backend"
              << std::setw(10) << "Min (ms)"
              << std::setw(10) << "Avg (ms)"
              << std::setw(12) << "Median (ms)"
              << std::setw(10) << "Max (ms)"
              << std::setw(10) << "Stddev"
              << std::setw(8) << "FPS" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    for (const auto& r : results) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::setw(20) << std::left << r.name
                  << std::setw(10) << r.min()
                  << std::setw(10) << r.avg()
                  << std::setw(12) << r.median()
                  << std::setw(10) << r.max()
                  << std::setw(10) << r.stddev()
                  << std::setw(8) << std::setprecision(1) << r.fps() << std::endl;
    }
    std::cout << std::endl;
    
    // Speedup
    if (results.size() >= 2) {
        double cpu_avg = results[0].avg();
        std::cout << "Speedup vs CPU:" << std::endl;
        for (size_t i = 1; i < results.size(); i++) {
            double speedup = cpu_avg / results[i].avg();
            double improvement = (cpu_avg - results[i].avg()) / cpu_avg * 100.0;
            std::cout << "  " << results[i].name << ": "
                      << std::fixed << std::setprecision(2) << speedup << "x"
                      << " (" << std::showpos << improvement << std::noshowpos << "%)" << std::endl;
        }
        std::cout << std::endl;
    }
    
    // Recommendation
    auto fastest = std::min_element(results.begin(), results.end(),
        [](const BenchResult& a, const BenchResult& b) { return a.avg() < b.avg(); });
    
    std::cout << "Fastest: " << fastest->name << " - "
              << std::fixed << std::setprecision(2) << fastest->avg() << " ms ("
              << fastest->fps() << " fps)" << std::endl;
    std::cout << std::endl;
    std::cout << "Note: First inference includes shader compilation (~50-100ms overhead)" << std::endl;
    std::cout << "      This benchmark includes warmup to measure real-world cached performance" << std::endl;
    
    return 0;
}

} // namespace faceid
