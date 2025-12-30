#include "cli_common.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include <dirent.h>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <sys/stat.h>
#include <unistd.h>

// STB_IMAGE_IMPLEMENTATION is defined in cmd_test_image.cpp
// We just need to declare the functions we use
extern "C" {
    unsigned char *stbi_load(char const *filename, int *x, int *y, int *comp, int req_comp);
    void stbi_image_free(void *retval_from_stbi_load);
}

namespace faceid {

struct ModelBenchmark {
    std::string name;
    std::string param_path;
    std::string bin_path;
    size_t dimension;
    size_t file_size_mb;
    double detection_time_ms;
    double encoding_time_ms;
    double total_time_ms;
    double fps;
    int detection_count;
    float optimal_threshold;  // Auto-detected optimal similarity threshold
    bool success;
};

struct DetectionModelBenchmark {
    std::string name;
    std::string param_path;
    std::string bin_path;
    size_t file_size_kb;
    double detection_time_ms;
    double fps;
    int detection_count;
    float optimal_confidence;  // Auto-detected optimal confidence
    bool success;
};

struct CombinationBenchmark {
    std::string detection_name;
    std::string recognition_name;
    double detection_time_ms;
    double encoding_time_ms;
    double total_time_ms;
    double fps;
    bool success;
};

// Parse model output dimension
static size_t parseModelDimension(const std::string& param_path) {
    std::ifstream file(param_path);
    if (!file.is_open()) return 0;
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("InnerProduct") != std::string::npos && line.find("out0") != std::string::npos) {
            size_t pos = line.find("0=");
            if (pos != std::string::npos) {
                std::string dim_str;
                pos += 2;
                while (pos < line.length() && isdigit(line[pos])) {
                    dim_str += line[pos++];
                }
                return std::stoull(dim_str);
            }
        }
    }
    return 0;
}

// Get file size in MB
static size_t getFileSizeMB(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        return st.st_size / (1024 * 1024);
    }
    return 0;
}

// Get file size in KB
static size_t getFileSizeKB(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        return st.st_size / 1024;
    }
    return 0;
}

// Format milliseconds with 1 decimal place
static std::string formatMs(double ms) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << ms;
    return oss.str();
}

// Truncate string to max_width with "..." if too long
static std::string truncate(const std::string& str, size_t max_width) {
    if (str.length() <= max_width) {
        return str;
    }
    if (max_width <= 3) {
        return str.substr(0, max_width);
    }
    return str.substr(0, max_width - 3) + "...";
}

// Find optimal detection confidence threshold for a model
// Returns -1.0 if no suitable threshold found
static float findOptimalConfidence(FaceDetector& detector, const ImageView& processed_frame, 
                                   int img_width, int img_height, bool show_progress = false) {
    // Helper to check if face is valid (reasonable size and position)
    auto isValidFace = [](const Rect& face, int img_w, int img_h) -> bool {
        // Check if face is reasonably sized (not too small or too large)
        float face_area_percent = (face.width * face.height * 100.0f) / (img_w * img_h);
        if (face_area_percent < 1.0f || face_area_percent > 80.0f) return false;
        
        // Check aspect ratio (faces are roughly 0.7-1.3 ratio)
        float aspect_ratio = static_cast<float>(face.width) / face.height;
        if (aspect_ratio < 0.5f || aspect_ratio > 2.0f) return false;
        
        // Check if face is within frame bounds
        if (face.x < 0 || face.y < 0 || 
            face.x + face.width > img_w || face.y + face.height > img_h) return false;
        
        return true;
    };
    
    // Helper to count valid faces at a given confidence
    auto countValidFaces = [&](float conf) -> int {
        auto faces = detector.detectFaces(processed_frame, false, conf);
        int valid_count = 0;
        for (const auto& face : faces) {
            if (isValidFace(face, img_width, img_height)) {
                valid_count++;
            }
        }
        return valid_count;
    };
    
    // Binary search for optimal confidence threshold
    // Start with a lower bound to support models like YuNet that multiply scores
    float low = 0.02f;
    float high = 0.95f;
    float optimal_confidence = -1.0f;
    int target_face_count = 1;  // We want exactly 1 valid face
    
    // First, check if we can detect any faces at low threshold
    if (countValidFaces(low) == 0) {
        if (show_progress) {
            std::cout << "  No faces detected even at threshold=" << std::fixed << std::setprecision(2) << low << std::endl;
        }
        return -1.0f;  // Cannot detect any faces
    }
    
    // Binary search to find highest threshold that still detects target_face_count faces
    while (high - low > 0.01f) {
        float mid = (low + high) / 2.0f;
        int face_count = countValidFaces(mid);
        
        if (face_count >= target_face_count) {
            optimal_confidence = mid;
            low = mid;
        } else {
            high = mid;
        }
    }
    
    if (show_progress && optimal_confidence > 0) {
        std::cout << "  Auto-detected optimal confidence: " << std::fixed << std::setprecision(2) 
                  << optimal_confidence << std::endl;
    }
    
    return optimal_confidence;
}

// Find optimal recognition similarity threshold for a model
// Returns -1.0 if encoding fails, otherwise returns a model-appropriate threshold
static float findOptimalRecognitionThreshold(FaceDetector& detector, const ImageView& processed_frame, 
                                             bool show_progress = false) {
    // Check if we can generate encodings successfully
    auto faces = detector.detectFaces(processed_frame);
    if (faces.empty()) {
        if (show_progress) {
            std::cout << "  No faces detected for threshold test" << std::endl;
        }
        return -1.0f;
    }
    
    if (show_progress) {
        std::cout << "  Detected " << faces.size() << " face(s), attempting encoding..." << std::endl;
    }
    
    auto encodings = detector.encodeFaces(processed_frame, faces);
    if (encodings.empty()) {
        if (show_progress) {
            std::cout << "  Failed to generate encodings (encodeFaces returned empty)" << std::endl;
        }
        return -1.0f;
    }
    
    // Return a reasonable default threshold
    // Note: Actual optimal threshold would require a reference face to compare against,
    // which we don't have in the benchmark context. The threshold is used during
    // face matching (similarity comparison), not during encoding.
    float optimal_threshold = 0.40f;  // Standard middle-ground threshold
    
    if (show_progress) {
        std::cout << "  Encoding successful (" << encodings.size() << " encoding(s) generated)" << std::endl;
        std::cout << "  Using standard threshold: " << std::fixed << std::setprecision(2) 
                  << optimal_threshold << std::endl;
    }
    
    return optimal_threshold;
}

int cmd_bench(const std::string& test_dir, bool show_detail) {
    std::cout << "=== FaceID Model Benchmark ===" << std::endl;
    std::cout << "Scanning directory: " << test_dir << "\n" << std::endl;
    
    // Scan for all models
    std::vector<ModelBenchmark> recognition_models;
    std::vector<DetectionModelBenchmark> detection_models;
    
    DIR* dir = opendir(test_dir.c_str());
    if (!dir) {
        std::cerr << "Error: Cannot open directory: " << test_dir << std::endl;
        std::cerr << "\nUsage: faceid bench <model_directory>" << std::endl;
        std::cerr << "Example: faceid bench /tmp/models" << std::endl;
        return 1;
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        
        // Skip if not a param file
        if (filename.find(".param") == std::string::npos) continue;
        
        std::string param_path = test_dir + "/" + filename;
        std::string bin_path;
        std::string base_name;
        
        if (filename.find(".ncnn.param") != std::string::npos) {
            // Model uses .ncnn.param extension
            base_name = filename.substr(0, filename.length() - 11);  // Remove ".ncnn.param"
            bin_path = test_dir + "/" + base_name + ".ncnn.bin";
        } else if (filename.find(".param") != std::string::npos) {
            // Standard .param extension
            base_name = filename.substr(0, filename.length() - 6);  // Remove ".param"
            bin_path = test_dir + "/" + base_name + ".bin";
        } else {
            continue;
        }
        
        // Check if .bin exists
        std::ifstream bin_check(bin_path);
        if (!bin_check.good()) continue;
        
        size_t dimension = parseModelDimension(param_path);
        
        if (dimension >= 64 && dimension <= 2048) {
            // Recognition model
            ModelBenchmark bench;
            bench.name = base_name;
            bench.param_path = param_path;
            bench.bin_path = bin_path;
            bench.dimension = dimension;
            bench.file_size_mb = getFileSizeMB(bin_path);
            bench.success = false;
            recognition_models.push_back(bench);
            std::cout << "  Found recognition: " << base_name << " (" << dimension << "D, " << bench.file_size_mb << " MB)" << std::endl;
        } else if (dimension == 0) {
            // Detection model
            DetectionModelBenchmark bench;
            bench.name = base_name;
            bench.param_path = param_path;
            bench.bin_path = bin_path;
            bench.file_size_kb = getFileSizeKB(bin_path);
            bench.success = false;
            detection_models.push_back(bench);
            std::cout << "  Found detection: " << base_name << " (" << bench.file_size_kb << " KB)" << std::endl;
        }
    }
    closedir(dir);
    
    std::cout << "\nFound " << detection_models.size() << " detection model(s)" << std::endl;
    std::cout << "Found " << recognition_models.size() << " recognition model(s)" << std::endl;
    std::cout << std::endl;
    
    if (detection_models.empty() && recognition_models.empty()) {
        std::cerr << "No valid models found!" << std::endl;
        std::cerr << "\nPlace .ncnn.param and .ncnn.bin files (or .param/.bin) in: " << test_dir << std::endl;
        return 1;
    }
    
    // Sort by name for consistent output
    std::sort(recognition_models.begin(), recognition_models.end(),
              [](const ModelBenchmark& a, const ModelBenchmark& b) {
                  return a.name < b.name;
              });
    std::sort(detection_models.begin(), detection_models.end(),
              [](const DetectionModelBenchmark& a, const DetectionModelBenchmark& b) {
                  return a.name < b.name;
              });
    
    // Initialize camera
    std::cout << "\n=== Loading Test Image ===" << std::endl;
    
    // Try to load static test image first
    std::string test_image_path = test_dir + "/face-test/single-face.jpg";
    Image test_frame;
    bool using_static_image = false;
    
    // Try to load with stb_image
    int img_w, img_h, channels;
    unsigned char* img_data = stbi_load(test_image_path.c_str(), &img_w, &img_h, &channels, 3);
    
    if (img_data) {
        std::cout << "Loaded test image: " << test_image_path << " (" << img_w << "x" << img_h << ")" << std::endl;
        test_frame = Image(img_w, img_h, 3);
        memcpy(test_frame.data(), img_data, img_w * img_h * 3);
        stbi_image_free(img_data);
        using_static_image = true;
    } else {
        std::cout << "Static test image not found, using camera..." << std::endl;
        
        // Fall back to camera capture
        // Get camera settings from config
        Config& config = Config::getInstance();
        auto device = config.getString("camera", "device").value_or("/dev/video0");
        int width = config.getInt("camera", "width").value_or(640);
        int height = config.getInt("camera", "height").value_or(480);
        
        std::cout << "Initializing camera: " << device << " (" << width << "x" << height << ")" << std::endl;
        Camera camera(device);
        if (!camera.open(width, height)) {
            std::cerr << "Error: Failed to open camera" << std::endl;
            std::cerr << "Please ensure your camera is connected or provide test image at:" << std::endl;
            std::cerr << "  " << test_image_path << std::endl;
            return 1;
        }
        
        // Capture test frame ONCE - will be reused for all benchmarks
        if (!camera.read(test_frame)) {
            std::cerr << "Error: Failed to capture frame" << std::endl;
            return 1;
        }
        camera.close();
    }
    
    std::cout << "Test frame: " << test_frame.width() << "x" << test_frame.height() 
              << " channels=" << test_frame.channels() << std::endl;
    
    // Save test frame for debugging
    if (show_detail) {
        std::string debug_path = "/tmp/faceid_bench_frame.jpg";
        // Simple PPM save for debugging (can be converted with: convert frame.ppm frame.jpg)
        std::ofstream ofs(debug_path.c_str(), std::ios::binary);
        if (ofs) {
            ofs << "P6\n" << test_frame.width() << " " << test_frame.height() << "\n255\n";
            // Convert BGR to RGB for PPM
            for (int i = 0; i < test_frame.width() * test_frame.height(); i++) {
                ofs << test_frame.data()[i*3 + 2];  // R (from B)
                ofs << test_frame.data()[i*3 + 1];  // G
                ofs << test_frame.data()[i*3 + 0];  // B (from R)
            }
            ofs.close();
            std::cout << "Debug: Saved test frame to " << debug_path << " (PPM format)" << std::endl;
        }
    }
    
    // Test if face can be detected with installed detection model
    {
        FaceDetector test_detector;
        if (!test_detector.loadModels()) {
            std::cerr << "Warning: Could not load installed detection model" << std::endl;
        } else {
            Image processed = test_detector.preprocessFrame(test_frame.view());
            auto test_faces = test_detector.detectFaces(processed.view());
            if (test_faces.empty()) {
                std::cerr << "\nWARNING: No faces detected in captured frame with installed model!" << std::endl;
                std::cerr << "This usually means:" << std::endl;
                std::cerr << "  1. No face visible in camera view" << std::endl;
                std::cerr << "  2. Frame is too dark" << std::endl;
                std::cerr << "  3. Detection threshold is too high" << std::endl;
                std::cerr << "\nBenchmark will likely show all failures.\n" << std::endl;
            } else {
                std::cout << "Pre-check: Detected " << test_faces.size() << " face(s) with installed model (" 
                          << test_detector.getDetectionModelType() << ")" << std::endl;
            }
        }
    }
    
    std::cout << "Using " << (using_static_image ? "static test image" : "captured frame") 
              << " for all benchmarks (more consistent results)\n" << std::endl;
    
    // Benchmark detection models first
    if (!detection_models.empty()) {
        if (show_detail) {
            std::cout << "=== Benchmarking Detection Models ===" << std::endl;
            std::cout << "Running 20 iterations per model (with 5-iteration warmup)...\n" << std::endl;
        } else {
            std::cout << "Benchmarking detection models..." << std::endl;
        }
        
        for (auto& model : detection_models) {
            if (show_detail) {
                std::cout << "Testing: " << model.name << " (" << model.file_size_kb << " KB)" << std::endl;
            }
            
            FaceDetector detector;
            
            // Get base path without extension (loadModels appends .param/.bin)
            std::string base_path = model.param_path;
            size_t ext_pos = base_path.rfind(".param");
            if (ext_pos != std::string::npos) {
                base_path = base_path.substr(0, ext_pos);
            }
            
            // Load with empty recognition model path, but explicit detection model path
            if (!detector.loadModels("", base_path)) {
                if (show_detail) {
                    std::cout << "  ✗ Failed to load model" << std::endl;
                }
                continue;
            }
            
            if (show_detail) {
                std::cout << "  Model loaded: " << detector.getDetectionModelType() << std::endl;
            }
            
            // Preprocess frame once
            Image processed = detector.preprocessFrame(test_frame.view());
            
            // Auto-detect optimal confidence threshold
            float optimal_conf = findOptimalConfidence(detector, processed.view(), 
                                                       test_frame.width(), test_frame.height(), 
                                                       show_detail);
            
            if (optimal_conf < 0) {
                if (show_detail) {
                    std::cout << "  ✗ Could not find optimal confidence (no faces detected)" << std::endl;
                }
                model.optimal_confidence = -1.0f;
                continue;
            }
            
            model.optimal_confidence = optimal_conf;
            
            if (show_detail) {
                std::cout << "  Using confidence: " << std::fixed << std::setprecision(2) << optimal_conf << std::endl;
            }
            
            // Warmup (5 iterations with optimal confidence)
            for (int i = 0; i < 5; i++) {
                Image proc = detector.preprocessFrame(test_frame.view());
                detector.detectFaces(proc.view(), false, optimal_conf);
            }
            
            // Benchmark (20 iterations) - measure full detection pipeline with optimal confidence
            int total_detections = 0;
            double total_detect_time = 0.0;
            
            for (int i = 0; i < 20; i++) {
                auto detect_start = std::chrono::high_resolution_clock::now();
                Image proc = detector.preprocessFrame(test_frame.view());
                auto faces = detector.detectFaces(proc.view(), false, optimal_conf);
                auto detect_end = std::chrono::high_resolution_clock::now();
                
                double detect_time = std::chrono::duration<double, std::milli>(detect_end - detect_start).count();
                total_detect_time += detect_time;
                total_detections += faces.size();
            }
            
            model.detection_time_ms = total_detect_time / 20.0;
            model.fps = (model.detection_time_ms > 0) ? 1000.0 / model.detection_time_ms : 0.0;
            model.detection_count = total_detections / 20;
            model.success = (total_detections > 0);
            
            if (show_detail) {
                std::cout << "  Detection:   " << std::fixed << std::setprecision(1) << model.detection_time_ms << " ms" << std::endl;
                std::cout << "  FPS:         " << std::fixed << std::setprecision(1) << model.fps << std::endl;
                std::cout << "  Faces/frame: " << model.detection_count << std::endl;
                std::cout << "  ✓ Success\n" << std::endl;
            }
            
        }
        
        if (!show_detail) {
            std::cout << "Done.\n" << std::endl;
        }
    }
    
    // Benchmark recognition models
    if (!recognition_models.empty()) {
        if (show_detail) {
            std::cout << "=== Benchmarking Recognition Models ===" << std::endl;
            std::cout << "Running 10 iterations per model (with 5-iteration warmup)...\n" << std::endl;
        } else {
            std::cout << "Benchmarking recognition models..." << std::endl;
        }
        
        // Load default detection model first (needed for face detection)
        FaceDetector temp_detector;
        if (!temp_detector.loadModels()) {
            std::cerr << "Error: Failed to load default detection model" << std::endl;
            std::cerr << "Please ensure detection model is available in: " << MODELS_DIR << std::endl;
            return 1;
        }
        
        for (auto& model : recognition_models) {
            if (show_detail) {
                std::cout << "Testing: " << model.name << " (" << model.dimension << "D, " << model.file_size_mb << " MB)" << std::endl;
            }
            
            FaceDetector detector;
            
            // Get base path without extension (loadModels appends .param/.bin or .ncnn.param/.ncnn.bin)
            std::string base_path = model.param_path;
            size_t ext_pos = base_path.rfind(".ncnn.param");
            if (ext_pos == std::string::npos) {
                ext_pos = base_path.rfind(".param");
            }
            if (ext_pos != std::string::npos) {
                base_path = base_path.substr(0, ext_pos);
            }
            
            if (!detector.loadModels(base_path)) {
                if (show_detail) {
                    std::cout << "  ✗ Failed to load model" << std::endl;
                }
                continue;
            }
            
            if (show_detail) {
                std::cout << "  Model loaded successfully: " << base_path << std::endl;
            }
            
            // Preprocess frame
            Image processed = detector.preprocessFrame(test_frame.view());
            
            // Check if faces are detected
            auto test_faces = detector.detectFaces(processed.view());
            if (test_faces.empty()) {
                if (show_detail) {
                    std::cout << "  ✗ No faces detected in test frame" << std::endl;
                    std::cout << "     Please position your face in front of the camera and try again." << std::endl;
                }
                continue;
            }
            if (show_detail) {
                std::cout << "  Detected " << test_faces.size() << " face(s)" << std::endl;
            }
            
            // Auto-detect optimal similarity threshold
            float optimal_threshold = findOptimalRecognitionThreshold(detector, processed.view(), show_detail);
            
            if (optimal_threshold < 0) {
                if (show_detail) {
                    std::cout << "  ✗ Could not find optimal threshold (recognition failed)" << std::endl;
                }
                model.optimal_threshold = -1.0f;
                continue;
            }
            
            model.optimal_threshold = optimal_threshold;
            
            if (show_detail) {
                std::cout << "  Using threshold: " << std::fixed << std::setprecision(2) << optimal_threshold << std::endl;
            }
            
            // Warmup (5 frames)
            for (int i = 0; i < 5; i++) {
                auto faces = detector.detectFaces(processed.view());
                if (!faces.empty()) {
                    detector.encodeFaces(processed.view(), faces);
                }
            }
            
            // Benchmark (10 frames)
            int total_detections = 0;
            double total_detect_time = 0.0;
            double total_encode_time = 0.0;
            
            for (int i = 0; i < 10; i++) {
                auto detect_start = std::chrono::high_resolution_clock::now();
                auto faces = detector.detectFaces(processed.view());
                auto detect_end = std::chrono::high_resolution_clock::now();
                
                double detect_time = std::chrono::duration<double, std::milli>(detect_end - detect_start).count();
                total_detect_time += detect_time;
                
                if (!faces.empty()) {
                    total_detections += faces.size();
                    
                    auto encode_start = std::chrono::high_resolution_clock::now();
                    auto encodings = detector.encodeFaces(processed.view(), faces);
                    auto encode_end = std::chrono::high_resolution_clock::now();
                    
                    double encode_time = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();
                    total_encode_time += encode_time;
                }
            }
            
            model.detection_time_ms = total_detect_time / 10.0;
            model.encoding_time_ms = total_detections > 0 ? total_encode_time / 10.0 : 0.0;
            model.total_time_ms = model.encoding_time_ms;
            model.fps = (model.total_time_ms > 0) ? 1000.0 / model.total_time_ms : 0.0;
            model.detection_count = total_detections / 10;
            model.success = (total_detections > 0);
            
            if (show_detail) {
                std::cout << "  Encoding:   " << std::fixed << std::setprecision(1) << model.encoding_time_ms << " ms" << std::endl;
                std::cout << "  FPS:        " << std::fixed << std::setprecision(1) << model.fps << std::endl;
                std::cout << "  ✓ Success\n" << std::endl;
            }
            
        }
        
        if (!show_detail) {
            std::cout << "Done.\n" << std::endl;
        }
    }
    
    // Benchmark all detection + recognition combinations
    std::vector<CombinationBenchmark> combinations;
    
    if (!detection_models.empty() && !recognition_models.empty()) {
        if (show_detail) {
            std::cout << "=== Benchmarking Detection + Recognition Combinations ===" << std::endl;
            std::cout << "Testing all combinations (5 iterations per pair)..." << std::endl;
            std::cout << "This shows complete pipeline performance.\n" << std::endl;
        } else {
            std::cout << "Benchmarking model combinations..." << std::endl;
        }
        
        for (auto& det_model : detection_models) {
            if (!det_model.success) continue;
            
            for (auto& rec_model : recognition_models) {
                if (!rec_model.success) continue;
                
                if (show_detail) {
                    std::cout << "Testing: " << det_model.name << " + " << rec_model.name << std::endl;
                }
                
                // Load detector with both models directly from their paths
                FaceDetector detector;
                
                // Get base paths without extensions
                // Detection models: remove .param or .ncnn.param
                std::string det_base = det_model.param_path;
                if (det_base.rfind(".ncnn.param") == det_base.length() - 11) {
                    det_base = det_base.substr(0, det_base.length() - 11);
                } else if (det_base.rfind(".param") == det_base.length() - 6) {
                    det_base = det_base.substr(0, det_base.length() - 6);
                }
                
                // Recognition models: remove .param or .ncnn.param
                std::string rec_base = rec_model.param_path;
                if (rec_base.rfind(".ncnn.param") == rec_base.length() - 11) {
                    rec_base = rec_base.substr(0, rec_base.length() - 11);
                } else if (rec_base.rfind(".param") == rec_base.length() - 6) {
                    rec_base = rec_base.substr(0, rec_base.length() - 6);
                }
                
                if (!detector.loadModels(rec_base, det_base)) {
                    if (show_detail) {
                        std::cout << "  ✗ Failed to load models" << std::endl;
                    }
                    continue;
                }
                
                // Warmup (2 iterations with same frame)
                for (int i = 0; i < 2; i++) {
                    Image proc = detector.preprocessFrame(test_frame.view());
                    auto faces = detector.detectFaces(proc.view());
                    if (!faces.empty()) {
                        detector.encodeFaces(proc.view(), faces);
                    }
                }
                
                // Benchmark (5 iterations)
                double total_detect_time = 0.0;
                double total_encode_time = 0.0;
                int successful_iterations = 0;
                
                for (int i = 0; i < 5; i++) {
                    // Measure detection (preprocessing + face detection)
                    auto detect_start = std::chrono::high_resolution_clock::now();
                    Image proc = detector.preprocessFrame(test_frame.view());
                    auto faces = detector.detectFaces(proc.view());
                    auto detect_end = std::chrono::high_resolution_clock::now();
                    
                    if (faces.empty()) continue;
                    
                    double detect_time = std::chrono::duration<double, std::milli>(detect_end - detect_start).count();
                    total_detect_time += detect_time;
                    
                    // Measure encoding
                    auto encode_start = std::chrono::high_resolution_clock::now();
                    auto encodings = detector.encodeFaces(proc.view(), faces);
                    auto encode_end = std::chrono::high_resolution_clock::now();
                    
                    double encode_time = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();
                    total_encode_time += encode_time;
                    successful_iterations++;
                }
                
                if (successful_iterations == 0) {
                    if (show_detail) {
                        std::cout << "  ✗ No faces detected" << std::endl;
                    }
                    continue;
                }
                
                CombinationBenchmark combo;
                combo.detection_name = det_model.name;
                combo.recognition_name = rec_model.name;
                combo.detection_time_ms = total_detect_time / successful_iterations;
                combo.encoding_time_ms = total_encode_time / successful_iterations;
                combo.total_time_ms = combo.detection_time_ms + combo.encoding_time_ms;
                combo.fps = (combo.total_time_ms > 0) ? 1000.0 / combo.total_time_ms : 0.0;
                combo.success = true;
                combinations.push_back(combo);
                
                if (show_detail) {
                    std::cout << "  Detection: " << std::fixed << std::setprecision(1) << combo.detection_time_ms << " ms" << std::endl;
                    std::cout << "  Encoding:  " << std::fixed << std::setprecision(1) << combo.encoding_time_ms << " ms" << std::endl;
                    std::cout << "  Total:     " << std::fixed << std::setprecision(1) << combo.total_time_ms << " ms" << std::endl;
                    std::cout << "  FPS:       " << std::fixed << std::setprecision(1) << combo.fps << std::endl;
                    std::cout << "  ✓ Success\n" << std::endl;
                }
            }
        }
        
        if (!show_detail) {
            std::cout << "Done.\n" << std::endl;
        } else {
            std::cout << std::endl;
        }
    }
    
    // Summary
    std::cout << "=== BENCHMARK SUMMARY ===" << std::endl;
    std::cout << std::endl;
    
    if (!detection_models.empty()) {
        std::cout << "Detection Models Performance:" << std::endl;
        std::cout << std::endl;
        std::cout << std::setw(30) << std::left << "Model" 
                  << std::setw(10) << "Size (KB)" 
                  << std::setw(10) << "Conf"
                  << std::setw(15) << "Detection" 
                  << std::setw(10) << "FPS" << std::endl;
        std::cout << std::string(75, '-') << std::endl;
        
        for (const auto& model : detection_models) {
            if (model.success) {
                std::cout << std::setw(30) << std::left << truncate(model.name, 30)
                          << std::setw(10) << model.file_size_kb
                          << std::setw(10) << (std::to_string(static_cast<int>(model.optimal_confidence * 100)) + "%")
                          << std::setw(15) << (formatMs(model.detection_time_ms) + " ms")
                          << std::setw(10) << (std::to_string(static_cast<int>(model.fps)) + " fps")
                          << std::endl;
            }
        }
        
        std::cout << std::endl;
        
        // Find fastest detection model
        auto fastest_detect = std::min_element(detection_models.begin(), detection_models.end(),
            [](const DetectionModelBenchmark& a, const DetectionModelBenchmark& b) {
                return a.success && (!b.success || a.detection_time_ms < b.detection_time_ms);
            });
        
        // Find smallest detection model
        auto smallest_detect = std::min_element(detection_models.begin(), detection_models.end(),
            [](const DetectionModelBenchmark& a, const DetectionModelBenchmark& b) {
                return a.success && (!b.success || a.file_size_kb < b.file_size_kb);
            });
        
        std::cout << "Detection Model Recommendations:" << std::endl;
        if (fastest_detect != detection_models.end() && fastest_detect->success) {
            std::cout << "  • Fastest:  " << fastest_detect->name << " (" 
                      << std::fixed << std::setprecision(1) << fastest_detect->detection_time_ms << " ms, " 
                      << static_cast<int>(fastest_detect->fps) << " fps)" << std::endl;
        }
        
        if (smallest_detect != detection_models.end() && smallest_detect->success) {
            std::cout << "  • Smallest: " << smallest_detect->name << " (" 
                      << smallest_detect->file_size_kb << " KB)" << std::endl;
        }
        
        std::cout << std::endl;
    }
    
    if (!recognition_models.empty()) {
        std::cout << "Recognition Models Performance:" << std::endl;
        std::cout << std::endl;
        std::cout << std::setw(45) << std::left << "Model" 
                  << std::setw(10) << "Dimension" 
                  << std::setw(10) << "Size (MB)" 
                  << std::setw(10) << "Thresh"
                  << std::setw(12) << "Encoding" 
                  << std::setw(12) << "Total" 
                  << std::setw(10) << "FPS" << std::endl;
        std::cout << std::string(109, '-') << std::endl;
        
        for (const auto& model : recognition_models) {
            if (model.success) {
                std::cout << std::setw(45) << std::left << truncate(model.name, 45)
                          << std::setw(10) << (std::to_string(model.dimension) + "D")
                          << std::setw(10) << model.file_size_mb
                          << std::setw(10) << (std::to_string(static_cast<int>(model.optimal_threshold * 100)) + "%")
                          << std::setw(12) << (formatMs(model.encoding_time_ms) + " ms")
                          << std::setw(12) << (formatMs(model.total_time_ms) + " ms")
                          << std::setw(10) << (std::to_string(static_cast<int>(model.fps)) + " fps")
                          << std::endl;
            }
        }
        
        std::cout << std::endl;
        
        // Recommendations
        std::cout << "Recognition Model Recommendations:" << std::endl;
        
        // Find fastest model
        auto fastest = std::min_element(recognition_models.begin(), recognition_models.end(),
            [](const ModelBenchmark& a, const ModelBenchmark& b) {
                return a.success && (!b.success || a.total_time_ms < b.total_time_ms);
            });
        
        // Find smallest model
        auto smallest = std::min_element(recognition_models.begin(), recognition_models.end(),
            [](const ModelBenchmark& a, const ModelBenchmark& b) {
                return a.success && (!b.success || a.file_size_mb < b.file_size_mb);
            });
        
        // Find highest dimension (best accuracy potential)
        auto highest_dim = std::max_element(recognition_models.begin(), recognition_models.end(),
            [](const ModelBenchmark& a, const ModelBenchmark& b) {
                return a.success && (!b.success || a.dimension < b.dimension);
            });
        
        if (fastest != recognition_models.end() && fastest->success) {
            std::cout << "  • Fastest:       " << fastest->name << " (" 
                      << std::fixed << std::setprecision(1) << fastest->total_time_ms << " ms, " 
                      << static_cast<int>(fastest->fps) << " fps)" << std::endl;
        }
        
        if (smallest != recognition_models.end() && smallest->success) {
            std::cout << "  • Smallest:      " << smallest->name << " (" 
                      << smallest->file_size_mb << " MB)" << std::endl;
        }
        
        if (highest_dim != recognition_models.end() && highest_dim->success) {
            std::cout << "  • Best Accuracy: " << highest_dim->name << " (" 
                      << highest_dim->dimension << "D, potentially more accurate)" << std::endl;
        }
        
        std::cout << std::endl;
    }
    
    if (!combinations.empty()) {
        std::cout << "Combined Pipeline Performance (Detection + Recognition):" << std::endl;
        std::cout << std::endl;
        std::cout << std::setw(28) << std::left << "Detection Model" 
                  << std::setw(40) << "Recognition Model" 
                  << std::setw(12) << "Detect" 
                  << std::setw(12) << "Encode" 
                  << std::setw(12) << "Total" 
                  << std::setw(10) << "FPS" << std::endl;
        std::cout << std::string(114, '-') << std::endl;
        
        // Sort by total time (fastest first)
        std::sort(combinations.begin(), combinations.end(),
                  [](const CombinationBenchmark& a, const CombinationBenchmark& b) {
                      return a.total_time_ms < b.total_time_ms;
                  });
        
        for (const auto& combo : combinations) {
            if (combo.success) {
                std::cout << std::setw(28) << std::left << truncate(combo.detection_name, 28)
                          << std::setw(40) << truncate(combo.recognition_name, 40)
                          << std::setw(12) << (formatMs(combo.detection_time_ms) + " ms")
                          << std::setw(12) << (formatMs(combo.encoding_time_ms) + " ms")
                          << std::setw(12) << (formatMs(combo.total_time_ms) + " ms")
                          << std::setw(10) << (std::to_string(static_cast<int>(combo.fps)) + " fps")
                          << std::endl;
            }
        }
        
        std::cout << std::endl;
        
        // Find best combinations
        if (!combinations.empty() && combinations[0].success) {
            std::cout << "Combined Recommendations:" << std::endl;
            std::cout << "  • Fastest Overall: " << combinations[0].detection_name << " + " << combinations[0].recognition_name 
                      << " (" << std::fixed << std::setprecision(1) << combinations[0].total_time_ms << " ms, " 
                      << static_cast<int>(combinations[0].fps) << " fps)" << std::endl;
            std::cout << std::endl;
        }
    }
    
    std::cout << "Benchmark complete!" << std::endl;
    
    if (!detection_models.empty()) {
        std::cout << "\nTo install a detection model:" << std::endl;
        std::cout << "  sudo cp " << test_dir << "/<model>.param /etc/faceid/models/detection.param" << std::endl;
        std::cout << "  sudo cp " << test_dir << "/<model>.bin /etc/faceid/models/detection.bin" << std::endl;
    }
    
    if (!recognition_models.empty()) {
        std::cout << "\nTo install a recognition model:" << std::endl;
        std::cout << "  sudo cp " << test_dir << "/<model>.ncnn.param /etc/faceid/models/recognition.param" << std::endl;
        std::cout << "  sudo cp " << test_dir << "/<model>.ncnn.bin /etc/faceid/models/recognition.bin" << std::endl;
        std::cout << "  sudo faceid add $(whoami)  # Re-enroll after changing models" << std::endl;
    }
    
    return 0;
}

} // namespace faceid
