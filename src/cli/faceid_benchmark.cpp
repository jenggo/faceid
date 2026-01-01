#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>
#include "../camera.h"
#include "../face_detector.h"
#include "../config.h"
#include "../models/binary_model.h"
#include "../models/model_cache.h"
#include "config_paths.h"

namespace faceid {

struct TimingStats {
    double min_ms = 999999.0;
    double max_ms = 0.0;
    double sum_ms = 0.0;
    int count = 0;
    
    void add(double ms) {
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
        sum_ms += ms;
        count++;
    }
    
    double avg() const {
        return count > 0 ? sum_ms / count : 0.0;
    }
};

static void printStatistics(const std::string& name, const TimingStats& stats) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  " << name << ":" << std::endl;
    std::cout << "    Min:     " << std::setw(8) << stats.min_ms << " ms" << std::endl;
    std::cout << "    Max:     " << std::setw(8) << stats.max_ms << " ms" << std::endl;
    std::cout << "    Average: " << std::setw(8) << stats.avg() << " ms" << std::endl;
    std::cout << "    Count:   " << std::setw(8) << stats.count << " samples" << std::endl;
}

} // namespace faceid

int main(int argc, char* argv[]) {
    using namespace faceid;
    
    // Parse command line arguments
    int num_frames = 100;  // Default number of frames to process
    int warmup_frames = 10;  // Warmup frames to skip
    std::string username = "";
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--frames" || arg == "-f") {
            if (i + 1 < argc) {
                num_frames = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: --frames requires a value" << std::endl;
                return 1;
            }
        } else if (arg == "--warmup" || arg == "-w") {
            if (i + 1 < argc) {
                warmup_frames = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: --warmup requires a value" << std::endl;
                return 1;
            }
        } else if (arg == "--user" || arg == "-u") {
            if (i + 1 < argc) {
                username = argv[++i];
            } else {
                std::cerr << "Error: --user requires a username" << std::endl;
                return 1;
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "FaceID Benchmark - Headless Performance Testing\n" << std::endl;
            std::cout << "Usage: faceid-benchmark [OPTIONS]\n" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -f, --frames N     Number of frames to process (default: 100)" << std::endl;
            std::cout << "  -w, --warmup N     Number of warmup frames to skip (default: 10)" << std::endl;
            std::cout << "  -u, --user NAME    Test against specific user (optional)" << std::endl;
            std::cout << "  -h, --help         Show this help message" << std::endl;
            std::cout << "\nExample:" << std::endl;
            std::cout << "  faceid-benchmark --frames 200 --warmup 20 --user john" << std::endl;
            return 0;
        }
    }
    
    std::cout << "=== FaceID Benchmark ===" << std::endl;
    std::cout << "Frames to process: " << num_frames << std::endl;
    std::cout << "Warmup frames:     " << warmup_frames << std::endl;
    if (!username.empty()) {
        std::cout << "Testing user:      " << username << std::endl;
    }
    std::cout << std::endl;
    
    // Load ALL users for face matching
    auto& cache = ModelCache::getInstance();
    std::vector<BinaryFaceModel> all_models = cache.loadAllUsersParallel(4);
    
    if (all_models.empty()) {
        std::cerr << "Error: No face models found for any user" << std::endl;
        std::cerr << "Run: sudo faceid add <username>" << std::endl;
        return 1;
    }
    
    std::cout << "Loaded " << all_models.size() << " enrolled user(s)" << std::endl;
    
    // Display unique usernames
    std::vector<std::string> unique_usernames;
    for (const auto& model : all_models) {
        if (std::find(unique_usernames.begin(), unique_usernames.end(), model.username) == unique_usernames.end()) {
            unique_usernames.push_back(model.username);
        }
    }
    
    std::cout << "Enrolled users: ";
    for (size_t i = 0; i < unique_usernames.size(); i++) {
        std::cout << unique_usernames[i];
        if (i < unique_usernames.size() - 1) std::cout << ", ";
    }
    std::cout << "\n" << std::endl;
    
    // Load configuration
    faceid::Config& config = faceid::Config::getInstance();
    std::string config_path = std::string(CONFIG_DIR) + "/faceid.conf";
    config.load(config_path);
    
    auto device = config.getString("camera", "device").value_or("/dev/video0");
    auto width = config.getInt("camera", "width").value_or(640);
    auto height = config.getInt("camera", "height").value_or(480);
    double threshold = config.getDouble("recognition", "threshold").value_or(0.6);
    int tracking_interval = config.getInt("face_detection", "tracking_interval").value_or(10);
    
    std::cout << "Camera: " << device << " (" << width << "x" << height << ")" << std::endl;
    std::cout << "Recognition threshold: " << threshold << std::endl;
    std::cout << "Tracking interval: " << tracking_interval << " frames" << std::endl;
    std::cout << std::endl;
    
    // Initialize camera
    Camera camera(device);
    if (!camera.open(width, height)) {
        std::cerr << "Error: Failed to open camera" << std::endl;
        return 1;
    }
    
    // Initialize face detector
    faceid::FaceDetector detector;
    
    if (!detector.loadModels()) {
        std::cerr << "Error: Failed to load face recognition model" << std::endl;
        std::cerr << "Expected files: " << MODELS_DIR << "/sface.param and sface.bin" << std::endl;
        return 1;
    }
    
    std::cout << "Camera and models initialized successfully\n" << std::endl;
    
    // Timing statistics
    TimingStats camera_stats;
    TimingStats preprocess_stats;
    TimingStats detection_stats;
    TimingStats encoding_stats;
    TimingStats matching_stats;
    TimingStats total_stats;
    
    // Recognition statistics
    int frames_with_faces = 0;
    int total_faces_detected = 0;
    int total_faces_recognized = 0;
    
    std::cout << "Starting benchmark..." << std::endl;
    std::cout << "Progress: " << std::flush;
    
    auto benchmark_start = std::chrono::steady_clock::now();
    
    for (int frame_num = 0; frame_num < warmup_frames + num_frames; frame_num++) {
        bool is_warmup = (frame_num < warmup_frames);
        
        // Progress indicator
        if (!is_warmup && (frame_num - warmup_frames) % 10 == 0) {
            std::cout << "." << std::flush;
        }
        
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        // 1. Capture frame
        auto capture_start = std::chrono::high_resolution_clock::now();
        faceid::Image frame;
        if (!camera.read(frame)) {
            std::cerr << "\nFailed to read frame from camera" << std::endl;
            continue;
        }
        auto capture_end = std::chrono::high_resolution_clock::now();
        double capture_time = std::chrono::duration<double, std::milli>(capture_end - capture_start).count();
        
        // 2. Preprocess frame
        auto preprocess_start = std::chrono::high_resolution_clock::now();
        faceid::Image processed_frame = detector.preprocessFrame(frame.view());
        auto preprocess_end = std::chrono::high_resolution_clock::now();
        double preprocess_time = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();
        
        // 3. Detect faces
        auto detect_start = std::chrono::high_resolution_clock::now();
        auto faces = detector.detectOrTrackFaces(processed_frame.view(), tracking_interval);
        auto detect_end = std::chrono::high_resolution_clock::now();
        double detection_time = std::chrono::duration<double, std::milli>(detect_end - detect_start).count();
        
        double encoding_time = 0.0;
        double matching_time = 0.0;
        
        // 4. Process detected faces
        if (!faces.empty()) {
            if (!is_warmup) {
                frames_with_faces++;
                total_faces_detected += faces.size();
            }
            
            // 5. Encode faces
            auto encode_start = std::chrono::high_resolution_clock::now();
            auto encodings = detector.encodeFaces(processed_frame.view(), faces);
            auto encode_end = std::chrono::high_resolution_clock::now();
            encoding_time = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();
            
            // 6. Match faces
            auto match_start = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < encodings.size() && i < faces.size(); i++) {
                double best_distance = 999.0;
                std::string best_match = "";
                
                for (const auto& model : all_models) {
                    if (!username.empty() && model.username != username) {
                        continue;  // Skip if testing specific user
                    }
                    
                    for (const auto& stored_encoding : model.encodings) {
                        double distance = detector.compareFaces(stored_encoding, encodings[i]);
                        if (distance < best_distance) {
                            best_distance = distance;
                            best_match = model.username;
                        }
                    }
                }
                
                if (best_distance < threshold && !is_warmup) {
                    total_faces_recognized++;
                }
            }
            auto match_end = std::chrono::high_resolution_clock::now();
            matching_time = std::chrono::duration<double, std::milli>(match_end - match_start).count();
        }
        
        auto frame_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
        
        // Record statistics (skip warmup frames)
        if (!is_warmup) {
            camera_stats.add(capture_time);
            preprocess_stats.add(preprocess_time);
            detection_stats.add(detection_time);
            if (!faces.empty()) {
                encoding_stats.add(encoding_time);
                matching_stats.add(matching_time);
            }
            total_stats.add(total_time);
        }
    }
    
    auto benchmark_end = std::chrono::steady_clock::now();
    double benchmark_duration = std::chrono::duration<double>(benchmark_end - benchmark_start).count();
    
    std::cout << " Done!\n" << std::endl;
    
    // Print results
    std::cout << "=== Benchmark Results ===" << std::endl;
    std::cout << "Total benchmark time: " << std::fixed << std::setprecision(2) 
              << benchmark_duration << " seconds" << std::endl;
    std::cout << "Frames processed: " << num_frames << " (+" << warmup_frames << " warmup)" << std::endl;
    std::cout << "Overall FPS: " << std::fixed << std::setprecision(2) 
              << (num_frames / benchmark_duration) << std::endl;
    std::cout << std::endl;
    
    std::cout << "=== Detection Statistics ===" << std::endl;
    std::cout << "Frames with faces: " << frames_with_faces << " / " << num_frames 
              << " (" << std::fixed << std::setprecision(1) 
              << (100.0 * frames_with_faces / num_frames) << "%)" << std::endl;
    std::cout << "Total faces detected: " << total_faces_detected << std::endl;
    std::cout << "Total faces recognized: " << total_faces_recognized << std::endl;
    if (total_faces_detected > 0) {
        std::cout << "Recognition rate: " << std::fixed << std::setprecision(1) 
                  << (100.0 * total_faces_recognized / total_faces_detected) << "%" << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "=== Timing Statistics (per frame) ===" << std::endl;
    printStatistics("Camera capture", camera_stats);
    printStatistics("Frame preprocessing", preprocess_stats);
    printStatistics("Face detection", detection_stats);
    if (encoding_stats.count > 0) {
        printStatistics("Face encoding", encoding_stats);
    }
    if (matching_stats.count > 0) {
        printStatistics("Face matching", matching_stats);
    }
    printStatistics("Total per frame", total_stats);
    std::cout << std::endl;
    
    std::cout << "=== Pipeline Breakdown ===" << std::endl;
    double total_avg = total_stats.avg();
    if (total_avg > 0) {
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "  Camera:       " << std::setw(5) << (100.0 * camera_stats.avg() / total_avg) << "%" << std::endl;
        std::cout << "  Preprocessing:" << std::setw(5) << (100.0 * preprocess_stats.avg() / total_avg) << "%" << std::endl;
        std::cout << "  Detection:    " << std::setw(5) << (100.0 * detection_stats.avg() / total_avg) << "%" << std::endl;
        if (encoding_stats.count > 0) {
            std::cout << "  Encoding:     " << std::setw(5) << (100.0 * encoding_stats.avg() / total_avg) << "%" << std::endl;
        }
        if (matching_stats.count > 0) {
            std::cout << "  Matching:     " << std::setw(5) << (100.0 * matching_stats.avg() / total_avg) << "%" << std::endl;
        }
    }
    std::cout << std::endl;
    
    std::cout << "Benchmark completed successfully!" << std::endl;
    
    return 0;
}
