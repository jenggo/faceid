#include <iostream>
#include <string>
#include <fstream>
#include <thread>
#include <chrono>
#include <dirent.h>
#include <json/json.h>
#include "config_paths.h"
#include "../config.h"
#include "../face_detector.h"
#include "../logger.h"

// Suppress warnings from OpenCV headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include "../camera.h"
#pragma GCC diagnostic pop

using namespace faceid;

void print_usage() {
    std::cout << "FaceID - Linux Face Authentication System" << std::endl;
    std::cout << "Version: " << VERSION << std::endl << std::endl;
    std::cout << "Usage: faceid <command> [options]" << std::endl << std::endl;
    std::cout << "Commands:" << std::endl;
    std::cout << "  add <username>      Add face model for user" << std::endl;
    std::cout << "  remove <username>   Remove face model for user" << std::endl;
    std::cout << "  list                List all enrolled users" << std::endl;
    std::cout << "  test <username>     Test face recognition" << std::endl;
    std::cout << "  devices             List available camera devices" << std::endl;
    std::cout << "  version             Show version information" << std::endl;
    std::cout << "  help                Show this help message" << std::endl;
}

int cmd_devices() {
    auto devices = Camera::listDevices();
    if (devices.empty()) {
        std::cerr << "No camera devices found" << std::endl;
        return 1;
    }
    
    std::cout << "Available camera devices:" << std::endl;
    for (const auto& device : devices) {
        std::cout << "  " << device << std::endl;
    }
    return 0;
}

int cmd_add(const std::string& username) {
    std::cout << "Adding face model for user: " << username << std::endl;
    
    // Load configuration
    faceid::Config& config = faceid::Config::getInstance();
    std::string config_path = std::string(CONFIG_DIR) + "/faceid.conf";
    if (!config.load(config_path)) {
        std::cerr << "Warning: Could not load config from " << config_path << std::endl;
        std::cerr << "Using default values" << std::endl;
    }
    
    // Get camera settings
    auto device = config.getString("camera", "device").value_or("/dev/video0");
    auto width = config.getInt("camera", "width").value_or(640);
    auto height = config.getInt("camera", "height").value_or(480);
    
    std::cout << "Using camera: " << device << " (" << width << "x" << height << ")" << std::endl;
    
    // Initialize camera
    Camera camera(device);
    if (!camera.open(width, height)) {
        std::cerr << "Error: Failed to open camera " << device << std::endl;
        std::cerr << "Available devices:" << std::endl;
        for (const auto& dev : Camera::listDevices()) {
            std::cerr << "  " << dev << std::endl;
        }
        return 1;
    }
    
    std::cout << "Camera opened successfully!" << std::endl;
    
    // Initialize face detector
    faceid::FaceDetector detector;
    std::string recognition_model = std::string(CONFIG_DIR) + "/models/face_recognition_sface_2021dec.onnx";
    
    std::cout << "Loading face recognition model..." << std::endl;
    if (!detector.loadModels(recognition_model)) {
        std::cerr << "Error: Failed to load face recognition model" << std::endl;
        std::cerr << "Expected file: " << recognition_model << std::endl;
        std::cerr << std::endl;
        std::cerr << "Run: sudo make install-models" << std::endl;
        return 1;
    }
    
    std::cout << "Models loaded successfully!" << std::endl;
    std::cout << std::endl;
    std::cout << "Please look at the camera and press Enter when ready..." << std::endl;
    std::cin.get();
    
    // Capture and process multiple frames
    const int num_samples = 5;
    std::vector<faceid::FaceEncoding> encodings;
    
    std::cout << "Capturing " << num_samples << " face samples..." << std::endl;
    
    for (int i = 0; i < num_samples; i++) {
        std::cout << "  Sample " << (i + 1) << "/" << num_samples << "... " << std::flush;
        
        cv::Mat frame;
        if (!camera.read(frame)) {
            std::cerr << "Failed to read frame" << std::endl;
            continue;
        }
        
        // Preprocess frame
        frame = detector.preprocessFrame(frame);
        
        // Detect faces
        auto faces = detector.detectFaces(frame, true);
        if (faces.empty()) {
            std::cout << "No face detected, retrying..." << std::endl;
            i--;  // Retry this sample
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }
        
        if (faces.size() > 1) {
            std::cout << "Multiple faces detected, please ensure only one person is visible" << std::endl;
            i--;  // Retry this sample
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }
        
        // Encode face
        auto face_encodings = detector.encodeFaces(frame, faces);
        if (face_encodings.empty()) {
            std::cout << "Failed to encode face, retrying..." << std::endl;
            i--;  // Retry this sample
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }
        
        encodings.push_back(face_encodings[0]);
        std::cout << "OK" << std::endl;
        
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    
    if (encodings.empty()) {
        std::cerr << "Error: Failed to capture any face samples" << std::endl;
        return 1;
    }
    
    std::cout << "Successfully captured " << encodings.size() << " samples!" << std::endl;
    
    // Save to JSON
    Json::Value model_data;
    model_data["username"] = username;
    model_data["timestamp"] = static_cast<Json::Int64>(std::time(nullptr));
    model_data["num_samples"] = static_cast<int>(encodings.size());
    
    Json::Value encodings_array(Json::arrayValue);
    for (const auto& encoding : encodings) {
        Json::Value encoding_array(Json::arrayValue);
        // SFace encoding is cv::Mat (128x1 float matrix)
        for (int i = 0; i < encoding.rows * encoding.cols; i++) {
            encoding_array.append(encoding.at<float>(i));
        }
        encodings_array.append(encoding_array);
    }
    model_data["encodings"] = encodings_array;
    
    // Write to file
    std::string model_path = std::string(MODELS_DIR) + "/" + username + ".json";
    std::ofstream model_file(model_path);
    if (!model_file.is_open()) {
        std::cerr << "Error: Failed to create model file: " << model_path << std::endl;
        return 1;
    }
    
    Json::StyledWriter writer;
    model_file << writer.write(model_data);
    model_file.close();
    
    std::cout << std::endl;
    std::cout << "✓ Face model saved successfully!" << std::endl;
    std::cout << "  File: " << model_path << std::endl;
    std::cout << "  Samples: " << encodings.size() << std::endl;
    std::cout << std::endl;
    std::cout << "You can now use face authentication for user: " << username << std::endl;
    
    return 0;
}

int cmd_remove(const std::string& username) {
    std::string model_path = std::string(MODELS_DIR) + "/" + username + ".json";
    if (std::remove(model_path.c_str()) == 0) {
        std::cout << "✓ Removed face model for user: " << username << std::endl;
        return 0;
    } else {
        std::cerr << "✗ Failed to remove face model (file may not exist)" << std::endl;
        return 1;
    }
}

int cmd_list() {
    std::string models_dir = MODELS_DIR;
    DIR* dir = opendir(models_dir.c_str());
    if (!dir) {
        std::cerr << "Error: Cannot open models directory: " << models_dir << std::endl;
        return 1;
    }
    
    std::cout << "Enrolled users:" << std::endl;
    int count = 0;
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (filename.size() > 5 && filename.substr(filename.size() - 5) == ".json") {
            std::string username = filename.substr(0, filename.size() - 5);
            std::cout << "  " << username << std::endl;
            count++;
        }
    }
    closedir(dir);
    
    if (count == 0) {
        std::cout << "  (none)" << std::endl;
    }
    std::cout << "Total: " << count << " user(s)" << std::endl;
    
    return 0;
}

int cmd_test(const std::string& username) {
    std::cout << "Testing face recognition for user: " << username << std::endl;
    
    // Load user model
    std::string model_path = std::string(MODELS_DIR) + "/" + username + ".json";
    std::ifstream model_file(model_path);
    if (!model_file.is_open()) {
        std::cerr << "Error: No face model found for user: " << username << std::endl;
        std::cerr << "Run: sudo faceid add " << username << std::endl;
        return 1;
    }
    
    Json::Value model_data;
    Json::Reader reader;
    if (!reader.parse(model_file, model_data)) {
        std::cerr << "Error: Failed to parse face model" << std::endl;
        return 1;
    }
    model_file.close();
    
    int num_samples = model_data.get("num_samples", 0).asInt();
    std::cout << "Loaded model with " << num_samples << " samples" << std::endl;
    
    // Load stored encodings
    std::vector<faceid::FaceEncoding> stored_encodings;
    const Json::Value& encodings_array = model_data["encodings"];
    for (const auto& enc_json : encodings_array) {
        // SFace encoding is 128D float vector as cv::Mat
        cv::Mat encoding(128, 1, CV_32F);
        for (int i = 0; i < 128 && i < static_cast<int>(enc_json.size()); i++) {
            encoding.at<float>(i) = enc_json[i].asFloat();
        }
        stored_encodings.push_back(encoding);
    }
    
    // Load configuration
    faceid::Config& config = faceid::Config::getInstance();
    std::string config_path = std::string(CONFIG_DIR) + "/faceid.conf";
    config.load(config_path);
    
    auto device = config.getString("camera", "device").value_or("/dev/video0");
    auto width = config.getInt("camera", "width").value_or(640);
    auto height = config.getInt("camera", "height").value_or(480);
    double threshold = config.getDouble("recognition", "threshold").value_or(0.6);
    
    std::cout << "Using camera: " << device << std::endl;
    std::cout << "Recognition threshold: " << threshold << std::endl;
    std::cout << std::endl;
    
    // Initialize camera
    Camera camera(device);
    if (!camera.open(width, height)) {
        std::cerr << "Error: Failed to open camera" << std::endl;
        return 1;
    }
    
    // Initialize face detector
    faceid::FaceDetector detector;
    std::string recognition_model = std::string(CONFIG_DIR) + "/models/face_recognition_sface_2021dec.onnx";
    
    if (!detector.loadModels(recognition_model)) {
        std::cerr << "Error: Failed to load face recognition model" << std::endl;
        std::cerr << "Expected: " << recognition_model << std::endl;
        return 1;
    }
    
    std::cout << "Please look at the camera..." << std::endl;
    std::cout << "Testing for 5 seconds..." << std::endl;
    
    auto start = std::chrono::steady_clock::now();
    bool recognized = false;
    
    while (std::chrono::duration_cast<std::chrono::seconds>(
           std::chrono::steady_clock::now() - start).count() < 5) {
        
        cv::Mat frame;
        if (!camera.read(frame)) {
            continue;
        }
        
        frame = detector.preprocessFrame(frame);
        auto faces = detector.detectFaces(frame, true);
        
        if (faces.empty()) {
            std::cout << "." << std::flush;
            continue;
        }
        
        auto encodings = detector.encodeFaces(frame, faces);
        if (encodings.empty()) {
            continue;
        }
        
        // Compare with stored encodings
        double min_distance = 999.0;
        for (const auto& stored : stored_encodings) {
            double distance = detector.compareFaces(stored, encodings[0]);
            if (distance < min_distance) {
                min_distance = distance;
            }
        }
        
        std::cout << std::endl;
        std::cout << "Face detected! Distance: " << min_distance;
        
        if (min_distance < threshold) {
            std::cout << " ✓ MATCH" << std::endl;
            recognized = true;
            break;
        } else {
            std::cout << " ✗ NO MATCH" << std::endl;
        }
    }
    
    std::cout << std::endl;
    if (recognized) {
        std::cout << "✓ Face recognition successful for user: " << username << std::endl;
        return 0;
    } else {
        std::cout << "✗ Face recognition failed" << std::endl;
        return 1;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    std::string command = argv[1];
    
    if (command == "help" || command == "--help" || command == "-h") {
        print_usage();
        return 0;
    }
    
    if (command == "version" || command == "--version" || command == "-v") {
        std::cout << "FaceID version " << VERSION << std::endl;
        return 0;
    }
    
    if (command == "devices") {
        return cmd_devices();
    }
    
    if (command == "list") {
        return cmd_list();
    }
    
    if (command == "add") {
        if (argc < 3) {
            std::cerr << "Error: username required" << std::endl;
            return 1;
        }
        return cmd_add(argv[2]);
    }
    
    if (command == "remove") {
        if (argc < 3) {
            std::cerr << "Error: username required" << std::endl;
            return 1;
        }
        return cmd_remove(argv[2]);
    }
    
    if (command == "test") {
        if (argc < 3) {
            std::cerr << "Error: username required" << std::endl;
            return 1;
        }
        return cmd_test(argv[2]);
    }
    
    std::cerr << "Unknown command: " << command << std::endl;
    print_usage();
    return 1;
}
