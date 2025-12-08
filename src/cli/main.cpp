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
    std::cout << "  add <username> [face_id]      Add face model for user (default: 'default')" << std::endl;
    std::cout << "  remove <username> [face_id]   Remove specific face or all faces" << std::endl;
    std::cout << "  list [username]               List all enrolled users or user's faces" << std::endl;
    std::cout << "  test <username>               Test face recognition" << std::endl;
    std::cout << "  devices                       List available camera devices" << std::endl;
    std::cout << "  version                       Show version information" << std::endl;
    std::cout << "  help                          Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  faceid add jenggo             # Add default face" << std::endl;
    std::cout << "  faceid add jenggo glasses     # Add face with glasses" << std::endl;
    std::cout << "  faceid list jenggo            # List all faces for jenggo" << std::endl;
    std::cout << "  faceid remove jenggo glasses  # Remove 'glasses' face only" << std::endl;
    std::cout << "  faceid remove jenggo          # Remove ALL faces for jenggo" << std::endl;
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

int cmd_add(const std::string& username, const std::string& face_id = "default") {
    std::cout << "Adding face model '" << face_id << "' for user: " << username << std::endl;
    
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
    
    // Create preview window
    const std::string window_name = "FaceID - Face Enrollment Preview";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 640, 480);
    
    std::cout << std::endl;
    std::cout << "📷 Preview window opened - adjust your position to show your face clearly" << std::endl;
    std::cout << "   Press 'q' in the preview window to cancel" << std::endl;
    std::cout << std::endl;
    
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
        cv::Mat processed_frame = detector.preprocessFrame(frame);
        
        // Detect faces
        auto faces = detector.detectFaces(processed_frame, true);
        
        // Draw visualization on original frame
        cv::Mat display_frame = frame.clone();
        
        // Draw detected face rectangles
        for (const auto& face : faces) {
            cv::Scalar color;
            
            if (faces.size() == 1) {
                color = cv::Scalar(0, 255, 0);  // Green for good detection
            } else {
                color = cv::Scalar(0, 0, 255);  // Red for multiple faces
            }
            
            // Draw rectangle around face
            cv::rectangle(display_frame, face, color, 2);
        }
        
        // Draw status text
        std::string status_text;
        cv::Scalar status_color;
        
        if (faces.empty()) {
            status_text = "No face detected - position yourself in frame";
            status_color = cv::Scalar(0, 165, 255);  // Orange
        } else if (faces.size() > 1) {
            status_text = "Multiple faces (" + std::to_string(faces.size()) + ") - only one person should be visible";
            status_color = cv::Scalar(0, 0, 255);  // Red
        } else {
            status_text = "Face detected - capturing sample " + std::to_string(i + 1) + "/" + std::to_string(num_samples);
            status_color = cv::Scalar(0, 255, 0);  // Green
        }
        
        // Draw status banner at top
        cv::rectangle(display_frame, cv::Point(0, 0), cv::Point(display_frame.cols, 40), 
                     cv::Scalar(0, 0, 0), -1);
        cv::putText(display_frame, status_text,
                   cv::Point(10, 25),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2);
        
        // Show progress bar at bottom
        int progress_width = (display_frame.cols * (i + 1)) / num_samples;
        cv::rectangle(display_frame, 
                     cv::Point(0, display_frame.rows - 10), 
                     cv::Point(progress_width, display_frame.rows),
                     cv::Scalar(0, 255, 0), -1);
        
        // Display the frame
        cv::imshow(window_name, display_frame);
        
        // Check for quit key (short wait to keep display responsive)
        int key = cv::waitKey(30);
        if (key == 'q' || key == 'Q' || key == 27) {  // q or ESC
            std::cout << std::endl << "Cancelled by user" << std::endl;
            cv::destroyWindow(window_name);
            return 1;
        }
        
        // Validate detection
        if (faces.empty()) {
            std::cout << "No face detected, retrying..." << std::endl;
            i--;  // Retry this sample
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }
        
        // If multiple faces, select the largest one (closest to camera)
        cv::Rect selected_face;
        if (faces.size() > 1) {
            std::cout << "Multiple faces detected (" << faces.size() << "), selecting largest... " << std::flush;
            
            // Find the largest face by area
            int max_area = 0;
            for (const auto& face : faces) {
                int area = face.width * face.height;
                if (area > max_area) {
                    max_area = area;
                    selected_face = face;
                }
            }
        } else {
            selected_face = faces[0];
        }
        
        // Encode the selected face
        std::vector<cv::Rect> single_face = {selected_face};
        auto face_encodings = detector.encodeFaces(processed_frame, single_face);
        if (face_encodings.empty()) {
            std::cout << "Failed to encode face, retrying..." << std::endl;
            i--;  // Retry this sample
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }
        
        encodings.push_back(face_encodings[0]);
        std::cout << "✓ OK" << std::endl;
        
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    
    // Close preview window
    cv::destroyWindow(window_name);
    std::cout << std::endl;
    
    if (encodings.empty()) {
        std::cerr << "Error: Failed to capture any face samples" << std::endl;
        return 1;
    }
    
    std::cout << "Successfully captured " << encodings.size() << " samples!" << std::endl;
    
    // Load existing model or create new one
    std::string model_path = std::string(MODELS_DIR) + "/" + username + ".json";
    Json::Value model_data;
    
    std::ifstream existing_file(model_path);
    if (existing_file.is_open()) {
        Json::Reader reader;
        reader.parse(existing_file, model_data);
        existing_file.close();
        std::cout << "Loaded existing model file" << std::endl;
    } else {
        // New user
        model_data["username"] = username;
        model_data["timestamp"] = static_cast<Json::Int64>(std::time(nullptr));
    }
    
    // Create face data
    Json::Value face_data;
    face_data["num_samples"] = static_cast<int>(encodings.size());
    face_data["created"] = static_cast<Json::Int64>(std::time(nullptr));
    
    Json::Value encodings_array(Json::arrayValue);
    for (const auto& encoding : encodings) {
        Json::Value encoding_array(Json::arrayValue);
        // SFace encoding is cv::Mat (128x1 float matrix)
        for (int i = 0; i < encoding.rows * encoding.cols; i++) {
            encoding_array.append(encoding.at<float>(i));
        }
        encodings_array.append(encoding_array);
    }
    face_data["encodings"] = encodings_array;
    
    // Add or update face
    if (!model_data.isMember("faces")) {
        model_data["faces"] = Json::Value(Json::objectValue);
    }
    model_data["faces"][face_id] = face_data;
    
    // Update timestamp
    model_data["timestamp"] = static_cast<Json::Int64>(std::time(nullptr));
    
    // Write to file
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
    std::cout << "  Face ID: " << face_id << std::endl;
    std::cout << "  Samples: " << encodings.size() << std::endl;
    
    // Show total faces for this user
    int total_faces = model_data["faces"].size();
    std::cout << "  Total faces for " << username << ": " << total_faces << std::endl;
    std::cout << std::endl;
    std::cout << "You can now use face authentication for user: " << username << std::endl;
    
    return 0;
}

int cmd_remove(const std::string& username, const std::string& face_id = "") {
    std::string model_path = std::string(MODELS_DIR) + "/" + username + ".json";
    
    // If no face_id specified, remove entire user model
    if (face_id.empty()) {
        if (std::remove(model_path.c_str()) == 0) {
            std::cout << "✓ Removed ALL face models for user: " << username << std::endl;
            return 0;
        } else {
            std::cerr << "✗ Failed to remove face model (file may not exist)" << std::endl;
            return 1;
        }
    }
    
    // Remove specific face_id
    std::ifstream model_file(model_path);
    if (!model_file.is_open()) {
        std::cerr << "Error: No face model found for user: " << username << std::endl;
        return 1;
    }
    
    Json::Value model_data;
    Json::Reader reader;
    if (!reader.parse(model_file, model_data)) {
        std::cerr << "Error: Failed to parse face model" << std::endl;
        return 1;
    }
    model_file.close();
    
    // Check if face_id exists
    if (!model_data.isMember("faces") || !model_data["faces"].isMember(face_id)) {
        std::cerr << "Error: Face ID '" << face_id << "' not found for user: " << username << std::endl;
        return 1;
    }
    
    // Remove the face_id
    model_data["faces"].removeMember(face_id);
    model_data["timestamp"] = static_cast<Json::Int64>(std::time(nullptr));
    
    // If no faces left, remove entire file
    if (model_data["faces"].empty()) {
        std::remove(model_path.c_str());
        std::cout << "✓ Removed last face '" << face_id << "', deleted entire model for user: " << username << std::endl;
        return 0;
    }
    
    // Write updated model
    std::ofstream out_file(model_path);
    if (!out_file.is_open()) {
        std::cerr << "Error: Failed to update model file" << std::endl;
        return 1;
    }
    
    Json::StyledWriter writer;
    out_file << writer.write(model_data);
    out_file.close();
    
    int remaining = model_data["faces"].size();
    std::cout << "✓ Removed face '" << face_id << "' for user: " << username << std::endl;
    std::cout << "  Remaining faces: " << remaining << std::endl;
    
    return 0;
}

int cmd_list(const std::string& username = "") {
    std::string models_dir = MODELS_DIR;
    
    // List faces for specific user
    if (!username.empty()) {
        std::string model_path = models_dir + "/" + username + ".json";
        std::ifstream model_file(model_path);
        if (!model_file.is_open()) {
            std::cerr << "Error: No face model found for user: " << username << std::endl;
            return 1;
        }
        
        Json::Value model_data;
        Json::Reader reader;
        if (!reader.parse(model_file, model_data)) {
            std::cerr << "Error: Failed to parse face model" << std::endl;
            return 1;
        }
        model_file.close();
        
        std::cout << "Faces for user: " << username << std::endl;
        
        if (model_data.isMember("faces")) {
            for (const auto& face_id : model_data["faces"].getMemberNames()) {
                const Json::Value& face_data = model_data["faces"][face_id];
                int samples = face_data.get("num_samples", 0).asInt();
                Json::Int64 created = face_data.get("created", 0).asInt64();
                
                std::cout << "  " << face_id;
                std::cout << " (" << samples << " samples";
                if (created > 0) {
                    std::time_t time = static_cast<std::time_t>(created);
                    char buffer[80];
                    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&time));
                    std::cout << ", created: " << buffer;
                }
                std::cout << ")" << std::endl;
            }
            std::cout << "Total: " << model_data["faces"].size() << " face(s)" << std::endl;
        } else {
            std::cout << "  (none)" << std::endl;
        }
        
        return 0;
    }
    
    // List all users
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
            std::string user = filename.substr(0, filename.size() - 5);
            
            // Load model to count faces
            std::string model_path = models_dir + "/" + filename;
            std::ifstream model_file(model_path);
            int face_count = 0;
            
            if (model_file.is_open()) {
                Json::Value model_data;
                Json::Reader reader;
                if (reader.parse(model_file, model_data) && model_data.isMember("faces")) {
                    face_count = model_data["faces"].size();
                }
                model_file.close();
            }
            
            std::cout << "  " << user << " (" << face_count << " face(s))" << std::endl;
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
    
    // Load stored encodings from all faces
    std::vector<faceid::FaceEncoding> stored_encodings;
    
    if (model_data.isMember("faces")) {
        // New multi-face format
        for (const auto& face_id : model_data["faces"].getMemberNames()) {
            const Json::Value& face_data = model_data["faces"][face_id];
            const Json::Value& encodings_array = face_data["encodings"];
            
            for (const auto& enc_json : encodings_array) {
                cv::Mat encoding(128, 1, CV_32F);
                for (int i = 0; i < 128 && i < static_cast<int>(enc_json.size()); i++) {
                    encoding.at<float>(i) = enc_json[i].asFloat();
                }
                stored_encodings.push_back(encoding);
            }
        }
        std::cout << "Loaded " << stored_encodings.size() << " encoding(s) from " 
                  << model_data["faces"].size() << " face(s)" << std::endl;
    } else if (model_data.isMember("encodings")) {
        // Old single-face format (backward compatibility)
        const Json::Value& encodings_array = model_data["encodings"];
        for (const auto& enc_json : encodings_array) {
            cv::Mat encoding(128, 1, CV_32F);
            for (int i = 0; i < 128 && i < static_cast<int>(enc_json.size()); i++) {
                encoding.at<float>(i) = enc_json[i].asFloat();
            }
            stored_encodings.push_back(encoding);
        }
        std::cout << "Loaded " << stored_encodings.size() << " encoding(s) (old format)" << std::endl;
    }
    
    if (stored_encodings.empty()) {
        std::cerr << "Error: No face encodings found" << std::endl;
        return 1;
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
        
        // Compare with all stored encodings
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
        if (argc >= 3) {
            return cmd_list(argv[2]);  // List faces for specific user
        }
        return cmd_list();  // List all users
    }
    
    if (command == "add") {
        if (argc < 3) {
            std::cerr << "Error: username required" << std::endl;
            return 1;
        }
        if (argc >= 4) {
            return cmd_add(argv[2], argv[3]);  // username + face_id
        }
        return cmd_add(argv[2]);  // username only (default face_id)
    }
    
    if (command == "remove") {
        if (argc < 3) {
            std::cerr << "Error: username required" << std::endl;
            return 1;
        }
        if (argc >= 4) {
            return cmd_remove(argv[2], argv[3]);  // Remove specific face
        }
        return cmd_remove(argv[2]);  // Remove all faces
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
