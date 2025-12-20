#include <iostream>
#include <string>
#include <fstream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <dirent.h>
#include <json/json.h>
#include "config_paths.h"
#include "../config.h"
#include "../face_detector.h"
#include "../camera.h"
#include "../display.h"

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
    std::cout << "  show                          Show live camera view with face detection" << std::endl;
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
    std::cout << "  faceid show                   # Live camera preview with detection" << std::endl;
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
    int tracking_interval = config.getInt("face_detection", "tracking_interval").value_or(10);
    
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
    faceid::Display display("FaceID - Face Enrollment Preview", 640, 480);
    
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
        
        faceid::Image frame;
        if (!camera.read(frame)) {
            std::cerr << "Failed to read frame" << std::endl;
            continue;
        }
        
        // Preprocess frame
        faceid::Image processed_frame = detector.preprocessFrame(frame.view());
        
        // Detect faces (with tracking optimization)
        auto faces = detector.detectOrTrackFaces(processed_frame.view(), tracking_interval);
        
        // Draw visualization on original frame
        faceid::Image display_frame = frame.clone();
        
        // Draw detected face rectangles (adjust coordinates for SDL flip)
        for (const auto& face : faces) {
            faceid::Color color = (faces.size() == 1) 
                ? faceid::Color::Green()  // Green for good detection
                : faceid::Color::Red();   // Red for multiple faces
            
            // Draw rectangle at ORIGINAL position (SDL will flip it correctly)
            faceid::drawRectangle(display_frame, face.x, face.y, 
                                 face.width, face.height, color, 2);
        }
        
        // Draw status text
        std::string status_text;
        faceid::Color status_color = faceid::Color::Black();  // Initialize with default
        
        if (faces.empty()) {
            status_text = "No face detected - position yourself in frame";
            status_color = faceid::Color::Orange();
        } else if (faces.size() > 1) {
            status_text = "Multiple faces (" + std::to_string(faces.size()) + ") - only one person should be visible";
            status_color = faceid::Color::Red();
        } else {
            status_text = "Face detected - capturing sample " + std::to_string(i + 1) + "/" + std::to_string(num_samples);
            status_color = faceid::Color::Green();
        }
        
        // Draw status banner at top
        faceid::drawFilledRectangle(display_frame, 0, 0, display_frame.width(), 40, faceid::Color::Black());
        std::string status_text_reversed = status_text;
        std::reverse(status_text_reversed.begin(), status_text_reversed.end());
        int status_width = status_text_reversed.length() * 8;
        faceid::drawText(display_frame, status_text_reversed, display_frame.width() - 10 - status_width, 10, status_color, 1.0);
        
        // Show progress bar at bottom
        int progress_width = (display_frame.width() * (i + 1)) / num_samples;
        faceid::drawFilledRectangle(display_frame, 0, display_frame.height() - 10, 
                                   progress_width, 10, faceid::Color::Green());
        
        // Display the frame (SDL will flip horizontally)
        display.show(display_frame);
        
        // Check for quit key (short wait to keep display responsive)
        int key = display.waitKey(30);
        if (key == 'q' || key == 'Q' || key == 27 || !display.isOpen()) {  // q or ESC or window closed
            std::cout << std::endl << "Cancelled by user" << std::endl;
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
        faceid::Rect selected_face;
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
        std::vector<faceid::Rect> single_face = {selected_face};
        auto face_encodings = detector.encodeFaces(processed_frame.view(), single_face);
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
    
    // Display window will close automatically when display object goes out of scope
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
        // Encoding is std::vector<float> (128D)
        for (float val : encoding) {
            encoding_array.append(val);
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

int cmd_show() {
    std::cout << "Starting live camera preview with face detection..." << std::endl;
    std::cout << "Press 'q' or ESC to quit" << std::endl << std::endl;
    
    // Load configuration
    faceid::Config& config = faceid::Config::getInstance();
    std::string config_path = std::string(CONFIG_DIR) + "/faceid.conf";
    if (!config.load(config_path)) {
        std::cerr << "Warning: Could not load config, using defaults" << std::endl;
    }
    
    // Get camera settings
    auto device = config.getString("camera", "device").value_or("/dev/video0");
    auto width = config.getInt("camera", "width").value_or(640);
    auto height = config.getInt("camera", "height").value_or(480);
    int tracking_interval = config.getInt("face_detection", "tracking_interval").value_or(10);
    
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
    
    // Initialize face detector
    faceid::FaceDetector detector;
    std::string recognition_model = std::string(CONFIG_DIR) + "/models/face_recognition_sface_2021dec.onnx";
    
    std::cout << "Loading face detection model..." << std::endl;
    if (!detector.loadModels(recognition_model)) {
        std::cerr << "Error: Failed to load face detection model" << std::endl;
        std::cerr << "Expected file: " << recognition_model << std::endl;
        std::cerr << "Run: sudo make install-models" << std::endl;
        return 1;
    }
    
    std::cout << "Models loaded successfully!" << std::endl;
    
    // Create preview window
    faceid::Display display("FaceID - Live Camera View", 800, 600);
    
    std::cout << "\nLive preview started. Press 'q' or ESC in the preview window to quit.\n" << std::endl;
    
    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();
    
    while (display.isOpen()) {
        faceid::Image frame;
        if (!camera.read(frame)) {
            std::cerr << "Failed to read frame from camera" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        // Preprocess and detect faces
        faceid::Image processed_frame = detector.preprocessFrame(frame.view());
        auto faces = detector.detectOrTrackFaces(processed_frame.view(), tracking_interval);
        
        // Clone frame for drawing
        faceid::Image display_frame = frame.clone();
        
        // Draw detected faces (adjust coordinates for SDL's horizontal flip)
        for (size_t i = 0; i < faces.size(); i++) {
            const auto& face = faces[i];
            
            // Color based on face index (primary face is green, others are yellow)
            faceid::Color color = (i == 0) 
                ? faceid::Color::Green()   // Green for primary face
                : faceid::Color::Yellow(); // Yellow for additional faces
            
            // Draw rectangle at ORIGINAL position (SDL will flip it correctly)
            faceid::drawRectangle(display_frame, face.x, face.y, 
                                 face.width, face.height, color, 2);
            
            // Label faces - reverse text and calculate mirrored position
            std::string label = (i == 0) ? "Face 1 (Primary)" : "Face " + std::to_string(i + 1);
            std::reverse(label.begin(), label.end());
            int text_width = label.length() * 8;
            // Text position needs to be mirrored: what appears at face.x will show at (width - face.x) after flip
            int text_x = display_frame.width() - (face.x + text_width);
            faceid::drawText(display_frame, label, text_x, face.y - 10, color, 1.0);
        }
        
        // Calculate FPS
        frame_count++;
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
        if (elapsed > 0) {
            double fps = static_cast<double>(frame_count) / elapsed;
            
            // Draw info banner at top
            faceid::drawFilledRectangle(display_frame, 0, 0, display_frame.width(), 70, faceid::Color::Black());
            
            // Face count (reversed text for SDL flip)
            std::string info_text = "Detected faces: " + std::to_string(faces.size());
            std::reverse(info_text.begin(), info_text.end());
            int info_width = info_text.length() * 8;
            faceid::drawText(display_frame, info_text, display_frame.width() - 10 - info_width, 10, faceid::Color::White(), 1.0);
            
            // FPS (reversed text for SDL flip)
            std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps));
            std::reverse(fps_text.begin(), fps_text.end());
            int fps_width = fps_text.length() * 8;
            faceid::drawText(display_frame, fps_text, display_frame.width() - 10 - fps_width, 25, faceid::Color::Green(), 1.0);
            
            // Resolution (reversed text for SDL flip)
            std::string res_text = std::to_string(display_frame.width()) + "x" + std::to_string(display_frame.height());
            std::reverse(res_text.begin(), res_text.end());
            faceid::drawText(display_frame, res_text, 10, 10, faceid::Color::Gray(), 1.0);
        }
        
        // Draw help text at bottom (reversed text for SDL flip)
        faceid::drawFilledRectangle(display_frame, 0, display_frame.height() - 30, 
                                   display_frame.width(), 30, faceid::Color::Black());
        std::string help_text = "Press 'q' or ESC to quit";
        std::reverse(help_text.begin(), help_text.end());
        int help_width = help_text.length() * 8;
        faceid::drawText(display_frame, help_text, display_frame.width() - 10 - help_width, display_frame.height() - 20, 
                        faceid::Color::White(), 1.0);
        
        // Display the frame (SDL will flip horizontally)
        display.show(display_frame);
        
        // Check for quit key
        int key = display.waitKey(30);
        if (key == 'q' || key == 'Q' || key == 27) {  // q or ESC
            break;
        }
    }
    
    std::cout << "\nLive preview stopped." << std::endl;
    std::cout << "Total frames processed: " << frame_count << std::endl;
    
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
                faceid::FaceEncoding encoding(128);
                for (int i = 0; i < 128 && i < static_cast<int>(enc_json.size()); i++) {
                    encoding[i] = enc_json[i].asFloat();
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
            faceid::FaceEncoding encoding(128);
            for (int i = 0; i < 128 && i < static_cast<int>(enc_json.size()); i++) {
                encoding[i] = enc_json[i].asFloat();
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
    int tracking_interval = config.getInt("face_detection", "tracking_interval").value_or(10);
    
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
    double detection_time_ms = 0.0;
    double recognition_time_ms = 0.0;
    
    while (std::chrono::duration_cast<std::chrono::seconds>(
           std::chrono::steady_clock::now() - start).count() < 5) {
        
        faceid::Image frame;
        if (!camera.read(frame)) {
            continue;
        }
        
        faceid::Image processed_frame = detector.preprocessFrame(frame.view());
        
        // Time face detection
        auto detect_start = std::chrono::high_resolution_clock::now();
        auto faces = detector.detectOrTrackFaces(processed_frame.view(), tracking_interval);
        auto detect_end = std::chrono::high_resolution_clock::now();
        detection_time_ms = std::chrono::duration<double, std::milli>(detect_end - detect_start).count();
        
        if (faces.empty()) {
            std::cout << "." << std::flush;
            continue;
        }
        
        // Time face recognition
        auto recog_start = std::chrono::high_resolution_clock::now();
        auto encodings = detector.encodeFaces(processed_frame.view(), faces);
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
        auto recog_end = std::chrono::high_resolution_clock::now();
        recognition_time_ms = std::chrono::duration<double, std::milli>(recog_end - recog_start).count();
        
        std::cout << std::endl;
        std::cout << "Face detected! Distance: " << min_distance;
        
        if (min_distance < threshold) {
            std::cout << " ✓ MATCH" << std::endl;
            std::cout << std::endl;
            std::cout << "Performance:" << std::endl;
            std::cout << "  Detection:    " << std::fixed << std::setprecision(2) << detection_time_ms << " ms" << std::endl;
            std::cout << "  Recognition:  " << std::fixed << std::setprecision(2) << recognition_time_ms << " ms" << std::endl;
            std::cout << "  Total:        " << std::fixed << std::setprecision(2) << (detection_time_ms + recognition_time_ms) << " ms" << std::endl;
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
    
    if (command == "show") {
        return cmd_show();
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
