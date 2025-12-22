#include <iostream>
#include <string>
#include <fstream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <dirent.h>
#include "../models/binary_model.h"
#include "config_paths.h"
#include "../config.h"
#include "../face_detector.h"
#include "../camera.h"
#include "../display.h"
#include "commands.h"
#include "cli_common.h"

namespace faceid {

using namespace faceid;

int cmd_add(const std::string& username, const std::string& face_id) {
    std::cout << "Adding face model '" << face_id << "' for user: " << username << std::endl;
    
    // Check for existing models
    auto existing_files = cli::findUserModelFiles(username);
    if (!existing_files.empty()) {
        std::cout << std::endl;
        std::cout << "Note: User '" << username << "' already has " << existing_files.size() << " face model(s):" << std::endl;
        for (const auto& file : existing_files) {
            // Extract just the filename
            size_t last_slash = file.find_last_of('/');
            std::string filename = (last_slash != std::string::npos) ? file.substr(last_slash + 1) : file;
            std::cout << "  - " << filename << std::endl;
        }
        std::cout << "This will add an additional face model: " << username << "." << face_id << ".bin" << std::endl;
        std::cout << std::endl;
    }
    
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
    
    std::cout << "Loading face recognition model..." << std::endl;
    if (!detector.loadModels()) {  // Use default MODELS_DIR/sface path
        std::cerr << "Error: Failed to load face recognition model" << std::endl;
        std::cerr << "Expected files: " << MODELS_DIR << "/sface.param and sface.bin" << std::endl;
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
    
    // Create model for this face
    std::string model_path = std::string(MODELS_DIR) + "/" + username + "." + face_id + ".bin";
    BinaryFaceModel model_data;
    model_data.username = username;
    model_data.face_ids.push_back(face_id);
    model_data.encodings = encodings;
    model_data.timestamp = static_cast<uint32_t>(std::time(nullptr));
    model_data.valid = true;
    
    // Write to file
    if (!BinaryModelLoader::saveUserModel(model_path, model_data)) {
        std::cerr << "Error: Failed to save face model file: " << model_path << std::endl;
        return 1;
    }
    
    std::cout << std::endl;
    std::cout << "✓ Face model saved successfully!" << std::endl;
    std::cout << "  File: " << model_path << std::endl;
    std::cout << "  Face ID: " << face_id << std::endl;
    std::cout << "  Samples: " << encodings.size() << std::endl;
    
    // Show total faces for this user
    int total_faces = 1;  // Since we create one file per face
    std::cout << "  Total faces for " << username << ": " << total_faces << " (this session)" << std::endl;
    std::cout << std::endl;
    std::cout << "You can now use face authentication for user: " << username << std::endl;
    
    return 0;
}

} // namespace faceid
