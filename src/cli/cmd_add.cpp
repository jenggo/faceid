#include <iostream>
#include <string>
#include <fstream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <dirent.h>
#include <cmath>
#include "../models/binary_model.h"
#include "config_paths.h"
#include "../config.h"
#include "../face_detector.h"
#include "../camera.h"
#include "../display.h"
#include "commands.h"
#include "cli_common.h"
#include "cli_helpers.h"

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
    
     // Create preview window with actual camera dimensions
    faceid::Display display("FaceID - Face Enrollment Preview", width, height);
    
    std::cout << std::endl;
    std::cout << "📷 Preview window opened - adjust your position to show your face clearly" << std::endl;
    std::cout << "   Press 'q' in the preview window to cancel" << std::endl;
    std::cout << std::endl;
    
    // Step 1: Auto-detect optimal detection confidence
    float optimal_confidence = findOptimalDetectionConfidence(camera, detector, display);
    if (optimal_confidence < 0.0f) {
        std::cerr << "Failed to determine optimal confidence" << std::endl;
        return 1;
    }
    
     // Get model-aware consistency threshold
     float consistency_threshold = getConsistencyThreshold(detector);
     std::cout << "Using consistency threshold: " << std::fixed << std::setprecision(3) 
               << consistency_threshold << " (model: " << detector.getModelName() << ")" << std::endl;
     std::cout << std::endl;
     
     // Storage for all consistency results
     struct SampleData {
         std::vector<std::vector<float>> all_encodings;  // All 5 frames
         std::vector<faceid::Rect> face_rects;
         int best_frame_index;
         float quality_score;
     };
     std::vector<SampleData> all_samples;
     
     // Capture and process multiple frames
     const int num_samples = 5;
     std::vector<faceid::FaceEncoding> encodings;
     
     std::cout << "Capturing " << num_samples << " face samples..." << std::endl;
     std::cout << "Tip: Move your head slightly between samples for better recognition" << std::endl;
     std::cout << std::endl;
     
     // Prompts to encourage variation
     const std::string prompts[] = {
         "(Look straight at camera)",
         "(Turn head slightly left)",
         "(Turn head slightly right)",
         "(Tilt head slightly up)",
         "(Neutral expression)"
     };
    
    for (int i = 0; i < num_samples; i++) {
        std::cout << "  Sample " << (i + 1) << "/" << num_samples << " " << prompts[i] << "... " << std::flush;
        
        // Phase 1: Wait for valid face detection (exactly 1 face)
        // Keep showing live preview until user is properly positioned
        bool face_detected = false;
        faceid::Image last_valid_frame;
        std::vector<faceid::Rect> last_valid_faces;
        
        while (!face_detected) {
            // Read and display current frame
            faceid::Image frame;
            if (!camera.read(frame)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
            
            // Preprocess frame
            faceid::Image processed_frame = detector.preprocessFrame(frame.view());
            
            // Detect faces (with tracking optimization, using optimal confidence)
            auto faces = detector.detectOrTrackFaces(processed_frame.view(), tracking_interval, optimal_confidence);
            
            // Draw visualization on original frame
            faceid::Image display_frame = frame.clone();
            
            // Draw detected face rectangles
            for (const auto& face : faces) {
                faceid::Color color = (faces.size() == 1) 
                    ? faceid::Color::Green()  // Green for good detection
                    : faceid::Color::Red();   // Red for multiple faces
                
                faceid::drawRectangle(display_frame, face.x, face.y, 
                                     face.width, face.height, color, 2);
                
                // Draw facial landmarks if available (5-point landmarks)
                if (face.hasLandmarks()) {
                    // Define colors for each landmark
                    faceid::Color landmark_colors[] = {
                        faceid::Color(0, 255, 255),    // Left eye - Cyan
                        faceid::Color(0, 255, 255),    // Right eye - Cyan  
                        faceid::Color(255, 0, 0),      // Nose - Blue
                        faceid::Color(255, 0, 255),    // Left mouth - Magenta
                        faceid::Color(255, 0, 255)     // Right mouth - Magenta
                    };
                    
                    for (size_t j = 0; j < face.landmarks.size() && j < 5; j++) {
                        const auto& pt = face.landmarks[j];
                        int px = static_cast<int>(pt.x);
                        int py = static_cast<int>(pt.y);
                        faceid::drawCircle(display_frame, px, py, 3, landmark_colors[j]);
                    }
                }
            }
            
            // Draw status text
            std::string status_text;
            faceid::Color status_color = faceid::Color::White();
            
            if (faces.empty()) {
                status_text = prompts[i] + " - Waiting for face...";
                status_color = faceid::Color::Orange();
            } else if (faces.size() > 1) {
                status_text = prompts[i] + " - Multiple faces detected, show only one";
                status_color = faceid::Color::Red();
            } else {
                // Exactly 1 face detected - ready to start countdown!
                status_text = prompts[i] + " - Face detected! Get ready...";
                status_color = faceid::Color::Green();
                face_detected = true;
                last_valid_frame = frame.clone();
                last_valid_faces = faces;
            }
            
            // Draw status banner at top
            faceid::drawFilledRectangle(display_frame, 0, 0, display_frame.width(), 40, faceid::Color::Black());
            std::string status_text_reversed = status_text;
            std::reverse(status_text_reversed.begin(), status_text_reversed.end());
            int status_width = status_text_reversed.length() * 8;
            faceid::drawText(display_frame, status_text_reversed, display_frame.width() - 10 - status_width, 10, status_color, 1.0);
            
            // Show progress bar at bottom
            int progress_width = (display_frame.width() * i) / num_samples;
            faceid::drawFilledRectangle(display_frame, 0, display_frame.height() - 10, 
                                       progress_width, 10, faceid::Color::Green());
            
            // Display the frame
            display.show(display_frame);
            
            // Check for quit key
            int key = display.waitKey(50);
            if (key == 'q' || key == 'Q' || key == 27 || !display.isOpen()) {
                std::cout << std::endl << "Cancelled by user" << std::endl;
                return 1;
            }
        }
        
        // Phase 2: Countdown with live preview (3 seconds)
        // Give user time to adjust pose according to the prompt
        auto countdown_start = std::chrono::steady_clock::now();
        const int prep_time_ms = 3000;  // 3 seconds preparation time
        
        while (true) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - countdown_start).count();
            
            if (elapsed >= prep_time_ms) break;
            
            // Read and display current frame
            faceid::Image frame;
            if (!camera.read(frame)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
            
            // Preprocess frame
            faceid::Image processed_frame = detector.preprocessFrame(frame.view());
            
            // Detect faces (with tracking optimization, using optimal confidence)
            auto faces = detector.detectOrTrackFaces(processed_frame.view(), tracking_interval, optimal_confidence);
            
            // Draw visualization on original frame
            faceid::Image display_frame = frame.clone();
            
            // Draw detected face rectangles
            for (const auto& face : faces) {
                faceid::Color color = (faces.size() == 1) 
                    ? faceid::Color::Green()  // Green for good detection
                    : faceid::Color::Red();   // Red for multiple faces
                
                faceid::drawRectangle(display_frame, face.x, face.y, 
                                     face.width, face.height, color, 2);
                
                // Draw facial landmarks if available
                if (face.hasLandmarks()) {
                    faceid::Color landmark_colors[] = {
                        faceid::Color(0, 255, 255),    // Left eye - Cyan
                        faceid::Color(0, 255, 255),    // Right eye - Cyan  
                        faceid::Color(255, 0, 0),      // Nose - Blue
                        faceid::Color(255, 0, 255),    // Left mouth - Magenta
                        faceid::Color(255, 0, 255)     // Right mouth - Magenta
                    };
                    
                    for (size_t j = 0; j < face.landmarks.size() && j < 5; j++) {
                        const auto& pt = face.landmarks[j];
                        int px = static_cast<int>(pt.x);
                        int py = static_cast<int>(pt.y);
                        faceid::drawCircle(display_frame, px, py, 3, landmark_colors[j]);
                    }
                }
            }
            
            // Draw countdown
            int remaining_sec = (prep_time_ms - elapsed) / 1000 + 1;
            std::string status_text = prompts[i] + " - Capturing in " + std::to_string(remaining_sec) + "s...";
            faceid::Color status_color = (faces.size() == 1) ? faceid::Color::Green() : faceid::Color::Orange();
            
            // Draw status banner at top
            faceid::drawFilledRectangle(display_frame, 0, 0, display_frame.width(), 40, faceid::Color::Black());
            std::string status_text_reversed = status_text;
            std::reverse(status_text_reversed.begin(), status_text_reversed.end());
            int status_width = status_text_reversed.length() * 8;
            faceid::drawText(display_frame, status_text_reversed, display_frame.width() - 10 - status_width, 10, status_color, 1.0);
            
            // Show progress bar at bottom (gradually fills during countdown)
            int progress_width = (display_frame.width() * (i * prep_time_ms + elapsed)) / (num_samples * prep_time_ms);
            faceid::drawFilledRectangle(display_frame, 0, display_frame.height() - 10, 
                                       progress_width, 10, faceid::Color::Green());
            
            // Display the frame
            display.show(display_frame);
            
            // Check for quit key
            int key = display.waitKey(50);
            if (key == 'q' || key == 'Q' || key == 27 || !display.isOpen()) {
                std::cout << std::endl << "Cancelled by user" << std::endl;
                return 1;
            }
        }
        
        // Phase 3: Consistency validation - capture 5 stable consecutive frames
        ConsistencyResult consistency_result = validateFrameConsistency(
             camera, detector, display,
             consistency_threshold,
             i,  // sample index
             prompts[i],
             num_samples,
             optimal_confidence,
             tracking_interval
         );
         
         if (!consistency_result.is_consistent) {
             std::cout << "Failed to capture consistent frames, retrying..." << std::endl;
             i--;  // Retry this sample
             std::this_thread::sleep_for(std::chrono::milliseconds(500));
             continue;
         }
         
         // Store all 5 encodings from this sample
         SampleData sample_data;
         sample_data.all_encodings = consistency_result.encodings;
         sample_data.face_rects = consistency_result.face_rects;
         sample_data.best_frame_index = consistency_result.best_frame_index;
         sample_data.quality_score = consistency_result.best_quality_score;
         all_samples.push_back(sample_data);
         
         std::cout << "✓ OK (quality: " << std::fixed << std::setprecision(2) 
                   << (consistency_result.best_quality_score * 100) << "%, "
                   << "avg distance: " << std::setprecision(3) << consistency_result.average_distance << ")" 
                   << std::endl;
    }
    
     // Display window will close automatically when display object goes out of scope
     std::cout << std::endl;
     
     if (all_samples.empty()) {
         std::cerr << "Error: Failed to capture any face samples" << std::endl;
         return 1;
     }
     
     std::cout << "Successfully captured " << num_samples << " samples with " 
               << (num_samples * 5) << " total frames!" << std::endl;
     
     // Flatten all encodings for storage (all 5 frames from each of 5 samples = 25 encodings)
     for (const auto& sample : all_samples) {
         for (const auto& encoding : sample.all_encodings) {
             encodings.push_back(encoding);
         }
     }
     
     std::cout << "Total encodings stored: " << encodings.size() << std::endl;
     std::cout << std::endl;
     
     // Step 2: Calculate optimal recognition threshold
     std::cout << "=== Calculating Optimal Recognition Threshold ===" << std::endl;
     std::cout << "Comparing samples to find best threshold..." << std::endl;
     
     std::vector<float> all_distances;
     for (size_t i = 0; i < encodings.size(); i++) {
         for (size_t j = i + 1; j < encodings.size(); j++) {
             float dist = cosineDistance(encodings[i], encodings[j]);
             all_distances.push_back(dist);
         }
     }
     
     // Find the maximum distance between any two samples (same person)
     float max_intra_distance = 0.0f;
     if (!all_distances.empty()) {
         max_intra_distance = *std::max_element(all_distances.begin(), all_distances.end());
     }
     
     // Set threshold with safety margin (20% above max intra-distance)
     float optimal_threshold = max_intra_distance * 1.2f;
     
     // Clamp to reasonable range
     if (optimal_threshold < 0.15f) optimal_threshold = 0.15f;
     if (optimal_threshold > 0.65f) {
         std::cout << "⚠ Warning: High recognition threshold (" << std::fixed << std::setprecision(2) 
                   << optimal_threshold << ") - enrollment conditions may not be optimal" << std::endl;
         std::cout << "  Consider re-enrolling with better lighting/camera positioning" << std::endl;
         optimal_threshold = 0.65f;  // Clamp for usability
     }
     
     std::cout << "✓ Optimal recognition threshold calculated: " << std::fixed << std::setprecision(2) 
               << optimal_threshold << std::endl;
     std::cout << "  Based on variation across " << encodings.size() << " frames" << std::endl;
     std::cout << "  Max intra-person distance: " << std::fixed << std::setprecision(4) 
               << max_intra_distance << std::endl;
     
     // Quality warnings
     if (max_intra_distance > 0.5f) {
         std::cout << std::endl;
         std::cout << "⚠ Warning: Large variation between frames detected (" 
                   << std::fixed << std::setprecision(3) << max_intra_distance << ")" << std::endl;
         std::cout << "  This may indicate poor lighting or camera conditions" << std::endl;
         std::cout << "  Recognition may be less reliable - consider re-enrolling" << std::endl;
     }
    
    // Create model for this face (save to FACES_DIR)
    std::string model_path = std::string(FACES_DIR) + "/" + username + "." + face_id + ".bin";
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
    
    // Step 3: Update config file with optimal values
    if (!updateConfigFile(config_path, optimal_confidence, optimal_threshold)) {
        std::cerr << "Warning: Could not update config file" << std::endl;
        std::cerr << "You may need to manually set these values in " << config_path << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "You can now use face authentication for user: " << username << std::endl;
    
    return 0;
}

} // namespace faceid
