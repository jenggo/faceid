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
#include "../daemon/config.h"
#include "../camera.h"
#include "../display.h"
#include "dbus_client_cli.h"
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
    std::string config_path = std::string(CONFIG_DIR) + "/faceid.conf";
    auto config_ptr = ::Config::load();
    if (!config_ptr) {
        std::cerr << "Warning: Could not load config, using defaults" << std::endl;
    }
    ::Config& config = config_ptr ? *config_ptr : ::Config::instance();
    
    // Get camera settings
    auto device = config.camera.device;
    auto width = config.camera.width;
    auto height = config.camera.height;
    int tracking_interval = config.face_detection.tracking_interval;
    
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
    
    // Initialize D-Bus client for daemon communication
    DBusClientCLI& dbus_client = DBusClientCLI::instance();
    if (!dbus_client.connect()) {
        std::cerr << "Error: Failed to connect to FaceID daemon via D-Bus" << std::endl;
        std::cerr << "Make sure the daemon is running: sudo systemctl status faceid-daemon" << std::endl;
        return 1;
    }
    
    std::cout << "Connected to FaceID daemon successfully!" << std::endl;
     std::cout << std::endl;
     std::cout << "Please look at the camera and press Enter when ready..." << std::endl;
     std::cin.get();
    
     // Create preview window with actual camera dimensions
    faceid::Display display("FaceID - Face Enrollment Preview", width, height);
    
    std::cout << std::endl;
    std::cout << "📷 Preview window opened - adjust your position to show your face clearly" << std::endl;
    std::cout << "   Press 'q' in the preview window to cancel" << std::endl;
    std::cout << std::endl;
    
      // Step 1: Auto-detect optimal detection confidence and quality threshold
     faceid::OptimizationResult optimization = findOptimalDetectionConfidence(camera, dbus_client, display);
     float optimal_confidence = optimization.detection_confidence;
     float min_quality_threshold = optimization.min_face_quality;
     if (optimal_confidence < 0.0f) {
         std::cerr << "Failed to determine optimal confidence" << std::endl;
         return 1;
     }
     
      // Get consistency threshold (simplified - no detector dependency)
      float consistency_threshold = getConsistencyThreshold();
      std::cout << "Using consistency threshold: " << std::fixed << std::setprecision(3) 
                << consistency_threshold << std::endl;
      std::cout << std::endl;
     
     // Storage for all consistency results
     struct SampleData {
         std::vector<std::vector<float>> all_encodings;  // All 5 frames encodings
         std::vector<faceid::Rect> face_rects;
         std::vector<std::shared_ptr<faceid::Image>> frames;  // PHASE 5: Frames for augmentation
         int best_frame_index;
         float quality_score;
     };
     std::vector<SampleData> all_samples;
     
      // PHASE 5: Determine number of samples based on extended enrollment setting
      bool enable_extended_enrollment = config.enrollment.extended_enrollment;
     const int num_samples = 5;  // Always capture 5 poses
     
     if (enable_extended_enrollment) {
         std::cout << "Extended enrollment enabled - will generate synthetic lighting variations" << std::endl;
         std::cout << "This improves recognition accuracy without requiring special lighting setup" << std::endl;
         std::cout << std::endl;
     }
     
     // Capture and process multiple frames
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
         "(Neutral expression)",
         // PHASE 5: Extended enrollment lighting-specific poses
         "(Face bright light source - e.g. desk lamp)",
         "(Turn away from light - create shadow)",
         "(Face window - backlit condition)"
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
             
             // Encode frame as JPEG for D-Bus transmission
             auto jpeg_data = DBusClientCLI::encode_frame_as_jpeg(frame);
             if (jpeg_data.empty()) {
                 std::this_thread::sleep_for(std::chrono::milliseconds(50));
                 continue;
             }
             
             // Detect faces via daemon
             auto detect_result = dbus_client.detect_face(jpeg_data);
             const auto& faces = detect_result.boxes;
            
            // Draw visualization on original frame
            faceid::Image display_frame = frame.clone();
            
            // Draw detected face rectangles with centering correction
            for (const auto& face : faces) {
                faceid::Color color = (faces.size() == 1) 
                    ? faceid::Color::Green()  // Green for good detection
                    : faceid::Color::Red();   // Red for multiple faces
                
                faceid::drawFaceBoundingBox(display_frame, face, color, 2);
                
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
            
            // Encode frame as JPEG for D-Bus transmission
            auto jpeg_data = DBusClientCLI::encode_frame_as_jpeg(frame);
            if (jpeg_data.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
            
            // Detect faces via daemon
            auto detect_result = dbus_client.detect_face(jpeg_data);
            const auto& faces = detect_result.boxes;
            
            // Draw visualization on original frame
            faceid::Image display_frame = frame.clone();
            
            // Draw detected face rectangles with centering correction
            for (const auto& face : faces) {
                faceid::Color color = (faces.size() == 1) 
                    ? faceid::Color::Green()  // Green for good detection
                    : faceid::Color::Red();   // Red for multiple faces
                
                faceid::drawFaceBoundingBox(display_frame, face, color, 2);
                
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
              camera, dbus_client, display,
              consistency_threshold,
              i,  // sample index
              prompts[i],
              num_samples,
              optimal_confidence,
              tracking_interval,
              min_quality_threshold  // Pass camera-adaptive quality
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
          sample_data.frames = consistency_result.frames;  // PHASE 5: Store frames
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
     
      // PHASE 5: Synthetic lighting augmentation disabled for now
      // Will be re-enabled when daemon gains synthetic augmentation support
      // NOTE: Daemon-based augmentation would require D-Bus methods like:
      //   - dbus_client.apply_bright_lighting(png_data, bbox)
      //   - dbus_client.apply_dim_lighting(png_data, bbox)
      if (enable_extended_enrollment) {
          std::cout << std::endl;
          std::cout << "Note: Synthetic lighting augmentation is currently disabled." << std::endl;
          std::cout << "Will be available when daemon-based augmentation is implemented." << std::endl;
          std::cout << std::endl;
      }
     
     std::cout << std::endl;
     
     // Organize encodings in V2 format: [pose][encoding_variant]
     // Each of the 5 samples represents a different pose/head position
     // Each sample has 5 encoding variants (the 5 consecutive frames captured)
     // With extended enrollment: 15 samples (5 poses × 3 lighting conditions)
     std::vector<std::vector<FaceEncoding>> sample_encodings;
     std::vector<std::vector<float>> quality_scores;
     
     sample_encodings.reserve(num_samples);
     quality_scores.reserve(num_samples);
     
     for (const auto& sample : all_samples) {
         std::vector<FaceEncoding> pose_encodings;
         std::vector<float> pose_qualities;
         
         pose_encodings.reserve(sample.all_encodings.size());
         pose_qualities.reserve(sample.all_encodings.size());
         
         // Calculate quality score for each encoding variant
         for (size_t i = 0; i < sample.all_encodings.size(); ++i) {
             const auto& encoding = sample.all_encodings[i];
             
             // Calculate encoding norm
             float norm = 0.0f;
             for (float val : encoding) {
                 norm += val * val;
             }
             norm = std::sqrt(norm);
             
             // Use sharpness if available, otherwise default to 1.0
             float sharpness = 100.0f;  // Default good sharpness
             
             // Calculate quality score (60% norm + 40% sharpness)
             float quality = calculateFrameQualityScore(norm, sharpness);
             
             pose_encodings.push_back(encoding);
             pose_qualities.push_back(quality);
         }
         
         sample_encodings.push_back(pose_encodings);
         quality_scores.push_back(pose_qualities);
     }
     
     size_t total_encodings = 0;
     for (const auto& pose : sample_encodings) {
         total_encodings += pose.size();
     }
     
     std::cout << "Organized into V2 format: " << sample_encodings.size() << " poses, " 
               << total_encodings << " total encodings" << std::endl;
     std::cout << std::endl;
     
     // Step 2: Calculate optimal recognition threshold
     std::cout << "=== Calculating Optimal Recognition Threshold ===" << std::endl;
     std::cout << "Comparing samples to find best threshold..." << std::endl;
     
     // Flatten encodings for distance calculation
     std::vector<FaceEncoding> all_encodings_flat;
     for (const auto& pose : sample_encodings) {
         for (const auto& encoding : pose) {
             all_encodings_flat.push_back(encoding);
         }
     }
     
     std::vector<float> all_distances;
     for (size_t i = 0; i < all_encodings_flat.size(); i++) {
         for (size_t j = i + 1; j < all_encodings_flat.size(); j++) {
             float dist = cosineDistance(all_encodings_flat[i], all_encodings_flat[j]);
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
     std::cout << "  Based on variation across " << all_encodings_flat.size() << " frames" << std::endl;
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
    
    // Create model for this face (save to user's faces directory) in V2 format
    std::string model_path = cli::getFacesDir(username) + "/" + username + "." + face_id + ".bin";
    BinaryFaceModel model_data;
    model_data.version = 2;  // V2 format with multi-sample encodings
    model_data.username = username;
    model_data.face_ids.push_back(face_id);
    model_data.sample_encodings = sample_encodings;
    model_data.quality_scores = quality_scores;
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
    std::cout << "  Format: V2 (multi-sample)" << std::endl;
    std::cout << "  Poses: " << sample_encodings.size() << std::endl;
    std::cout << "  Total encodings: " << total_encodings << std::endl;
    
    // Show total faces for this user
    int total_faces = 1;  // Since we create one file per face
    std::cout << "  Total faces for " << username << ": " << total_faces << " (this session)" << std::endl;
    std::cout << std::endl;
    
    // Step 3: Update config file with optimal values
    // Save global optimal confidence (applies to all users)
    if (!updateConfigFile(config_path, optimal_confidence, optimal_threshold)) {
        std::cerr << "Warning: Could not update global config values" << std::endl;
    }
    
    // Save per-user threshold (Quick Win #3: Per-User Thresholds)
    if (!faceid::savePerUserThreshold(config_path, username, optimal_threshold)) {
        std::cerr << "Warning: Could not save per-user threshold" << std::endl;
        std::cerr << "You may need to manually add this to " << config_path << ":" << std::endl;
        std::cerr << "  " << username << ".threshold = " << std::fixed << std::setprecision(2) 
                  << optimal_threshold << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "You can now use face authentication for user: " << username << std::endl;
    
    return 0;
}

} // namespace faceid
