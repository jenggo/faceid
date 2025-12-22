#include "commands.h"
#include "cli_common.h"

namespace faceid {

using namespace faceid;

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
    
    std::cout << "Loading face detection model..." << std::endl;
    if (!detector.loadModels()) {  // Use default MODELS_DIR/sface path
        std::cerr << "Error: Failed to load face detection model" << std::endl;
        std::cerr << "Expected files: " << MODELS_DIR << "/sface.param and sface.bin" << std::endl;
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

} // namespace faceid
