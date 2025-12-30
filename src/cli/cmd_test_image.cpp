#include "commands.h"
#include "cli_common.h"
#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <libyuv.h>
#include "../models/binary_model.h"

// stb_image for loading JPG/PNG images (header-only library)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace faceid {

using namespace faceid;

// Helper: Resize image to target resolution (fast bilinear scaling using libyuv)
static faceid::Image resizeImage(const faceid::Image& src, int target_width, int target_height) {
    // Skip resize if already at target resolution
    if (src.width() == target_width && src.height() == target_height) {
        return src.clone();
    }
    
    // Convert BGR to ARGB
    faceid::Image src_argb(src.width(), src.height(), 4);
    libyuv::RGB24ToARGB(src.data(), src.stride(), src_argb.data(), src_argb.stride(), 
                        src.width(), src.height());
    
    // Scale to target resolution
    faceid::Image dst_argb(target_width, target_height, 4);
    libyuv::ARGBScale(
        src_argb.data(), src_argb.stride(),
        src_argb.width(), src_argb.height(),
        dst_argb.data(), dst_argb.stride(),
        dst_argb.width(), dst_argb.height(),
        libyuv::kFilterBilinear
    );
    
    // Convert back to BGR
    faceid::Image result(target_width, target_height, 3);
    libyuv::ARGBToRGB24(dst_argb.data(), dst_argb.stride(), result.data(), result.stride(), 
                        target_width, target_height);
    return result;
}

// Helper: Calculate L2 norm of a vector
static float calculateNorm(const std::vector<float>& vec) {
    float sum = 0.0f;
    for (float val : vec) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

// Helper: Calculate cosine distance
static float cosineDistance(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    float dot = 0.0f;
    for (size_t i = 0; i < vec1.size(); i++) {
        dot += vec1[i] * vec2[i];
    }
    // Clamp dot product to [-1, 1] to handle floating point precision errors
    if (dot > 1.0f) dot = 1.0f;
    if (dot < -1.0f) dot = -1.0f;
    return 1.0f - dot;
}

// Helper: Load image using stb_image
static bool loadImage(const std::string& path, faceid::Image& out_image) {
    int width, height, channels;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 3);  // Force RGB
    
    if (!data) {
        std::cerr << "Error: Failed to load image: " << path << std::endl;
        std::cerr << "  Reason: " << stbi_failure_reason() << std::endl;
        return false;
    }
    
    // Create faceid::Image and copy data
    faceid::Image img(width, height, 3);
    std::memcpy(img.data(), data, width * height * 3);
    
    // Free stb_image data
    stbi_image_free(data);
    
    out_image = std::move(img);
    return true;
}

// Helper: Encode a single face from image
static bool encodeFaceFromImage(FaceDetector& detector, const std::string& image_path, 
                                std::vector<FaceEncoding>& out_encodings, float confidence_threshold,
                                int target_width, int target_height) {
    // Load image
    faceid::Image frame;
    if (!loadImage(image_path, frame)) {
        return false;
    }

    std::cout << "  Original size: " << frame.width() << "x" << frame.height() << std::endl;
    
    // SCRFD does its own aspect-ratio preserving resize, so skip pre-resizing for SCRFD
    faceid::Image resized_frame;
    if (detector.getDetectionModelType() == "SCRFD") {
        std::cout << "  Using original resolution (SCRFD handles its own resizing)" << std::endl;
        resized_frame = frame.clone();
    } else {
        // Other models need pre-resizing to camera resolution
        resized_frame = resizeImage(frame, target_width, target_height);
        std::cout << "  Resized to: " << resized_frame.width() << "x" << resized_frame.height() 
                  << " (camera resolution)" << std::endl;
    }

    // Preprocess and detect faces
    faceid::Image processed_frame = detector.preprocessFrame(resized_frame.view());
    auto faces = detector.detectFaces(processed_frame.view(), false, confidence_threshold);

    if (faces.empty()) {
        std::cerr << "Error: No faces detected in enrollment image: " << image_path << std::endl;
        return false;
    }

    if (faces.size() > 1) {
        std::cout << "Warning: Multiple faces detected in enrollment image. Using first face." << std::endl;
    }

    // Encode the first face
    auto encodings = detector.encodeFaces(processed_frame.view(), faces);
    if (encodings.empty()) {
        std::cerr << "Error: Failed to encode face from: " << image_path << std::endl;
        return false;
    }

    out_encodings = encodings;
    return true;
}

// Helper: Validate if a detected face is likely a real face
static bool isValidFace(const Rect& face, int img_width, int img_height, 
                       const std::vector<float>& encoding) {
    // Check 1: Face size (should be 10-80% of image width)
    float face_width_ratio = (float)face.width / img_width;
    if (face_width_ratio < 0.10f || face_width_ratio > 0.80f) {
        return false;
    }
    
    // Check 2: Aspect ratio (faces should be roughly 1:1 to 1:1.5 - width:height)
    float aspect_ratio = (float)face.width / face.height;
    if (aspect_ratio < 0.6f || aspect_ratio > 1.8f) {
        return false;
    }
    
    // Check 3: Position (face center should be in middle 80% of image)
    float face_center_x = (face.x + face.width / 2.0f) / img_width;
    float face_center_y = (face.y + face.height / 2.0f) / img_height;
    if (face_center_x < 0.1f || face_center_x > 0.9f || 
        face_center_y < 0.1f || face_center_y > 0.9f) {
        return false;
    }
    
    // Check 4: Encoding quality (L2 norm should be close to 1.0 for normalized embeddings)
    if (!encoding.empty()) {
        float norm = 0.0f;
        for (float val : encoding) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        
        // Good face encodings should have norm between 0.95 and 1.05
        if (norm < 0.90f || norm > 1.10f) {
            return false;
        }
    }
    
    return true;
}

// Helper: Test detection at different confidence thresholds using binary search
// Returns the optimal confidence threshold found, or -1.0f if not found
static float testDetectionConfidence(FaceDetector& detector, const std::string& image_path,
                                   int target_width, int target_height) {
    std::cout << std::endl;
    std::cout << "=== Detection Confidence Analysis ===" << std::endl;
    std::cout << "Finding optimal confidence threshold..." << std::endl;
    std::cout << std::endl;
    
    // Load image once
    faceid::Image frame;
    if (!loadImage(image_path, frame)) {
        return -1.0f;
    }
    
    // SCRFD does its own resizing, others need pre-resize
    faceid::Image resized_frame;
    if (detector.getDetectionModelType() == "SCRFD") {
        resized_frame = frame.clone();
    } else {
        resized_frame = resizeImage(frame, target_width, target_height);
    }
    
    faceid::Image processed_frame = detector.preprocessFrame(resized_frame.view());
    
    int img_width = resized_frame.width();
    int img_height = resized_frame.height();
    
    // Helper lambda to count valid faces at a given confidence
    auto countValidFaces = [&](float conf) -> int {
        auto faces = detector.detectFaces(processed_frame.view(), false, conf);
        auto encodings = detector.encodeFaces(processed_frame.view(), faces);
        
        int valid_count = 0;
        for (size_t i = 0; i < faces.size(); i++) {
            std::vector<float> encoding = (i < encodings.size()) ? encodings[i] : std::vector<float>();
            if (isValidFace(faces[i], img_width, img_height, encoding)) {
                valid_count++;
            }
        }
        return valid_count;
    };
    
    // Binary search for optimal confidence threshold
    float low = 0.30f;
    float high = 0.99f;
    float found_confidence = -1.0f;
    std::vector<Rect> found_faces;
    std::vector<std::vector<float>> found_encodings;
    
    // First, do a coarse linear search to find a good starting range
    float coarse_step = 0.10f;
    for (float conf = low; conf <= high; conf += coarse_step) {
        int valid_count = countValidFaces(conf);
        
        if (valid_count == 1) {
            // Found a good candidate, now refine with binary search
            low = std::max(0.30f, conf - coarse_step);
            high = std::min(0.99f, conf + coarse_step);
            break;
        } else if (valid_count == 0) {
            // Went too high
            high = conf;
            break;
        }
    }
    
    // Binary search refinement with 0.01 precision
    while (high - low > 0.01f) {
        float mid = (low + high) / 2.0f;
        int valid_count = countValidFaces(mid);
        
        if (valid_count == 1) {
            // Found it! Try to find the lowest confidence that still works
            found_confidence = mid;
            
            // Get the actual faces for this confidence
            auto faces = detector.detectFaces(processed_frame.view(), false, mid);
            auto encodings = detector.encodeFaces(processed_frame.view(), faces);
            found_faces = faces;
            found_encodings = encodings;
            
            high = mid;  // Try to find lower confidence
        } else if (valid_count > 1) {
            // Too many faces, increase confidence
            low = mid;
        } else {
            // No faces, decrease confidence
            high = mid;
        }
    }
    
    // If not found yet, try the final candidate
    if (found_confidence < 0.0f) {
        int valid_count = countValidFaces(low);
        if (valid_count == 1) {
            found_confidence = low;
            auto faces = detector.detectFaces(processed_frame.view(), false, low);
            auto encodings = detector.encodeFaces(processed_frame.view(), faces);
            found_faces = faces;
            found_encodings = encodings;
        }
    }
    
    // Report result
    if (found_confidence > 0.0f) {
        std::cout << "✓ Found optimal confidence: " << std::fixed << std::setprecision(2) 
                  << found_confidence << std::endl;
        std::cout << "  Detected exactly 1 valid face" << std::endl;
        std::cout << std::endl;
        
        // Show face details
        for (size_t i = 0; i < found_faces.size(); i++) {
            const auto& face = found_faces[i];
            std::vector<float> encoding = (i < found_encodings.size()) ? 
                found_encodings[i] : std::vector<float>();
            
            if (isValidFace(face, img_width, img_height, encoding)) {
                std::cout << "Face details:" << std::endl;
                std::cout << "  Position: (" << face.x << ", " << face.y << ")" << std::endl;
                std::cout << "  Size: " << face.width << "x" << face.height << " pixels" << std::endl;
                std::cout << "  Size ratio: " << std::fixed << std::setprecision(1) 
                          << ((float)face.width / img_width * 100.0f) << "% of image width" << std::endl;
                
                // Show encoding L2 norm
                if (!encoding.empty()) {
                    float norm = 0.0f;
                    for (float val : encoding) {
                        norm += val * val;
                    }
                    norm = std::sqrt(norm);
                    std::cout << "  Encoding L2 norm: " << std::fixed << std::setprecision(4) 
                              << norm << std::endl;
                }
                break;
            }
        }
        
        std::cout << std::endl;
        std::cout << "Recommendation: Update config/faceid.conf with:" << std::endl;
        std::cout << "  [recognition]" << std::endl;
        std::cout << "  confidence = " << std::fixed << std::setprecision(2) 
                  << found_confidence << "  # For " << detector.getDetectionModelType() << std::endl;
    } else {
        std::cerr << "✗ FAILED: Could not find optimal confidence threshold" << std::endl;
        std::cerr << "  Tested range: 0.30 to 0.99" << std::endl;
        std::cerr << "  The enrollment image does not contain exactly 1 detectable face." << std::endl;
        std::cerr << std::endl;
        std::cerr << "Possible issues:" << std::endl;
        std::cerr << "  - Enrollment image contains multiple people" << std::endl;
        std::cerr << "  - Face in image is too small/large" << std::endl;
        std::cerr << "  - Image quality is poor" << std::endl;
        std::cerr << "  - Detection model not suitable for this image" << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "Detection model: " << detector.getDetectionModelName() 
              << " (" << detector.getDetectionModelType() << ")" << std::endl;
    
    return found_confidence;
}

int cmd_test_image(const std::vector<std::string>& args) {
    // Parse arguments with flags
    std::string enrollment_image_path;
    std::string test_image_path;
    float confidence_threshold = 0.0f;  // 0 = use config default
    bool verbose = false;
    
    for (size_t i = 0; i < args.size(); i++) {
        if (args[i] == "--enroll" && i + 1 < args.size()) {
            enrollment_image_path = args[i + 1];
            i++;
        } else if (args[i] == "--test" && i + 1 < args.size()) {
            test_image_path = args[i + 1];
            i++;
        } else if (args[i] == "--confidence" && i + 1 < args.size()) {
            try {
                confidence_threshold = std::stof(args[i + 1]);
                if (confidence_threshold < 0.0f || confidence_threshold > 1.0f) {
                    std::cerr << "Error: --confidence must be between 0.0 and 1.0" << std::endl;
                    return 1;
                }
            } catch (...) {
                std::cerr << "Error: Invalid confidence value: " << args[i + 1] << std::endl;
                return 1;
            }
            i++;
        } else if (args[i] == "--verbose" || args[i] == "-v") {
            verbose = true;
        }
    }
    
    if (enrollment_image_path.empty() || test_image_path.empty()) {
        std::cerr << "Usage: faceid image test --enroll <enrollment_image> --test <test_image> [options]" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  --enroll <image>       Face image to use as enrolled reference" << std::endl;
        std::cerr << "  --test <image>         Image with faces to test against reference" << std::endl;
        std::cerr << "  --confidence <0.0-1.0> Detection confidence threshold (default: from config)" << std::endl;
        std::cerr << "                         Higher values = stricter detection, fewer false positives" << std::endl;
        std::cerr << "                         RetinaFace/YuNet: 0.8 recommended, SCRFD/UltraFace: 0.5" << std::endl;
        std::cerr << "  --verbose, -v          Show detailed analysis and debug information" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Example: faceid image test --enroll single-face.jpg --test two-faces.jpg --confidence 0.9" << std::endl;
        std::cerr << "  This will enroll the face from single-face.jpg and test" << std::endl;
        std::cerr << "  all faces in two-faces.jpg against it with 90% confidence threshold." << std::endl;
        return 1;
    }

    std::string test_username = "test-user";  // Hardcoded test username

    if (verbose) {
        std::cout << "=== FaceID Static Image Test ===" << std::endl;
        std::cout << "Enrollment image: " << enrollment_image_path << std::endl;
        std::cout << "Test image: " << test_image_path << std::endl;
        std::cout << "Test username: " << test_username << std::endl << std::endl;
    }

    // Load configuration
    faceid::Config& config = faceid::Config::getInstance();
    std::string config_path = std::string(CONFIG_DIR) + "/faceid.conf";
    config.load(config_path);

    double recognition_threshold = config.getDouble("recognition", "threshold").value_or(0.6);
    double detection_confidence_config = config.getDouble("recognition", "confidence").value_or(0.8);
    
    // Load camera resolution for image normalization
    int camera_width = config.getInt("camera", "width").value_or(640);
    int camera_height = config.getInt("camera", "height").value_or(480);
    
    // Use CLI override if provided, otherwise use config value
    float detection_confidence = (confidence_threshold > 0.0f) ? confidence_threshold : static_cast<float>(detection_confidence_config);

    if (verbose) {
        std::cout << "Configuration Thresholds:" << std::endl;
        std::cout << "  Recognition threshold: " << recognition_threshold << std::endl;
        std::cout << "  Detection confidence: " << detection_confidence;
        if (confidence_threshold > 0.0f) {
            std::cout << " (CLI override)";
        } else {
            std::cout << " (from config)";
        }
        std::cout << std::endl;
        std::cout << "  Camera resolution: " << camera_width << "x" << camera_height 
                  << " (images will be resized to match)" << std::endl << std::endl;

        // Initialize face detector
        std::cout << "Loading face detection and recognition models..." << std::endl;
    }
    faceid::FaceDetector detector;
    if (!detector.loadModels()) {
        std::cerr << "Error: Failed to load face recognition model" << std::endl;
        return 1;
    }
    
    std::string models_dir = std::string(MODELS_DIR);
    if (verbose) {
        std::cout << "✓ Models loaded successfully" << std::endl;
        std::cout << "  Models directory: " << models_dir << std::endl;
        std::cout << "  Detection model: " << detector.getDetectionModelName() << " (" << detector.getDetectionModelType() << ")" << std::endl;
        std::cout << "  Recognition model: " << detector.getModelName() << " (" << detector.getRecognitionModelType() << ", " << detector.getEncodingDimension() << "D)" << std::endl;
        std::cout << "  Encoding dimension: " << detector.getEncodingDimension() << "D" << std::endl;
        std::cout << std::endl;
    }
    
    // Run confidence analysis if confidence not explicitly set by user
    float optimal_detection_confidence = -1.0f;
    if (confidence_threshold <= 0.0f) {
        optimal_detection_confidence = testDetectionConfidence(detector, enrollment_image_path, camera_width, camera_height);
    }

    // Step 1: Encode the enrollment face (single-face.jpg)
    if (verbose) {
        std::cout << "=== Step 1: Encoding Reference Face ===" << std::endl;
    }
    std::vector<FaceEncoding> reference_encodings;
    if (!encodeFaceFromImage(detector, enrollment_image_path, reference_encodings, detection_confidence, 
                            camera_width, camera_height)) {
        // If enrollment failed and we haven't run confidence analysis yet, run it now
        if (confidence_threshold > 0.0f) {
            testDetectionConfidence(detector, enrollment_image_path, camera_width, camera_height);
        }
        return 1;
    }
    std::cout << "✓ Reference face encoded successfully" << std::endl;
    std::cout << "  Encoding dimension: " << reference_encodings[0].size() << "D" << std::endl;
    std::cout << "  L2 norm: " << std::fixed << std::setprecision(4) 
              << calculateNorm(reference_encodings[0]) << std::endl;
    std::cout << std::endl;

    // Step 2: Load and detect faces in test image
    std::cout << "=== Step 2: Loading Test Image ===" << std::endl;
    faceid::Image test_frame;
    if (!loadImage(test_image_path, test_frame)) {
        return 1;
    }
    std::cout << "✓ Test image loaded" << std::endl;
    std::cout << "  Original size: " << test_frame.width() << "x" << test_frame.height() << std::endl;
    
    // SCRFD does its own resizing, others need pre-resize
    faceid::Image resized_test;
    if (detector.getDetectionModelType() == "SCRFD") {
        std::cout << "  Using original resolution (SCRFD handles its own resizing)" << std::endl;
        resized_test = test_frame.clone();
    } else {
        // Resize to camera resolution for consistent face detection
        resized_test = resizeImage(test_frame, camera_width, camera_height);
        std::cout << "  Resized to: " << resized_test.width() << "x" << resized_test.height() 
                  << " (camera resolution)" << std::endl;
    }
    std::cout << std::endl;

    // Step 3: Detect faces
    std::cout << "=== Step 3: Detecting Faces ===" << std::endl;
    faceid::Image processed_frame = detector.preprocessFrame(resized_test.view());
    
    auto detect_start = std::chrono::high_resolution_clock::now();
    auto detected_faces = detector.detectFaces(processed_frame.view(), false, detection_confidence);
    auto detect_end = std::chrono::high_resolution_clock::now();
    double detection_time = std::chrono::duration<double, std::milli>(detect_end - detect_start).count();

    std::cout << "✓ Detection complete" << std::endl;
    std::cout << "  Faces detected: " << detected_faces.size() << std::endl;
    std::cout << "  Detection time: " << std::fixed << std::setprecision(2) 
              << detection_time << " ms" << std::endl;
    std::cout << std::endl;

    if (detected_faces.empty()) {
        std::cout << "Warning: No faces detected in test image" << std::endl;
        return 0;
    }

    // Step 4: Encode and compare each detected face
    std::cout << "=== Step 4: Face Recognition Analysis ===" << std::endl;
    auto encode_start = std::chrono::high_resolution_clock::now();
    auto test_encodings = detector.encodeFaces(processed_frame.view(), detected_faces);
    auto encode_end = std::chrono::high_resolution_clock::now();
    double encoding_time = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();

    std::cout << "✓ Encoding complete" << std::endl;
    std::cout << "  Encoding time: " << std::fixed << std::setprecision(2) 
              << encoding_time << " ms (for " << test_encodings.size() << " faces)" << std::endl;
    std::cout << std::endl;

    // Track false positives
    int verified_count = 0;
    int rejected_count = 0;

    // Analyze each detected face
    for (size_t i = 0; i < detected_faces.size() && i < test_encodings.size(); i++) {
        const auto& face = detected_faces[i];
        const auto& encoding = test_encodings[i];

        std::cout << "Face #" << (i + 1) << ":" << std::endl;
        std::cout << "  Bounding box: [" << face.x << ", " << face.y << ", " 
                  << (face.x + face.width) << ", " << (face.y + face.height) << "]" << std::endl;
        std::cout << "  Size: " << face.width << "x" << face.height << " pixels" << std::endl;

        // Encoding info
        std::cout << "  Encoding:" << std::endl;
        std::cout << "    Dimension: " << encoding.size() << "D" << std::endl;
        std::cout << "    L2 norm: " << std::fixed << std::setprecision(4) 
                  << calculateNorm(encoding) << std::endl;
        
        // Show first few values of encoding vector
        std::cout << "    First 10 values: [";
        for (size_t j = 0; j < std::min(size_t(10), encoding.size()); j++) {
            std::cout << std::fixed << std::setprecision(3) << encoding[j];
            if (j < std::min(size_t(10), encoding.size()) - 1) std::cout << ", ";
        }
        std::cout << "...]" << std::endl;

        // Compare with reference encoding
        float distance = cosineDistance(reference_encodings[0], encoding);
        bool is_match = distance < recognition_threshold;

        std::cout << "  Recognition:" << std::endl;
        std::cout << "    Cosine distance: " << std::fixed << std::setprecision(4) << distance << std::endl;
        std::cout << "    Threshold: " << recognition_threshold << std::endl;
        std::cout << "    Match confidence: " << std::fixed << std::setprecision(1) 
                  << ((1.0f - distance) * 100.0f) << "%" << std::endl;
        
        if (is_match) {
            std::cout << "    Result: ✓ VERIFIED (distance " << std::fixed << std::setprecision(4) 
                      << distance << " < " << recognition_threshold << ")" << std::endl;
            verified_count++;
        } else {
            std::cout << "    Result: ✗ REJECTED (distance " << std::fixed << std::setprecision(4) 
                      << distance << " > " << recognition_threshold << ")" << std::endl;
            rejected_count++;
        }

        std::cout << std::endl;
    }

    // Step 5: Summary and false positive analysis
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "Total faces detected: " << detected_faces.size() << std::endl;
    std::cout << "Verified: " << verified_count << std::endl;
    std::cout << "Rejected: " << rejected_count << std::endl;
    std::cout << std::endl;

    // False positive rate analysis
    // Expected: 1 face should verify (left face), rest should be rejected
    int expected_matches = 1;
    int false_positives = std::max(0, verified_count - expected_matches);
    int false_negatives = (verified_count == 0 && detected_faces.size() > 0) ? 1 : 0;

    std::cout << "=== False Positive Analysis ===" << std::endl;
    if (false_positives > 0) {
        std::cout << "⚠ WARNING: " << false_positives << " false positive(s) detected!" << std::endl;
        std::cout << "  Expected: " << expected_matches << " verified face(s)" << std::endl;
        std::cout << "  Actual: " << verified_count << " verified face(s)" << std::endl;
        std::cout << "  False positive rate: " << std::fixed << std::setprecision(1) 
                  << (float(false_positives) / detected_faces.size() * 100.0f) << "%" << std::endl;
        std::cout << std::endl;
        std::cout << "Recommendation: Increase recognition threshold in config/faceid.conf" << std::endl;
        std::cout << "  Current threshold: " << recognition_threshold << std::endl;
        std::cout << "  Suggested threshold: Try values between " 
                  << std::fixed << std::setprecision(2) 
                  << (recognition_threshold - 0.1) << " and " << (recognition_threshold - 0.05) << std::endl;
    } else if (false_negatives > 0) {
        std::cout << "⚠ WARNING: Expected face was not verified (false negative)" << std::endl;
        std::cout << "  This means the legitimate user was rejected" << std::endl;
        std::cout << std::endl;
        std::cout << "Recommendation: Decrease recognition threshold in config/faceid.conf" << std::endl;
        std::cout << "  Current threshold: " << recognition_threshold << std::endl;
        std::cout << "  Suggested threshold: Try values between " 
                  << std::fixed << std::setprecision(2) 
                  << (recognition_threshold + 0.05) << " and " << (recognition_threshold + 0.1) << std::endl;
    } else if (verified_count == expected_matches && rejected_count == (int)detected_faces.size() - expected_matches) {
        std::cout << "✓ PASS: Recognition working correctly" << std::endl;
        std::cout << "  No false positives detected" << std::endl;
        std::cout << "  Expected face verified successfully" << std::endl;
        std::cout << "  False positive rate: 0%" << std::endl;
        std::cout << std::endl;
        std::cout << "Current threshold (" << recognition_threshold << ") is appropriate." << std::endl;
    } else {
        std::cout << "ℹ Unexpected result pattern" << std::endl;
        std::cout << "  Review the individual face results above" << std::endl;
    }

    // Step 6: Automatic Recognition Threshold Finder
    std::cout << std::endl;
    std::cout << "=== Automatic Recognition Threshold Finder ===" << std::endl;
    
    // Collect all distances with their face indices
    std::vector<std::pair<float, size_t>> distance_pairs;
    for (size_t i = 0; i < detected_faces.size() && i < test_encodings.size(); i++) {
        const auto& encoding = test_encodings[i];
        float distance = cosineDistance(reference_encodings[0], encoding);
        distance_pairs.push_back({distance, i});
    }
    
    // Sort by distance
    std::sort(distance_pairs.begin(), distance_pairs.end());
    
    // Find optimal threshold
    // Strategy: Find the largest gap between consecutive distances
    // The threshold should be in the middle of that gap
    float optimal_threshold = -1.0f;
    
    if (distance_pairs.size() >= 2) {
        float max_gap = 0.0f;
        float gap_midpoint = -1.0f;
        
        // Find the largest gap between the first face and any subsequent face
        for (size_t i = 0; i < distance_pairs.size() - 1; i++) {
            float gap = distance_pairs[i + 1].first - distance_pairs[i].first;
            
            // We want the gap after the first face (index 0 in sorted list)
            // This will separate the enrolled face from all other faces
            if (i == 0 && gap > max_gap) {
                max_gap = gap;
                gap_midpoint = (distance_pairs[i].first + distance_pairs[i + 1].first) / 2.0f;
            }
        }
        
        // Verify this threshold gives us exactly 1 verified face
        if (gap_midpoint > 0) {
            int verified_count = 0;
            for (const auto& pair : distance_pairs) {
                if (pair.first < gap_midpoint) {
                    verified_count++;
                }
            }
            
            if (verified_count == 1) {
                optimal_threshold = gap_midpoint;
            }
        }
    } else if (distance_pairs.size() == 1) {
        // Only one face, recommend threshold slightly above its distance
        optimal_threshold = distance_pairs[0].first + 0.05f;
    }
    
    if (optimal_threshold > 0) {
        std::cout << "✓ Optimal recognition threshold found: " << std::fixed << std::setprecision(2) 
                  << optimal_threshold << std::endl;
        std::cout << "  This threshold will verify exactly 1 face (the enrolled face)" << std::endl;
        std::cout << std::endl;
        std::cout << "Recommendation: Update config/faceid.conf with:" << std::endl;
        std::cout << "  [recognition]" << std::endl;
        std::cout << "  threshold = " << std::fixed << std::setprecision(2) << optimal_threshold << std::endl;
        
        // Also show the confidence recommendation from earlier
        std::string model_type = detector.getDetectionModelType();
        float recommended_confidence = 0.8f;  // Default for RetinaFace/YuNet
        if (model_type == "SCRFD" || model_type == "UltraFace") {
            recommended_confidence = 0.5f;
        }
        
        // Use the optimal confidence found earlier if available
        if (optimal_detection_confidence > 0.0f) {
            recommended_confidence = optimal_detection_confidence;
        }
        
        std::cout << "  confidence = " << std::fixed << std::setprecision(2) << recommended_confidence 
                  << "  # For " << model_type << std::endl;
    } else {
        std::cout << "⚠ Could not automatically determine optimal threshold" << std::endl;
        if (!distance_pairs.empty()) {
            std::cout << "  Closest match distance: " << std::fixed << std::setprecision(4) 
                      << distance_pairs[0].first << std::endl;
            std::cout << "  Try setting threshold to: " << std::fixed << std::setprecision(2) 
                      << (distance_pairs[0].first + 0.05f) << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "=== Performance Summary ===" << std::endl;
    std::cout << "Detection time: " << std::fixed << std::setprecision(2) << detection_time << " ms" << std::endl;
    std::cout << "Encoding time: " << encoding_time << " ms" << std::endl;
    std::cout << "Total time: " << (detection_time + encoding_time) << " ms" << std::endl;
    std::cout << std::endl;
    
    std::cout << "=== Models Information ===" << std::endl;
    std::cout << "Models directory: " << models_dir << std::endl;
    std::cout << "Detection model: " << detector.getDetectionModelName() << " (" << detector.getDetectionModelType() << ")" << std::endl;
    std::cout << "Recognition model: " << detector.getModelName() << " (" << detector.getRecognitionModelType() << ", " << detector.getEncodingDimension() << "D)" << std::endl;
    std::cout << std::endl;

    return 0;
}

} // namespace faceid
