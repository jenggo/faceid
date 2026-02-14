#pragma once

#include <string>
#include <vector>
#include <memory>
#include <systemd/sd-bus.h>
#include "../image.h"

/**
 * CLI D-Bus Client - Communicates with system daemon for face processing
 * 
 * This client is used by cmd_add.cpp and cmd_test.cpp to send frames to the daemon
 * for ML processing, avoiding duplicated FaceDetector code in CLI.
 */
class DBusClientCLI {
public:
    struct DetectResult {
        std::vector<faceid::Rect> boxes;
        double quality;
    };
    
    struct EmbeddingResult {
        std::vector<float> embedding;  // 512-dimensional face encoding
    };
    
    struct VerifyResult {
        bool match;
        double confidence;
    };
    
    static DBusClientCLI& instance();
    
    /**
     * Connect to the system D-Bus daemon
     * Returns true if connection successful
     */
    bool connect();
    
    /**
     * Disconnect from the daemon
     */
    void disconnect();
    
    /**
     * Check if connected
     */
    bool is_connected() const { return bus_ != nullptr; }
    
    /**
     * Encode frame as JPEG
     * Returns JPEG-encoded bytes suitable for D-Bus transmission
     */
    static std::vector<uint8_t> encode_frame_as_jpeg(const faceid::Image& frame);
    
    /**
     * Send JPEG frame to daemon for face detection
     * Returns detection results (bounding boxes and quality metric)
     */
    DetectResult detect_face(const std::vector<uint8_t>& jpeg_data);
    
    /**
     * Generate embedding for a face in the frame
     * @param jpeg_data JPEG-encoded frame bytes
     * @param bbox Bounding box of the face region (x, y, width, height)
     * @return 512-dimensional embedding vector
     */
    EmbeddingResult generate_embedding(const std::vector<uint8_t>& jpeg_data,
                                      const faceid::Rect& bbox);
    
    /**
     * Verify a face against stored embeddings
     * @param username User to verify against
     * @param jpeg_data JPEG-encoded frame bytes
     * @return Match result (match boolean, confidence score 0.0-1.0)
     */
    VerifyResult verify_face(const std::string& username,
                            const std::vector<uint8_t>& jpeg_data);
    
    /**
     * Save embedding for a user
     * @param username Username for enrollment
     * @param embedding 512-dimensional face encoding
     * @param pose Description of face pose (e.g., "center", "left", "right")
     */
    bool save_embedding(const std::string& username,
                       const std::vector<float>& embedding,
                       const std::string& pose);
    
    ~DBusClientCLI();
    
private:
    DBusClientCLI();
    
    sd_bus* bus_ = nullptr;
};
