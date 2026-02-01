#pragma once

#include <memory>
#include <string>

/**
 * Camera Manager - Singleton for managing camera lifecycle
 * 
 * Responsibilities:
 * - Open camera device (lazy on first use, or at daemon start)
 * - Provide frame capture interface
 * - Handle camera disconnects/reconnects gracefully
 * - Report camera availability status
 */
class CameraManager {
public:
    static CameraManager& instance();
    
    /**
     * Initialize camera (can be called multiple times)
     * Returns true if camera is ready, false otherwise
     */
    bool initialize();
    
    /**
     * Check if camera is ready to use
     */
    bool is_ready() const { return camera_ready_; }
    
    /**
     * Get camera device path (e.g., /dev/video0)
     */
    std::string get_device_path() const { return device_path_; }
    
    /**
     * Release camera resources
     */
    void shutdown();
    
    ~CameraManager();
    
private:
    CameraManager() = default;
    
    bool camera_ready_ = false;
    std::string device_path_;
    int camera_fd_ = -1;  // File descriptor for camera device
    
    bool find_and_open_camera();
};
