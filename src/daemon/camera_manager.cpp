#include "camera_manager.h"
#include <syslog.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

CameraManager::CameraManager() : camera_ready_(false) {
    // Default to /dev/video0, will be configurable later
    device_path_ = "/dev/video0";
}

CameraManager& CameraManager::instance() {
    static CameraManager instance;
    return instance;
}

bool CameraManager::initialize() {
    if (camera_ready_) {
        return true;  // Already initialized
    }

    // Create camera instance
    camera_ = std::make_unique<faceid::Camera>(device_path_);
    
    // Try to open camera with configured resolution
    if (!camera_->open(width_, height_)) {
        syslog(LOG_ERR, "CameraManager: Failed to open camera at %s", device_path_.c_str());
        camera_.reset();
        return false;
    }

    camera_ready_ = true;
    syslog(LOG_INFO, "CameraManager: Camera opened successfully (%s, %dx%d)", 
           device_path_.c_str(), width_, height_);
    return true;
}

faceid::Image CameraManager::capture_frame() {
    if (!camera_ready_ || !camera_) {
        syslog(LOG_WARNING, "CameraManager: Camera not initialized");
        return faceid::Image();
    }
    
    faceid::Image frame;
    if (!camera_->read(frame)) {
        syslog(LOG_ERR, "CameraManager: Failed to capture frame");
        return faceid::Image();
    }
    
    return frame;  // Move semantics - no copy
}

void CameraManager::shutdown() {
    if (camera_) {
        camera_->close();
        camera_.reset();
    }
    camera_ready_ = false;
    syslog(LOG_INFO, "CameraManager: Camera shutdown");
}

CameraManager::~CameraManager() {
    shutdown();
}
