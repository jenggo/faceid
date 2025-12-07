// Suppress warnings from OpenCV headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include "camera.h"
#pragma GCC diagnostic pop

#include <sys/types.h>
#include <dirent.h>
#include <cstring>
#include <algorithm>

namespace faceid {

Camera::Camera(const std::string& device_path) 
    : device_path_(device_path) {
}

Camera::~Camera() {
    close();
}

bool Camera::open() {
    capture_.open(device_path_, cv::CAP_V4L2);
    
    if (!capture_.isOpened()) {
        return false;
    }
    
    // Enable hardware acceleration optimizations
    // Set MJPEG codec for hardware-accelerated decoding on most webcams
    capture_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    
    // Set reasonable FPS for authentication (we don't need high framerate)
    capture_.set(cv::CAP_PROP_FPS, 30);
    
    // Disable auto-exposure for faster, more consistent capture
    capture_.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);  // Manual mode
    
    return true;
}

bool Camera::open(int width, int height) {
    capture_.open(device_path_, cv::CAP_V4L2);
    if (!capture_.isOpened()) {
        return false;
    }
    
    // Set resolution first
    capture_.set(cv::CAP_PROP_FRAME_WIDTH, width);
    capture_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    
    // Enable hardware acceleration optimizations
    // Set MJPEG codec for hardware-accelerated decoding
    capture_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    
    // Set FPS based on resolution
    // Lower resolution = can use higher FPS without CPU penalty
    int target_fps = (width <= 640) ? 30 : 24;
    capture_.set(cv::CAP_PROP_FPS, target_fps);
    
    // Disable auto-exposure for faster, more consistent capture
    capture_.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);  // Manual mode
    
    // Enable hardware buffering if available
    capture_.set(cv::CAP_PROP_BUFFERSIZE, 1);  // Minimal buffering for low latency
    
    return true;
}

void Camera::close() {
    if (capture_.isOpened()) {
        capture_.release();
    }
}

bool Camera::isOpened() const {
    return capture_.isOpened();
}

bool Camera::read(cv::Mat& frame) {
    return capture_.read(frame);
}

std::vector<std::string> Camera::listDevices() {
    std::vector<std::string> devices;
    
    DIR* dir = opendir("/dev");
    if (!dir) {
        return devices;
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (strncmp(entry->d_name, "video", 5) == 0) {
            devices.push_back(std::string("/dev/") + entry->d_name);
        }
    }
    closedir(dir);
    
    std::sort(devices.begin(), devices.end());
    return devices;
}

} // namespace faceid
