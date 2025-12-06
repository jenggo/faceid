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
    return capture_.isOpened();
}

bool Camera::open(int width, int height) {
    capture_.open(device_path_, cv::CAP_V4L2);
    if (!capture_.isOpened()) {
        return false;
    }
    
    capture_.set(cv::CAP_PROP_FRAME_WIDTH, width);
    capture_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    
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
