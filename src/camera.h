#ifndef FACEID_CAMERA_H
#define FACEID_CAMERA_H

#include <string>
#include <vector>

// Suppress warnings from OpenCV headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/opencv.hpp>
#pragma GCC diagnostic pop

namespace faceid {

class Camera {
public:
    Camera(const std::string& device_path = "/dev/video0");
    ~Camera();
    
    bool open();
    bool open(int width, int height);
    void close();
    bool isOpened() const;
    
    bool read(cv::Mat& frame);
    
    std::string getDevicePath() const { return device_path_; }
    
    static std::vector<std::string> listDevices();

private:
    std::string device_path_;
    cv::VideoCapture capture_;
};

} // namespace faceid

#endif // FACEID_CAMERA_H
