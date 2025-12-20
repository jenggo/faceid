#ifndef FACEID_CAMERA_H
#define FACEID_CAMERA_H

#include "image.h"
#include <string>
#include <vector>
#include <turbojpeg.h>

namespace faceid {

class Camera {
public:
    Camera(const std::string& device_path = "/dev/video0");
    ~Camera();
    
    bool open();
    bool open(int width, int height);
    void close();
    bool isOpened() const;
    
    // Read frame into provided Image buffer (reuses allocation if same size)
    bool read(Image& frame);
    
    std::string getDevicePath() const { return device_path_; }
    
    static std::vector<std::string> listDevices();

private:
    std::string device_path_;
    int fd_ = -1;
    int width_ = 640;
    int height_ = 480;
    
    struct Buffer {
        void* start = nullptr;
        size_t length = 0;
    };
    std::vector<Buffer> buffers_;
    
    tjhandle tjhandle_ = nullptr;
    bool streaming_ = false;
};

} // namespace faceid

#endif // FACEID_CAMERA_H
