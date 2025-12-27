#include "camera.h"
#include "logger.h"
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <cstring>
#include <cerrno>
#include <algorithm>
#include <linux/videodev2.h>

namespace faceid {

Camera::Camera(const std::string& device_path) 
    : device_path_(device_path), fd_(-1), width_(640), height_(480), 
      tjhandle_(nullptr), streaming_(false) {
}

Camera::~Camera() {
    close();
}

bool Camera::open() {
    return open(640, 480);
}

bool Camera::open(int width, int height) {
    if (fd_ >= 0) {
        close();
    }
    
    width_ = width;
    height_ = height;
    
    // Open device
    fd_ = ::open(device_path_.c_str(), O_RDWR);
    if (fd_ < 0) {
        Logger::getInstance().error("Failed to open camera device: " + device_path_ + 
                                   " (errno: " + std::to_string(errno) + " - " + strerror(errno) + ")");
        return false;
    }
    
    // Query capabilities
    struct v4l2_capability cap;
    if (ioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0) {
        Logger::getInstance().error("VIDIOC_QUERYCAP failed for device: " + device_path_);
        close();
        return false;
    }
    
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        Logger::getInstance().error("Device does not support video capture: " + device_path_);
        close();
        return false;
    }
    
    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        Logger::getInstance().error("Device does not support streaming: " + device_path_);
        close();
        return false;
    }
    
    // Set format to MJPEG
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width_;
    fmt.fmt.pix.height = height_;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    
    if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
        Logger::getInstance().error("VIDIOC_S_FMT failed (MJPEG format) for device: " + device_path_);
        close();
        return false;
    }
    
    // Update actual resolution
    width_ = fmt.fmt.pix.width;
    height_ = fmt.fmt.pix.height;
    
    // Set framerate to 30 FPS
    struct v4l2_streamparm parm;
    memset(&parm, 0, sizeof(parm));
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = 30;
    if (ioctl(fd_, VIDIOC_S_PARM, &parm) < 0) {
        // Not critical, continue anyway
    }
    
    // Request buffers (use 4 buffers for smooth streaming)
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    
    if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
        Logger::getInstance().error("VIDIOC_REQBUFS failed for device: " + device_path_);
        close();
        return false;
    }
    
    if (req.count < 2) {
        Logger::getInstance().error("Insufficient buffer memory for device: " + device_path_);
        close();
        return false;
    }
    
    // Map buffers
    buffers_.resize(req.count);
    for (unsigned int i = 0; i < req.count; i++) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        
        if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
            Logger::getInstance().error("VIDIOC_QUERYBUF failed for device: " + device_path_);
            close();
            return false;
        }
        
        buffers_[i].length = buf.length;
        buffers_[i].start = mmap(NULL, buf.length,
                                PROT_READ | PROT_WRITE,
                                MAP_SHARED,
                                fd_, buf.m.offset);
        
        if (buffers_[i].start == MAP_FAILED) {
            Logger::getInstance().error("mmap failed for buffer " + std::to_string(i) + " on device: " + device_path_);
            close();
            return false;
        }
    }
    
    // Queue all buffers
    for (unsigned int i = 0; i < buffers_.size(); i++) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        
        if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
            Logger::getInstance().error("VIDIOC_QBUF failed during initialization for device: " + device_path_);
            close();
            return false;
        }
    }
    
    // Start streaming
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
        Logger::getInstance().error("VIDIOC_STREAMON failed for device: " + device_path_);
        close();
        return false;
    }
    streaming_ = true;
    
    // Initialize TurboJPEG decompressor
    tjhandle_ = tjInitDecompress();
    if (!tjhandle_) {
        Logger::getInstance().error("Failed to initialize TurboJPEG for device: " + device_path_);
        close();
        return false;
    }
    
    return true;
}

void Camera::close() {
    // Stop streaming
    if (streaming_ && fd_ >= 0) {
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl(fd_, VIDIOC_STREAMOFF, &type);
        streaming_ = false;
    }
    
    // Unmap buffers
    for (auto& buf : buffers_) {
        if (buf.start != MAP_FAILED && buf.start != nullptr) {
            munmap(buf.start, buf.length);
        }
    }
    buffers_.clear();
    
    // Close device
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    
    // Destroy TurboJPEG handle
    if (tjhandle_) {
        tjDestroy(tjhandle_);
        tjhandle_ = nullptr;
    }
}

bool Camera::isOpened() const {
    return fd_ >= 0 && streaming_;
}

bool Camera::read(Image& frame) {
    if (!isOpened()) {
        return false;
    }
    
    // Dequeue buffer
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    
    if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
        Logger::getInstance().debug("VIDIOC_DQBUF failed: " + std::string(strerror(errno)));
        return false;
    }
    
    if (buf.index >= buffers_.size()) {
        Logger::getInstance().error("Invalid buffer index from V4L2");
        ioctl(fd_, VIDIOC_QBUF, &buf);
        return false;
    }
    
    // Decompress MJPEG using TurboJPEG
    int jpeg_width, jpeg_height, jpeg_subsamp, jpeg_colorspace;
    if (tjDecompressHeader3(tjhandle_, 
                           (unsigned char*)buffers_[buf.index].start,
                           buf.bytesused,
                           &jpeg_width, &jpeg_height, 
                           &jpeg_subsamp, &jpeg_colorspace) < 0) {
        Logger::getInstance().debug("tjDecompressHeader3 failed: " + std::string(tjGetErrorStr()));
        ioctl(fd_, VIDIOC_QBUF, &buf);
        return false;
    }
    
    // Allocate/reallocate frame if needed (reuses memory if same size)
    if (frame.empty() || frame.width() != jpeg_width || frame.height() != jpeg_height) {
        frame = Image(jpeg_width, jpeg_height, 3);  // BGR = 3 channels
    }
    
    // Decompress to RGB directly into our aligned Image buffer
    if (tjDecompress2(tjhandle_,
                     (unsigned char*)buffers_[buf.index].start,
                     buf.bytesused,
                     frame.data(),
                     jpeg_width, 0, jpeg_height,
                     TJPF_BGR,
                     TJFLAG_FASTDCT) < 0) {
        Logger::getInstance().debug("tjDecompress2 failed: " + std::string(tjGetErrorStr()));
        ioctl(fd_, VIDIOC_QBUF, &buf);
        return false;
    }
    
    // Requeue buffer
    if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
        Logger::getInstance().debug("VIDIOC_QBUF (requeue) failed");
        return false;
    }
    
    return true;
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
