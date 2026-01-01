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
    
    // Set format - try GREY first (IR cameras are superior), then MJPEG (RGB), then YUYV (fallback)
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width_;
    fmt.fmt.pix.height = height_;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    
    // Try GREY first (IR cameras - best for face recognition!)
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_GREY;
    if (ioctl(fd_, VIDIOC_S_FMT, &fmt) == 0) {
        format_ = FORMAT_GREY;
        Logger::getInstance().info("Camera format: GREY (IR camera) - Optimal for face recognition!");
    } else {
        // Try MJPEG (RGB cameras)
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
        if (ioctl(fd_, VIDIOC_S_FMT, &fmt) == 0) {
            format_ = FORMAT_MJPEG;
            Logger::getInstance().info("Camera format: MJPEG (RGB camera)");
        } else {
            // Try YUYV as fallback (uncompressed RGB)
            fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
            if (ioctl(fd_, VIDIOC_S_FMT, &fmt) == 0) {
                format_ = FORMAT_YUYV;
                Logger::getInstance().info("Camera format: YUYV (RGB camera, uncompressed)");
            } else {
                Logger::getInstance().error("No supported format found for device: " + device_path_);
                close();
                return false;
            }
        }
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
    
    // Initialize TurboJPEG decompressor (only for MJPEG format)
    if (format_ == FORMAT_MJPEG) {
        tjhandle_ = tjInitDecompress();
        if (!tjhandle_) {
            Logger::getInstance().error("Failed to initialize TurboJPEG for device: " + device_path_);
            close();
            return false;
        }
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
    
    bool success = false;
    
    // Handle different camera formats
    if (format_ == FORMAT_MJPEG) {
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
        
        // Decompress to BGR directly into our aligned Image buffer
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
        success = true;
        
    } else if (format_ == FORMAT_GREY) {
        // IR camera - 8-bit grayscale, convert to BGR for compatibility
        // Allocate/reallocate frame if needed
        if (frame.empty() || frame.width() != width_ || frame.height() != height_) {
            frame = Image(width_, height_, 3);  // BGR = 3 channels
        }
        
        // Copy grayscale data and replicate to all 3 channels (B=G=R for grayscale)
        unsigned char* grey_data = (unsigned char*)buffers_[buf.index].start;
        unsigned char* bgr_data = frame.data();
        
        for (int i = 0; i < width_ * height_; i++) {
            unsigned char grey_value = grey_data[i];
            bgr_data[i * 3 + 0] = grey_value;  // B
            bgr_data[i * 3 + 1] = grey_value;  // G
            bgr_data[i * 3 + 2] = grey_value;  // R
        }
        success = true;
        
    } else if (format_ == FORMAT_YUYV) {
        // YUV 4:2:2 format - convert to BGR
        // Allocate/reallocate frame if needed
        if (frame.empty() || frame.width() != width_ || frame.height() != height_) {
            frame = Image(width_, height_, 3);  // BGR = 3 channels
        }
        
        // Convert YUYV to BGR
        unsigned char* yuyv_data = (unsigned char*)buffers_[buf.index].start;
        unsigned char* bgr_data = frame.data();
        
        for (int i = 0; i < width_ * height_ / 2; i++) {
            int y0 = yuyv_data[i * 4 + 0];
            int u  = yuyv_data[i * 4 + 1];
            int y1 = yuyv_data[i * 4 + 2];
            int v  = yuyv_data[i * 4 + 3];
            
            // Convert YUV to RGB (simplified, fast conversion)
            int c = y0 - 16;
            int d = u - 128;
            int e = v - 128;
            
            int r0 = (298 * c + 409 * e + 128) >> 8;
            int g0 = (298 * c - 100 * d - 208 * e + 128) >> 8;
            int b0 = (298 * c + 516 * d + 128) >> 8;
            
            c = y1 - 16;
            int r1 = (298 * c + 409 * e + 128) >> 8;
            int g1 = (298 * c - 100 * d - 208 * e + 128) >> 8;
            int b1 = (298 * c + 516 * d + 128) >> 8;
            
            // Clamp values
            r0 = r0 < 0 ? 0 : (r0 > 255 ? 255 : r0);
            g0 = g0 < 0 ? 0 : (g0 > 255 ? 255 : g0);
            b0 = b0 < 0 ? 0 : (b0 > 255 ? 255 : b0);
            r1 = r1 < 0 ? 0 : (r1 > 255 ? 255 : r1);
            g1 = g1 < 0 ? 0 : (g1 > 255 ? 255 : g1);
            b1 = b1 < 0 ? 0 : (b1 > 255 ? 255 : b1);
            
            // Store as BGR
            bgr_data[i * 6 + 0] = b0;
            bgr_data[i * 6 + 1] = g0;
            bgr_data[i * 6 + 2] = r0;
            bgr_data[i * 6 + 3] = b1;
            bgr_data[i * 6 + 4] = g1;
            bgr_data[i * 6 + 5] = r1;
        }
        success = true;
    }
    
    // Requeue buffer
    if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
        Logger::getInstance().debug("VIDIOC_QBUF (requeue) failed");
        return false;
    }
    
    return success;
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
