#include "camera_manager.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <syslog.h>
#include <cstring>
#include <cerrno>

CameraManager& CameraManager::instance() {
    static CameraManager instance;
    return instance;
}

bool CameraManager::initialize() {
    if (camera_ready_) {
        return true;  // Already initialized
    }

    if (!find_and_open_camera()) {
        syslog(LOG_ERR, "Failed to initialize camera");
        return false;
    }

    camera_ready_ = true;
    syslog(LOG_INFO, "✓ Camera initialized (%s)", device_path_.c_str());
    return true;
}

bool CameraManager::find_and_open_camera() {
    // Try common video device paths
    const char* device_paths[] = {
        "/dev/video0",
        "/dev/video1",
        "/dev/video2",
        nullptr
    };

    for (int i = 0; device_paths[i]; i++) {
        struct stat st;
        if (stat(device_paths[i], &st) != 0) {
            continue;
        }

        // Check if it's a character device
        if (!S_ISCHR(st.st_mode)) {
            continue;
        }

        // Try to open it
        int fd = open(device_paths[i], O_RDONLY | O_NONBLOCK);
        if (fd >= 0) {
            camera_fd_ = fd;
            device_path_ = device_paths[i];
            syslog(LOG_INFO, "Found camera at %s", device_path_.c_str());
            return true;
        }
    }

    syslog(LOG_WARNING, "No camera device found at standard paths");
    return false;
}

void CameraManager::shutdown() {
    if (camera_fd_ >= 0) {
        close(camera_fd_);
        camera_fd_ = -1;
    }
    camera_ready_ = false;
    syslog(LOG_INFO, "Camera shutdown");
}

CameraManager::~CameraManager() {
    shutdown();
}
