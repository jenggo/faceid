#include "input_monitor.h"
#include "../logger.h"
#include <libevdev/libevdev.h>
#include <linux/input.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/select.h>
#include <fstream>
#include <sstream>
#include <cstring>

namespace faceid {

InputMonitor::InputMonitor() 
    : last_activity_raw_(std::chrono::steady_clock::now().time_since_epoch().count()) {
}

InputMonitor::~InputMonitor() {
    stop();
}

bool InputMonitor::initialize() {
    Logger& logger = Logger::getInstance();
    logger.info("Initializing input monitor with libevdev");
    
    // Scan for input devices
    scanInputDevices();
    
    if (devices_.empty()) {
        logger.error("No input devices found for monitoring");
        return false;
    }
    
    logger.info("Input monitor initialized with " + std::to_string(devices_.size()) + " device(s)");
    return true;
}

void InputMonitor::start() {
    if (running_.load()) {
        return;  // Already running
    }
    
    Logger::getInstance().info("Starting input monitoring thread");
    running_.store(true);
    event_thread_ = std::thread(&InputMonitor::eventLoop, this);
}

void InputMonitor::stop() {
    if (!running_.load()) {
        return;  // Already stopped
    }
    
    Logger& logger = Logger::getInstance();
    logger.info("Stopping input monitoring thread");
    
    running_.store(false);
    
    if (event_thread_.joinable()) {
        event_thread_.join();
    }
    
    closeAllDevices();
}

void InputMonitor::setActivityCallback(std::function<void()> callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    activity_callback_ = std::move(callback);
}

std::chrono::steady_clock::time_point InputMonitor::getLastActivity() const {
    return std::chrono::steady_clock::time_point(
        std::chrono::steady_clock::duration(last_activity_raw_.load())
    );
}

size_t InputMonitor::getDeviceCount() const {
    return devices_.size();
}

void InputMonitor::scanInputDevices() {
    Logger& logger = Logger::getInstance();
    
    // Close existing devices first
    closeAllDevices();
    
    // Parse /proc/bus/input/devices to find relevant input devices
    std::ifstream devices_file("/proc/bus/input/devices");
    if (!devices_file.is_open()) {
        logger.error("Failed to open /proc/bus/input/devices");
        return;
    }
    
    std::string line;
    std::string current_name;
    std::string current_handlers;
    unsigned int current_ev_caps = 0;
    
    auto process_device = [&]() {
        // Check if this device has input capabilities we care about
        // EV_KEY (0x01) = keyboard/mouse buttons
        // EV_REL (0x02) = relative axes (mouse movement)
        // EV_ABS (0x08) = absolute axes (touchpad/touchscreen)
        bool has_key = (current_ev_caps & 0x0001) != 0;
        bool has_rel = (current_ev_caps & 0x0002) != 0;
        bool has_abs = (current_ev_caps & 0x0008) != 0;
        
        // Skip devices without input capabilities
        if (!has_key && !has_rel && !has_abs) {
            return;
        }
        
        // Skip non-input devices by name
        if (current_name.find("Power Button") != std::string::npos ||
            current_name.find("Sleep Button") != std::string::npos ||
            current_name.find("Lid Switch") != std::string::npos ||
            current_name.find("Video Bus") != std::string::npos ||
            current_name.find("HDMI") != std::string::npos ||
            current_name.find("Audio") != std::string::npos ||
            current_name.find("Headphone") != std::string::npos ||
            current_name.find("Mic") != std::string::npos) {
            return;
        }
        
        // Extract event device path from handlers
        size_t event_pos = current_handlers.find("event");
        if (event_pos != std::string::npos) {
            size_t end_pos = current_handlers.find_first_of(" \t\n", event_pos);
            std::string event_name = current_handlers.substr(event_pos, end_pos - event_pos);
            std::string event_path = "/dev/input/" + event_name;
            
            // Try to open device with libevdev
            int fd = open(event_path.c_str(), O_RDONLY | O_NONBLOCK);
            if (fd < 0) {
                logger.debug("Failed to open " + event_path + ": " + std::string(strerror(errno)));
                return;
            }
            
            libevdev* dev = nullptr;
            int rc = libevdev_new_from_fd(fd, &dev);
            if (rc < 0) {
                logger.debug("Failed to create libevdev for " + event_path + ": " + std::string(strerror(-rc)));
                close(fd);
                return;
            }
            
            // Successfully opened device
            DeviceHandle handle;
            handle.fd = fd;
            handle.dev = dev;
            handle.path = event_path;
            handle.name = current_name;
            devices_.push_back(std::move(handle));
            
            logger.info("Input device opened: " + event_path + " (" + current_name + ")");
        }
    };
    
    while (std::getline(devices_file, line)) {
        if (line.empty()) {
            // Empty line = end of device block
            process_device();
            current_name.clear();
            current_handlers.clear();
            current_ev_caps = 0;
            continue;
        }
        
        if (line[0] == 'N' && line.find("N: Name=") == 0) {
            // Name line: N: Name="Device Name"
            size_t start = line.find('"');
            size_t end = line.rfind('"');
            if (start != std::string::npos && end != std::string::npos && end > start) {
                current_name = line.substr(start + 1, end - start - 1);
            }
        } else if (line[0] == 'H' && line.find("H: Handlers=") == 0) {
            // Handlers line: H: Handlers=kbd event4 mouse0
            current_handlers = line;
        } else if (line[0] == 'B' && line.find("B: EV=") == 0) {
            // Event capabilities: B: EV=120013
            size_t eq_pos = line.find('=');
            if (eq_pos != std::string::npos) {
                const char* caps_str = line.c_str() + eq_pos + 1;
                current_ev_caps = static_cast<unsigned int>(strtoul(caps_str, nullptr, 16));
            }
        }
    }
    
    // Process last device
    process_device();
    
    devices_file.close();
    
    last_device_scan_ = std::chrono::steady_clock::now();
    
    logger.info("Input device scan complete: " + std::to_string(devices_.size()) + " device(s) found");
}

void InputMonitor::closeAllDevices() {
    for (auto& handle : devices_) {
        if (handle.dev) {
            libevdev_free(handle.dev);
            handle.dev = nullptr;
        }
        if (handle.fd >= 0) {
            close(handle.fd);
            handle.fd = -1;
        }
    }
    devices_.clear();
}

void InputMonitor::notifyActivity() {
    // Update last activity timestamp
    auto now = std::chrono::steady_clock::now();
    last_activity_raw_.store(now.time_since_epoch().count());
    
    // Invoke callback if set (non-blocking)
    std::lock_guard<std::mutex> lock(callback_mutex_);
    if (activity_callback_) {
        activity_callback_();
    }
}

void InputMonitor::eventLoop() {
    Logger& logger = Logger::getInstance();
    logger.info("Input monitoring event loop started");
    
    while (running_.load()) {
        // Check if we should rescan devices (to detect hotplugged devices)
        auto now = std::chrono::steady_clock::now();
        auto time_since_scan = std::chrono::duration_cast<std::chrono::minutes>(
            now - last_device_scan_
        ).count();
        
        if (time_since_scan >= device_rescan_interval_.count()) {
            logger.info("Rescanning input devices (periodic hotplug check)");
            scanInputDevices();
            
            if (devices_.empty()) {
                logger.warning("No input devices available after rescan");
                std::this_thread::sleep_for(std::chrono::seconds(5));
                continue;
            }
        }
        
        // Build fd_set for select()
        fd_set readfds;
        FD_ZERO(&readfds);
        int max_fd = -1;
        
        for (const auto& handle : devices_) {
            if (handle.fd >= 0) {
                FD_SET(handle.fd, &readfds);
                if (handle.fd > max_fd) {
                    max_fd = handle.fd;
                }
            }
        }
        
        if (max_fd < 0) {
            // No valid file descriptors
            logger.warning("No valid file descriptors for select()");
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }
        
        // Wait for input events (with timeout)
        struct timeval timeout;
        timeout.tv_sec = 1;   // 1 second timeout
        timeout.tv_usec = 0;
        
        int ret = select(max_fd + 1, &readfds, nullptr, nullptr, &timeout);
        
        if (ret < 0) {
            if (errno == EINTR) {
                continue;  // Interrupted by signal
            }
            logger.error("select() failed: " + std::string(strerror(errno)));
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }
        
        if (ret == 0) {
            // Timeout - no events
            continue;
        }
        
        // Process events from ready file descriptors
        bool activity_detected = false;
        
        for (const auto& handle : devices_) {
            if (handle.fd < 0 || !FD_ISSET(handle.fd, &readfds)) {
                continue;
            }
            
            // Read events from this device
            struct input_event ev;
            int rc;
            
            while ((rc = libevdev_next_event(handle.dev, LIBEVDEV_READ_FLAG_NORMAL, &ev)) >= 0) {
                if (rc == LIBEVDEV_READ_STATUS_SUCCESS) {
                    // Check if this is an input event we care about
                    bool is_input = false;
                    
                    switch (ev.type) {
                        case EV_KEY:
                            // Keyboard key or mouse button
                            if (ev.value != 0) {  // Key press or mouse button down
                                is_input = true;
                            }
                            break;
                            
                        case EV_REL:
                            // Relative axis movement (mouse/trackball)
                            if (ev.code == REL_X || ev.code == REL_Y ||
                                ev.code == REL_WHEEL || ev.code == REL_HWHEEL) {
                                // Mouse movement or scroll wheel (CRITICAL: this fixes Bluetooth mouse!)
                                is_input = true;
                            }
                            break;
                            
                        case EV_ABS:
                            // Absolute axis movement (touchpad/touchscreen)
                            is_input = true;
                            break;
                    }
                    
                    if (is_input) {
                        activity_detected = true;
                        logger.debug("Input activity detected: " + handle.name + 
                                    " (type=" + std::to_string(ev.type) + 
                                    ", code=" + std::to_string(ev.code) + ")");
                    }
                }
            }
            
            if (rc == -EAGAIN) {
                // No more events from this device
                continue;
            }
            
            if (rc < 0 && rc != -EAGAIN) {
                logger.warning("Error reading from " + handle.path + ": " + std::string(strerror(-rc)));
            }
        }
        
        if (activity_detected) {
            notifyActivity();
        }
    }
    
    logger.info("Input monitoring event loop stopped");
}

} // namespace faceid
