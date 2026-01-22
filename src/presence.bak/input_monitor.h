#ifndef FACEID_PRESENCE_INPUT_MONITOR_H
#define FACEID_PRESENCE_INPUT_MONITOR_H

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <functional>
#include <mutex>
#include <chrono>

// Forward declare libevdev structure to avoid including libevdev.h in header
struct libevdev;

namespace faceid {

/**
 * Input Monitor using libevdev
 * 
 * Event-driven input monitoring that detects keyboard, mouse, and touchpad activity
 * Fixes the Bluetooth mouse scrolling detection issue present in interrupt-based monitoring
 * 
 * Features:
 * - Event-driven (no polling overhead)
 * - Detects all input types: keyboard, mouse, scroll, touchpad
 * - Automatic hotplug detection (USB/Bluetooth devices)
 * - Thread-safe activity callbacks
 * - Graceful degradation if libevdev is unavailable
 */
class InputMonitor {
public:
    InputMonitor();
    ~InputMonitor();
    
    /**
     * Initialize input monitoring
     * Scans /dev/input/event* devices and opens them for monitoring
     * 
     * @return true if at least one input device was opened successfully
     */
    bool initialize();
    
    /**
     * Start the input monitoring thread
     * Begins listening for input events
     */
    void start();
    
    /**
     * Stop the input monitoring thread
     * Closes all open device handles
     */
    void stop();
    
    /**
     * Check if monitoring is active
     */
    bool isRunning() const { return running_.load(); }
    
    /**
     * Register callback for activity events
     * Callback is invoked (from monitoring thread) whenever input activity is detected
     * 
     * @param callback Function to call on input activity
     */
    void setActivityCallback(std::function<void()> callback);
    
    /**
     * Get the timestamp of the last detected input activity
     * 
     * @return Time point of last activity
     */
    std::chrono::steady_clock::time_point getLastActivity() const;
    
    /**
     * Get number of monitored devices
     */
    size_t getDeviceCount() const;
    
private:
    /**
     * Main event loop (runs in separate thread)
     * Uses select() to monitor multiple devices simultaneously
     */
    void eventLoop();
    
    /**
     * Scan /dev/input/event* devices and open relevant ones
     * Filters out non-input devices (power buttons, lid switches, etc.)
     */
    void scanInputDevices();
    
    /**
     * Close all open device handles
     */
    void closeAllDevices();
    
    /**
     * Notify activity (thread-safe)
     * Updates last activity timestamp and invokes callback if set
     */
    void notifyActivity();
    
    // Device information structure
    struct DeviceHandle {
        int fd;                 // File descriptor
        libevdev* dev;          // libevdev device handle
        std::string path;       // Device path (/dev/input/eventX)
        std::string name;       // Device name (for logging)
    };
    
    std::vector<DeviceHandle> devices_;
    std::thread event_thread_;
    std::atomic<bool> running_{false};
    std::function<void()> activity_callback_;
    std::mutex callback_mutex_;
    
    // Last activity tracking
    std::atomic<std::chrono::steady_clock::time_point::rep> last_activity_raw_;
    
    // Rescan devices periodically to detect hotplugged devices (USB/Bluetooth)
    std::chrono::steady_clock::time_point last_device_scan_;
    std::chrono::minutes device_rescan_interval_{5};  // Rescan every 5 minutes
};

} // namespace faceid

#endif // FACEID_PRESENCE_INPUT_MONITOR_H
