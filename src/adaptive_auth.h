#ifndef FACEID_ADAPTIVE_AUTH_H
#define FACEID_ADAPTIVE_AUTH_H

#include <cstdint>
#include <ctime>
#include <string>
#include <pthread.h>

namespace faceid {

// Maximum frame size for shared memory
// Production-grade: Use 8 MB to accommodate various resolutions
// - 640x480x3 = 921,600 bytes (~900 KB)
// - 640x360x3 = 691,200 bytes (~675 KB) - IR cameras
// - 1280x720x3 = 2,764,800 bytes (~2.6 MB) - HD cameras
// - 1920x1080x3 = 6,220,800 bytes (~5.9 MB) - Full HD cameras
constexpr size_t MAX_FRAME_SIZE = 8 * 1024 * 1024;  // 8 MB

// Helper function to calculate frame size
inline size_t calculateFrameSize(uint32_t width, uint32_t height, uint32_t channels = 3) {
    return static_cast<size_t>(width) * height * channels;
}

// Validate frame size fits in buffer
inline bool validateFrameSize(uint32_t width, uint32_t height, uint32_t channels = 3) {
    size_t required = calculateFrameSize(width, height, channels);
    return required <= MAX_FRAME_SIZE && required > 0;
}

// Shared memory structure for adaptive authentication
struct AdaptiveAuthState {
    // Failure tracking
    uint32_t consecutive_failures;
    time_t last_failure_time;
    
    // Optimization request/status
    volatile bool optimization_requested;
    volatile bool optimization_in_progress;
    volatile bool optimization_complete;
    time_t optimization_start_time;
    time_t last_optimization_time;
    
    // Frame data (captured on Nth failure)
    uint32_t frame_width;
    uint32_t frame_height;
    uint32_t frame_channels;
    uint8_t frame_data[MAX_FRAME_SIZE];
    
    // Optimization results
    float new_confidence;
    float new_threshold;
    
    // Process synchronization
    pthread_mutex_t mutex;
    pthread_mutexattr_t mutex_attr;
};

// Shared memory manager for adaptive authentication
class AdaptiveAuthManager {
public:
    AdaptiveAuthManager();
    ~AdaptiveAuthManager();
    
    // Initialize/attach to shared memory
    bool initialize();
    
    // PAM module interface
    void recordFailure();
    void recordSuccess();
    bool shouldTriggerOptimization(int threshold = 5);
    void captureFrame(const uint8_t* data, int width, int height, int channels);
    bool hasNewOptimalValues();
    bool getOptimalValues(float& confidence, float& threshold);
    bool isOptimizationInProgress();
    
    // Daemon/worker interface
    bool hasOptimizationRequest();
    bool getFrameData(uint8_t* buffer, int& width, int& height, int& channels);
    void startOptimization();
    void completeOptimization(float confidence, float threshold);
    void failOptimization();
    
    // Utilities
    void reset();
    uint32_t getFailureCount();
    
private:
    int shm_fd_;
    AdaptiveAuthState* state_;
    bool is_owner_;
    
    void lock();
    void unlock();
};

} // namespace faceid

#endif // FACEID_ADAPTIVE_AUTH_H
