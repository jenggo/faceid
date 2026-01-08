#ifndef FACEID_ADAPTIVE_AUTH_H
#define FACEID_ADAPTIVE_AUTH_H

#include <cstdint>
#include <ctime>
#include <string>
#include <pthread.h>

namespace faceid {

// Forward declaration
struct AdaptiveAuthState;

// Maximum frame size for shared memory (safety upper limit)
// This is an upper bound; actual allocation is based on configured camera resolution
// - 640x480x3 = 921,600 bytes (~900 KB)
// - 640x360x3 = 691,200 bytes (~675 KB) - IR cameras  
// - 1280x720x3 = 2,764,800 bytes (~2.6 MB) - HD cameras
// - 1920x1080x3 = 6,220,800 bytes (~5.9 MB) - Full HD cameras
constexpr size_t MAX_FRAME_SIZE = 8 * 1024 * 1024;  // 8 MB upper limit

// Calculate actual frame size needed based on resolution
inline size_t calculateFrameSize(uint32_t width, uint32_t height, uint32_t channels = 3) {
    return static_cast<size_t>(width) * height * channels;
}

// Forward declaration helper - actual implementation after struct definition
size_t calculateSharedMemorySize(uint32_t width, uint32_t height, uint32_t channels = 3);

// Validate frame size fits in upper limit
inline bool validateFrameSize(uint32_t width, uint32_t height, uint32_t channels = 3) {
    size_t required = calculateFrameSize(width, height, channels);
    return required <= MAX_FRAME_SIZE && required > 0;
}

// Shared memory structure for adaptive authentication
// Note: This struct has a variable-size buffer at the end (frame_data)
// Actual allocation size is: sizeof(AdaptiveAuthState) - 1 + actual_frame_size
// The frame_data[1] is a placeholder; real size is determined at runtime
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
    uint32_t allocated_frame_size;  // Actual allocated size for validation
    
    // Optimization results
    float new_confidence;
    float new_threshold;
    
    // Process synchronization
    pthread_mutex_t mutex;
    pthread_mutexattr_t mutex_attr;
    
    // Variable-size frame buffer (struct hack for C++ compatibility)
    // Actual size is allocated_frame_size bytes, not just 1 byte
    // This MUST be the last member of the struct
    uint8_t frame_data[1];
};

// Calculate actual shared memory size needed (struct + frame buffer)
// Defined here after struct is complete
inline size_t calculateSharedMemorySize(uint32_t width, uint32_t height, uint32_t channels) {
    size_t frame_size = calculateFrameSize(width, height, channels);
    // Struct already includes 1 byte for frame_data[1], so subtract 1 and add actual frame_size
    return sizeof(AdaptiveAuthState) - 1 + frame_size;
}

// Shared memory manager for adaptive authentication
class AdaptiveAuthManager {
public:
    AdaptiveAuthManager();
    ~AdaptiveAuthManager();
    
    // Initialize/attach to shared memory
    // width, height, channels: camera resolution (default 640x360x3 for IR cameras)
    bool initialize(uint32_t width = 640, uint32_t height = 360, uint32_t channels = 3);
    
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
    size_t shm_size_;  // Actual allocated shared memory size
    
    void lock();
    void unlock();
};

} // namespace faceid

#endif // FACEID_ADAPTIVE_AUTH_H
