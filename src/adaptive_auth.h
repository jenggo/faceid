#ifndef FACEID_ADAPTIVE_AUTH_H
#define FACEID_ADAPTIVE_AUTH_H

#include <cstdint>
#include <ctime>
#include <string>
#include <pthread.h>

namespace faceid {

// Maximum frame size for shared memory (640x480x3)
constexpr size_t MAX_FRAME_SIZE = 640 * 480 * 3;

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
