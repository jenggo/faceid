#include "adaptive_auth.h"
#include "logger.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <errno.h>

namespace faceid {

static const char* SHM_NAME = "/faceid_adaptive_auth";

AdaptiveAuthManager::AdaptiveAuthManager() 
    : shm_fd_(-1), state_(nullptr), is_owner_(false) {
}

AdaptiveAuthManager::~AdaptiveAuthManager() {
    if (state_ != nullptr) {
        // Cleanup mutex if we're the owner
        if (is_owner_) {
            pthread_mutex_destroy(&state_->mutex);
        }
        munmap(state_, sizeof(AdaptiveAuthState));
        state_ = nullptr;
    }
    
    if (shm_fd_ != -1) {
        close(shm_fd_);
        shm_fd_ = -1;
    }
}

bool AdaptiveAuthManager::initialize() {
    // Try to open existing shared memory first
    shm_fd_ = shm_open(SHM_NAME, O_RDWR, 0666);
    
    if (shm_fd_ == -1) {
        // Doesn't exist, create it
        shm_fd_ = shm_open(SHM_NAME, O_CREAT | O_RDWR | O_EXCL, 0666);
        if (shm_fd_ == -1) {
            // Race condition: someone else created it
            shm_fd_ = shm_open(SHM_NAME, O_RDWR, 0666);
            if (shm_fd_ == -1) {
                Logger::getInstance().error("Failed to open shared memory: " + std::string(strerror(errno)));
                return false;
            }
            is_owner_ = false;
        } else {
            is_owner_ = true;
            
            // Set size
            if (ftruncate(shm_fd_, sizeof(AdaptiveAuthState)) == -1) {
                Logger::getInstance().error("Failed to set shared memory size: " + std::string(strerror(errno)));
                close(shm_fd_);
                shm_fd_ = -1;
                shm_unlink(SHM_NAME);
                return false;
            }
        }
    }
    
    // Map shared memory
    state_ = static_cast<AdaptiveAuthState*>(
        mmap(nullptr, sizeof(AdaptiveAuthState), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0)
    );
    
    if (state_ == MAP_FAILED) {
        Logger::getInstance().error("Failed to map shared memory: " + std::string(strerror(errno)));
        close(shm_fd_);
        shm_fd_ = -1;
        if (is_owner_) {
            shm_unlink(SHM_NAME);
        }
        return false;
    }
    
    // Initialize mutex if we're the owner
    if (is_owner_) {
        pthread_mutexattr_init(&state_->mutex_attr);
        pthread_mutexattr_setpshared(&state_->mutex_attr, PTHREAD_PROCESS_SHARED);
        pthread_mutex_init(&state_->mutex, &state_->mutex_attr);
        
        // Initialize state
        std::memset(state_, 0, sizeof(AdaptiveAuthState));
    }
    
    Logger::getInstance().debug("Adaptive auth shared memory initialized (owner: " + 
                               std::string(is_owner_ ? "yes" : "no") + ")");
    return true;
}

void AdaptiveAuthManager::lock() {
    if (state_ != nullptr) {
        pthread_mutex_lock(&state_->mutex);
    }
}

void AdaptiveAuthManager::unlock() {
    if (state_ != nullptr) {
        pthread_mutex_unlock(&state_->mutex);
    }
}

void AdaptiveAuthManager::recordFailure() {
    if (state_ == nullptr) return;
    
    lock();
    state_->consecutive_failures++;
    state_->last_failure_time = time(nullptr);
    unlock();
    
    Logger::getInstance().debug("Adaptive auth: Failure recorded (total: " + 
                               std::to_string(state_->consecutive_failures) + ")");
}

void AdaptiveAuthManager::recordSuccess() {
    if (state_ == nullptr) return;
    
    lock();
    state_->consecutive_failures = 0;
    state_->optimization_complete = false;
    unlock();
    
    Logger::getInstance().debug("Adaptive auth: Success recorded, counter reset");
}

bool AdaptiveAuthManager::shouldTriggerOptimization(int threshold) {
    if (state_ == nullptr) return false;
    
    lock();
    bool should_trigger = (state_->consecutive_failures >= static_cast<uint32_t>(threshold) &&
                          !state_->optimization_requested &&
                          !state_->optimization_in_progress);
    unlock();
    
    return should_trigger;
}

void AdaptiveAuthManager::captureFrame(const uint8_t* data, int width, int height, int channels) {
    if (state_ == nullptr) return;
    
    lock();
    
    state_->frame_width = width;
    state_->frame_height = height;
    state_->frame_channels = channels;
    
    size_t frame_size = width * height * channels;
    if (frame_size > MAX_FRAME_SIZE) {
        Logger::getInstance().error("Frame too large for shared memory: " + std::to_string(frame_size));
        unlock();
        return;
    }
    
    std::memcpy(state_->frame_data, data, frame_size);
    state_->optimization_requested = true;
    
    unlock();
    
    Logger::getInstance().info("Adaptive auth: Frame captured (" + std::to_string(width) + "x" + 
                              std::to_string(height) + "x" + std::to_string(channels) + 
                              "), optimization requested");
}

bool AdaptiveAuthManager::hasNewOptimalValues() {
    if (state_ == nullptr) return false;
    
    lock();
    bool has_values = state_->optimization_complete;
    unlock();
    return has_values;
}

bool AdaptiveAuthManager::getOptimalValues(float& confidence, float& threshold) {
    if (state_ == nullptr) return false;
    
    lock();
    if (state_->optimization_complete) {
        confidence = state_->new_confidence;
        threshold = state_->new_threshold;
        unlock();
        return true;
    }
    unlock();
    return false;
}

bool AdaptiveAuthManager::hasOptimizationRequest() {
    if (state_ == nullptr) return false;
    
    lock();
    bool has_request = state_->optimization_requested && !state_->optimization_in_progress;
    unlock();
    return has_request;
}

bool AdaptiveAuthManager::getFrameData(uint8_t* buffer, int& width, int& height, int& channels) {
    if (state_ == nullptr) return false;
    
    lock();
    if (!state_->optimization_requested) {
        unlock();
        return false;
    }
    
    width = state_->frame_width;
    height = state_->frame_height;
    channels = state_->frame_channels;
    
    size_t frame_size = width * height * channels;
    std::memcpy(buffer, state_->frame_data, frame_size);
    
    unlock();
    return true;
}

void AdaptiveAuthManager::startOptimization() {
    if (state_ == nullptr) return;
    
    lock();
    state_->optimization_requested = false;
    state_->optimization_in_progress = true;
    state_->optimization_start_time = time(nullptr);
    unlock();
    
    Logger::getInstance().info("Adaptive auth: Optimization started");
}

void AdaptiveAuthManager::completeOptimization(float confidence, float threshold) {
    if (state_ == nullptr) return;
    
    lock();
    state_->new_confidence = confidence;
    state_->new_threshold = threshold;
    state_->optimization_in_progress = false;
    state_->optimization_complete = true;
    state_->last_optimization_time = time(nullptr);
    state_->consecutive_failures = 0;  // Reset counter
    unlock();
    
    Logger::getInstance().info("Adaptive auth: Optimization complete (confidence: " + 
                              std::to_string(confidence) + ", threshold: " + 
                              std::to_string(threshold) + ")");
}

void AdaptiveAuthManager::failOptimization() {
    if (state_ == nullptr) return;
    
    lock();
    state_->optimization_in_progress = false;
    state_->optimization_requested = false;
    // Don't reset failure counter - let it accumulate
    unlock();
    
    Logger::getInstance().warning("Adaptive auth: Optimization failed");
}

void AdaptiveAuthManager::reset() {
    if (state_ == nullptr) return;
    
    lock();
    state_->consecutive_failures = 0;
    state_->optimization_requested = false;
    state_->optimization_in_progress = false;
    state_->optimization_complete = false;
    unlock();
    
    Logger::getInstance().info("Adaptive auth: State reset");
}

uint32_t AdaptiveAuthManager::getFailureCount() {
    if (state_ == nullptr) return 0;
    
    lock();
    uint32_t count = state_->consecutive_failures;
    unlock();
    return count;
}

bool AdaptiveAuthManager::isOptimizationInProgress() {
    if (state_ == nullptr) return false;
    
    lock();
    bool in_progress = state_->optimization_in_progress || state_->optimization_requested;
    unlock();
    return in_progress;
}

} // namespace faceid
