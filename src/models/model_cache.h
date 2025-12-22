#ifndef MODEL_CACHE_H
#define MODEL_CACHE_H

#include "binary_model.h"
#include <map>
#include <string>
#include <vector>
#include <mutex>

namespace faceid {

class ModelCache {
public:
    static ModelCache& getInstance();  // Singleton pattern
    
    // Check if a user has any enrolled face models (lightweight check)
    bool hasUserModel(const std::string& username);
    
    // Load single user model (with caching)
    bool loadUserModel(const std::string& username, BinaryFaceModel& model);
    
    // Load multiple users in parallel (4 threads)
    std::vector<BinaryFaceModel> loadUsersParallel(
        const std::vector<std::string>& usernames,
        int num_threads = 4
    );
    
    // Load all enrolled users in parallel
    std::vector<BinaryFaceModel> loadAllUsersParallel(int num_threads = 4);
    
    // Preload a user model into cache
    void preloadUser(const std::string& username);
    
    // Clear the cache
    void clearCache();
    
    // Get cache size (number of cached models)
    size_t getCacheSize() const;
    
    // Get cache statistics
    struct CacheStats {
        size_t hits;
        size_t misses;
        size_t total_loads;
    };
    CacheStats getStats() const;

private:
    ModelCache() = default;
    ~ModelCache() = default;
    ModelCache(const ModelCache&) = delete;
    ModelCache& operator=(const ModelCache&) = delete;
    
    std::map<std::string, BinaryFaceModel> cache_;
    mutable std::mutex cache_mutex_;
    
    // Statistics
    size_t cache_hits_ = 0;
    size_t cache_misses_ = 0;
};

} // namespace faceid

#endif // MODEL_CACHE_H