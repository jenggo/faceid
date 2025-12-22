#include "model_cache.h"
#include "binary_model.h"
#include <thread>
#include <mutex>
#include <dirent.h>
#include <fnmatch.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include "config_paths.h"

namespace faceid {

ModelCache& ModelCache::getInstance() {
    static ModelCache instance;
    return instance;
}

// Helper function to find user model files (both username.bin and username.*.bin)
static std::vector<std::string> findUserModelFiles(const std::string& username) {
    std::vector<std::string> files;
    std::string models_dir = MODELS_DIR;
    
    DIR* dir = opendir(models_dir.c_str());
    if (!dir) {
        return files;
    }
    
    // Patterns: username.bin and username.*.bin
    std::vector<std::string> patterns = {
        username + ".bin",
        username + ".*.bin"
    };
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        
        for (const auto& pattern : patterns) {
            if (fnmatch(pattern.c_str(), filename.c_str(), 0) == 0) {
                files.push_back(models_dir + std::string("/") + filename);
                break;
            }
        }
    }
    closedir(dir);
    
    std::sort(files.begin(), files.end());
    return files;
}

bool ModelCache::hasUserModel(const std::string& username) {
    auto files = findUserModelFiles(username);
    return !files.empty();
}

bool ModelCache::loadUserModel(const std::string& username, BinaryFaceModel& model) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    // Check cache first
    auto it = cache_.find(username);
    if (it != cache_.end()) {
        model = it->second;
        cache_hits_++;
        return true;
    }
    
    cache_misses_++;
    
    // Find all model files for this user
    auto files = findUserModelFiles(username);
    if (files.empty()) {
        return false;
    }
    
    // If only one file, load it directly
    if (files.size() == 1) {
        if (BinaryModelLoader::loadUserModel(files[0], model)) {
            cache_[username] = model;
            return true;
        }
        return false;
    }
    
    // Multiple files - merge all encodings into one model
    BinaryFaceModel merged_model;
    merged_model.username = username;
    merged_model.timestamp = 0;
    merged_model.valid = false;
    
    for (const auto& filepath : files) {
        BinaryFaceModel file_model;
        if (BinaryModelLoader::loadUserModel(filepath, file_model) && file_model.valid) {
            // Merge encodings
            merged_model.encodings.insert(
                merged_model.encodings.end(),
                file_model.encodings.begin(),
                file_model.encodings.end()
            );
            
            // Merge face_ids
            merged_model.face_ids.insert(
                merged_model.face_ids.end(),
                file_model.face_ids.begin(),
                file_model.face_ids.end()
            );
            
            // Keep earliest timestamp
            if (merged_model.timestamp == 0 || file_model.timestamp < merged_model.timestamp) {
                merged_model.timestamp = file_model.timestamp;
            }
            
            merged_model.valid = true;
        }
    }
    
    if (merged_model.valid && !merged_model.encodings.empty()) {
        model = merged_model;
        cache_[username] = merged_model;
        return true;
    }
    
    return false;
}

std::vector<BinaryFaceModel> ModelCache::loadUsersParallel(
    const std::vector<std::string>& usernames,
    int num_threads
) {
    std::vector<BinaryFaceModel> results(usernames.size());
    
    // Divide usernames into chunks
    std::vector<std::vector<size_t>> chunks(num_threads);
    for (size_t i = 0; i < usernames.size(); ++i) {
        chunks[i % num_threads].push_back(i);
    }
    
    // Create threads
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this, &usernames, &results, &chunks, t]() {
            for (size_t idx : chunks[t]) {
                loadUserModel(usernames[idx], results[idx]);
            }
        });
    }
    
    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    return results;
}

std::vector<BinaryFaceModel> ModelCache::loadAllUsersParallel(int num_threads) {
    std::vector<std::string> usernames;
    
    // Scan MODELS_DIR for *.bin files
    DIR* dir = opendir(MODELS_DIR);
    if (!dir) {
        std::cerr << "Failed to open models directory: " << MODELS_DIR << std::endl;
        return {};
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (filename.length() > 4 && filename.substr(filename.length() - 4) == ".bin") {
            std::string username = filename.substr(0, filename.length() - 4);
            usernames.push_back(username);
        }
    }
    closedir(dir);
    
    return loadUsersParallel(usernames, num_threads);
}

void ModelCache::preloadUser(const std::string& username) {
    BinaryFaceModel model;
    loadUserModel(username, model);
}

void ModelCache::clearCache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.clear();
}

size_t ModelCache::getCacheSize() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return cache_.size();
}

ModelCache::CacheStats ModelCache::getStats() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return {cache_hits_, cache_misses_, cache_hits_ + cache_misses_};
}

} // namespace faceid