#pragma once

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include "../face_detector.h"
#include "../models/model_cache.h"

/**
 * Model Store - Manages face recognition models
 * 
 * Responsibilities:
 * - Load user face models from ~/.local/share/faceid/models/
 * - Cache models in memory (two-tier: disk -> memory)
 * - Provide model lookup by username
 * - Handle corrupt model files gracefully
 * - Report model availability
 * - Thread-safe for parallel authentication
 */
class ModelStore {
public:
    static ModelStore& instance();
    
    /**
     * Load all available user models (scan directory)
     * Returns number of models found
     */
    int load_models();
    
    /**
     * Get user face encodings (loads from disk/cache on demand)
     * Returns all face encodings for the user
     */
    std::vector<faceid::FaceEncoding> get_user_model(const std::string& username);
    
    /**
     * Check if a model exists for the given user
     */
    bool has_model(const std::string& username) const;
    
    /**
     * Get all enrolled usernames
     */
    std::vector<std::string> get_enrolled_users() const;
    
    /**
     * Delete a user's model file
     */
    bool delete_model(const std::string& username);
    
    /**
     * Invalidate memory cache for a user (after enrollment)
     */
    void invalidate_cache(const std::string& username);
    
    /**
     * Get number of loaded models
     */
    int count() const { return models_loaded_; }
    
    /**
     * Get model directory path
     */
    std::string get_model_dir() const;
    
    ~ModelStore();
    
private:
    ModelStore() = default;
    
    int models_loaded_ = 0;
    std::map<std::string, std::string> user_model_paths_;  // username -> file path
    
    // Memory cache: username -> face encodings (flattened from all poses)
    std::map<std::string, std::vector<faceid::FaceEncoding>> memory_cache_;
    mutable std::mutex cache_mutex_;
    
    std::string expand_xdg_paths(const std::string& path);
    bool is_valid_model_file(const std::string& path);
};
