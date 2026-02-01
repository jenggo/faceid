#pragma once

#include <memory>
#include <string>
#include <vector>
#include <map>

/**
 * Model Store - Manages face recognition models
 * 
 * Responsibilities:
 * - Load user face models from ~/.local/share/faceid/models/
 * - Cache models in memory
 * - Provide model lookup by username
 * - Handle corrupt model files gracefully
 * - Report model availability
 */
class ModelStore {
public:
    static ModelStore& instance();
    
    /**
     * Load all available user models
     * Returns number of models loaded
     */
    int load_models();
    
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
    
    std::string expand_xdg_paths(const std::string& path);
    bool is_valid_model_file(const std::string& path);
};
