#include "model_store.h"
#include "../models/binary_model.h"
#include <syslog.h>
#include <cstdlib>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>

ModelStore& ModelStore::instance() {
    static ModelStore instance;
    return instance;
}

std::vector<faceid::FaceEncoding> ModelStore::get_user_model(const std::string& username) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    // Check memory cache first
    auto it = memory_cache_.find(username);
    if (it != memory_cache_.end()) {
        syslog(LOG_DEBUG, "ModelStore: Cache hit for user %s", username.c_str());
        return it->second;
    }
    
    // Check if user has a model file
    auto path_it = user_model_paths_.find(username);
    if (path_it == user_model_paths_.end()) {
        syslog(LOG_INFO, "ModelStore: No model file for user %s", username.c_str());
        return {};
    }
    
    // Load from disk using ModelCache
    faceid::BinaryFaceModel binary_model;
    if (!faceid::ModelCache::getInstance().loadUserModel(username, binary_model)) {
        syslog(LOG_WARNING, "ModelStore: Failed to load model for user %s", username.c_str());
        return {};
    }
    
    // Flatten all encodings from all poses into a single vector
    std::vector<faceid::FaceEncoding> all_encodings;
    for (const auto& pose_samples : binary_model.sample_encodings) {
        for (const auto& encoding : pose_samples) {
            all_encodings.push_back(encoding);
        }
    }
    
    if (all_encodings.empty()) {
        syslog(LOG_WARNING, "ModelStore: Model file for user %s contains no encodings", username.c_str());
        return {};
    }
    
    // Cache in memory for subsequent auth
    memory_cache_[username] = all_encodings;
    
    syslog(LOG_INFO, "ModelStore: Loaded %zu encodings for user %s", 
           all_encodings.size(), username.c_str());
    
    return all_encodings;
}

void ModelStore::invalidate_cache(const std::string& username) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    memory_cache_.erase(username);
    syslog(LOG_INFO, "ModelStore: Invalidated cache for user %s", username.c_str());
}

std::string ModelStore::expand_xdg_paths(const std::string& path) {
    if (path.empty()) return "";
    
    // Handle ~ expansion
    if (path[0] == '~') {
        const char* home = getenv("HOME");
        if (!home) {
            home = "/tmp";  // Fallback
        }
        return std::string(home) + path.substr(1);
    }
    
    return path;
}

std::string ModelStore::get_model_dir() const {
    const char* xdg_data_home = getenv("XDG_DATA_HOME");
    std::string base_path;
    
    if (xdg_data_home && *xdg_data_home) {
        base_path = xdg_data_home;
    } else {
        const char* home = getenv("HOME");
        if (!home) home = "/tmp";
        base_path = std::string(home) + "/.local/share";
    }
    
    return base_path + "/faceid/models";
}

bool ModelStore::is_valid_model_file(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        return false;
    }
    
    // Check it's a regular file and has reasonable size (at least 1KB)
    if (!S_ISREG(st.st_mode) || st.st_size < 1024) {
        return false;
    }
    
    return true;
}

int ModelStore::load_models() {
    user_model_paths_.clear();
    models_loaded_ = 0;
    
    std::string model_dir = get_model_dir();
    DIR* dir = opendir(model_dir.c_str());
    
    if (!dir) {
        syslog(LOG_WARNING, "Model directory not found: %s", model_dir.c_str());
        return 0;
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        // Skip . and ..
        if (entry->d_name[0] == '.') continue;
        
        // Look for .faceid files
        size_t len = strlen(entry->d_name);
        if (len < 7 || strcmp(entry->d_name + len - 7, ".faceid") != 0) {
            continue;
        }
        
        std::string full_path = model_dir + "/" + entry->d_name;
        
        if (!is_valid_model_file(full_path)) {
            syslog(LOG_WARNING, "Invalid model file: %s", full_path.c_str());
            continue;
        }
        
        // Extract username (remove .faceid extension)
        std::string username(entry->d_name);
        username = username.substr(0, username.length() - 7);
        
        user_model_paths_[username] = full_path;
        models_loaded_++;
        syslog(LOG_DEBUG, "Loaded model for user: %s", username.c_str());
    }
    
    closedir(dir);
    
    if (models_loaded_ > 0) {
        syslog(LOG_INFO, "✓ Loaded %d face models", models_loaded_);
    }
    
    return models_loaded_;
}

bool ModelStore::has_model(const std::string& username) const {
    return user_model_paths_.find(username) != user_model_paths_.end();
}

std::vector<std::string> ModelStore::get_enrolled_users() const {
    std::vector<std::string> users;
    for (const auto& pair : user_model_paths_) {
        users.push_back(pair.first);
    }
    return users;
}

bool ModelStore::delete_model(const std::string& username) {
    auto it = user_model_paths_.find(username);
    if (it == user_model_paths_.end()) {
        syslog(LOG_WARNING, "Model not found for user: %s", username.c_str());
        return false;
    }
    
    const std::string& path = it->second;
    if (unlink(path.c_str()) != 0) {
        syslog(LOG_ERR, "Failed to delete model: %s (%s)", path.c_str(), strerror(errno));
        return false;
    }
    
    user_model_paths_.erase(it);
    models_loaded_--;
    syslog(LOG_INFO, "Deleted model for user: %s", username.c_str());
    return true;
}

ModelStore::~ModelStore() {
    // Cleanup happens automatically
}
