#include <iostream>
#include <cstdio>
#include "../models/binary_model.h"
#include "commands.h"
#include "cli_common.h"

namespace faceid {

using namespace faceid;
using namespace faceid::cli;

int cmd_remove(const std::string& username, const std::string& face_id) {
    std::string faces_dir = FACES_DIR;
    
    // If no face_id specified, remove entire user model
    if (face_id.empty()) {
        // Use centralized file discovery
        auto files = cli::findUserModelFiles(username);
        
        if (files.empty()) {
            std::cerr << "✗ No face model files found for user: " << username << std::endl;
            std::cerr << "  Looked in: " << faces_dir << std::endl;
            return 1;
        }
        
        int removed_count = 0;
        for (const auto& filepath : files) {
            if (std::remove(filepath.c_str()) == 0) {
                // Extract filename for display
                size_t pos = filepath.find_last_of('/');
                std::string filename = (pos != std::string::npos) ? filepath.substr(pos + 1) : filepath;
                std::cout << "  Removed: " << filename << std::endl;
                removed_count++;
            } else {
                size_t pos = filepath.find_last_of('/');
                std::string filename = (pos != std::string::npos) ? filepath.substr(pos + 1) : filepath;
                std::cerr << "  Warning: Failed to remove " << filename << " (permission denied?)" << std::endl;
            }
        }
        
        if (removed_count > 0) {
            std::cout << "✓ Removed " << removed_count << " face model file(s) for user: " << username << std::endl;
            return 0;
        } else {
            std::cerr << "✗ Failed to remove any files (permission denied?)" << std::endl;
            return 1;
        }
    }
    
    // Remove specific face_id
    std::string model_path = faces_dir + "/" + username + "." + face_id + ".bin";
    BinaryFaceModel model;
    
    if (!BinaryModelLoader::loadUserModel(model_path, model)) {
        std::cerr << "Error: No face model found for user: " << username 
                  << ", face ID: " << face_id << std::endl;
        return 1;
    }
    
    if (std::remove(model_path.c_str()) == 0) {
        std::cout << "✓ Removed face '" << face_id << "' for user: " << username << std::endl;
        return 0;
    } else {
        std::cerr << "✗ Failed to remove face model file" << std::endl;
        return 1;
    }
}

} // namespace faceid
