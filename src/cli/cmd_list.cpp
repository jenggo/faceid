#include "commands.h"
#include "cli_common.h"
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <ctime>
#include "../models/binary_model.h"

namespace faceid {

using namespace faceid;
using namespace faceid::cli;

int cmd_list(const std::string& username) {
    std::string models_dir = MODELS_DIR;
    
    // List faces for specific user
    if (!username.empty()) {
        auto files = findUserModelFiles(username);
        
        if (files.empty()) {
            std::cerr << "Error: No face model found for user: " << username << std::endl;
            return 1;
        }
        
        std::cout << "Faces for user: " << username << std::endl;
        
        for (const auto& filepath : files) {
            BinaryFaceModel model;
            if (BinaryModelLoader::loadUserModel(filepath, model) && model.valid) {
                // Extract filename
                size_t pos = filepath.find_last_of('/');
                std::string filename = (pos != std::string::npos) ? filepath.substr(pos + 1) : filepath;
                
                // Extract face_id from filename
                std::string face_id = "default";
                if (filename != username + ".bin") {
                    // Format: username.face_id.bin
                    size_t dot_pos = filename.find_last_of('.');
                    size_t prev_dot = filename.find_last_of('.', dot_pos - 1);
                    if (prev_dot != std::string::npos) {
                        face_id = filename.substr(prev_dot + 1, dot_pos - prev_dot - 1);
                    }
                }
                
                int samples = model.encodings.size();
                
                std::cout << "  " << face_id;
                std::cout << " (" << samples << " samples";
                if (model.timestamp > 0) {
                    std::cout << ", created: " << formatTimestamp(model.timestamp);
                }
                std::cout << ")" << std::endl;
            }
        }
        std::cout << "Total: " << files.size() << " face(s)" << std::endl;
        
        return 0;
    }
    
    // List all users
    auto users = getEnrolledUsers();
    
    std::cout << "Enrolled users:" << std::endl;
    
    if (users.empty()) {
        std::cout << "  (none)" << std::endl;
    } else {
        for (const auto& user : users) {
            auto files = findUserModelFiles(user);
            std::cout << "  " << user << " (" << files.size() << " face(s))" << std::endl;
        }
    }
    std::cout << "Total: " << users.size() << " user(s)" << std::endl;
    
    return 0;
}

} // namespace faceid
