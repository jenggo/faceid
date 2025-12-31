#include "commands.h"
#include "cli_common.h"
#include "../face_detector.h"
#include "../config.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <unistd.h>
#include <sys/stat.h>
#include <libgen.h>

namespace faceid {

enum class ModelPurpose {
    DETECTION,
    RECOGNITION,
    UNKNOWN
};

// Determine if a model is for detection or recognition
// Uses existing FaceDetector methods (DRY principle)
static ModelPurpose determineModelPurpose(const std::string& param_path) {
    FaceDetector detector;
    
    // Try as detection model first
    DetectionModelType det_type = detector.detectModelType(param_path);
    if (det_type != DetectionModelType::UNKNOWN) {
        return ModelPurpose::DETECTION;
    }
    
    // Try as recognition model
    size_t output_dim = detector.parseModelOutputDim(param_path);
    if (output_dim >= 64 && output_dim <= 2048) {
        return ModelPurpose::RECOGNITION;
    }
    
    return ModelPurpose::UNKNOWN;
}

static std::string getModelPurposeName(ModelPurpose purpose) {
    switch (purpose) {
        case ModelPurpose::DETECTION: return "Detection";
        case ModelPurpose::RECOGNITION: return "Recognition";
        default: return "Unknown";
    }
}

static bool fileExists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

static bool copyFile(const std::string& src, const std::string& dst) {
    std::ifstream src_file(src, std::ios::binary);
    if (!src_file.good()) {
        std::cerr << "Error: Cannot read source file: " << src << std::endl;
        return false;
    }
    
    std::ofstream dst_file(dst, std::ios::binary);
    if (!dst_file.good()) {
        std::cerr << "Error: Cannot write to destination file: " << dst << std::endl;
        return false;
    }
    
    dst_file << src_file.rdbuf();
    
    if (!dst_file.good()) {
        std::cerr << "Error: Copy operation failed" << std::endl;
        return false;
    }
    
    return true;
}

// Extract base model name from path
// Example: "/path/to/sface_2021dec_int8bq.ncnn.param" -> "sface_2021dec_int8bq"
static std::string extractBaseName(const std::string& path) {
    // Get the filename from the full path
    std::string path_copy = path;
    char* path_cstr = const_cast<char*>(path_copy.c_str());
    std::string filename = basename(path_cstr);
    
    // Strip extensions
    std::vector<std::string> extensions = {".ncnn.param", ".ncnn.bin", ".param", ".bin", ".ncnn"};
    for (const auto& ext : extensions) {
        size_t pos = filename.rfind(ext);
        if (pos != std::string::npos && pos == filename.length() - ext.length()) {
            filename = filename.substr(0, pos);
            break;
        }
    }
    
    return filename;
}

// Read .use file (key=value format)
static std::map<std::string, std::string> readUseFile(const std::string& path) {
    std::map<std::string, std::string> data;
    
    std::ifstream file(path);
    if (!file.good()) {
        return data; // Return empty map if file doesn't exist
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Parse key=value
        size_t eq_pos = line.find('=');
        if (eq_pos != std::string::npos) {
            std::string key = line.substr(0, eq_pos);
            std::string value = line.substr(eq_pos + 1);
            data[key] = value;
        }
    }
    
    return data;
}

// Write .use file (key=value format)
static bool writeUseFile(const std::string& path, const std::map<std::string, std::string>& data) {
    std::ofstream file(path);
    if (!file.good()) {
        std::cerr << "Warning: Could not write .use file: " << path << std::endl;
        return false;
    }
    
    file << "# FaceID model metadata" << std::endl;
    file << "# This file tracks original model names after installation" << std::endl;
    
    for (const auto& pair : data) {
        file << pair.first << "=" << pair.second << std::endl;
    }
    
    return file.good();
}

int cmd_use(const std::string& model_path) {
    if (model_path.empty()) {
        std::cerr << "Error: absolute model path required" << std::endl;
        std::cerr << "Usage: faceid use <absolute_path_to_model>" << std::endl;
        std::cerr << "Example: faceid use /home/user/models/mnet-retinaface.param" << std::endl;
        std::cerr << "         faceid use /etc/faceid/models/sface_2021dec_int8bq.ncnn.param" << std::endl;
        std::cerr << std::endl;
        std::cerr << "This command will:" << std::endl;
        std::cerr << "  1. Auto-detect if the model is for detection or recognition" << std::endl;
        std::cerr << "  2. Copy the model to /etc/faceid/models/detection.* or recognition.*" << std::endl;
        return 1;
    }
    
    // Require absolute path
    if (model_path[0] != '/') {
        std::cerr << "Error: Absolute path required (must start with /)" << std::endl;
        std::cerr << "You provided: " << model_path << std::endl;
        std::cerr << std::endl;
        std::cerr << "Example with absolute path:" << std::endl;
        std::cerr << "  faceid use /home/user/models/mnet-retinaface.param" << std::endl;
        std::cerr << "  faceid use $(pwd)/models/yunet.param" << std::endl;
        return 1;
    }
    
    // Strip common extensions to get base path
    std::string base_path = model_path;
    std::vector<std::string> extensions = {".bin", ".param", ".ncnn.bin", ".ncnn.param", ".ncnn"};
    for (const auto& ext : extensions) {
        size_t pos = base_path.rfind(ext);
        if (pos != std::string::npos && pos == base_path.length() - ext.length()) {
            base_path = base_path.substr(0, pos);
            break;
        }
    }
    
    // Try both .param and .ncnn.param
    std::string source_param = base_path + ".param";
    std::string source_bin = base_path + ".bin";
    
    if (!fileExists(source_param)) {
        source_param = base_path + ".ncnn.param";
        source_bin = base_path + ".ncnn.bin";
    }
    
    // Verify model files exist
    if (!fileExists(source_param)) {
        std::cerr << "Error: Model param file not found: " << source_param << std::endl;
        std::cerr << "Also tried: " << base_path << ".param" << std::endl;
        return 1;
    }
    
    if (!fileExists(source_bin)) {
        std::cerr << "Error: Model binary file not found: " << source_bin << std::endl;
        return 1;
    }
    
    std::cout << "Found model files:" << std::endl;
    std::cout << "  Param:  " << source_param << std::endl;
    std::cout << "  Binary: " << source_bin << std::endl;
    std::cout << std::endl;
    
    // Detect model type using existing FaceDetector methods
    std::cout << "Detecting model type..." << std::endl;
    ModelPurpose model_purpose = determineModelPurpose(source_param);
    
    if (model_purpose == ModelPurpose::UNKNOWN) {
        std::cerr << "Error: Could not determine model type (detection or recognition)" << std::endl;
        std::cerr << "This model may not be a valid face detection or recognition model." << std::endl;
        return 1;
    }
    
    std::cout << "✓ Model type: " << getModelPurposeName(model_purpose) << std::endl;
    std::cout << std::endl;
    
    // Determine target files
    std::string target_base = (model_purpose == ModelPurpose::DETECTION) ? "detection" : "recognition";
    std::string models_dir = std::string(MODELS_DIR);
    std::string target_param = models_dir + "/" + target_base + ".param";
    std::string target_bin = models_dir + "/" + target_base + ".bin";
    
    // Check write permissions
    if (access(models_dir.c_str(), W_OK) != 0) {
        std::cout << "Note: Switching models requires write access to " << models_dir << std::endl;
        std::cout << "You need to run with sudo:" << std::endl;
        std::cout << "  sudo faceid use " << model_path << std::endl;
        return 1;
    }
    
    // Copy new model files (will overwrite existing)
    std::cout << "Installing model files..." << std::endl;
    
    if (!copyFile(source_param, target_param)) {
        std::cerr << "Error: Failed to copy param file" << std::endl;
        return 1;
    }
    std::cout << "  ✓ Copied: " << source_param << " -> " << target_param << std::endl;
    
    if (!copyFile(source_bin, target_bin)) {
        std::cerr << "Error: Failed to copy binary file" << std::endl;
        return 1;
    }
    std::cout << "  ✓ Copied: " << source_bin << " -> " << target_bin << std::endl;
    
    // Update .use metadata file
    std::string use_file = models_dir + "/.use";
    std::string base_model_name = extractBaseName(source_param);
    
    std::map<std::string, std::string> use_data = readUseFile(use_file);
    use_data[target_base] = base_model_name;
    
    if (writeUseFile(use_file, use_data)) {
        std::cout << "  ✓ Updated metadata: " << base_model_name << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "✓ Successfully switched " << getModelPurposeName(model_purpose) << " model" << std::endl;
    std::cout << std::endl;
    std::cout << "Test the new model with:" << std::endl;
    std::cout << "  faceid show         # Live camera test" << std::endl;
    std::cout << "  faceid test <user>  # Recognition test" << std::endl;
    
    return 0;
}

} // namespace faceid
