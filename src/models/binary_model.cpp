#include "binary_model.h"
#include "../logger.h"
#include <fstream>
#include <cstring>
#include <algorithm>

namespace faceid {

bool BinaryModelLoader::loadUserModel(const std::string& path, BinaryFaceModel& model) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        faceid::Logger::getInstance().error("Failed to open model file: " + path);
        return false;
    }

    model.valid = false;

    // Read username (16 bytes, null-padded)
    model.username = readNullPaddedString(file, 16);
    if (model.username.empty()) {
        faceid::Logger::getInstance().error("Invalid username in model file: " + path);
        return false;
    }

    // Read version and encoding dimension
    uint32_t file_version = readUint32LE(file);
    if (!file) return false;
    
    uint32_t encoding_dim = readUint32LE(file);
    if (!file) return false;
    
    // Only support V2 format
    if (file_version != 2) {
        faceid::Logger::getInstance().error("Unsupported model version " + std::to_string(file_version) + 
                                          " (only V2 supported): " + path);
        return false;
    }
    
    model.version = file_version;
    
    if (encoding_dim == 0 || encoding_dim > 2048) {
        encoding_dim = ENCODING_DIM;
        faceid::Logger::getInstance().warning("Invalid encoding dim, assuming " + 
                                             std::to_string(encoding_dim) + "D: " + path);
    }
    
    faceid::Logger::getInstance().info("Loading V2 format model: " + path);
    
    // Skip remaining reserved (40 bytes = 48 - 8 already read)
    file.seekg(40, std::ios::cur);
    if (!file) return false;
    
    size_t dynamic_encoding_size = encoding_dim * sizeof(float);

    // Read timestamp (uint32_t, little-endian)
    model.timestamp = readUint32LE(file);
    if (!file) return false;

    // Skip reserved (4 bytes)
    file.seekg(4, std::ios::cur);
    if (!file) return false;

    // Read face count
    uint32_t face_count = readUint32LE(file);
    if (!file || face_count == 0) {
        faceid::Logger::getInstance().error("Invalid face count in model file: " + path);
        return false;
    }

    // Read face ID label (36 bytes, null-terminated)
    std::string face_id_label = readNullPaddedString(file, FACE_ID_LABEL_SIZE);
    if (face_id_label.empty()) {
        faceid::Logger::getInstance().error("Invalid face ID label in model file: " + path);
        return false;
    }
    model.face_ids.push_back(face_id_label);

    // Skip reserved/metadata (8 bytes)
    file.seekg(8, std::ios::cur);
    if (!file) return false;

    // Read V2 format: multi-sample encodings with quality scores
    // Format: num_poses, then for each pose: num_variants, [encoding1, quality1, encoding2, quality2, ...]
    
    uint32_t num_poses = readUint32LE(file);
    if (!file || num_poses == 0) {
        faceid::Logger::getInstance().error("Invalid pose count in V2 model file: " + path);
        return false;
    }
    
    model.sample_encodings.resize(num_poses);
    model.quality_scores.resize(num_poses);
    
    for (uint32_t pose_idx = 0; pose_idx < num_poses; ++pose_idx) {
        uint32_t num_variants = readUint32LE(file);
        if (!file || num_variants == 0) {
            faceid::Logger::getInstance().error("Invalid variant count for pose " + std::to_string(pose_idx) + 
                                               " in V2 model file: " + path);
            return false;
        }
        
        model.sample_encodings[pose_idx].resize(num_variants);
        model.quality_scores[pose_idx].resize(num_variants);
        
        for (uint32_t var_idx = 0; var_idx < num_variants; ++var_idx) {
            // Read encoding
            FaceEncoding encoding(encoding_dim);
            file.read(reinterpret_cast<char*>(encoding.data()), dynamic_encoding_size);
            if (!file) {
                faceid::Logger::getInstance().error("Failed to read encoding at pose " + std::to_string(pose_idx) + 
                                                   " variant " + std::to_string(var_idx) + ": " + path);
                return false;
            }
            model.sample_encodings[pose_idx][var_idx] = encoding;
            
            // Read quality score (float)
            float quality;
            file.read(reinterpret_cast<char*>(&quality), sizeof(float));
            if (!file) {
                faceid::Logger::getInstance().error("Failed to read quality score at pose " + std::to_string(pose_idx) + 
                                                   " variant " + std::to_string(var_idx) + ": " + path);
                return false;
            }
            model.quality_scores[pose_idx][var_idx] = quality;
        }
    }
    
    faceid::Logger::getInstance().info("Loaded V2 model: " + std::to_string(num_poses) + " poses, " + 
                                      std::to_string(model.getTotalEncodingCount()) + " total encodings");

    model.valid = !model.sample_encodings.empty();
    return model.valid;
}

bool BinaryModelLoader::saveUserModel(const std::string& path, const BinaryFaceModel& model) {
    // Validate model data based on version
    if (!model.valid || model.face_ids.empty()) {
        faceid::Logger::getInstance().error("Invalid model data - cannot save to: " + path);
        return false;
    }
    
    // Always save as V2 format (even if loaded as V1)
    if (model.sample_encodings.empty()) {
        faceid::Logger::getInstance().error("No sample encodings to save (V2 format required): " + path);
        return false;
    }

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        faceid::Logger::getInstance().error("Failed to open file for writing: " + path);
        return false;
    }

    // Determine encoding dimension from first encoding
    size_t encoding_dim = ENCODING_DIM;
    if (!model.sample_encodings.empty() && !model.sample_encodings[0].empty()) {
        encoding_dim = model.sample_encodings[0][0].size();
    }

    // Write username (16 bytes, null-padded)
    writeNullPaddedString(file, model.username, 16);

    // Write version (4 bytes) - always V2
    writeUint32LE(file, 2);
    
    // Write encoding dimension (4 bytes)
    writeUint32LE(file, static_cast<uint32_t>(encoding_dim));
    
    // Write remaining reserved (40 bytes = 48 - 8 already written)
    char zeros[48] = {0};
    file.write(zeros, 40);

    // Write timestamp (4 bytes)
    writeUint32LE(file, model.timestamp);

    // Write reserved (4 bytes)
    file.write(zeros, 4);

    // Write face count (legacy field, use total encoding count)
    uint32_t total_encodings = model.getTotalEncodingCount();
    writeUint32LE(file, total_encodings);

    // Write face ID label (36 bytes, null-terminated)
    writeNullPaddedString(file, model.face_ids[0], FACE_ID_LABEL_SIZE);

    // Write reserved/metadata (8 bytes)
    file.write(zeros, 8);

    // Write V2 format data: multi-sample encodings with quality scores
    // Format: num_poses, then for each pose: num_variants, [encoding1, quality1, encoding2, quality2, ...]
    
    uint32_t num_poses = model.sample_encodings.size();
    writeUint32LE(file, num_poses);
    
    size_t encoding_size = encoding_dim * sizeof(float);
    
    for (size_t pose_idx = 0; pose_idx < num_poses; ++pose_idx) {
        const auto& pose_encodings = model.sample_encodings[pose_idx];
        const auto& pose_qualities = model.quality_scores[pose_idx];
        
        if (pose_encodings.size() != pose_qualities.size()) {
            faceid::Logger::getInstance().error("Encoding/quality size mismatch at pose " + 
                                               std::to_string(pose_idx) + ": " + path);
            return false;
        }
        
        uint32_t num_variants = pose_encodings.size();
        writeUint32LE(file, num_variants);
        
        for (size_t var_idx = 0; var_idx < num_variants; ++var_idx) {
            const auto& encoding = pose_encodings[var_idx];
            
            if (encoding.size() != encoding_dim) {
                faceid::Logger::getInstance().error("Inconsistent encoding dimension at pose " + 
                                                   std::to_string(pose_idx) + " variant " + 
                                                   std::to_string(var_idx) + ": " + path);
                return false;
            }
            
            // Write encoding
            file.write(reinterpret_cast<const char*>(encoding.data()), encoding_size);
            
            // Write quality score
            float quality = pose_qualities[var_idx];
            file.write(reinterpret_cast<const char*>(&quality), sizeof(float));
        }
    }
    
    if (!file.good()) {
        faceid::Logger::getInstance().error("Failed to write model data to: " + path);
        return false;
    }

    faceid::Logger::getInstance().info("Saved V2 model: " + std::to_string(num_poses) + " poses, " + 
                                      std::to_string(total_encodings) + " total encodings to: " + path);
    return true;
}

bool BinaryModelLoader::validateBinaryFile(const std::string& path) {
    BinaryFaceModel model;
    if (!loadUserModel(path, model)) {
        return false;
    }

    // File size validation
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    
    size_t expected_size = getModelFileSize(model);
    
    if (file_size != expected_size) {
        faceid::Logger::getInstance().warning("File size mismatch: expected " + std::to_string(expected_size) + 
                                            " bytes, got " + std::to_string(file_size) + " bytes for " + path);
        // Don't fail validation for size mismatch - file format may have changed
        // Just warn and continue
    }

    return true;
}

size_t BinaryModelLoader::getModelFileSize(const BinaryFaceModel& model) {
    if (model.sample_encodings.empty()) {
        return HEADER_SIZE;
    }
    
    // V2 format calculation
    size_t encoding_dim = model.sample_encodings[0][0].size();
    size_t data_size = HEADER_SIZE;
    
    // Add size for num_poses field (4 bytes)
    data_size += sizeof(uint32_t);
    
    // For each pose: num_variants (4 bytes) + encodings + quality scores
    for (const auto& pose_encodings : model.sample_encodings) {
        data_size += sizeof(uint32_t);  // num_variants
        
        size_t num_variants = pose_encodings.size();
        data_size += num_variants * encoding_dim * sizeof(float);  // all encodings
        data_size += num_variants * sizeof(float);                 // all quality scores
    }
    
    return data_size;
}

uint32_t BinaryModelLoader::readUint32LE(std::ifstream& file) {
    uint32_t value;
    file.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

void BinaryModelLoader::writeUint32LE(std::ofstream& file, uint32_t value) {
    file.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

std::string BinaryModelLoader::readNullPaddedString(std::ifstream& file, size_t max_len) {
    std::vector<char> buffer(max_len);
    file.read(buffer.data(), max_len);
    if (!file) return "";

    // Find null terminator
    auto it = std::find(buffer.begin(), buffer.end(), '\0');
    return std::string(buffer.begin(), it);
}

void BinaryModelLoader::writeNullPaddedString(std::ofstream& file, const std::string& str, size_t max_len) {
    std::vector<char> buffer(max_len, 0);
    size_t copy_len = std::min(str.size(), max_len - 1);
    std::memcpy(buffer.data(), str.c_str(), copy_len);
    file.write(buffer.data(), max_len);
}

} // namespace faceid