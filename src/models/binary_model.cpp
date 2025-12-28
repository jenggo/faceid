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

    // Read encoding dimension from reserved space (uint32_t at offset 16)
    uint32_t encoding_dim = readUint32LE(file);
    if (!file) return false;
    
    // If dimension is 0 or invalid, assume legacy format with hardcoded ENCODING_DIM
    if (encoding_dim == 0 || encoding_dim > 2048) {
        encoding_dim = ENCODING_DIM;
        faceid::Logger::getInstance().warning("Model file uses legacy format, assuming " + 
                                             std::to_string(encoding_dim) + "D encodings: " + path);
    }
    
    size_t dynamic_encoding_size = encoding_dim * sizeof(float);

    // Skip remaining reserved (44 bytes = 48 - 4 already read)
    file.seekg(44, std::ios::cur);
    if (!file) return false;

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

    // Read encodings - read all available encodings until EOF
    // Note: face_count is not the number of encodings, but metadata field
    model.encodings.reserve(50);  // Reserve for typical case
    while (file) {
        FaceEncoding encoding(encoding_dim);
        file.read(reinterpret_cast<char*>(encoding.data()), dynamic_encoding_size);
        if (!file) {
            // Handle partial encoding at end of file
            if (file.gcount() > 0 && file.eof()) {
                size_t bytes_read = file.gcount();
                size_t floats_read = bytes_read / sizeof(float);
                if (floats_read > 0) {
                    encoding.resize(floats_read);
                    model.encodings.push_back(encoding);
                }
            }
            break;
        }
        model.encodings.push_back(encoding);
    }

    model.valid = !model.encodings.empty();
    return model.valid;
}

bool BinaryModelLoader::saveUserModel(const std::string& path, const BinaryFaceModel& model) {
    if (!model.valid || model.encodings.empty() || model.face_ids.empty()) {
        faceid::Logger::getInstance().error("Invalid model data - cannot save to: " + path);
        return false;
    }

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        faceid::Logger::getInstance().error("Failed to open file for writing: " + path);
        return false;
    }

    // Write username (16 bytes, null-padded)
    writeNullPaddedString(file, model.username, 16);

    // Write encoding dimension at offset 16 (4 bytes)
    size_t encoding_dim = model.encodings.empty() ? ENCODING_DIM : model.encodings[0].size();
    writeUint32LE(file, static_cast<uint32_t>(encoding_dim));
    
    // Write remaining reserved (44 bytes = 48 - 4 already written)
    char zeros[44] = {0};
    file.write(zeros, 44);

    // Write timestamp
    writeUint32LE(file, model.timestamp);

    // Write reserved (4 bytes)
    file.write(zeros, 4);

    // Write face count
    uint32_t face_count = model.encodings.size();
    writeUint32LE(file, face_count);

    // Write face ID label (36 bytes, null-terminated)
    writeNullPaddedString(file, model.face_ids[0], FACE_ID_LABEL_SIZE);

    // Write reserved/metadata (8 bytes)
    file.write(zeros, 8);

    // Write encodings
    size_t encoding_size = encoding_dim * sizeof(float);
    
    for (const auto& encoding : model.encodings) {
        if (encoding.size() != encoding_dim) {
            faceid::Logger::getInstance().error("Inconsistent encoding dimensions in model: " + path);
            return false;
        }
        file.write(reinterpret_cast<const char*>(encoding.data()), encoding_size);
    }

    return file.good();
}

bool BinaryModelLoader::validateBinaryFile(const std::string& path) {
    BinaryFaceModel model;
    if (!loadUserModel(path, model)) {
        return false;
    }

    // Additional validation: check reserved areas are zeros
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;

    // Skip username
    file.seekg(16, std::ios::beg);

    // Check first reserved (16 bytes)
    char buffer[16];
    file.read(buffer, 16);
    for (char c : buffer) {
        if (c != 0) return false;
    }

    // Check second reserved (32 bytes)
    file.read(buffer, 16); // reuse buffer
    for (int i = 0; i < 32; i += 16) {
        file.read(buffer, 16);
        for (char c : buffer) {
            if (c != 0) return false;
        }
    }

    // Skip timestamp and reserved
    file.seekg(8, std::ios::cur);

    // Skip face count and label
    file.seekg(4 + FACE_ID_LABEL_SIZE, std::ios::cur);

    // Check reserved/metadata (8 bytes)
    file.read(buffer, 8);
    for (char c : buffer) {
        if (c != 0) return false;
    }

    // Check file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    size_t expected_size = HEADER_SIZE + model.encodings.size() * ENCODING_SIZE;
    if (file_size != expected_size) {
        return false;
    }

    return true;
}

size_t BinaryModelLoader::getModelFileSize(const BinaryFaceModel& model) {
    return HEADER_SIZE + model.encodings.size() * ENCODING_SIZE;
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