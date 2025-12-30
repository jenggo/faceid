#ifndef FACEID_BINARY_MODEL_H
#define FACEID_BINARY_MODEL_H

#include <string>
#include <vector>
#include <cstdint>
#include "../encoding_config.h"  // FACE_ENCODING_DIM constant
#include "../face_detector.h"

namespace faceid {

// Binary face model structure
struct BinaryFaceModel {
    std::string username;
    std::vector<std::string> face_ids;  // Face ID labels (typically one per model)
    std::vector<FaceEncoding> encodings;  // Face encodings (dimension set by FACE_ENCODING_DIM)
    uint32_t timestamp;
    bool valid;
};

// Binary model loader class
class BinaryModelLoader {
public:
    static constexpr size_t HEADER_SIZE = 120;  // 0x78
    // NOTE: Encoding dimension is stored in each binary file and detected at runtime.
    // ENCODING_DIM below is only a fallback for legacy files without dimension header.
    static constexpr size_t ENCODING_DIM = FACE_ENCODING_DIM;
    static constexpr size_t FACE_ID_LABEL_SIZE = 36;

    // Load user model from binary file
    static bool loadUserModel(const std::string& path, BinaryFaceModel& model);

    // Save user model to binary file
    static bool saveUserModel(const std::string& path, const BinaryFaceModel& model);

    // Validate binary file format and integrity
    static bool validateBinaryFile(const std::string& path);

    // Get expected file size for a model
    static size_t getModelFileSize(const BinaryFaceModel& model);

private:
    // Helper functions for endianness handling
    static uint32_t readUint32LE(std::ifstream& file);
    static void writeUint32LE(std::ofstream& file, uint32_t value);

    // Helper to read null-padded string
    static std::string readNullPaddedString(std::ifstream& file, size_t max_len);

    // Helper to write null-padded string
    static void writeNullPaddedString(std::ofstream& file, const std::string& str, size_t max_len);
};

} // namespace faceid

#endif // FACEID_BINARY_MODEL_H