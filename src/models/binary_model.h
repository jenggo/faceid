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
    uint32_t version = 2;  // V2: Multi-sample encoding support with quality scores
    std::string username;
    std::vector<std::string> face_ids;  // Face ID labels (typically one per model)
    
    // Multi-sample encoding support - [pose][encoding_variant]
    std::vector<std::vector<FaceEncoding>> sample_encodings;  // Multiple encodings per pose
    std::vector<std::vector<float>> quality_scores;           // Quality score for each encoding
    
    uint32_t timestamp;
    bool valid;
    
    // Helper: Get total number of encodings across all samples
    size_t getTotalEncodingCount() const {
        size_t total = 0;
        for (const auto& sample : sample_encodings) {
            total += sample.size();
        }
        return total;
    }
};

// Binary model loader class
class BinaryModelLoader {
public:
    static constexpr size_t HEADER_SIZE = 120;  // 0x78
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