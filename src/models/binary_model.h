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
    uint32_t version = 2;  // Version 2: multi-encoding support, Version 1: legacy single-encoding
    std::string username;
    std::vector<std::string> face_ids;  // Face ID labels (typically one per model)
    
    // Version 2 (NEW): Multi-sample encoding support - [pose][encoding_variant]
    std::vector<std::vector<FaceEncoding>> sample_encodings;  // Multiple encodings per pose
    std::vector<std::vector<float>> quality_scores;           // Quality score for each encoding
    
    // Version 1 (LEGACY): Single encoding per pose - kept for backwards compatibility
    std::vector<FaceEncoding> encodings;  // Deprecated: used only for v1 files
    
    uint32_t timestamp;
    bool valid;
    
    // Helper: Get total number of encodings across all samples
    size_t getTotalEncodingCount() const {
        if (version == 2 && !sample_encodings.empty()) {
            size_t total = 0;
            for (const auto& sample : sample_encodings) {
                total += sample.size();
            }
            return total;
        }
        return encodings.size();  // Legacy format
    }
    
    // Helper: Check if using new multi-sample format
    bool isMultiSampleFormat() const {
        return version == 2 && !sample_encodings.empty();
    }
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