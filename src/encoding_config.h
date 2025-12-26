#ifndef FACEID_ENCODING_CONFIG_H
#define FACEID_ENCODING_CONFIG_H

#include <cstddef>

namespace faceid {

// ============================================================================
// FACE ENCODING DIMENSION - SINGLE SOURCE OF TRUTH
// ============================================================================
// Change this constant if you switch to a different face recognition model:
// - SFace model: 512D (current)
// - ArcFace model: typically 512D
// - FaceNet model: typically 128D or 512D depending on variant
// 
// After changing this constant:
// 1. Rebuild the project: make build && sudo make install
// 2. Re-enroll all users: sudo faceid add <username>
// 3. Delete old .bin files from /etc/faceid/faces/
// ============================================================================
constexpr size_t FACE_ENCODING_DIM = 512;  // Current model: SFace (512D)

} // namespace faceid

#endif // FACEID_ENCODING_CONFIG_H
