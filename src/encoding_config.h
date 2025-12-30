#ifndef FACEID_ENCODING_CONFIG_H
#define FACEID_ENCODING_CONFIG_H

#include <cstddef>

namespace faceid {

// ============================================================================
// FACE ENCODING DIMENSION - FALLBACK DEFAULT
// ============================================================================
// This is a FALLBACK value used when:
// 1. Model dimension cannot be auto-detected from .param file
// 2. Legacy binary files without dimension header are loaded
// 
// The actual dimension is AUTO-DETECTED from the loaded model's output layer.
// 
// Common model dimensions:
// - SFace: 128D
// - MobileFaceNet: 192D  
// - ArcFace-R34: 256D
// - ArcFace-R50 / WebFace / Glint360K / MS1M models: 512D
// 
// NOTE: You can use any model - the code will detect its dimension automatically.
// Old and new enrollments with different dimensions can coexist.
// ============================================================================
constexpr size_t FACE_ENCODING_DIM = 512;  // Fallback default (auto-detected in practice)

} // namespace faceid

#endif // FACEID_ENCODING_CONFIG_H
