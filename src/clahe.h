// Standalone CLAHE (Contrast Limited Adaptive Histogram Equalization)
// Extracted and adapted from OpenCV 4.x (Apache 2.0 License)
// Original Copyright (C) 2013, NVIDIA Corporation
// Original Copyright (C) 2014, Itseez Inc.
//
// Modifications for standalone use without OpenCV dependencies

#ifndef FACEID_CLAHE_H
#define FACEID_CLAHE_H

#include <cstdint>

namespace faceid {

// Standalone CLAHE implementation
// Works on single-channel 8-bit grayscale images
class CLAHE {
public:
    CLAHE(double clipLimit = 2.0, int tilesX = 8, int tilesY = 8);
    ~CLAHE();
    
    // Delete copy/move to ensure single ownership of buffers
    CLAHE(const CLAHE&) = delete;
    CLAHE& operator=(const CLAHE&) = delete;
    CLAHE(CLAHE&&) = delete;
    CLAHE& operator=(CLAHE&&) = delete;
    
    // Apply CLAHE to grayscale image
    // Input: src_data (width x height, stride = width)
    // Output: dst_data (same size)
    void apply(const uint8_t* src_data, uint8_t* dst_data, 
               int width, int height, int src_stride, int dst_stride);
    
    void setClipLimit(double clipLimit);
    double getClipLimit() const;
    
    void setTilesGridSize(int tilesX, int tilesY);
    
private:
    double clipLimit_;
    int tilesX_;
    int tilesY_;
    
    // Internal buffers
    uint8_t* lut_;
    int lutSize_;
    uint8_t* srcExt_;
    int srcExtWidth_;
    int srcExtHeight_;
};

} // namespace faceid

#endif // FACEID_CLAHE_H
