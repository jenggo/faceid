// Standalone CLAHE (Contrast Limited Adaptive Histogram Equalization)
// Extracted and adapted from OpenCV 4.x (Apache 2.0 License)
// Original Copyright (C) 2013, NVIDIA Corporation
// Original Copyright (C) 2014, Itseez Inc.

#include "clahe.h"
#include <algorithm>
#include <cmath>

namespace faceid {

// Helper: saturate cast for uint8_t
inline uint8_t saturate_cast(float value) {
    int v = static_cast<int>(value + 0.5f);
    return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

CLAHE::CLAHE(double clipLimit, int tilesX, int tilesY)
    : clipLimit_(clipLimit), tilesX_(tilesX), tilesY_(tilesY),
      lut_(nullptr), lutSize_(0), srcExt_(nullptr), 
      srcExtWidth_(0), srcExtHeight_(0) {
}

void CLAHE::setClipLimit(double clipLimit) {
    clipLimit_ = clipLimit;
}

double CLAHE::getClipLimit() const {
    return clipLimit_;
}

void CLAHE::setTilesGridSize(int tilesX, int tilesY) {
    tilesX_ = tilesX;
    tilesY_ = tilesY;
}

void CLAHE::apply(const uint8_t* src_data, uint8_t* dst_data,
                  int width, int height, int src_stride, int dst_stride) {
    
    const int histSize = 256;
    
    // Calculate tile size
    int tileWidth = width / tilesX_;
    int tileHeight = height / tilesY_;
    
    // Handle non-divisible dimensions
    const uint8_t* srcForLut = src_data;
    int srcForLutStride = src_stride;
    
    if (width % tilesX_ != 0 || height % tilesY_ != 0) {
        // Need to extend the image
        srcExtWidth_ = width + (tilesX_ - (width % tilesX_));
        srcExtHeight_ = height + (tilesY_ - (height % tilesY_));
        
        if (srcExt_) delete[] srcExt_;
        srcExt_ = new uint8_t[srcExtWidth_ * srcExtHeight_];
        
        // Copy with border replication
        for (int y = 0; y < srcExtHeight_; ++y) {
            for (int x = 0; x < srcExtWidth_; ++x) {
                int sy = std::min(y, height - 1);
                int sx = std::min(x, width - 1);
                srcExt_[y * srcExtWidth_ + x] = src_data[sy * src_stride + sx];
            }
        }
        
        srcForLut = srcExt_;
        srcForLutStride = srcExtWidth_;
        tileWidth = srcExtWidth_ / tilesX_;
        tileHeight = srcExtHeight_ / tilesY_;
    }
    
    const int tileSizeTotal = tileWidth * tileHeight;
    const float lutScale = static_cast<float>(histSize - 1) / tileSizeTotal;
    
    int clipLimit = 0;
    if (clipLimit_ > 0.0) {
        clipLimit = static_cast<int>(clipLimit_ * tileSizeTotal / histSize);
        clipLimit = std::max(clipLimit, 1);
    }
    
    // Allocate LUT
    lutSize_ = tilesX_ * tilesY_ * histSize;
    if (lut_) delete[] lut_;
    lut_ = new uint8_t[lutSize_];
    
    // Step 1: Calculate LUT for each tile
    for (int ty = 0; ty < tilesY_; ++ty) {
        for (int tx = 0; tx < tilesX_; ++tx) {
            int tileIdx = ty * tilesX_ + tx;
            uint8_t* tileLut = lut_ + tileIdx * histSize;
            
            // Calculate histogram for this tile
            int tileHist[256] = {0};
            
            int tileX = tx * tileWidth;
            int tileY = ty * tileHeight;
            
            for (int y = 0; y < tileHeight; ++y) {
                const uint8_t* row = srcForLut + (tileY + y) * srcForLutStride + tileX;
                for (int x = 0; x < tileWidth; ++x) {
                    tileHist[row[x]]++;
                }
            }
            
            // Clip histogram
            if (clipLimit > 0) {
                int clipped = 0;
                for (int i = 0; i < histSize; ++i) {
                    if (tileHist[i] > clipLimit) {
                        clipped += tileHist[i] - clipLimit;
                        tileHist[i] = clipLimit;
                    }
                }
                
                // Redistribute clipped pixels
                int redistBatch = clipped / histSize;
                int residual = clipped - redistBatch * histSize;
                
                for (int i = 0; i < histSize; ++i) {
                    tileHist[i] += redistBatch;
                }
                
                if (residual != 0) {
                    int residualStep = std::max(histSize / residual, 1);
                    for (int i = 0; i < histSize && residual > 0; i += residualStep, residual--) {
                        tileHist[i]++;
                    }
                }
            }
            
            // Calculate cumulative distribution (LUT)
            int sum = 0;
            for (int i = 0; i < histSize; ++i) {
                sum += tileHist[i];
                tileLut[i] = saturate_cast(sum * lutScale);
            }
        }
    }
    
    // Step 2: Interpolate and apply LUT
    float inv_tw = 1.0f / tileWidth;
    float inv_th = 1.0f / tileHeight;
    
    for (int y = 0; y < height; ++y) {
        uint8_t* dstRow = dst_data + y * dst_stride;
        const uint8_t* srcRow = src_data + y * src_stride;
        
        float tyf = y * inv_th - 0.5f;
        int ty1 = static_cast<int>(std::floor(tyf));
        int ty2 = ty1 + 1;
        float ya = tyf - ty1;
        float ya1 = 1.0f - ya;
        
        ty1 = std::max(ty1, 0);
        ty2 = std::min(ty2, tilesY_ - 1);
        
        const uint8_t* lutPlane1 = lut_ + ty1 * tilesX_ * histSize;
        const uint8_t* lutPlane2 = lut_ + ty2 * tilesX_ * histSize;
        
        for (int x = 0; x < width; ++x) {
            float txf = x * inv_tw - 0.5f;
            int tx1 = static_cast<int>(std::floor(txf));
            int tx2 = tx1 + 1;
            float xa = txf - tx1;
            float xa1 = 1.0f - xa;
            
            tx1 = std::max(tx1, 0);
            tx2 = std::min(tx2, tilesX_ - 1);
            
            int srcVal = srcRow[x];
            
            int ind1 = tx1 * histSize + srcVal;
            int ind2 = tx2 * histSize + srcVal;
            
            float res = (lutPlane1[ind1] * xa1 + lutPlane1[ind2] * xa) * ya1 +
                       (lutPlane2[ind1] * xa1 + lutPlane2[ind2] * xa) * ya;
            
            dstRow[x] = saturate_cast(res);
        }
    }
}

CLAHE::~CLAHE() {
    if (lut_) delete[] lut_;
    if (srcExt_) delete[] srcExt_;
}

} // namespace faceid
