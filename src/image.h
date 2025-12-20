/*
 * Type-Safe Image Classes for FaceID
 * 
 * Design Philosophy:
 * - Explicit ownership at type level (like Rust)
 * - Prevent accidental copies (compile-time safety)
 * - Zero-cost abstractions
 * - Cache-friendly (64-byte alignment)
 * 
 * Classes:
 * - ImageView: Non-owning view (can't copy, only reference)
 * - Image: Owning image (move-only, explicit clone)
 * - Rect: Bounding rectangle
 */

#ifndef FACEID_IMAGE_H
#define FACEID_IMAGE_H

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace faceid {

// Forward declarations
class Image;
class ImageView;

// ========== Rect: Bounding Rectangle ==========

struct Rect {
    int x, y, width, height;
    
    constexpr Rect() noexcept : x(0), y(0), width(0), height(0) {}
    constexpr Rect(int x_, int y_, int w, int h) noexcept 
        : x(x_), y(y_), width(w), height(h) {}
    
    // Intersection with bounds (clip to frame)
    Rect& operator&=(const Rect& bounds) noexcept {
        int x2 = std::min(x + width, bounds.x + bounds.width);
        int y2 = std::min(y + height, bounds.y + bounds.height);
        x = std::max(x, bounds.x);
        y = std::max(y, bounds.y);
        width = std::max(0, x2 - x);
        height = std::max(0, y2 - y);
        return *this;
    }
    
    constexpr bool empty() const noexcept { 
        return width <= 0 || height <= 0; 
    }
    
    constexpr int area() const noexcept { 
        return width * height; 
    }
    
    // Center point
    constexpr int centerX() const noexcept { 
        return x + width / 2; 
    }
    
    constexpr int centerY() const noexcept { 
        return y + height / 2; 
    }
};

// ========== ImageView: Non-Owning View ==========

class ImageView {
public:
    // Construct view from existing data (non-owning)
    ImageView(uint8_t* data, int width, int height, int channels, int stride = 0) noexcept
        : data_(data), width_(width), height_(height), 
          channels_(channels), stride_(stride > 0 ? stride : width * channels) {}
    
    // Delete copy (views can't be copied - forces explicit intent)
    ImageView(const ImageView&) = delete;
    ImageView& operator=(const ImageView&) = delete;
    
    // Allow move (cheap pointer copy)
    ImageView(ImageView&& other) noexcept
        : data_(other.data_), width_(other.width_), height_(other.height_),
          channels_(other.channels_), stride_(other.stride_) {
        other.data_ = nullptr;
    }
    
    ImageView& operator=(ImageView&& other) noexcept {
        data_ = other.data_;
        width_ = other.width_;
        height_ = other.height_;
        channels_ = other.channels_;
        stride_ = other.stride_;
        other.data_ = nullptr;
        return *this;
    }
    
    // Accessors (cv::Mat compatible)
    uint8_t* data() noexcept { return data_; }
    const uint8_t* data() const noexcept { return data_; }
    constexpr int cols() const noexcept { return width_; }
    constexpr int rows() const noexcept { return height_; }
    constexpr int width() const noexcept { return width_; }
    constexpr int height() const noexcept { return height_; }
    constexpr int step() const noexcept { return stride_; }
    constexpr int stride() const noexcept { return stride_; }
    constexpr int channels() const noexcept { return channels_; }
    constexpr bool empty() const noexcept { return data_ == nullptr || width_ == 0 || height_ == 0; }
    constexpr size_t size() const noexcept { return width_ * height_ * channels_; }
    
    // ROI extraction (returns new non-owning view)
    ImageView roi(int x, int y, int w, int h) const noexcept {
        uint8_t* roi_data = data_ + y * stride_ + x * channels_;
        return ImageView(roi_data, w, h, channels_, stride_);
    }
    
    ImageView roi(const Rect& rect) const noexcept {
        return roi(rect.x, rect.y, rect.width, rect.height);
    }
    
    // Explicit deep copy (returns owning Image)
    Image clone() const;

private:
    uint8_t* data_;
    int width_;
    int height_;
    int channels_;
    int stride_;
};

// ========== Image: Owning Image (Move-Only) ==========

class Image {
public:
    // Default constructor (empty image)
    Image() noexcept 
        : data_(nullptr), width_(0), height_(0), channels_(0), stride_(0) {}
    
    // Allocating constructor (owns data, 64-byte aligned)
    Image(int width, int height, int channels) 
        : width_(width), height_(height), channels_(channels), 
          stride_(width * channels) {
        
        if (width <= 0 || height <= 0 || channels <= 0) {
            throw std::invalid_argument("Image dimensions must be positive");
        }
        
        // Allocate aligned memory (64-byte for cache line + AVX-512)
        size_t size = stride_ * height_;
        size_t aligned_size = (size + 63) & ~63;  // Round up to 64-byte boundary
        
        #ifdef _WIN32
            data_ = static_cast<uint8_t*>(_aligned_malloc(aligned_size, 64));
            if (!data_) throw std::bad_alloc();
        #else
            if (posix_memalign(reinterpret_cast<void**>(&data_), 64, aligned_size) != 0) {
                throw std::bad_alloc();
            }
        #endif
        
        // Initialize to zero
        std::memset(data_, 0, aligned_size);
    }
    
    // Destructor
    ~Image() noexcept {
        if (data_) {
            #ifdef _WIN32
                _aligned_free(data_);
            #else
                free(data_);
            #endif
        }
    }
    
    // Delete copy (images can't be copied - must use explicit clone())
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    
    // Move semantics (transfer ownership)
    Image(Image&& other) noexcept
        : data_(other.data_), width_(other.width_), height_(other.height_),
          channels_(other.channels_), stride_(other.stride_) {
        other.data_ = nullptr;
        other.width_ = 0;
        other.height_ = 0;
    }
    
    Image& operator=(Image&& other) noexcept {
        if (this != &other) {
            if (data_) {
                #ifdef _WIN32
                    _aligned_free(data_);
                #else
                    free(data_);
                #endif
            }
            
            data_ = other.data_;
            width_ = other.width_;
            height_ = other.height_;
            channels_ = other.channels_;
            stride_ = other.stride_;
            
            other.data_ = nullptr;
            other.width_ = 0;
            other.height_ = 0;
        }
        return *this;
    }
    
    // Accessors (cv::Mat compatible)
    uint8_t* data() noexcept { return data_; }
    const uint8_t* data() const noexcept { return data_; }
    constexpr int cols() const noexcept { return width_; }
    constexpr int rows() const noexcept { return height_; }
    constexpr int width() const noexcept { return width_; }
    constexpr int height() const noexcept { return height_; }
    constexpr int step() const noexcept { return stride_; }
    constexpr int stride() const noexcept { return stride_; }
    constexpr int channels() const noexcept { return channels_; }
    constexpr bool empty() const noexcept { return data_ == nullptr || width_ == 0 || height_ == 0; }
    constexpr size_t size() const noexcept { return width_ * height_ * channels_; }
    
    // Get non-owning view (safe: view lifetime must be < image lifetime)
    ImageView view() noexcept {
        return ImageView(data_, width_, height_, channels_, stride_);
    }
    
    const ImageView view() const noexcept {
        return ImageView(const_cast<uint8_t*>(data_), width_, height_, channels_, stride_);
    }
    
    // ROI extraction (returns non-owning view)
    ImageView roi(int x, int y, int w, int h) noexcept {
        uint8_t* roi_data = data_ + y * stride_ + x * channels_;
        return ImageView(roi_data, w, h, channels_, stride_);
    }
    
    ImageView roi(const Rect& rect) noexcept {
        return roi(rect.x, rect.y, rect.width, rect.height);
    }
    
    // Explicit deep copy
    Image clone() const {
        if (empty()) {
            return Image();
        }
        
        Image copy(width_, height_, channels_);
        
        // Copy row by row (handles stride)
        for (int y = 0; y < height_; y++) {
            std::memcpy(
                copy.data_ + y * copy.stride_,
                data_ + y * stride_,
                width_ * channels_
            );
        }
        
        return copy;
    }

private:
    uint8_t* data_;
    int width_;
    int height_;
    int channels_;
    int stride_;
};

// ImageView::clone() implementation (needs Image definition)
inline Image ImageView::clone() const {
    if (empty()) {
        return Image();
    }
    
    Image copy(width_, height_, channels_);
    
    // Copy row by row (handles stride)
    for (int y = 0; y < height_; y++) {
        std::memcpy(
            copy.data() + y * copy.stride(),
            data_ + y * stride_,
            width_ * channels_
        );
    }
    
    return copy;
}

} // namespace faceid

#endif // FACEID_IMAGE_H
