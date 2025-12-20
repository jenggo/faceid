/*
 * SDL2-Based Display Wrapper Implementation
 */

#include "display.h"
#include <algorithm>
#include <iostream>

namespace faceid {

// ========== Static Members ==========

int Display::sdl_init_count_ = 0;
std::mutex Display::sdl_init_mutex_;

// ========== Simple 8x8 Bitmap Font Data ==========

// Each character is 8x8 pixels, stored as 8 bytes (1 byte per row)
static const uint8_t FONT_8X8[128][8] = {
    // ASCII 0-31 (control characters - blank)
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 0
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 1
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 2
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 3
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 4
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 5
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 6
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 7
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 8
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 9
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 10
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 11
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 12
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 13
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 14
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 15
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 16
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 17
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 18
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 19
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 20
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 21
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 22
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 23
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 24
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 25
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 26
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 27
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 28
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 29
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 30
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 31
    // ASCII 32-127 (printable characters)
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // Space (32)
    {0x18, 0x3C, 0x3C, 0x18, 0x18, 0x00, 0x18, 0x00}, // ! (33)
    {0x36, 0x36, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // " (34)
    {0x36, 0x36, 0x7F, 0x36, 0x7F, 0x36, 0x36, 0x00}, // # (35)
    {0x0C, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x0C, 0x00}, // $ (36)
    {0x00, 0x63, 0x33, 0x18, 0x0C, 0x66, 0x63, 0x00}, // % (37)
    {0x1C, 0x36, 0x1C, 0x6E, 0x3B, 0x33, 0x6E, 0x00}, // & (38)
    {0x06, 0x06, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00}, // ' (39)
    {0x18, 0x0C, 0x06, 0x06, 0x06, 0x0C, 0x18, 0x00}, // ( (40)
    {0x06, 0x0C, 0x18, 0x18, 0x18, 0x0C, 0x06, 0x00}, // ) (41)
    {0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00}, // * (42)
    {0x00, 0x0C, 0x0C, 0x3F, 0x0C, 0x0C, 0x00, 0x00}, // + (43)
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x06}, // , (44)
    {0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0x00}, // - (45)
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00}, // . (46)
    {0x60, 0x30, 0x18, 0x0C, 0x06, 0x03, 0x01, 0x00}, // / (47)
    {0x3E, 0x63, 0x73, 0x7B, 0x6F, 0x67, 0x3E, 0x00}, // 0 (48)
    {0x0C, 0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x3F, 0x00}, // 1 (49)
    {0x1E, 0x33, 0x30, 0x1C, 0x06, 0x33, 0x3F, 0x00}, // 2 (50)
    {0x1E, 0x33, 0x30, 0x1C, 0x30, 0x33, 0x1E, 0x00}, // 3 (51)
    {0x38, 0x3C, 0x36, 0x33, 0x7F, 0x30, 0x78, 0x00}, // 4 (52)
    {0x3F, 0x03, 0x1F, 0x30, 0x30, 0x33, 0x1E, 0x00}, // 5 (53)
    {0x1C, 0x06, 0x03, 0x1F, 0x33, 0x33, 0x1E, 0x00}, // 6 (54)
    {0x3F, 0x33, 0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x00}, // 7 (55)
    {0x1E, 0x33, 0x33, 0x1E, 0x33, 0x33, 0x1E, 0x00}, // 8 (56)
    {0x1E, 0x33, 0x33, 0x3E, 0x30, 0x18, 0x0E, 0x00}, // 9 (57)
    {0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x00}, // : (58)
    {0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x06}, // ; (59)
    {0x18, 0x0C, 0x06, 0x03, 0x06, 0x0C, 0x18, 0x00}, // < (60)
    {0x00, 0x00, 0x3F, 0x00, 0x00, 0x3F, 0x00, 0x00}, // = (61)
    {0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06, 0x00}, // > (62)
    {0x1E, 0x33, 0x30, 0x18, 0x0C, 0x00, 0x0C, 0x00}, // ? (63)
    {0x3E, 0x63, 0x7B, 0x7B, 0x7B, 0x03, 0x1E, 0x00}, // @ (64)
    {0x0C, 0x1E, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x00}, // A (65)
    {0x3F, 0x66, 0x66, 0x3E, 0x66, 0x66, 0x3F, 0x00}, // B (66)
    {0x3C, 0x66, 0x03, 0x03, 0x03, 0x66, 0x3C, 0x00}, // C (67)
    {0x1F, 0x36, 0x66, 0x66, 0x66, 0x36, 0x1F, 0x00}, // D (68)
    {0x7F, 0x46, 0x16, 0x1E, 0x16, 0x46, 0x7F, 0x00}, // E (69)
    {0x7F, 0x46, 0x16, 0x1E, 0x16, 0x06, 0x0F, 0x00}, // F (70)
    {0x3C, 0x66, 0x03, 0x03, 0x73, 0x66, 0x7C, 0x00}, // G (71)
    {0x33, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x33, 0x00}, // H (72)
    {0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00}, // I (73)
    {0x78, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E, 0x00}, // J (74)
    {0x67, 0x66, 0x36, 0x1E, 0x36, 0x66, 0x67, 0x00}, // K (75)
    {0x0F, 0x06, 0x06, 0x06, 0x46, 0x66, 0x7F, 0x00}, // L (76)
    {0x63, 0x77, 0x7F, 0x7F, 0x6B, 0x63, 0x63, 0x00}, // M (77)
    {0x63, 0x67, 0x6F, 0x7B, 0x73, 0x63, 0x63, 0x00}, // N (78)
    {0x1C, 0x36, 0x63, 0x63, 0x63, 0x36, 0x1C, 0x00}, // O (79)
    {0x3F, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x0F, 0x00}, // P (80)
    {0x1E, 0x33, 0x33, 0x33, 0x3B, 0x1E, 0x38, 0x00}, // Q (81)
    {0x3F, 0x66, 0x66, 0x3E, 0x36, 0x66, 0x67, 0x00}, // R (82)
    {0x1E, 0x33, 0x07, 0x0E, 0x38, 0x33, 0x1E, 0x00}, // S (83)
    {0x3F, 0x2D, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00}, // T (84)
    {0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x3F, 0x00}, // U (85)
    {0x33, 0x33, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00}, // V (86)
    {0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00}, // W (87)
    {0x63, 0x63, 0x36, 0x1C, 0x1C, 0x36, 0x63, 0x00}, // X (88)
    {0x33, 0x33, 0x33, 0x1E, 0x0C, 0x0C, 0x1E, 0x00}, // Y (89)
    {0x7F, 0x63, 0x31, 0x18, 0x4C, 0x66, 0x7F, 0x00}, // Z (90)
    {0x1E, 0x06, 0x06, 0x06, 0x06, 0x06, 0x1E, 0x00}, // [ (91)
    {0x03, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x40, 0x00}, // backslash (92)
    {0x1E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x1E, 0x00}, // ] (93)
    {0x08, 0x1C, 0x36, 0x63, 0x00, 0x00, 0x00, 0x00}, // ^ (94)
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF}, // _ (95)
    {0x0C, 0x0C, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00}, // ` (96)
    {0x00, 0x00, 0x1E, 0x30, 0x3E, 0x33, 0x6E, 0x00}, // a (97)
    {0x07, 0x06, 0x06, 0x3E, 0x66, 0x66, 0x3B, 0x00}, // b (98)
    {0x00, 0x00, 0x1E, 0x33, 0x03, 0x33, 0x1E, 0x00}, // c (99)
    {0x38, 0x30, 0x30, 0x3e, 0x33, 0x33, 0x6E, 0x00}, // d (100)
    {0x00, 0x00, 0x1E, 0x33, 0x3f, 0x03, 0x1E, 0x00}, // e (101)
    {0x1C, 0x36, 0x06, 0x0f, 0x06, 0x06, 0x0F, 0x00}, // f (102)
    {0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x1F}, // g (103)
    {0x07, 0x06, 0x36, 0x6E, 0x66, 0x66, 0x67, 0x00}, // h (104)
    {0x0C, 0x00, 0x0E, 0x0C, 0x0C, 0x0C, 0x1E, 0x00}, // i (105)
    {0x30, 0x00, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E}, // j (106)
    {0x07, 0x06, 0x66, 0x36, 0x1E, 0x36, 0x67, 0x00}, // k (107)
    {0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00}, // l (108)
    {0x00, 0x00, 0x33, 0x7F, 0x7F, 0x6B, 0x63, 0x00}, // m (109)
    {0x00, 0x00, 0x1F, 0x33, 0x33, 0x33, 0x33, 0x00}, // n (110)
    {0x00, 0x00, 0x1E, 0x33, 0x33, 0x33, 0x1E, 0x00}, // o (111)
    {0x00, 0x00, 0x3B, 0x66, 0x66, 0x3E, 0x06, 0x0F}, // p (112)
    {0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x78}, // q (113)
    {0x00, 0x00, 0x3B, 0x6E, 0x66, 0x06, 0x0F, 0x00}, // r (114)
    {0x00, 0x00, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x00}, // s (115)
    {0x08, 0x0C, 0x3E, 0x0C, 0x0C, 0x2C, 0x18, 0x00}, // t (116)
    {0x00, 0x00, 0x33, 0x33, 0x33, 0x33, 0x6E, 0x00}, // u (117)
    {0x00, 0x00, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00}, // v (118)
    {0x00, 0x00, 0x63, 0x6B, 0x7F, 0x7F, 0x36, 0x00}, // w (119)
    {0x00, 0x00, 0x63, 0x36, 0x1C, 0x36, 0x63, 0x00}, // x (120)
    {0x00, 0x00, 0x33, 0x33, 0x33, 0x3E, 0x30, 0x1F}, // y (121)
    {0x00, 0x00, 0x3F, 0x19, 0x0C, 0x26, 0x3F, 0x00}, // z (122)
    {0x38, 0x0C, 0x0C, 0x07, 0x0C, 0x0C, 0x38, 0x00}, // { (123)
    {0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x00}, // | (124)
    {0x07, 0x0C, 0x0C, 0x38, 0x0C, 0x0C, 0x07, 0x00}, // } (125)
    {0x6E, 0x3B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // ~ (126)
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // DEL (127)
};

// ========== Display Class Implementation ==========

Display::Display(const char* window_name, int width, int height)
    : window_name_(window_name)
    , width_(width)
    , height_(height)
    , is_open_(false)
    , window_(nullptr)
    , renderer_(nullptr)
    , texture_(nullptr)
    , last_key_(-1)
    , quit_requested_(false)
{
    initSDL();
    if (width > 0 && height > 0) {
        createWindow(width, height);
    }
}

Display::~Display() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (texture_) {
        SDL_DestroyTexture(texture_);
        texture_ = nullptr;
    }
    
    if (renderer_) {
        SDL_DestroyRenderer(renderer_);
        renderer_ = nullptr;
    }
    
    if (window_) {
        SDL_DestroyWindow(window_);
        window_ = nullptr;
    }
    
    // Cleanup SDL if this is the last Display instance
    std::lock_guard<std::mutex> sdl_lock(sdl_init_mutex_);
    sdl_init_count_--;
    if (sdl_init_count_ == 0) {
        SDL_Quit();
    }
}

void Display::initSDL() {
    std::lock_guard<std::mutex> lock(sdl_init_mutex_);
    
    if (sdl_init_count_ == 0) {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            std::cerr << "SDL initialization failed: " << SDL_GetError() << std::endl;
            return;
        }
    }
    
    sdl_init_count_++;
}

void Display::createWindow(int width, int height) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    width_ = width;
    height_ = height;
    
    // Create window
    window_ = SDL_CreateWindow(
        window_name_.c_str(),
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        width,
        height,
        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE
    );
    
    if (!window_) {
        std::cerr << "Failed to create window: " << SDL_GetError() << std::endl;
        return;
    }
    
    // Create renderer with hardware acceleration
    renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!renderer_) {
        std::cerr << "Failed to create renderer: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window_);
        window_ = nullptr;
        return;
    }
    
    is_open_ = true;
}

void Display::createTexture(int width, int height) {
    // Must be called with mutex_ locked
    
    if (texture_) {
        SDL_DestroyTexture(texture_);
    }
    
    // Create texture for BGR24 format (we'll convert to RGB during upload)
    texture_ = SDL_CreateTexture(
        renderer_,
        SDL_PIXELFORMAT_RGB24,
        SDL_TEXTUREACCESS_STREAMING,
        width,
        height
    );
    
    if (!texture_) {
        std::cerr << "Failed to create texture: " << SDL_GetError() << std::endl;
    }
}

void Display::updateTexture(const uint8_t* data, int width, int height, int stride) {
    // Must be called with mutex_ locked
    
    if (!texture_ || width != width_ || height != height_) {
        width_ = width;
        height_ = height;
        createTexture(width, height);
    }
    
    if (!texture_) return;
    
    // Lock texture for update
    void* pixels;
    int pitch;
    if (SDL_LockTexture(texture_, nullptr, &pixels, &pitch) < 0) {
        std::cerr << "Failed to lock texture: " << SDL_GetError() << std::endl;
        return;
    }
    
    // Convert BGR to RGB and copy to texture
    uint8_t* dst = static_cast<uint8_t*>(pixels);
    for (int y = 0; y < height; y++) {
        const uint8_t* src_row = data + y * stride;
        uint8_t* dst_row = dst + y * pitch;
        
        for (int x = 0; x < width; x++) {
            // BGR -> RGB conversion
            dst_row[x * 3 + 0] = src_row[x * 3 + 2];  // R
            dst_row[x * 3 + 1] = src_row[x * 3 + 1];  // G
            dst_row[x * 3 + 2] = src_row[x * 3 + 0];  // B
        }
    }
    
    SDL_UnlockTexture(texture_);
}

void Display::render() {
    // Must be called with mutex_ locked
    
    if (!renderer_ || !texture_) return;
    
    // Clear screen
    SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
    SDL_RenderClear(renderer_);
    
    // Render texture with horizontal flip for mirror mode
    // Text will be backwards, but this is handled by drawing text in reverse
    SDL_RenderCopyEx(renderer_, texture_, nullptr, nullptr, 
                     0.0, nullptr, SDL_FLIP_HORIZONTAL);
    
    // Present
    SDL_RenderPresent(renderer_);
}

void Display::show(const faceid::Image& img) {
    if (img.empty() || img.channels() != 3) {
        std::cerr << "Invalid image: must be non-empty 3-channel BGR image" << std::endl;
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Create window if not created yet
    if (!window_) {
        createWindow(img.width(), img.height());
    }
    
    updateTexture(img.data(), img.width(), img.height(), img.stride());
    render();
    handleEvents();
}

void Display::show(const faceid::ImageView& view) {
    if (view.empty() || view.channels() != 3) {
        std::cerr << "Invalid image view: must be non-empty 3-channel BGR image" << std::endl;
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Create window if not created yet
    if (!window_) {
        createWindow(view.width(), view.height());
    }
    
    updateTexture(view.data(), view.width(), view.height(), view.stride());
    render();
    handleEvents();
}

void Display::handleEvents() {
    // Must be called with mutex_ locked
    
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_QUIT:
                is_open_ = false;
                quit_requested_ = true;
                break;
                
            case SDL_KEYDOWN:
                // Store key code for waitKey()
                if (event.key.keysym.sym < 128) {
                    last_key_ = event.key.keysym.sym;
                } else if (event.key.keysym.sym == SDLK_ESCAPE) {
                    last_key_ = 27;  // ESC
                }
                break;
                
            case SDL_WINDOWEVENT:
                if (event.window.event == SDL_WINDOWEVENT_CLOSE) {
                    is_open_ = false;
                    quit_requested_ = true;
                }
                break;
        }
    }
}

int Display::waitKey(int delay_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!is_open_) {
        return -1;
    }
    
    last_key_ = -1;
    
    if (delay_ms == 0) {
        // Wait forever for a key
        while (last_key_ == -1 && is_open_) {
            SDL_Event event;
            if (SDL_WaitEvent(&event)) {
                if (event.type == SDL_QUIT) {
                    is_open_ = false;
                    quit_requested_ = true;
                    return -1;
                } else if (event.type == SDL_KEYDOWN) {
                    if (event.key.keysym.sym < 128) {
                        last_key_ = event.key.keysym.sym;
                    } else if (event.key.keysym.sym == SDLK_ESCAPE) {
                        last_key_ = 27;
                    }
                }
            }
        }
    } else {
        // Wait with timeout
        auto start = SDL_GetTicks();
        while (last_key_ == -1 && is_open_) {
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) {
                    is_open_ = false;
                    quit_requested_ = true;
                    return -1;
                } else if (event.type == SDL_KEYDOWN) {
                    if (event.key.keysym.sym < 128) {
                        last_key_ = event.key.keysym.sym;
                    } else if (event.key.keysym.sym == SDLK_ESCAPE) {
                        last_key_ = 27;
                    }
                }
            }
            
            auto now = SDL_GetTicks();
            if (now - start >= static_cast<uint32_t>(delay_ms)) {
                break;
            }
            
            SDL_Delay(1);  // Short sleep to avoid busy-wait
        }
    }
    
    return last_key_;
}

bool Display::isOpen() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return is_open_ && !quit_requested_;
}

void Display::resize(int width, int height) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!window_) {
        createWindow(width, height);
    } else {
        SDL_SetWindowSize(window_, width, height);
        width_ = width;
        height_ = height;
    }
}

// ========== Drawing Functions ==========

void drawHLine(faceid::Image& img, int x1, int x2, int y, const Color& color) {
    if (y < 0 || y >= img.height()) return;
    
    x1 = std::max(0, x1);
    x2 = std::min(img.width() - 1, x2);
    
    if (x1 > x2) std::swap(x1, x2);
    
    uint8_t* row = img.data() + y * img.stride();
    for (int x = x1; x <= x2; x++) {
        uint8_t* pixel = row + x * img.channels();
        pixel[0] = color.b;
        pixel[1] = color.g;
        pixel[2] = color.r;
    }
}

void drawVLine(faceid::Image& img, int x, int y1, int y2, const Color& color) {
    if (x < 0 || x >= img.width()) return;
    
    y1 = std::max(0, y1);
    y2 = std::min(img.height() - 1, y2);
    
    if (y1 > y2) std::swap(y1, y2);
    
    for (int y = y1; y <= y2; y++) {
        uint8_t* pixel = img.data() + y * img.stride() + x * img.channels();
        pixel[0] = color.b;
        pixel[1] = color.g;
        pixel[2] = color.r;
    }
}

void drawRectangle(faceid::Image& img, int x, int y, int width, int height,
                   const Color& color, int thickness) {
    if (width <= 0 || height <= 0) return;
    
    // Draw rectangle outline with thickness
    for (int t = 0; t < thickness; t++) {
        // Top and bottom
        drawHLine(img, x, x + width - 1, y + t, color);
        drawHLine(img, x, x + width - 1, y + height - 1 - t, color);
        
        // Left and right
        drawVLine(img, x + t, y, y + height - 1, color);
        drawVLine(img, x + width - 1 - t, y, y + height - 1, color);
    }
}

void drawFilledRectangle(faceid::Image& img, int x, int y, int width, int height,
                        const Color& color) {
    if (width <= 0 || height <= 0) return;
    
    // Clip to image bounds
    int x1 = std::max(0, x);
    int y1 = std::max(0, y);
    int x2 = std::min(img.width(), x + width);
    int y2 = std::min(img.height(), y + height);
    
    // Fill rectangle
    for (int row = y1; row < y2; row++) {
        uint8_t* pixels = img.data() + row * img.stride();
        for (int col = x1; col < x2; col++) {
            uint8_t* pixel = pixels + col * img.channels();
            pixel[0] = color.b;
            pixel[1] = color.g;
            pixel[2] = color.r;
        }
    }
}

void drawText(faceid::Image& img, const std::string& text, int x, int y,
             const Color& color, double scale) {
    if (text.empty() || scale <= 0) return;
    
    int char_width = static_cast<int>(8 * scale);
    
    int cursor_x = x;
    
    for (char c : text) {
        // Convert to unsigned to handle full ASCII range
        unsigned char uc = static_cast<unsigned char>(c);
        if (uc >= 128) {
            uc = '?';  // Replace non-ASCII with question mark
        }
        
        const uint8_t* glyph = FONT_8X8[uc];
        
        // Draw character
        for (int row = 0; row < 8; row++) {
            uint8_t row_bits = glyph[row];
            for (int col = 0; col < 8; col++) {
                if (row_bits & (1 << (7 - col))) {
                    // Draw pixel(s) for this bit (with scaling)
                    for (int sy = 0; sy < scale; sy++) {
                        for (int sx = 0; sx < scale; sx++) {
                            int px = cursor_x + static_cast<int>(col * scale) + sx;
                            int py = y + static_cast<int>(row * scale) + sy;
                            drawPixel(img, px, py, color);
                        }
                    }
                }
            }
        }
        
        cursor_x += char_width;
    }
}

void flipHorizontal(faceid::Image& img) {
    if (img.empty()) return;
    
    const int width = img.width();
    const int height = img.height();
    const int channels = img.channels();
    const int stride = img.stride();
    uint8_t* data = img.data();
    
    // Flip each row
    for (int y = 0; y < height; y++) {
        uint8_t* row = data + y * stride;
        
        // Swap pixels from both ends moving towards center
        for (int x = 0; x < width / 2; x++) {
            int x_mirror = width - 1 - x;
            
            // Swap all channels
            for (int c = 0; c < channels; c++) {
                std::swap(row[x * channels + c], row[x_mirror * channels + c]);
            }
        }
    }
}

} // namespace faceid
