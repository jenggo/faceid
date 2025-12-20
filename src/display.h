/*
 * SDL2-Based Display Wrapper for FaceID
 * 
 * Provides OpenCV-compatible API for displaying images and basic drawing.
 * Designed to replace opencv_highgui with hardware-accelerated SDL2 rendering.
 * 
 * Features:
 * - BGR format support (OpenCV-compatible)
 * - Hardware-accelerated rendering
 * - Basic drawing primitives (rectangles, text)
 * - Event handling (keyboard, window close)
 * - Thread-safe operations
 */

#ifndef FACEID_DISPLAY_H
#define FACEID_DISPLAY_H

#include <string>
#include <mutex>
#include <SDL2/SDL.h>
#include "image.h"

namespace faceid {

// ========== Color Struct (BGR order like OpenCV) ==========

struct Color {
    uint8_t b, g, r;
    
    constexpr Color(uint8_t b_, uint8_t g_, uint8_t r_) noexcept 
        : b(b_), g(g_), r(r_) {}
    
    // Pre-defined colors (BGR order)
    static constexpr Color Red()     { return Color(0, 0, 255); }
    static constexpr Color Green()   { return Color(0, 255, 0); }
    static constexpr Color Blue()    { return Color(255, 0, 0); }
    static constexpr Color White()   { return Color(255, 255, 255); }
    static constexpr Color Black()   { return Color(0, 0, 0); }
    static constexpr Color Yellow()  { return Color(0, 255, 255); }
    static constexpr Color Cyan()    { return Color(255, 255, 0); }
    static constexpr Color Magenta() { return Color(255, 0, 255); }
    static constexpr Color Orange()  { return Color(0, 165, 255); }
    static constexpr Color Gray()    { return Color(128, 128, 128); }
};

// ========== Display Class (Main Window) ==========

class Display {
public:
    // Constructor: Create display window
    // Parameters:
    //   window_name: Window title
    //   width: Initial window width (0 = auto from first image)
    //   height: Initial window height (0 = auto from first image)
    Display(const char* window_name, int width = 640, int height = 480);
    
    // Destructor: Cleanup SDL resources
    ~Display();
    
    // Delete copy (display windows are unique)
    Display(const Display&) = delete;
    Display& operator=(const Display&) = delete;
    
    // Display an image (BGR format)
    void show(const faceid::Image& img);
    void show(const faceid::ImageView& view);
    
    // Wait for key press (OpenCV-compatible)
    // Parameters:
    //   delay_ms: Wait time in milliseconds (0 = wait forever, >0 = timeout)
    // Returns: ASCII key code, or -1 if no key pressed (timeout)
    //          27 for ESC, 'q' for 'q', etc.
    int waitKey(int delay_ms = 0);
    
    // Check if window is still open (user hasn't closed it)
    bool isOpen() const;
    
    // Resize window
    void resize(int width, int height);
    
private:
    void initSDL();
    void createWindow(int width, int height);
    void createTexture(int width, int height);
    void updateTexture(const uint8_t* data, int width, int height, int stride);
    void render();
    void handleEvents();
    
    std::string window_name_;
    int width_;
    int height_;
    bool is_open_;
    
    SDL_Window* window_;
    SDL_Renderer* renderer_;
    SDL_Texture* texture_;
    
    mutable std::mutex mutex_;  // Thread safety
    
    // Event tracking
    int last_key_;
    bool quit_requested_;
    
    // SDL initialization counter (shared across all Display instances)
    static int sdl_init_count_;
    static std::mutex sdl_init_mutex_;
};

// ========== Drawing Functions (Modify Image in-place) ==========

// Draw rectangle outline
void drawRectangle(faceid::Image& img, int x, int y, int width, int height,
                   const Color& color, int thickness = 1);

// Draw filled rectangle
void drawFilledRectangle(faceid::Image& img, int x, int y, int width, int height,
                        const Color& color);

// Draw text (simple 8x8 bitmap font)
// Parameters:
//   img: Image to draw on
//   text: Text string to draw
//   x, y: Top-left corner position
//   color: Text color
//   scale: Scale factor (1.0 = normal size, 2.0 = double size)
void drawText(faceid::Image& img, const std::string& text, int x, int y,
             const Color& color, double scale = 1.0);

// Helper: Draw a single pixel (with bounds checking)
inline void drawPixel(faceid::Image& img, int x, int y, const Color& color) {
    if (x >= 0 && x < img.width() && y >= 0 && y < img.height()) {
        uint8_t* pixel = img.data() + y * img.stride() + x * img.channels();
        pixel[0] = color.b;
        pixel[1] = color.g;
        pixel[2] = color.r;
    }
}

// Helper: Draw horizontal line
void drawHLine(faceid::Image& img, int x1, int x2, int y, const Color& color);

// Helper: Draw vertical line
void drawVLine(faceid::Image& img, int x, int y1, int y2, const Color& color);

// Helper: Flip image horizontally (in-place, for mirror mode)
void flipHorizontal(faceid::Image& img);

} // namespace faceid

#endif // FACEID_DISPLAY_H
