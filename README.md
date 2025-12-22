# FaceID - Linux Face Authentication System

Single-line PAM authentication combining face recognition, fingerprint, and IR emitter support. Replace multiple PAM configs (lid detection script + howdy + linux-enable-ir-emitter + fprintd) with: `auth sufficient pam_faceid.so`.
Tested in my T14 Gen4 Ryzen (Manjaro Plasma Wayland).

## Key Features

- **Parallel Biometric Auth**: Face + fingerprint run simultaneously, first to succeed wins (~50-500ms)
- **Fast Face Detection**: LibFaceDetection CNN (embedded model, no external files) + SFace (recognition)
- **Smart Lid Detection**: Skips biometric auth when lid closed, saves 5-second timeout
- **Shoulder Surfing Detection**: Detects multiple faces and blanks screen if someone looks over your shoulder
- **Live Camera Preview**: `faceid show` command displays real-time face detection visualization with SDL2 hardware acceleration
- **Single Binary**: C++20, no Python dependencies, low memory footprint (~50-100 MB)
- **D-Bus Fingerprint**: Integrates with fprintd, uses existing enrollments
- **Performance**: ~25ms total pipeline (2.6x faster than OpenCV), with AVX512 optimization
- **Comprehensive Logging**: Audit trail at `/var/log/faceid.log`
- **PAM Compatible**: Works with sudo, login, lock screen, GDM, LightDM
- **Zero OpenCV Dependencies**: 83% smaller binaries (3.3MB vs 20MB), specialized libraries for each task

## Quick Start

### 1. Install Dependencies

**Arch Linux:**
```bash
sudo pacman -S base-devel meson ninja ncnn pam libjpeg-turbo sdl2 libyuv glib2 fprintd
```

**Debian/Ubuntu:**
```bash
sudo apt-get install build-essential meson ninja-build libncnn-dev \
    libpam0g-dev libturbojpeg0-dev libsdl2-dev libyuv-dev libglib2.0-dev fprintd pkg-config
```

**Fedora:**
```bash
sudo dnf install gcc-c++ meson ninja-build ncnn-devel \
    pam-devel turbojpeg-devel SDL2-devel libyuv-devel glib2-devel fprintd pkgconfig
```

### 2. Build & Install

```bash
meson setup build
meson compile -C build
sudo meson install -C build
```

Or use Makefile: `make build && sudo make install`

### 3. Download Face Recognition Model

FaceID uses **NCNN format** models (`.param` + `.bin` files) for face recognition. Face detection uses LibFaceDetection with embedded model weights - no download needed!

**Note:** Pre-converted NCNN models for SFace are not officially available. You must convert the ONNX model yourself using the instructions below.

#### Convert ONNX to NCNN Format

```bash
# 1. Install NCNN tools (includes onnx2ncnn converter)
# Arch Linux:
sudo pacman -S ncnn

# Debian/Ubuntu (build from source):
git clone https://github.com/Tencent/ncnn.git
cd ncnn
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_BUILD_TOOLS=ON ..
make -j$(nproc)
sudo make install

# 2. Download ONNX model from OpenCV Model Zoo
cd /etc/faceid/models
sudo wget -O face_recognition_sface_2021dec.onnx \
    https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx

# 3. Convert ONNX to NCNN format
onnx2ncnn face_recognition_sface_2021dec.onnx sface.param sface.bin

# 4. Verify the files exist
ls -lh sface.param sface.bin
# Expected output:
#   sface.param (~9KB)   - NCNN model architecture
#   sface.bin (~37MB)    - NCNN model weights

# 5. Remove ONNX file (optional, saves 37MB)
sudo rm face_recognition_sface_2021dec.onnx
```

**Required Files:**
- `/etc/faceid/models/sface.param` (9KB) - Model architecture
- `/etc/faceid/models/sface.bin` (37MB) - Model weights

**Note:** The system will automatically load these files from `/etc/faceid/models/sface.{param,bin}` by default.

### 4. Enroll

```bash
# Face
sudo faceid add $(whoami)

# Fingerprint (optional, uses fprintd)
fprintd-enroll
```

### 5. Configure PAM

Add to `/etc/pam.d/sudo`:
```
auth       sufficient   pam_faceid.so
```

That's it! Test with `sudo ls`

## How It Works

### Parallel Authentication Flow
```
PAM Request → Lid Check
                ↓
         ┌──────┴──────┐
      CLOSED         OPEN
         ↓              ↓
    Password    Face + Fingerprint Threads
                       ↓
                 First Success Wins
                       ↓
                  PAM_SUCCESS
```

### Laptop Lid Detection
Automatically detects closed lid and skips biometric auth:
- **Methods**: `/proc/acpi/button/lid/*/state`, `/sys/class/input/*/lid_state`, systemd-logind D-Bus
- **Benefit**: Saves 5-second timeout when lid closed

### Fingerprint via D-Bus
Uses fprintd D-Bus API (`net.reactivated.Fprint`) instead of direct libfprint-2 access:
- Works with existing `fprintd-enroll` enrollments
- No storage path dependencies
- Proper device management

## Configuration

`/etc/faceid/faceid.conf`:

```ini
[camera]
device = /dev/video0
width = 640
height = 480

[recognition]
threshold = 0.6          # Lower = stricter matching
timeout = 5              # Authentication timeout (seconds)

[authentication]
enable_fingerprint = true
enable_optimizations = true
check_lid_state = true   # Skip biometric auth when lid closed

[no_peek]
enabled = true
min_face_distance_pixels = 80    # Min distance between faces to count as separate person
min_face_size_percent = 0.08     # Min face size (% of screen width) to trigger blanking
peek_detection_delay_seconds = 2 # Delay before blanking screen
unblank_delay_seconds = 3        # Grace period before re-enabling screen

[schedule]
enabled = false
active_days = 1,2,3,4,5          # Weekdays only (1=Monday, 7=Sunday)
time_start = 0900                # Start time (HHMM format)
time_end = 1700                  # End time (HHMM format)

[logging]
log_file = /var/log/faceid.log
log_level = INFO
```

## CLI Commands

```bash
faceid devices              # List cameras
faceid show                 # Live camera preview with face detection
sudo faceid add <user>      # Enroll face
sudo faceid remove <user>   # Remove face model
sudo faceid test <user>     # Test recognition (includes integrity checks)
sudo faceid list            # List enrolled users
tail -f /var/log/faceid.log # View auth logs
```

**Note:** `faceid test` automatically performs encoding integrity checks before the live camera test, verifying:
- Proper normalization of face encodings
- No NaN or Inf values in the data
- Correct dimensions (128D vectors)
- Self-similarity validation

## PAM Configuration Examples

### Recommended: Face/Fingerprint OR Password
`/etc/pam.d/sudo`:
```
auth       sufficient   pam_faceid.so
auth       required     pam_unix.so nullok
```

### Try Biometric then Password
```
auth       [success=2 default=ignore]  pam_faceid.so
auth       required     pam_unix.so nullok try_first_pass
```

**⚠️ IMPORTANT**: Always test in a new terminal! Keep a root shell open when making PAM changes.

## Common Issues

### Camera Not Detected
```bash
ls -l /dev/video*
sudo usermod -aG video $USER  # Add to video group
```

### Face Not Recognized  
- Ensure good lighting, look directly at camera
- Try re-enrolling: `sudo faceid add username`
- Adjust threshold in config (higher = more lenient)

### Model Loading Failed
```bash
# Check if NCNN model files exist
ls -lh /etc/faceid/models/sface.{param,bin}

# If missing, you need to convert ONNX to NCNN format (see installation step 3)
# Or download pre-converted models

# Check file permissions
sudo chmod 644 /etc/faceid/models/sface.{param,bin}
```

**Error: "Failed to load face recognition model"**
- Ensure both `sface.param` (9KB) and `sface.bin` (37MB) exist in `/etc/faceid/models/`
- If you have the ONNX file instead, convert it using `onnx2ncnn` (see installation instructions)
- The system uses NCNN format, not ONNX format directly

### Fingerprint Not Working
```bash
systemctl status fprintd            # Check service
fprintd-list $(whoami)              # Verify enrollment
fprintd-verify                      # Test standalone
grep "Fingerprint" /var/log/faceid.log  # Check logs
```

### Lid Detection Issues
```bash
cat /proc/acpi/button/lid/*/state   # Test detection
# Disable if needed: check_lid_state = false in config
```

### Recovery if Locked Out
1. Boot into recovery mode or live USB
2. Mount root: `sudo mount /dev/sdaX /mnt`
3. Edit `/mnt/etc/pam.d/sudo` - remove `pam_faceid.so` line
4. Reboot

## FAQ

**Q: Do both face and fingerprint need to be enrolled?**  
A: No! Any combination works: face only, fingerprint only, both (faster), or neither (falls to password).

**Q: Which is faster?**  
A: Face (~45-60ms) is now faster than fingerprint (~200-400ms), but both run in parallel so first to succeed wins.

**Q: What happens when lid is closed?**  
A: Biometric auth is skipped immediately (< 1ms), goes straight to password.

**Q: Can I use this with SSH?**  
A: No camera/sensor over SSH. Password will be used automatically.

**Q: How to unenroll?**  
- Face: `sudo faceid remove $(whoami)`
- Fingerprint: `fprintd-delete $(whoami)`

**Q: What is shoulder surfing detection?**  
A: The "no peek" feature detects multiple faces in the camera frame and blanks the screen if someone stands behind you. Adjustable via `min_face_distance_pixels` and `min_face_size_percent` in the `[no_peek]` config section.

**Q: What model format does FaceID use?**  
A: FaceID uses **NCNN format** (`.param` + `.bin` files) for face recognition. If you have an ONNX model, convert it using the `onnx2ncnn` tool (see installation step 3). Face detection uses LibFaceDetection with embedded weights, so no external model file is needed.

**Q: Can I use the ONNX model directly?**  
A: No. FaceID requires NCNN format for optimal performance. NCNN is specifically optimized for mobile/embedded devices and offers better speed than ONNX runtime on CPU. Use `onnx2ncnn` to convert your model.

**Q: Where can I get pre-converted NCNN models?**  
A: The SFace model can be converted from the [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface). Follow the conversion instructions in installation step 3.

**Q: How do I preview the camera and see face detection?**  
A: Run `faceid show` for a live camera feed with real-time face detection rectangles (green for primary face, yellow for additional faces). Shows FPS, face count, and resolution.

**Q: What does schedule-based presence do?**  
A: Presence detection can be limited to specific days/times (e.g., weekdays 9-5). Enable via `enabled = true` in `[schedule]` section and set `active_days` and time range.

## Uninstallation

```bash
# Remove PAM config first!
sudo nano /etc/pam.d/sudo  # Remove pam_faceid.so line

# Uninstall
sudo make uninstall
sudo rm -rf /etc/faceid
```

## Security Notice

⚠️ Face/fingerprint authentication is less secure than passwords. Can be fooled by photos or similar-looking people. Always use `sufficient` PAM rules with password fallback, never `required` alone.

## Technical Details

**Authentication Flow:**
1. Check lid state → skip if closed
2. Launch parallel threads: face (LibFaceDetection CNN + SFace recognition) + fingerprint (D-Bus → fprintd)
3. First success cancels the other → PAM_SUCCESS
4. Both timeout (5s) → PAM_FAILURE → password prompt

**Face Detection Pipeline:**
- **Detection**: LibFaceDetection CNN (embedded 436KB model weights, ~2.5ms with AVX512, 6x faster than OpenCV)
- **Recognition**: NCNN SFace/MobileFaceNet (NCNN format: `sface.param` + `sface.bin`, ~13ms, 128D encodings)
- **Preprocessing**: Custom CLAHE implementation (0.77ms, 6.6x faster than OpenCV)
- **Advantages**: Embedded model (no external files), SIMD optimized (AVX512/AVX2/NEON), zero OpenCV dependencies
- **Multi-face Analysis**: Calculates center-to-center distances and face sizes to distinguish multiple people
- **Speed**: 25ms total pipeline (2.6x faster than OpenCV-based implementation)
- **Camera**: V4L2 + TurboJPEG (33ms, 2x faster than cv::VideoCapture)
- **Display**: SDL2 hardware-accelerated rendering with mirror mode support

**Shoulder Surfing Detection ("No Peek"):**
- **Purpose**: Detects when additional faces appear in the frame (shoulder surfing attempt)
- **Method**: Compares face distances and sizes; if multiple distinct faces detected, blanks screen
- **Configurable**: `min_face_distance_pixels`, `min_face_size_percent`, detection/unblank delays
- **Behavior**: Screen blanks after `peek_detection_delay_seconds` if extra face persists; unblank delay prevents flickering
- **Security**: Complements biometric auth by protecting screen content from over-the-shoulder viewers

**Performance:**
- Auth time: ~50-500ms (parallel), ~50-100MB memory
- Total face auth: ~25ms pipeline (2.6x faster than OpenCV implementation)
  - Camera capture: 33ms (V4L2 + TurboJPEG, 2x faster than cv::VideoCapture)
  - Face tracking: 2.5ms (custom optical flow, 6x faster than cv::calcOpticalFlowPyrLK)
  - CLAHE preprocessing: 0.77ms (6.6x faster than cv::CLAHE)
  - Face detection: 2.5ms (LibFaceDetection with AVX512)
  - Face recognition: 13ms (NCNN SFace)
- Binary size: 3.3MB (83% smaller than OpenCV-based version)
- Optimizations: LibFaceDetection with AVX512/AVX2/NEON SIMD, custom CLAHE, libyuv image processing, V4L2 camera, SDL2 display
- Logging: `/var/log/faceid.log` with timestamps, durations, methods

**Storage:**
- Face models: `/etc/faceid/models/<user>.bin` or `/etc/faceid/models/<user>.<location>.bin` (binary format with 128D SFace encodings)
- Recognition model: `/etc/faceid/models/sface.param` + `/etc/faceid/models/sface.bin` (NCNN format)
- Fingerprints: fprintd database (`/var/lib/fprint/`, managed via D-Bus)

**Live Camera Preview (`faceid show`):**
- **Purpose**: Debug camera setup and visualize face detection in real-time
- **Display**: SDL2 hardware-accelerated rendering with mirror mode (natural camera view)
- **Visual Feedback**: Green rectangle for primary (closest) face, yellow for additional faces, reversed text overlays
- **Info Banner**: Shows detected face count, FPS, and resolution
- **Controls**: Press 'q' or ESC to quit
- **Use Cases**: Verify camera is working, test detection sensitivity, check lighting conditions
- **Performance**: Real-time 30+ FPS with face detection and overlay rendering

## Technical Architecture

**Zero OpenCV Dependencies:**

FaceID has completely eliminated OpenCV in favor of specialized, faster libraries:

| Component | Old (OpenCV) | New (Specialized) | Improvement |
|-----------|-------------|-------------------|-------------|
| Camera | `cv::VideoCapture` | V4L2 + TurboJPEG | 2.0x faster |
| Image ops | `cv::resize`, `cv::cvtColor` | libyuv | 3-5x faster |
| CLAHE | `cv::CLAHE` | Custom implementation | 6.6x faster |
| Optical flow | `cv::calcOpticalFlowPyrLK` | Custom Lucas-Kanade | 3.9x faster |
| Display | `cv::imshow` | SDL2 | Hardware accelerated |
| Data types | `cv::Mat`, `cv::Rect` | Custom `Image`/`ImageView` | Type-safe, move-only |

**Dependencies:**
- **NCNN**: Face recognition inference engine (optimized neural network framework)
- **TurboJPEG**: Fast MJPEG decompression for V4L2 camera
- **libyuv**: Hardware-accelerated image resizing and color conversion
- **SDL2**: Hardware-accelerated display rendering
- **GLib/GIO**: D-Bus communication for fingerprint
- **PAM**: Linux authentication framework

**Why No OpenCV?**
- 83% smaller binaries (3.3MB vs 20MB with OpenCV)
- 2.6x faster overall pipeline
- No unnecessary dependencies (OpenCV is massive and includes hundreds of unused algorithms)
- Better performance through specialized libraries (libyuv uses SIMD, SDL2 uses GPU)
- Cleaner, more maintainable code with purpose-built data structures

## Credits

Inspired by [Howdy](https://github.com/boltgolt/howdy) + [linux-enable-ir-emitter](https://github.com/EmixamPP/linux-enable-ir-emitter). Uses LibFaceDetection for face detection, NCNN SFace for recognition, fprintd for fingerprints. OpenCV completely eliminated in favor of V4L2, TurboJPEG, libyuv, and SDL2.

MIT License
