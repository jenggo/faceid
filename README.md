# FaceID - Linux Face Authentication System

Single-line PAM authentication combining face recognition, fingerprint, and IR emitter support. Replace multiple PAM configs (lid detection script + howdy + linux-enable-ir-emitter + fprintd) with: `auth sufficient pam_faceid.so`.
Tested in my T14 Gen4 Ryzen (Manjaro Plasma Wayland).

## Key Features

- **Parallel Biometric Auth**: Face + fingerprint run simultaneously, first to succeed wins (~50-500ms)
- **Fast Face Detection**: LibFaceDetection CNN (embedded model, no external files) + SFace (recognition)
- **Smart Lid Detection**: Skips biometric auth when lid closed, saves 5-second timeout
- **Shoulder Surfing Detection**: Detects multiple faces and blanks screen if someone looks over your shoulder
- **Live Camera Preview**: `faceid show` command displays real-time face detection visualization
- **Single Binary**: C++20, no Python dependencies, low memory footprint (~50-100 MB)
- **D-Bus Fingerprint**: Integrates with fprintd, uses existing enrollments
- **Performance**: ~45-60ms total (LibFaceDetection ~40ms + SFace ~13ms), with AVX512 optimization
- **Comprehensive Logging**: Audit trail at `/var/log/faceid.log`
- **PAM Compatible**: Works with sudo, login, lock screen, GDM, LightDM

## Quick Start

### 1. Install Dependencies

**Arch Linux:**
```bash
sudo pacman -S base-devel meson ninja opencv pam jsoncpp glib2 fprintd
```

**Debian/Ubuntu:**
```bash
sudo apt-get install build-essential meson ninja-build libopencv-dev \
    libpam0g-dev libjsoncpp-dev libglib2.0-dev fprintd pkg-config
```

**Fedora:**
```bash
sudo dnf install gcc-c++ meson ninja-build opencv-devel \
    pam-devel jsoncpp-devel glib2-devel fprintd pkgconfig
```

### 2. Build & Install

```bash
meson setup build
meson compile -C build
sudo meson install -C build
```

Or use Makefile: `make build && sudo make install`

### 3. Download Face Recognition Model

```bash
# Create directories
sudo mkdir -p /etc/faceid/models && cd /etc/faceid/models

# Download SFace face recognition model (37MB)
# NOTE: Face detection uses LibFaceDetection with embedded model weights - no download needed!
sudo wget -O face_recognition_sface_2021dec.onnx \
    https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

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
sudo faceid test <user>     # Test recognition
sudo faceid list            # List enrolled users
tail -f /var/log/faceid.log # View auth logs
```

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
- **Detection**: LibFaceDetection CNN (embedded 436KB model weights, ~40ms with AVX512)
- **Recognition**: OpenCV SFace/MobileFaceNet (ONNX model, 37MB, ~13ms, 128D encodings)
- **Advantages**: Embedded model (no external files), SIMD optimized (AVX512/AVX2/NEON), faster than OpenCV YuNet
- **Multi-face Analysis**: Calculates center-to-center distances and face sizes to distinguish multiple people
- **Speed**: 45-60ms total (faster than YuNet's 63ms)

**Shoulder Surfing Detection ("No Peek"):**
- **Purpose**: Detects when additional faces appear in the frame (shoulder surfing attempt)
- **Method**: Compares face distances and sizes; if multiple distinct faces detected, blanks screen
- **Configurable**: `min_face_distance_pixels`, `min_face_size_percent`, detection/unblank delays
- **Behavior**: Screen blanks after `peek_detection_delay_seconds` if extra face persists; unblank delay prevents flickering
- **Security**: Complements biometric auth by protecting screen content from over-the-shoulder viewers

**Performance:**
- Auth time: ~50-500ms (parallel), ~50-100MB memory
- Total face auth: ~45-60ms (LibFaceDetection 40ms + SFace 13ms)
- Optimizations: LibFaceDetection with AVX512/AVX2/NEON SIMD, SFace fast recognition, frame downscaling, detection caching
- Logging: `/var/log/faceid.log` with timestamps, durations, methods

**Storage:**
- Face models: `/etc/faceid/models/<user>.json` (128D SFace encodings)
- Fingerprints: fprintd database (`/var/lib/fprint/`, managed via D-Bus)

**Live Camera Preview (`faceid show`):**
- **Purpose**: Debug camera setup and visualize face detection in real-time
- **Display**: Green rectangle for primary (closest) face, yellow for additional faces
- **Info Banner**: Shows detected face count, FPS, and resolution
- **Controls**: Press 'q' or ESC to quit
- **Use Cases**: Verify camera is working, test detection sensitivity, check lighting conditions

## Credits

Inspired by [Howdy](https://github.com/boltgolt/howdy) + [linux-enable-ir-emitter](https://github.com/EmixamPP/linux-enable-ir-emitter). Uses OpenCV YuNet/SFace for face recognition, fprintd for fingerprints.

MIT License
