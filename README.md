# FaceID - Linux Face Authentication

Fast biometric PAM authentication with face recognition + fingerprint support. Single-line PAM config: `auth sufficient pam_faceid.so`

Tested on T14 Gen4 Ryzen (Manjaro Plasma Wayland)

## Key Features

- **Parallel Biometric Auth**: Face + fingerprint run simultaneously, first to succeed wins (~50-500ms)
- **Embedded Face Detection**: YuNet (primary) + RetinaFace (fallback) built into binary - no model downloads needed
- **Cascading Detection**: 3-stage detection (standard → aggressive preprocessing → fallback model) for robust face detection
- **Facial Landmarks**: Real-time visualization of 5-point landmarks (eyes, nose, mouth) in preview and test modes
- **Smart Face Matching**: Deduplicates multiple detections and validates uniqueness with margin check to prevent false positives
- **Smart Lid Detection**: Skips biometric auth when lid closed, saves 5-second timeout
- **Shoulder Surfing Detection**: Detects multiple faces and blanks screen if someone looks over your shoulder
- **Live Camera Preview**: `faceid show` command displays real-time face detection with landmarks and SDL2 hardware acceleration
- **Auto-Optimization**: Enrollment automatically finds optimal detection confidence and recognition threshold
- **Single Binary**: C++20, no Python dependencies, low memory footprint (~50-100 MB)
- **D-Bus Fingerprint**: Integrates with fprintd, uses existing enrollments
- **Performance**: ~7-15ms face detection, ~20-33ms recognition (model-dependent)
- **Comprehensive Logging**: Audit trail at `/var/log/faceid.log`
- **PAM Compatible**: Works with sudo, login, lock screen, GDM, LightDM
- **Zero OpenCV Dependencies**: 83% smaller binaries (3.3MB vs 20MB), specialized libraries for each task

## Quick Start

### Dependencies

**Arch**: `sudo pacman -S base-devel meson ninja ncnn pam libjpeg-turbo sdl2 libyuv glib2 fprintd systemd`  
**Debian/Ubuntu**: `sudo apt install build-essential meson ninja-build libncnn-dev libpam0g-dev libturbojpeg0-dev libsdl2-dev libyuv-dev libglib2.0-dev fprintd libsystemd-dev pkg-config`  
**Fedora**: `sudo dnf install gcc-c++ meson ninja-build ncnn-devel pam-devel turbojpeg-devel SDL2-devel libyuv-devel glib2-devel fprintd systemd-devel pkgconfig`

### Build & Install

```bash
make build && sudo make install
```

**Note**: Detection models (YuNet + RetinaFace) are embedded in the binary. Only recognition models need to be downloaded separately.

### Download Recognition Models

**Storage Location**: Models are stored in **per-user XDG directories** (not system-wide):
- Primary: `~/.local/share/faceid/models/` (XDG_DATA_HOME/faceid/models)
- Fallback: `~/.local/share/faceid/models/` if XDG_DATA_HOME is not set
- Temp fallback: `/tmp/faceid/models/` (session-only)

**Installation**:
```bash
# The installer (sudo make install) automatically sets up your user's models directory
# and copies recognition models to ~/.local/share/faceid/models/ if they're available

# To manually download a recognition model:
pip3 install pnnx
mkdir -p ~/.local/share/faceid/models
cd ~/.local/share/faceid/models
wget https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
pnnx face_recognition_sface_2021dec.onnx
mv face_recognition_sface_2021dec.ncnn.{param,bin} recognition.{param,bin}
```

### Model Selection Guide

FaceID supports multiple recognition models with different trade-offs between speed and accuracy.

#### Benchmark Your Models

```bash
# Download test models to /tmp/models/
# Then benchmark them:
faceid bench /tmp/models

# Or with a custom test image:
faceid bench --image face.jpg /tmp/models
```

#### Performance Comparison

| Model | Dimension | Size | Encoding Time | FPS | Best For |
|-------|-----------|------|---------------|-----|----------|
| **sface (int8)** | 128D | 18 MB | ~4.5 ms | 225 fps | ⚡ Speed, Real-time |
| **webface_r50** | 512D | 83 MB | ~20 ms | 50 fps | ⚖️ Balanced |
| **glint360k_r50_pfc** | 512D | 83 MB | ~20 ms | 50 fps | 🎯 Accuracy, Security |
| **ms1m_megaface_r50** | 512D | 83 MB | ~20 ms | 50 fps | 🎯 Accuracy, Security |
| **ms1mv2_r50_pfc** | 512D | 83 MB | ~20 ms | 49 fps | 🎯 Accuracy, Security |

#### Use Case Recommendations

**Real-time Authentication (PAM, Screen Lock)**:
- **Use:** sface (128D)
- **Why:** 4.5x faster encoding, instant unlock feel
- **Trade-off:** Lower accuracy with extreme lighting/angles

**High Security (Multi-user, Public Systems)**:
- **Use:** glint360k_r50_pfc (512D)
- **Why:** Trained on 360K identities, better at distinguishing similar faces
- **Trade-off:** 4x slower, larger file size

**Home/Office (Balanced)**:
- **Use:** sface for single-user, glint360k for multi-user

#### Install a Different Model

**Each user has their own model copy** in `~/.local/share/faceid/models/`:

```bash
# Option 1: Standard naming (recommended)
cp model.ncnn.param ~/.local/share/faceid/models/recognition.param
cp model.ncnn.bin ~/.local/share/faceid/models/recognition.bin

# Option 2: Legacy naming  
cp model.ncnn.param ~/.local/share/faceid/models/sface.param
cp model.ncnn.bin ~/.local/share/faceid/models/sface.bin

# Re-enroll after changing models (required)
faceid add $(whoami)
```

**Note:** Switching between different model dimensions (128D ↔ 512D) or different models always requires re-enrollment.

### Enroll & Configure

```bash
# Enable the system daemon (FaceID runs as a system service so PAM can reach it):
sudo systemctl daemon-reload && sudo systemctl enable --now faceid.service

# Enroll face (daemon handles ML; CLI captures frames and forwards them to the daemon):
sudo faceid add $(whoami)
# This automatically:
# - Finds optimal detection confidence for your face
# - Captures 5 diverse samples with guided prompts
# - Calculates optimal recognition threshold based on face matching
# - Validates matches with margin check to prevent false positives
# - Stores face data in ~/.local/share/faceid/faces/

# Enroll fingerprint (optional)
fprintd-enroll

# Add to /etc/pam.d/sudo
auth       sufficient   pam_faceid.so

# Test
sudo ls
```

## Commands

```bash
faceid devices              # List cameras
faceid show                 # Live preview with face detection and landmarks
faceid bench <directory>    # Benchmark recognition models (uses embedded image)
faceid bench --image <path> <directory>  # Benchmark with custom test image
faceid image test           # Test with static images (auto-finds optimal thresholds)
sudo faceid add <user>      # Enroll face (auto-optimizes settings)
sudo faceid test <user>     # Test authentication with timing metrics
sudo faceid list            # List enrolled users
sudo faceid remove <user>   # Remove user enrollment
```

## Face Matching Algorithm

FaceID uses a robust two-stage matching algorithm to prevent false positives:

1. **Distance Threshold Check**: Compares detected face encoding against all enrolled users
2. **Uniqueness Validation**: When multiple enrolled users exist, ensures the best match has a significant margin (≥5%) over the second-best match to confirm a unique, unambiguous match
3. **Deduplication**: Filters duplicate detections of the same face at different angles/positions before matching

This prevents scenarios where:
- Similar faces from different users are confused
- Multiple detections of the same face create false positives
- Border cases between two enrolled users are incorrectly accepted

## Facial Landmarks Visualization

When using `faceid show` or `faceid test`, detected facial landmarks are displayed:
- **Cyan circles**: Left and right eyes
- **Blue circle**: Nose tip
- **Magenta circles**: Left and right mouth corners

Landmarks help validate face detection quality and show which facial features are being tracked.

## Troubleshooting

**Camera not detected**: `ls -l /dev/video* && sudo usermod -aG video $USER`  
**Face not recognized**: Good lighting, re-enroll, or adjust threshold in config  
**Model loading failed**: Check `~/.local/share/faceid/models/` for recognition model files (e.g., `recognition.{param,bin}` or `sface.{param,bin}`). Models are per-user; you may need to download them if they weren't copied during installation.  
**Fingerprint issues**: `systemctl status fprintd && fprintd-verify`  
**Daemon not running**: `systemctl --user enable --now faceid-daemon && journalctl --user -u faceid-daemon -n 50`  
**Locked out**: Boot recovery, mount root, edit `/mnt/etc/pam.d/sudo`, remove pam_faceid.so line  
**False positives**: Re-enroll to update threshold, or manually adjust threshold in `~/.config/faceid/config.yaml`  
**Multiple faces detected**: Use `faceid show` to verify face detection is working correctly; deduplication filters duplicate detections

## Configuration

FaceID uses **user-level XDG configuration**:
- Primary: `~/.config/faceid/config.yaml`
- System defaults: `/etc/faceid/config.yaml` (reference only)
- Models: `~/.local/share/faceid/models/` (per-user, auto-created on install)
- Faces: `~/.local/share/faceid/faces/` (per-user enrollments)

Edit `~/.config/faceid/config.yaml`:
- `[recognition] threshold = 0.6` - Lower = stricter (auto-set during enrollment)
- `[recognition] confidence = 0.5` - Detection confidence (auto-set during enrollment)
- `[authentication] check_lid_state = true` - Skip auth when lid closed
- `[no_peek]` - Screen blanking when multiple faces detected
- `[logging] log_level = INFO` - View logs: `journalctl --user -u faceid-daemon -f`

**Note**: Values are automatically optimized during `faceid add` enrollment. Manual adjustment rarely needed.

## FAQ

**Enrollment**: Face and/or fingerprint (any combo works, both = faster)  
**Speed**: Face ~48ms, fingerprint ~200-400ms (parallel, first wins)  
**Models**: Detection models (YuNet + RetinaFace) embedded in binary. Recognition models from NCNN format (SFace via PNNX conversion)  
**SSH**: Falls back to password (no camera)  
**Unenroll**: `sudo faceid remove <user>` or `fprintd-delete <user>`  
**Landmarks**: 5-point landmarks (eyes, nose, mouth) shown in preview and test modes for visual verification  
**Matching robustness**: Deduplication + margin check prevents false positives with similar faces

## Technical Details

**Architecture**: User-level D-Bus service (systemd --user) with per-user XDG storage  
**Pipeline**: Lid check → parallel (face detection + recognition | fprintd) → first success wins  
**Detection**: YuNet (primary) + RetinaFace (fallback) embedded in binary with 3-stage cascading  
**Recognition Models**: SFace/MobileFaceNet/ArcFace variants (128D-512D encodings) stored in `~/.local/share/faceid/models/`  
**Performance**: ~7-15ms detection (YuNet), ~20-33ms recognition, ~50-100MB memory  
**Storage**: 
  - Models: `~/.local/share/faceid/models/` (per-user, ~18-83MB per user)
  - Faces: `~/.local/share/faceid/faces/` (per-user enrollments)
  - Config: `~/.config/faceid/config.yaml` (per-user settings)
  - Logs: `/var/log/faceid.log` (system-wide audit trail)
**Enrollment**: Binary search for optimal confidence (0.01 precision), 5 samples with 1s intervals, automatic threshold calculation with deduplication and margin validation  
**Face Matching**: Two-stage algorithm with threshold check + uniqueness validation + deduplication

**Zero OpenCV Stack**:
- Camera: V4L2 + TurboJPEG (2x faster than cv::VideoCapture)
- Image ops: libyuv (3-5x faster than cv::resize)
- CLAHE: Custom (6.6x faster than cv::CLAHE)
- Display: SDL2 (hardware accelerated)
- Result: 83% smaller binary (3.3MB vs 20MB)

**Dependencies**: NCNN (inference), TurboJPEG (MJPEG), libyuv (SIMD ops), SDL2 (display), GLib (D-Bus), libsystemd (session management), PAM

---

Inspired by [Howdy](https://github.com/boltgolt/howdy) + [linux-enable-ir-emitter](https://github.com/EmixamPP/linux-enable-ir-emitter) | MIT License
