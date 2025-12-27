# FaceID - Linux Face Authentication

Fast biometric PAM authentication with face recognition + fingerprint support. Single-line PAM config: `auth sufficient pam_faceid.so`

Tested on T14 Gen4 Ryzen (Manjaro Plasma Wayland)

## Key Features

- **Parallel Biometric Auth**: Face + fingerprint run simultaneously, first to succeed wins (~50-500ms)
- **Fast Face Detection**: RetinaFace (MobileNet backbone, 74.90% faster than LibFaceDetection) + SFace (recognition)
- **Smart Lid Detection**: Skips biometric auth when lid closed, saves 5-second timeout
- **Shoulder Surfing Detection**: Detects multiple faces and blanks screen if someone looks over your shoulder
- **Live Camera Preview**: `faceid show` command displays real-time face detection visualization with SDL2 hardware acceleration
- **Single Binary**: C++20, no Python dependencies, low memory footprint (~50-100 MB)
- **D-Bus Fingerprint**: Integrates with fprintd, uses existing enrollments
- **Performance**: ~15ms face detection (RetinaFace), ~33ms recognition (SFace)
- **Comprehensive Logging**: Audit trail at `/var/log/faceid.log`
- **PAM Compatible**: Works with sudo, login, lock screen, GDM, LightDM
- **Zero OpenCV Dependencies**: 83% smaller binaries (3.3MB vs 20MB), specialized libraries for each task

## Quick Start

### Dependencies

**Arch**: `sudo pacman -S base-devel meson ninja ncnn pam libjpeg-turbo sdl2 libyuv glib2 fprintd systemd`  
**Debian/Ubuntu**: `sudo apt install build-essential meson ninja-build libncnn-dev libpam0g-dev libturbojpeg0-dev libsdl2-dev libyuv-dev libglib2.0-dev fprintd libsystemd-dev pkg-config`  
**Fedora**: `sudo dnf install gcc-c++ meson ninja-build ncnn-devel pam-devel turbojpeg-devel SDL2-devel libyuv-devel glib2-devel fprintd systemd-devel pkgconfig`

### Build

```bash
make build && sudo make install
```

### Download Models

**RetinaFace (Detection)**:
```bash
cd /etc/faceid/models
sudo wget https://github.com/nihui/ncnn-assets/raw/master/models/mnet.25-opt.{param,bin}
```

**SFace (Recognition)** - convert from ONNX:
```bash
pip3 install pnnx
cd /etc/faceid/models
wget https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
pnnx face_recognition_sface_2021dec.onnx
sudo mv face_recognition_sface_2021dec.ncnn.{param,bin} sface.{param,bin}
```

### Enroll & Configure

```bash
# Enroll face
sudo faceid add $(whoami)

# Enroll fingerprint (optional)
fprintd-enroll

# Add to /etc/pam.d/sudo
auth       sufficient   pam_faceid.so

# Test
sudo ls
```

## Commands

```bash
faceid devices           # List cameras
faceid show              # Live preview
sudo faceid add <user>   # Enroll face
sudo faceid test <user>  # Test with timing
sudo faceid list         # List users
```

## Troubleshooting

**Camera not detected**: `ls -l /dev/video* && sudo usermod -aG video $USER`  
**Face not recognized**: Good lighting, re-enroll, or adjust threshold in config  
**Model loading failed**: Check `/etc/faceid/models/` for `mnet.25-opt.{param,bin}` and `sface.{param,bin}`  
**Fingerprint issues**: `systemctl status fprintd && fprintd-verify`  
**Locked out**: Boot recovery, mount root, edit `/mnt/etc/pam.d/sudo`, remove pam_faceid.so line

## Configuration

Edit `/etc/faceid/faceid.conf`:
- `[recognition] threshold = 0.6` - Lower = stricter (default: 0.6)
- `[authentication] check_lid_state = true` - Skip auth when lid closed
- `[no_peek]` - Screen blanking when multiple faces detected
- `[logging] log_level = INFO` - View logs: `tail -f /var/log/faceid.log`

## FAQ

**Enrollment**: Face and/or fingerprint (any combo works, both = faster)  
**Speed**: Face ~48ms, fingerprint ~200-400ms (parallel, first wins)  
**Models**: NCNN format only (RetinaFace from [ncnn-assets](https://github.com/nihui/ncnn-assets), SFace convert with PNNX)  
**SSH**: Falls back to password (no camera)  
**Unenroll**: `sudo faceid remove <user>` or `fprintd-delete <user>`

## Technical Details

**Pipeline**: Lid check → parallel (RetinaFace+SFace | fprintd) → first success wins  
**Detection**: RetinaFace MobileNet 0.25 (~15ms, multi-scale anchors, NCNN 4 threads)  
**Recognition**: SFace MobileFaceNet (~33ms, 512D encodings)  
**Performance**: ~48ms total face auth, ~50-100MB memory  
**Storage**: Face models in `/etc/faceid/models/<user>.bin`, logs in `/var/log/faceid.log`

**Zero OpenCV Stack**:
- Camera: V4L2 + TurboJPEG (2x faster than cv::VideoCapture)
- Image ops: libyuv (3-5x faster than cv::resize)
- CLAHE: Custom (6.6x faster than cv::CLAHE)
- Display: SDL2 (hardware accelerated)
- Result: 83% smaller binary (3.3MB vs 20MB)

**Dependencies**: NCNN (inference), TurboJPEG (MJPEG), libyuv (SIMD ops), SDL2 (display), GLib (D-Bus), libsystemd (session management), PAM

---

Inspired by [Howdy](https://github.com/boltgolt/howdy) + [linux-enable-ir-emitter](https://github.com/EmixamPP/linux-enable-ir-emitter) | MIT License
