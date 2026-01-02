project(
    'faceid',
    'cpp',
    version: '1.1.0',
    license: 'MIT',
    default_options: [
        'cpp_std=c++20',
        'buildtype=release',
        'warning_level=3',
        'prefix=/usr',
        'sysconfdir=/etc',
        'localstatedir=/var',
    ],
)

# Dependencies
# OpenCV completely eliminated! Replaced with:
# - Camera capture: V4L2 + TurboJPEG (49% faster than cv::VideoCapture)
# - Image processing: libyuv (3-5x faster than cv::resize, cv::cvtColor)
# - CLAHE: Custom implementation (6x faster than cv::CLAHE)
# - Optical flow: Custom Lucas-Kanade pyramid (3.9x faster than cv::calcOpticalFlowPyrLK)
# - Display: SDL2 (hardware accelerated, replaces cv::imshow)
# - Face detection: LibFaceDetection (embedded)
# - Face recognition: NCNN (SFace model)

# NCNN for face recognition (SFace model)
ncnn_dep = dependency('ncnn', required: true)

# TurboJPEG for fast MJPEG decompression (used by V4L2 camera backend)
turbojpeg_dep = dependency('libturbojpeg', required: true)

# libyuv for fast image resizing (3-5x faster than OpenCV resize)
# Declare as system dependency since it doesn't have pkg-config
libyuv_dep = declare_dependency(
    link_args: ['-lyuv']
)

pam_dep = dependency('pam', required: false)
if not pam_dep.found()
    pam_dep = declare_dependency(
        link_args: ['-lpam'],
    )
endif

# libsystemd for D-Bus communication with systemd-logind
# Requires version >= 232 for sd_bus_* API support
libsystemd_dep = dependency('libsystemd', required: true, version: '>=232')

# Configuration variables
config_dir = get_option('sysconfdir') / 'faceid'
models_dir = config_dir / 'models'
faces_dir = config_dir / 'faces'
log_dir = get_option('localstatedir') / 'log' / 'faceid'

conf_data = configuration_data()
conf_data.set('CONFIG_DIR', config_dir)
conf_data.set('MODELS_DIR', models_dir)
conf_data.set('FACES_DIR', faces_dir)
conf_data.set('LOG_DIR', log_dir)
conf_data.set('VERSION', meson.project_version())

configure_file(
    input: 'src/config_paths.h.in',
    output: 'config_paths.h',
    configuration: conf_data,
)

# Include directories
inc = include_directories('src', 'build')

# Subdirectories
subdir('src')

# Install empty directories
install_emptydir(config_dir)
install_emptydir(models_dir)
install_emptydir(faces_dir)
install_emptydir(log_dir)

# Config installation handled by Makefile using faceid-config-merge utility
# This allows preserving user settings during upgrades
# The Makefile runs: faceid-config-merge config/faceid.conf /etc/faceid/faceid.conf

# Install documentation
install_data(
    'README.md',
    install_dir: get_option('datadir') / 'doc' / 'faceid',
)

# Install systemd service
install_data(
    'systemd/faceid-presence.service',
    install_dir: get_option('prefix') / 'lib' / 'systemd' / 'system',
)

# Install systemd tmpfiles.d configuration
# Creates /run/faceid directory with proper permissions for lock files
install_data(
    'systemd/faceid-tmpfiles.conf',
    install_dir: get_option('prefix') / 'lib' / 'tmpfiles.d',
)
