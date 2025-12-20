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
# Only link the OpenCV modules we actually use:
# - core: Basic data structures (Mat, etc.)
# - imgproc: Image processing (resize, cvtColor, etc.)
# Note: Removed videoio - we now use raw V4L2 for camera capture (49% faster!)
# Note: We use NCNN for face recognition (no longer need opencv_dnn)
# Note: We use LibFaceDetection for detection (no need for objdetect, features2d, calib3d)
opencv_modules = ['opencv_core', 'opencv_imgproc', 'opencv_imgcodecs']
opencv_dep = dependency('opencv4', required: true, modules: opencv_modules)

# NCNN for face recognition (SFace model)
ncnn_dep = dependency('ncnn', required: true)

# TurboJPEG for fast MJPEG decompression (used by V4L2 camera backend)
turbojpeg_dep = dependency('libturbojpeg', required: true)

pam_dep = dependency('pam', required: false)
if not pam_dep.found()
    pam_dep = declare_dependency(
        link_args: ['-lpam'],
    )
endif

# Configuration variables
config_dir = get_option('sysconfdir') / 'faceid'
models_dir = config_dir / 'models'
log_dir = get_option('localstatedir') / 'log' / 'faceid'

conf_data = configuration_data()
conf_data.set('CONFIG_DIR', config_dir)
conf_data.set('MODELS_DIR', models_dir)
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
