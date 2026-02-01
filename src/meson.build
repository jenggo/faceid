# Compiler flags to suppress external library warnings
cpp_args = [
    '-Wno-unused-parameter',    # Suppress unused PAM parameter warnings
]

# Dependencies
gio_dep = dependency('gio-2.0', required: true)
glib_dep = dependency('glib-2.0', required: true)
sdl2_dep = dependency('sdl2', required: true)
libevdev_dep = dependency('libevdev', required: true)  # NEW: For input monitoring
yaml_dep = dependency('yaml-cpp', required: true)      # NEW: For YAML config parsing

# Core library
core_sources = files(
    'config.cpp',
    'camera.cpp',
    'face_detector.cpp',
    'clahe.cpp',
    'logger.cpp',
    'fingerprint_auth.cpp',
    'lid_detector.cpp',
    'display_detector.cpp',
    'systemd_helper.cpp',
    'adaptive_auth.cpp',
    'presence/guard.cpp',
    'presence/detector.cpp',
    'presence/input_monitor.cpp',      # NEW: Event-driven input monitoring
    'presence/peek_detector.cpp',       # NEW: Gaze-aware peek detection
    'presence/notification.cpp',        # NEW: Desktop notifications
    'models/binary_model.cpp',
    'models/model_cache.cpp',
    'detectors/retinaface.cpp',
    'detectors/yunet.cpp',
    'detectors/yunet_model_data.cpp',
    'detectors/retinaface_model_data.cpp',
)

core_lib = static_library(
    'faceid-core',
    sources: core_sources,
    include_directories: inc,
    dependencies: [ncnn_dep, turbojpeg_dep, libyuv_dep, gio_dep, glib_dep, libsystemd_dep, libevdev_dep],  # Added libevdev_dep
    cpp_args: cpp_args,
    link_args: ['-lpthread', '-lrt'],  # pthread for mutex, rt for shm_open
)

# PAM module
pam_sources = files(
    'pam/pam_faceid.cpp',
    'pam/dbus_client.cpp',
    'pam/context_detector.cpp'
)

pam_module = shared_module(
    'pam_faceid',
    sources: pam_sources,
    include_directories: inc,
    dependencies: [pam_dep, libsystemd_dep, gio_dep, glib_dep, libevdev_dep],
    link_with: core_lib,
    cpp_args: cpp_args,
    link_args: ['-ldl', '-lpthread'],  # For threading support
    name_prefix: '',
    install: true,
    install_dir: get_option('libdir') / 'security',
)

# CLI tool
cli_sources = files(
    'cli/main.cpp',
    'cli/cmd_devices.cpp',
    'cli/cmd_add.cpp',
    'cli/cmd_remove.cpp',
    'cli/cmd_list.cpp',
    'cli/cmd_show.cpp',
    'cli/cmd_test.cpp',
    'cli/cmd_test_image.cpp',
    'cli/cmd_bench.cpp',
    'cli/cmd_use.cpp',
    'cli/cmd_help.cpp',
    'display.cpp'
)

cli_exe = executable(
    'faceid',
    sources: cli_sources,
    include_directories: inc,
    dependencies: [ncnn_dep, turbojpeg_dep, libyuv_dep, sdl2_dep],
    link_with: core_lib,
    cpp_args: cpp_args,
    install: true,
)

# Presence detection daemon
presence_daemon_sources = files('presence/daemon.cpp')

presence_daemon_exe = executable(
    'faceid-presence',
    sources: presence_daemon_sources,
    include_directories: inc,
    dependencies: [ncnn_dep, turbojpeg_dep, libyuv_dep, libevdev_dep],  # Added libevdev_dep
    link_with: core_lib,
    cpp_args: cpp_args,
    link_args: ['-lpthread'],
    install: true,
    install_dir: get_option('bindir'),
)

# Config merge utility
config_merge_exe = executable(
    'faceid-config-merge',
    sources: files('config_merge.cpp'),
    include_directories: inc,
    cpp_args: cpp_args,
    install: true,
    install_dir: get_option('bindir'),
)

# FaceID Daemon (user-level D-Bus service)
daemon_sources = files(
    'daemon/main.cpp',
    'daemon/dbus_server.cpp',
    'daemon/validation.cpp',
    'daemon/config.cpp',
    'daemon/auth_manager.cpp',
    'daemon/camera_manager.cpp',
    'daemon/model_store.cpp',
    'daemon/face_detector_manager.cpp',
    'daemon/fingerprint_manager.cpp',
    'daemon/presence_monitor.cpp',
    'daemon/enroll_manager.cpp',
)

daemon_exe = executable(
    'faceid-daemon',
    sources: daemon_sources,
    include_directories: inc,
    dependencies: [libsystemd_dep, yaml_dep, gio_dep, glib_dep, ncnn_dep, turbojpeg_dep, libyuv_dep],
    link_with: core_lib,
    cpp_args: cpp_args,
    link_args: ['-lpthread'],
    install: true,
    install_dir: get_option('bindir'),
)
