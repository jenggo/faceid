# Compiler flags to suppress external library warnings
cpp_args = [
    '-Wno-overloaded-virtual',  # Suppress OpenCV warnings
    '-Wno-unused-parameter',    # Suppress unused PAM parameter warnings
]

# Dependencies
gio_dep = dependency('gio-2.0', required: true)
glib_dep = dependency('glib-2.0', required: true)

# Core library
core_sources = files(
    'config.cpp',
    'camera.cpp',
    'face_detector.cpp',
    'logger.cpp',
    'fingerprint_auth.cpp',
    'lid_detector.cpp',
    'display_detector.cpp',
    'presence/presence_guard.cpp',
    'presence/presence_detector.cpp',
)

core_lib = static_library(
    'faceid-core',
    sources: core_sources,
    include_directories: inc,
    dependencies: [opencv_dep, gio_dep, glib_dep],
    cpp_args: cpp_args,
)

# Dependencies
jsoncpp_dep = dependency('jsoncpp', required: true)

# PAM module
pam_sources = files('pam/pam_faceid.cpp')

pam_module = shared_module(
    'pam_faceid',
    sources: pam_sources,
    include_directories: inc,
    dependencies: [pam_dep, opencv_dep, jsoncpp_dep],
    link_with: core_lib,
    cpp_args: cpp_args,
    link_args: ['-ldl', '-lpthread'],  # For threading support
    name_prefix: '',
    install: true,
    install_dir: get_option('libdir') / 'security',
)

# CLI tool
cli_sources = files('cli/main.cpp')

cli_exe = executable(
    'faceid',
    sources: cli_sources,
    include_directories: inc,
    dependencies: [opencv_dep, jsoncpp_dep],
    link_with: core_lib,
    cpp_args: cpp_args,
    install: true,
)

# Presence detection daemon
presence_daemon_sources = files('presence/presence_daemon.cpp')

presence_daemon_exe = executable(
    'faceid-presence',
    sources: presence_daemon_sources,
    include_directories: inc,
    dependencies: [opencv_dep],
    link_with: core_lib,
    cpp_args: cpp_args,
    link_args: ['-lpthread'],
    install: true,
    install_dir: get_option('bindir'),
)
