# Compiler flags to suppress external library warnings
cpp_args = [
    '-Wno-overloaded-virtual',  # Suppress OpenCV warnings
    '-Wno-unused-parameter',    # Suppress unused PAM parameter warnings
]

# Detect CPU architecture for SIMD optimization
cpu_arch = host_machine.cpu_family()

# LibFaceDetection SIMD flags based on CPU
libface_cpp_args = cpp_args
if cpu_arch == 'x86_64'
    # Check for AVX512 support (best performance)
    if run_command('grep', '-q', 'avx512', '/proc/cpuinfo', check: false).returncode() == 0
        libface_cpp_args += ['-D_ENABLE_AVX512', '-mavx512f', '-mavx512dq', '-mavx512bw', '-mavx512vl']
        message('LibFaceDetection: Enabling AVX512 SIMD optimization')
    # Fallback to AVX2 if AVX512 not available
    elif run_command('grep', '-q', 'avx2', '/proc/cpuinfo', check: false).returncode() == 0
        libface_cpp_args += ['-D_ENABLE_AVX2', '-mavx2']
        message('LibFaceDetection: Enabling AVX2 SIMD optimization')
    else
        message('LibFaceDetection: No SIMD optimization (SSE only)')
    endif
elif cpu_arch in ['aarch64', 'arm']
    libface_cpp_args += ['-D_ENABLE_NEON', '-mfpu=neon']
    message('LibFaceDetection: Enabling NEON SIMD optimization')
else
    message('LibFaceDetection: No SIMD optimization for architecture: ' + cpu_arch)
endif

# Dependencies
gio_dep = dependency('gio-2.0', required: true)
glib_dep = dependency('glib-2.0', required: true)

# LibFaceDetection sources
libfacedetection_sources = files(
    'libfacedetection/facedetectcnn.cpp',
    'libfacedetection/facedetectcnn-data.cpp',
    'libfacedetection/facedetectcnn-model.cpp',
)

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
    sources: core_sources + libfacedetection_sources,
    include_directories: inc,
    dependencies: [opencv_dep, gio_dep, glib_dep],
    cpp_args: libface_cpp_args,
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
    cpp_args: libface_cpp_args,
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
    cpp_args: libface_cpp_args,
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
    cpp_args: libface_cpp_args,
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
