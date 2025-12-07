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
opencv_dep = dependency('opencv4', required: true)
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

# Install config with preservation logic via custom script
# Note: Meson doesn't support conditional install, so we use install_data
# and handle preservation in the Makefile
install_data(
    'config/faceid.conf',
    install_dir: config_dir,
    install_mode: 'rw-r--r--',
)

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
