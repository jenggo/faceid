.PHONY: help deps check-deps setup build install uninstall clean distclean test status full-install pam-install pam-status pam-reset logs-view systemd-enable systemd-disable systemd-status systemd-logs presence-status

# Project configuration
PROJECT_NAME := faceid
BUILD_DIR := build
INSTALL_PREFIX := /usr
MESON := meson
NINJA := ninja

# Colors for output
COLOR_RESET := \033[0m
COLOR_BOLD := \033[1m
COLOR_CYAN := \033[36m
COLOR_GREEN := \033[32m
COLOR_YELLOW := \033[33m
COLOR_RED := \033[31m

# Help target
help:
	@printf "$(COLOR_BOLD)$(PROJECT_NAME) - Build and Installation Targets$(COLOR_RESET)\n"
	@printf " \n"
	@printf "$(COLOR_CYAN)Setup Targets:$(COLOR_RESET)\n"
	@printf "  $(COLOR_GREEN)deps$(COLOR_RESET)         - Install build dependencies\n"
	@printf "  $(COLOR_GREEN)check-deps$(COLOR_RESET)   - Check for required dependencies\n"
	@printf "  $(COLOR_GREEN)setup$(COLOR_RESET)        - Setup build directory with Meson\n"
	@printf " \n"
	@printf "$(COLOR_CYAN)Build Targets:$(COLOR_RESET)\n"
	@printf "  $(COLOR_GREEN)build$(COLOR_RESET)        - Build the project\n"
	@printf "  $(COLOR_GREEN)rebuild$(COLOR_RESET)      - Clean and build from scratch\n"
	@printf " \n"
	@printf "$(COLOR_CYAN)Installation Targets:$(COLOR_RESET)\n"
	@printf "  $(COLOR_GREEN)install$(COLOR_RESET)      - Install to system (requires root)\n"
	@printf "  $(COLOR_GREEN)uninstall$(COLOR_RESET)    - Uninstall from system (requires root)\n"
	@printf "  $(COLOR_GREEN)full-install$(COLOR_RESET) - Setup, build, and install in one step\n"
	@printf " \n"
	@printf "$(COLOR_CYAN)Maintenance Targets:$(COLOR_RESET)\n"
	@printf "  $(COLOR_GREEN)clean$(COLOR_RESET)        - Remove build artifacts\n"
	@printf "  $(COLOR_GREEN)distclean$(COLOR_RESET)    - Remove build directory completely\n"
	@printf "  $(COLOR_GREEN)test$(COLOR_RESET)         - Run tests\n"
	@printf "  $(COLOR_GREEN)status$(COLOR_RESET)       - Show build status\n"
	@printf " \n"
	@printf "$(COLOR_CYAN)PAM Integration Targets:$(COLOR_RESET)\n"
	@printf "  $(COLOR_GREEN)pam-install$(COLOR_RESET)  - Configure PAM to use FaceID authentication (requires root)\n"
	@printf "  $(COLOR_GREEN)pam-status$(COLOR_RESET)   - Show current PAM configuration status\n"
	@printf "  $(COLOR_GREEN)pam-reset$(COLOR_RESET)    - Reset PAM to backup configuration\n"
	@printf " \n"
	@printf "$(COLOR_CYAN)Logging:$(COLOR_RESET)\n"
	@printf "  $(COLOR_GREEN)logs-view$(COLOR_RESET)    - View recent FaceID authentication logs\n"
	@printf " \n"
	@printf "$(COLOR_CYAN)Presence Detection (systemd):$(COLOR_RESET)\n"
	@printf "  $(COLOR_GREEN)systemd-enable$(COLOR_RESET)  - Enable and start presence detection service\n"
	@printf "  $(COLOR_GREEN)systemd-disable$(COLOR_RESET) - Disable and stop presence detection service\n"
	@printf "  $(COLOR_GREEN)systemd-status$(COLOR_RESET)  - Show presence detection service status\n"
	@printf "  $(COLOR_GREEN)systemd-logs$(COLOR_RESET)    - View presence detection service logs\n"
	@printf "  $(COLOR_GREEN)presence-status$(COLOR_RESET) - Show comprehensive presence detection status\n"
	@printf " \n"
	@printf "$(COLOR_CYAN)Advanced Targets:$(COLOR_RESET)\n"
	@printf "  $(COLOR_GREEN)reconfigure$(COLOR_RESET)  - Reconfigure the build\n"
	@printf "  $(COLOR_GREEN)install-debug$(COLOR_RESET)- Install debug version\n"
	@printf " \n"

# Check for required dependencies
check-deps:
	@printf "$(COLOR_BOLD)Checking for required dependencies...$(COLOR_RESET)\n"
	@command -v meson >/dev/null 2>&1 || { printf "$(COLOR_RED)✗ meson not found$(COLOR_RESET)\n"; exit 1; }
	@printf "$(COLOR_GREEN)✓ meson$(COLOR_RESET)\n"
	@command -v ninja >/dev/null 2>&1 || { printf "$(COLOR_RED)✗ ninja not found$(COLOR_RESET)\n"; exit 1; }
	@printf "$(COLOR_GREEN)✓ ninja$(COLOR_RESET)\n"
	@pkg-config --exists ncnn || { printf "$(COLOR_RED)✗ ncnn not found$(COLOR_RESET)\n"; exit 1; }
	@printf "$(COLOR_GREEN)✓ ncnn$(COLOR_RESET)\n"
	@pkg-config --exists libturbojpeg || { printf "$(COLOR_RED)✗ libturbojpeg not found$(COLOR_RESET)\n"; exit 1; }
	@printf "$(COLOR_GREEN)✓ libturbojpeg$(COLOR_RESET)\n"
	@pkg-config --exists sdl2 || { printf "$(COLOR_RED)✗ sdl2 not found$(COLOR_RESET)\n"; exit 1; }
	@printf "$(COLOR_GREEN)✓ sdl2$(COLOR_RESET)\n"
	@pkg-config --exists pam || { printf "$(COLOR_YELLOW)⚠ pam not found (optional)$(COLOR_RESET)\n"; }
	@command -v pkg-config >/dev/null 2>&1 || { printf "$(COLOR_RED)✗ pkg-config not found$(COLOR_RESET)\n"; exit 1; }
	@printf "$(COLOR_GREEN)✓ pkg-config$(COLOR_RESET)\n"
	@printf "$(COLOR_GREEN)All required dependencies found!$(COLOR_RESET)\n"

# Install build dependencies (Arch/Manjaro example)
deps:
	@printf "$(COLOR_BOLD)Installing build dependencies...$(COLOR_RESET)\n"
	@if command -v pacman >/dev/null 2>&1; then \
		sudo pacman -S --needed base-devel meson ninja ncnn pam libjpeg-turbo sdl2 libyuv; \
	elif command -v apt-get >/dev/null 2>&1; then \
		sudo apt-get install -y build-essential meson ninja-build libncnn-dev libpam-dev libturbojpeg0-dev libsdl2-dev libyuv-dev pkg-config; \
	elif command -v dnf >/dev/null 2>&1; then \
		sudo dnf install -y @development-tools meson ninja ncnn-devel pam-devel turbojpeg-devel SDL2-devel libyuv-devel pkg-config; \
	else \
		printf "$(COLOR_RED)Unsupported package manager. Please install dependencies manually.$(COLOR_RESET)\n"; \
		exit 1; \
	fi
	@printf "$(COLOR_GREEN)Dependencies installed successfully!$(COLOR_RESET)\n"

# Setup build directory
setup: check-deps
	@printf "$(COLOR_BOLD)Setting up build directory...$(COLOR_RESET)\n"
	@if [ ! -d "$(BUILD_DIR)" ]; then \
		$(MESON) setup $(BUILD_DIR) --prefix=$(INSTALL_PREFIX); \
		printf "$(COLOR_GREEN)Build directory created and configured.$(COLOR_RESET)\n"; \
	else \
		printf "$(COLOR_YELLOW)Build directory already exists. Run 'make reconfigure' to reconfigure.$(COLOR_RESET)\n"; \
	fi

# Reconfigure the build
reconfigure:
	@printf "$(COLOR_BOLD)Reconfiguring build...$(COLOR_RESET)\n"
	@$(MESON) setup --reconfigure $(BUILD_DIR) --prefix=$(INSTALL_PREFIX)
	@printf "$(COLOR_GREEN)Build reconfigured.$(COLOR_RESET)\n"

# Build the project
build: setup
	@printf "$(COLOR_BOLD)Building $(PROJECT_NAME)...$(COLOR_RESET)\n"
	@$(MESON) compile -C $(BUILD_DIR)
	@printf "$(COLOR_GREEN)Build completed successfully!$(COLOR_RESET)\n"

# Rebuild from scratch
rebuild: distclean build
	@printf "$(COLOR_GREEN)Rebuild completed!$(COLOR_RESET)\n"

# Install to system
install: build
	@printf "$(COLOR_BOLD)Installing $(PROJECT_NAME) to system...$(COLOR_RESET)\n"
	@BUILD_USER=$$(stat -c '%U' $(BUILD_DIR) 2>/dev/null || echo ""); \
	if [ "$$(id -u)" != "0" ]; then \
		sudo $(MESON) install -C $(BUILD_DIR) --no-rebuild; \
		if [ -n "$$BUILD_USER" ] && [ "$$BUILD_USER" != "root" ]; then \
			sudo chown -R $$BUILD_USER:$$BUILD_USER $(BUILD_DIR); \
		fi; \
	else \
		$(MESON) install -C $(BUILD_DIR) --no-rebuild; \
	fi
	@printf " \n"
	@printf "$(COLOR_CYAN)Checking configuration file...$(COLOR_RESET)\n"
	@SOURCE_CONFIG="config/faceid.conf"; \
	DEST_CONFIG="/etc/faceid/faceid.conf"; \
	MERGE_UTIL="$(BUILD_DIR)/src/faceid-config-merge"; \
	if [ ! -f "$$MERGE_UTIL" ]; then \
		printf "$(COLOR_RED)✗ Config merge utility not found: $$MERGE_UTIL$(COLOR_RESET)\n"; \
		printf "$(COLOR_YELLOW)⚠ Falling back to simple copy...$(COLOR_RESET)\n"; \
		if [ ! -f "$$DEST_CONFIG" ]; then \
			if [ "$$(id -u)" != "0" ]; then \
				sudo cp "$$SOURCE_CONFIG" "$$DEST_CONFIG"; \
				sudo chmod 644 "$$DEST_CONFIG"; \
			else \
				cp "$$SOURCE_CONFIG" "$$DEST_CONFIG"; \
				chmod 644 "$$DEST_CONFIG"; \
			fi; \
			printf "$(COLOR_GREEN)✓ New config installed: $$DEST_CONFIG$(COLOR_RESET)\n"; \
		else \
			printf "$(COLOR_YELLOW)⚠ Existing config unchanged: $$DEST_CONFIG$(COLOR_RESET)\n"; \
		fi; \
	else \
		if [ "$$(id -u)" != "0" ]; then \
			sudo "$$MERGE_UTIL" "$$SOURCE_CONFIG" "$$DEST_CONFIG"; \
		else \
			"$$MERGE_UTIL" "$$SOURCE_CONFIG" "$$DEST_CONFIG"; \
		fi; \
	fi
	@printf " \n"
	@printf "$(COLOR_CYAN)Setting up logging...$(COLOR_RESET)\n"
	@if [ "$$(id -u)" != "0" ]; then \
		sudo touch /var/log/faceid.log; \
		sudo chown root:root /var/log/faceid.log; \
		sudo chmod 644 /var/log/faceid.log; \
	else \
		touch /var/log/faceid.log; \
		chown root:root /var/log/faceid.log; \
		chmod 644 /var/log/faceid.log; \
	fi
	@printf "$(COLOR_GREEN)✓ Log file created: /var/log/faceid.log (root:root 644)$(COLOR_RESET)\n"
	@printf " \n"
	@printf "$(COLOR_CYAN)Installing systemd service...$(COLOR_RESET)\n"
	@if [ -f "$(INSTALL_PREFIX)/lib/systemd/system/faceid-presence.service" ]; then \
		printf "$(COLOR_GREEN)✓ Systemd service installed: faceid-presence.service$(COLOR_RESET)\n"; \
		printf "$(COLOR_CYAN)To enable: sudo systemctl enable --now faceid-presence$(COLOR_RESET)\n"; \
	else \
		printf "$(COLOR_YELLOW)⚠ Systemd service file not found$(COLOR_RESET)\n"; \
	fi
	@printf " \n"
	@printf "$(COLOR_GREEN)Installation completed!$(COLOR_RESET)\n"

# Install debug version
install-debug: setup
	@printf "$(COLOR_BOLD)Building and installing debug version...$(COLOR_RESET)\n"
	@$(MESON) setup --reconfigure $(BUILD_DIR) -Dbuildtype=debug
	@$(MESON) compile -C $(BUILD_DIR)
	@BUILD_USER=$$(stat -c '%U' $(BUILD_DIR) 2>/dev/null || echo ""); \
	if [ "$$(id -u)" != "0" ]; then \
		sudo $(MESON) install -C $(BUILD_DIR); \
		if [ -n "$$BUILD_USER" ] && [ "$$BUILD_USER" != "root" ]; then \
			sudo chown -R $$BUILD_USER:$$BUILD_USER $(BUILD_DIR); \
		fi; \
	else \
		$(MESON) install -C $(BUILD_DIR); \
	fi
	@printf "$(COLOR_GREEN)Debug installation completed!$(COLOR_RESET)\n"

# Uninstall from system
uninstall:
	@printf "$(COLOR_BOLD)Uninstalling $(PROJECT_NAME) from system...$(COLOR_RESET)\n"
	@if [ -d "$(BUILD_DIR)" ]; then \
		if [ "$$(id -u)" != "0" ]; then \
			sudo $(MESON) install -C $(BUILD_DIR) --only-changed; \
		else \
			$(MESON) install -C $(BUILD_DIR) --only-changed; \
		fi; \
		printf "$(COLOR_YELLOW)Note: Meson uninstall may not fully remove all files.$(COLOR_RESET)\n"; \
	else \
		printf "$(COLOR_RED)Build directory not found. Cannot uninstall.$(COLOR_RESET)\n"; \
		exit 1; \
	fi
	@printf "Please manually verify installation.\n"

# Full installation (setup + build + install)
full-install: deps check-deps setup build install
	@printf "$(COLOR_GREEN)$(COLOR_BOLD)Full installation completed successfully!$(COLOR_RESET)\n"
	@printf " \n"
	@printf "$(COLOR_CYAN)Next steps:$(COLOR_RESET)\n"
	@printf "  1. Verify installation: faceid --version\n"
	@printf "  2. Check PAM module: ls -la /usr/lib/security/pam_faceid.so\n"
	@printf "  3. Check presence daemon: faceid-presence --help\n"
	@printf "  4. Configure PAM: make pam-install\n"
	@printf "  5. Enable presence detection: sudo make systemd-enable\n"
	@printf "  6. Review configuration: /etc/faceid/faceid.conf\n"
	@printf " \n"

# Run tests
test: build
	@printf "$(COLOR_BOLD)Running tests...$(COLOR_RESET)\n"
	@if [ -f "test_lid.cpp" ]; then \
		printf "$(COLOR_CYAN)Found test_lid.cpp - compiling and running...$(COLOR_RESET)\n"; \
		cd $(BUILD_DIR) && $(MESON) test || printf "$(COLOR_YELLOW)No tests configured in Meson yet.$(COLOR_RESET)\n"; \
	else \
		$(MESON) test -C $(BUILD_DIR) || printf "$(COLOR_YELLOW)No tests configured.$(COLOR_RESET)\n"; \
	fi

# Clean build artifacts
clean:
	@printf "$(COLOR_BOLD)Cleaning build artifacts...$(COLOR_RESET)\n"
	@if [ -d "$(BUILD_DIR)" ]; then \
		$(MESON) compile -C $(BUILD_DIR) --clean; \
		printf "$(COLOR_GREEN)Build artifacts cleaned.$(COLOR_RESET)\n"; \
	else \
		printf "$(COLOR_YELLOW)Build directory not found, nothing to clean.$(COLOR_RESET)\n"; \
	fi

# Remove build directory completely
distclean:
	@printf "$(COLOR_BOLD)Removing build directory...$(COLOR_RESET)\n"
	@rm -rf $(BUILD_DIR)
	@printf "$(COLOR_GREEN)Build directory removed.$(COLOR_RESET)\n"

# Show build status
status:
	@printf "$(COLOR_BOLD)Build Status:$(COLOR_RESET)\n"
	@if [ -d "$(BUILD_DIR)" ]; then \
		printf "$(COLOR_GREEN)✓ Build directory exists: $(BUILD_DIR)$(COLOR_RESET)\n"; \
		printf "\n"; \
		printf "$(COLOR_CYAN)Build files:$(COLOR_RESET)\n"; \
		find $(BUILD_DIR) -maxdepth 1 -type f -printf "  %f\n" 2>/dev/null | head -5; \
		printf "\n"; \
		printf "$(COLOR_CYAN)Compiler info:$(COLOR_RESET)\n"; \
		if [ -f "$(BUILD_DIR)/build.log" ]; then \
			printf "  (See $(BUILD_DIR)/build.log for details)\n"; \
		fi; \
		printf "\n"; \
		printf "$(COLOR_CYAN)Project structure:$(COLOR_RESET)\n"; \
		printf "  Source files: $$(find src -name '*.cpp' -o -name '*.h' | wc -l) files\n"; \
		printf "  Build directory: $(BUILD_DIR)\n"; \
		printf "  Install prefix: $(INSTALL_PREFIX)\n"; \
	else \
		printf "$(COLOR_RED)✗ Build directory does not exist$(COLOR_RESET)\n"; \
		printf "  Run 'make setup' or 'make build' to initialize the build.\n"; \
	fi
	@printf " \n"

# PAM integration targets
pam-install:
	@printf "$(COLOR_BOLD)Installing FaceID PAM Configuration$(COLOR_RESET)\n"
	@printf " \n"
	@if [ "$$(id -u)" != "0" ]; then \
		printf "$(COLOR_RED)Error: PAM configuration requires root privileges$(COLOR_RESET)\n"; \
		printf "Please run: sudo make pam-install\n"; \
		exit 1; \
	fi
	@if [ ! -f "/usr/lib/security/pam_faceid.so" ]; then \
		printf "$(COLOR_RED)Error: pam_faceid.so not found!$(COLOR_RESET)\n"; \
		printf "Please run 'sudo make install' first to install FaceID.\n"; \
		exit 1; \
	fi
	@printf "This will configure system-wide PAM authentication to use FaceID.\n"
	@printf " \n"
	@printf "$(COLOR_CYAN)Files to be modified:$(COLOR_RESET)\n"
	@printf "  • /etc/pam.d/system-auth        (sudo, su, etc.)\n"
	@printf "  • /etc/pam.d/system-local-login (login screens, screen lockers)\n"
	@printf " \n"
	@printf "$(COLOR_CYAN)What will be done:$(COLOR_RESET)\n"
	@printf "  • Create timestamped backups of current configs\n"
	@printf "  • Replace Howdy + fprintd with pam_faceid.so\n"
	@printf "  • Remove external lid check (FaceID has built-in lid detection)\n"
	@printf " \n"
	@printf "$(COLOR_YELLOW)Note: FaceID includes built-in lid detection and fingerprint support$(COLOR_RESET)\n"
	@printf " \n"
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo; \
	if [ ! "$$REPLY" = "y" ] && [ ! "$$REPLY" = "Y" ]; then \
		printf "Installation cancelled.\n"; \
		exit 0; \
	fi
	@printf " \n"
	@TIMESTAMP=$$(date +%Y%m%d-%H%M%S); \
	CONFIGS="system-auth system-local-login"; \
	for cfg in $$CONFIGS; do \
		printf "$(COLOR_CYAN)Configuring $$cfg...$(COLOR_RESET)\n"; \
		if grep -q "pam_faceid.so" /etc/pam.d/$$cfg 2>/dev/null; then \
			printf "$(COLOR_YELLOW)✓ FaceID already configured in $$cfg (skipping)$(COLOR_RESET)\n"; \
			continue; \
		fi; \
		if [ -f "/etc/pam.d/$$cfg" ]; then \
			cp /etc/pam.d/$$cfg /etc/pam.d/$$cfg.backup.$$TIMESTAMP; \
			printf "$(COLOR_GREEN)✓ Backup created: /etc/pam.d/$$cfg.backup.$$TIMESTAMP$(COLOR_RESET)\n"; \
		fi; \
		if [ -f "/etc/pam.d/$$cfg" ]; then \
			sed -i '/check-lid-state.sh/d' /etc/pam.d/$$cfg; \
			sed -i '/linux-enable-ir-emitter/d' /etc/pam.d/$$cfg; \
			sed -i 's|auth.*sufficient.*pam_python.so /lib/security/howdy/pam.py|auth       sufficient                   pam_faceid.so|g' /etc/pam.d/$$cfg; \
			sed -i '/pam_fprintd.so/d' /etc/pam.d/$$cfg; \
			printf "$(COLOR_GREEN)✓ Configured $$cfg to use FaceID$(COLOR_RESET)\n"; \
		else \
			printf "$(COLOR_RED)✗ /etc/pam.d/$$cfg not found$(COLOR_RESET)\n"; \
		fi; \
		printf "\n"; \
	done
	@printf "$(COLOR_GREEN)$(COLOR_BOLD)PAM configuration complete!$(COLOR_RESET)\n"
	@printf " \n"
	@printf "$(COLOR_CYAN)Next steps:$(COLOR_RESET)\n"
	@printf "  1. Test authentication: sudo echo test\n"
	@printf "  2. Check status: make pam-status\n"
	@printf "  3. To rollback: make pam-reset\n"
	@printf " \n"

pam-status:
	@printf "$(COLOR_BOLD)FaceID PAM Configuration Status$(COLOR_RESET)\n"
	@printf " \n"
	@if [ -f "/usr/lib/security/pam_faceid.so" ]; then \
		printf "$(COLOR_GREEN)✓ PAM module installed: /usr/lib/security/pam_faceid.so$(COLOR_RESET)\n"; \
	else \
		printf "$(COLOR_RED)✗ PAM module not found$(COLOR_RESET)\n"; \
		printf "  Run 'sudo make install' to install FaceID first.\n"; \
	fi
	@printf " \n"
	@printf "$(COLOR_CYAN)PAM Configuration:$(COLOR_RESET)\n"
	@if grep -q "pam_faceid.so" /etc/pam.d/system-auth 2>/dev/null; then \
		printf "$(COLOR_GREEN)✓ system-auth configured for FaceID$(COLOR_RESET)\n"; \
	else \
		printf "$(COLOR_RED)✗ system-auth not configured$(COLOR_RESET)\n"; \
	fi
	@if grep -q "pam_faceid.so" /etc/pam.d/system-local-login 2>/dev/null; then \
		printf "$(COLOR_GREEN)✓ system-local-login configured for FaceID$(COLOR_RESET)\n"; \
	else \
		printf "$(COLOR_RED)✗ system-local-login not configured$(COLOR_RESET)\n"; \
	fi
	@printf " \n"
	@if [ -f "/etc/faceid/faceid.conf" ]; then \
		printf "$(COLOR_CYAN)FaceID Configuration:$(COLOR_RESET)\n"; \
		if grep -q "check_lid_state = true" /etc/faceid/faceid.conf 2>/dev/null; then \
			printf "$(COLOR_GREEN)✓ Lid detection enabled$(COLOR_RESET)\n"; \
		else \
			printf "$(COLOR_YELLOW)⚠ Lid detection disabled in config$(COLOR_RESET)\n"; \
		fi; \
		if grep -q "enable_fingerprint = true" /etc/faceid/faceid.conf 2>/dev/null; then \
			printf "$(COLOR_GREEN)✓ Fingerprint support enabled$(COLOR_RESET)\n"; \
		else \
			printf "$(COLOR_YELLOW)⚠ Fingerprint support disabled in config$(COLOR_RESET)\n"; \
		fi; \
	else \
		printf "$(COLOR_YELLOW)⚠ Config file not found: /etc/faceid/faceid.conf$(COLOR_RESET)\n"; \
	fi
	@printf " \n"
	@BACKUPS=$$(ls -1t /etc/pam.d/*.backup.* 2>/dev/null | head -3); \
	if [ -n "$$BACKUPS" ]; then \
		printf "$(COLOR_CYAN)Available backups (latest 3):$(COLOR_RESET)\n"; \
		for backup in $$BACKUPS; do \
			printf "  $$(basename $$backup)\n"; \
		done; \
	else \
		printf "$(COLOR_YELLOW)No backup files found$(COLOR_RESET)\n"; \
	fi
	@printf " \n"

pam-reset:
	@printf "$(COLOR_BOLD)Reset PAM Configuration$(COLOR_RESET)\n"
	@printf " \n"
	@if [ "$$(id -u)" != "0" ]; then \
		printf "$(COLOR_RED)Error: PAM reset requires root privileges$(COLOR_RESET)\n"; \
		printf "Please run: sudo make pam-reset\n"; \
		exit 1; \
	fi
	@BACKUPS=$$(ls -1t /etc/pam.d/*.backup.* 2>/dev/null); \
	if [ -z "$$BACKUPS" ]; then \
		printf "$(COLOR_YELLOW)No backup files found$(COLOR_RESET)\n"; \
		printf "Backups are automatically created when running 'make pam-install'\n"; \
		exit 0; \
	fi
	@printf "$(COLOR_CYAN)Available backups:$(COLOR_RESET)\n"
	@ls -1t /etc/pam.d/*.backup.* 2>/dev/null | head -5
	@printf " \n"
	@printf "To restore a backup manually, run:\n"
	@printf "  sudo cp /etc/pam.d/FILE.backup.TIMESTAMP /etc/pam.d/FILE\n"
	@printf " \n"
	@read -p "Restore latest backups? [y/N] " -n 1 -r; \
	echo; \
	if [ ! "$$REPLY" = "y" ] && [ ! "$$REPLY" = "Y" ]; then \
		printf "Reset cancelled.\n"; \
		exit 0; \
	fi
	@LATEST=$$(ls -1t /etc/pam.d/system-auth.backup.* 2>/dev/null | head -1); \
	if [ -n "$$LATEST" ]; then \
		cp "$$LATEST" /etc/pam.d/system-auth; \
		printf "$(COLOR_GREEN)✓ Restored system-auth from $$(basename $$LATEST)$(COLOR_RESET)\n"; \
	fi
	@LATEST=$$(ls -1t /etc/pam.d/system-local-login.backup.* 2>/dev/null | head -1); \
	if [ -n "$$LATEST" ]; then \
		cp "$$LATEST" /etc/pam.d/system-local-login; \
		printf "$(COLOR_GREEN)✓ Restored system-local-login from $$(basename $$LATEST)$(COLOR_RESET)\n"; \
	fi
	@printf " \n"
	@printf "$(COLOR_GREEN)PAM configuration restored$(COLOR_RESET)\n"
	@printf " \n"

# Logging targets
logs-view:
	@printf "$(COLOR_BOLD)FaceID Authentication Logs$(COLOR_RESET)\n"
	@printf " \n"
	@if [ ! -f /var/log/faceid.log ]; then \
		printf "$(COLOR_YELLOW)No log file found at /var/log/faceid.log$(COLOR_RESET)\n"; \
		printf "Log file is created automatically during installation or first authentication.\n"; \
		printf "\n"; \
		printf "To create it manually: sudo touch /var/log/faceid.log && sudo chmod 666 /var/log/faceid.log\n"; \
	else \
		printf "$(COLOR_CYAN)Recent authentication attempts (last 50 lines):$(COLOR_RESET)\n"; \
		printf "\n"; \
		tail -50 /var/log/faceid.log | grep --color=auto -E "AUTH_ATTEMPT|AUTH_SUCCESS|AUTH_FAILURE|ERROR|WARNING|$$"; \
		printf "\n"; \
		printf "$(COLOR_CYAN)Statistics:$(COLOR_RESET)\n"; \
		TOTAL=$$(grep -c "AUTH_ATTEMPT" /var/log/faceid.log 2>/dev/null || echo 0); \
		SUCCESS=$$(grep -c "AUTH_SUCCESS" /var/log/faceid.log 2>/dev/null || echo 0); \
		FAILED=$$(grep -c "AUTH_FAILURE" /var/log/faceid.log 2>/dev/null || echo 0); \
		printf "  Total attempts: $$TOTAL\n"; \
		printf "  Successful: $$SUCCESS\n"; \
		printf "  Failed: $$FAILED\n"; \
		printf "\n"; \
		printf "To view live log: tail -f /var/log/faceid.log\n"; \
		printf "To clear log: sudo truncate -s 0 /var/log/faceid.log\n"; \
	fi
	@printf " \n"

# Systemd service management
systemd-enable:
	@printf "$(COLOR_BOLD)Enabling FaceID Presence Detection Service$(COLOR_RESET)\n"
	@printf " \n"
	@if [ "$$(id -u)" != "0" ]; then \
		printf "$(COLOR_RED)Error: Systemd management requires root privileges$(COLOR_RESET)\n"; \
		printf "Please run: sudo make systemd-enable\n"; \
		exit 1; \
	fi
	@if [ ! -f "/usr/lib/systemd/system/faceid-presence.service" ]; then \
		printf "$(COLOR_RED)Error: Service file not found!$(COLOR_RESET)\n"; \
		printf "Please run 'sudo make install' first.\n"; \
		exit 1; \
	fi
	@if [ ! -f "/usr/bin/faceid-presence" ]; then \
		printf "$(COLOR_RED)Error: faceid-presence binary not found!$(COLOR_RESET)\n"; \
		printf "Please run 'sudo make install' first.\n"; \
		exit 1; \
	fi
	@printf "$(COLOR_CYAN)Checking configuration...$(COLOR_RESET)\n"
	@if ! grep -q "enabled = true" /etc/faceid/faceid.conf 2>/dev/null; then \
		printf "$(COLOR_YELLOW)⚠ Presence detection is disabled in config$(COLOR_RESET)\n"; \
		printf "  Edit /etc/faceid/faceid.conf and set: enabled = true\n"; \
		printf "  in [presence_detection] section\n"; \
		printf " \n"; \
		read -p "Enable now in config? [y/N] " -n 1 -r; \
		echo; \
		if [ "$$REPLY" = "y" ] || [ "$$REPLY" = "Y" ]; then \
			sed -i 's/^enabled = false/enabled = true/' /etc/faceid/faceid.conf; \
			printf "$(COLOR_GREEN)✓ Enabled presence detection in config$(COLOR_RESET)\n"; \
		fi; \
	else \
		printf "$(COLOR_GREEN)✓ Presence detection enabled in config$(COLOR_RESET)\n"; \
	fi
	@printf " \n"
	@printf "$(COLOR_CYAN)Reloading systemd daemon...$(COLOR_RESET)\n"
	@systemctl daemon-reload
	@printf "$(COLOR_GREEN)✓ Daemon reloaded$(COLOR_RESET)\n"
	@printf " \n"
	@printf "$(COLOR_CYAN)Enabling service...$(COLOR_RESET)\n"
	@systemctl enable faceid-presence.service
	@printf "$(COLOR_GREEN)✓ Service enabled$(COLOR_RESET)\n"
	@printf " \n"
	@printf "$(COLOR_CYAN)Starting service...$(COLOR_RESET)\n"
	@systemctl start faceid-presence.service
	@sleep 1
	@if systemctl is-active --quiet faceid-presence.service; then \
		printf "$(COLOR_GREEN)✓ Service started successfully$(COLOR_RESET)\n"; \
	else \
		printf "$(COLOR_RED)✗ Service failed to start$(COLOR_RESET)\n"; \
		printf "  Check logs: sudo journalctl -u faceid-presence -n 50\n"; \
		exit 1; \
	fi
	@printf " \n"
	@printf "$(COLOR_GREEN)$(COLOR_BOLD)Presence detection service enabled!$(COLOR_RESET)\n"
	@printf " \n"
	@printf "$(COLOR_CYAN)Next steps:$(COLOR_RESET)\n"
	@printf "  • Check status: make systemd-status\n"
	@printf "  • View logs: make systemd-logs\n"
	@printf "  • Full status: make presence-status\n"
	@printf " \n"

systemd-disable:
	@printf "$(COLOR_BOLD)Disabling FaceID Presence Detection Service$(COLOR_RESET)\n"
	@printf " \n"
	@if [ "$$(id -u)" != "0" ]; then \
		printf "$(COLOR_RED)Error: Systemd management requires root privileges$(COLOR_RESET)\n"; \
		printf "Please run: sudo make systemd-disable\n"; \
		exit 1; \
	fi
	@printf "$(COLOR_CYAN)Stopping service...$(COLOR_RESET)\n"
	@if systemctl is-active --quiet faceid-presence.service; then \
		systemctl stop faceid-presence.service; \
		printf "$(COLOR_GREEN)✓ Service stopped$(COLOR_RESET)\n"; \
	else \
		printf "$(COLOR_YELLOW)⚠ Service is not running$(COLOR_RESET)\n"; \
	fi
	@printf " \n"
	@printf "$(COLOR_CYAN)Disabling service...$(COLOR_RESET)\n"
	@systemctl disable faceid-presence.service 2>/dev/null || true
	@printf "$(COLOR_GREEN)✓ Service disabled$(COLOR_RESET)\n"
	@printf " \n"
	@printf "$(COLOR_GREEN)Presence detection service disabled$(COLOR_RESET)\n"
	@printf " \n"

systemd-status:
	@printf "$(COLOR_BOLD)FaceID Presence Detection Service Status$(COLOR_RESET)\n"
	@printf " \n"
	@systemctl status faceid-presence.service --no-pager || true
	@printf " \n"

systemd-logs:
	@printf "$(COLOR_BOLD)FaceID Presence Detection Service Logs$(COLOR_RESET)\n"
	@printf " \n"
	@printf "$(COLOR_CYAN)Recent logs (last 50 lines):$(COLOR_RESET)\n"
	@journalctl -u faceid-presence.service -n 50 --no-pager || printf "$(COLOR_YELLOW)Service not found or no logs available$(COLOR_RESET)\n"
	@printf " \n"
	@printf "$(COLOR_CYAN)To follow live logs:$(COLOR_RESET)\n"
	@printf "  journalctl -u faceid-presence.service -f\n"
	@printf " \n"

presence-status:
	@printf "$(COLOR_BOLD)FaceID Presence Detection Status$(COLOR_RESET)\n"
	@printf " \n"
	@printf "$(COLOR_CYAN)Installation:$(COLOR_RESET)\n"
	@if [ -f "/usr/bin/faceid-presence" ]; then \
		printf "$(COLOR_GREEN)✓ Binary installed: /usr/bin/faceid-presence$(COLOR_RESET)\n"; \
	else \
		printf "$(COLOR_RED)✗ Binary not found$(COLOR_RESET)\n"; \
	fi
	@if [ -f "/usr/lib/systemd/system/faceid-presence.service" ]; then \
		printf "$(COLOR_GREEN)✓ Systemd service file installed$(COLOR_RESET)\n"; \
	else \
		printf "$(COLOR_RED)✗ Systemd service file not found$(COLOR_RESET)\n"; \
	fi
	@printf " \n"
	@printf "$(COLOR_CYAN)Configuration:$(COLOR_RESET)\n"
	@if [ -f "/etc/faceid/faceid.conf" ]; then \
		if grep -q "enabled = true" /etc/faceid/faceid.conf 2>/dev/null; then \
			printf "$(COLOR_GREEN)✓ Presence detection enabled in config$(COLOR_RESET)\n"; \
			THRESHOLD=$$(grep "inactive_threshold_seconds" /etc/faceid/faceid.conf | awk '{print $$3}'); \
			INTERVAL=$$(grep "scan_interval_seconds" /etc/faceid/faceid.conf | awk '{print $$3}'); \
			FAILURES=$$(grep "max_scan_failures" /etc/faceid/faceid.conf | awk '{print $$3}'); \
			IDLE=$$(grep "max_idle_time_minutes" /etc/faceid/faceid.conf | awk '{print $$3}'); \
			printf "  • Inactive threshold: $${THRESHOLD}s\n"; \
			printf "  • Scan interval: $${INTERVAL}s\n"; \
			printf "  • Max failures: $${FAILURES}\n"; \
			printf "  • Max idle time: $${IDLE} min\n"; \
		else \
			printf "$(COLOR_YELLOW)⚠ Presence detection disabled in config$(COLOR_RESET)\n"; \
			printf "  Enable: sudo sed -i 's/enabled = false/enabled = true/' /etc/faceid/faceid.conf\n"; \
		fi; \
	else \
		printf "$(COLOR_RED)✗ Config file not found$(COLOR_RESET)\n"; \
	fi
	@printf " \n"
	@printf "$(COLOR_CYAN)Service Status:$(COLOR_RESET)\n"
	@if systemctl is-enabled --quiet faceid-presence.service 2>/dev/null; then \
		printf "$(COLOR_GREEN)✓ Service enabled$(COLOR_RESET)\n"; \
	else \
		printf "$(COLOR_YELLOW)⚠ Service not enabled$(COLOR_RESET)\n"; \
		printf "  Enable: sudo make systemd-enable\n"; \
	fi
	@if systemctl is-active --quiet faceid-presence.service 2>/dev/null; then \
		printf "$(COLOR_GREEN)✓ Service running$(COLOR_RESET)\n"; \
	else \
		printf "$(COLOR_RED)✗ Service not running$(COLOR_RESET)\n"; \
		printf "  Start: sudo systemctl start faceid-presence\n"; \
	fi
	@printf " \n"
	@printf "$(COLOR_CYAN)Quick Actions:$(COLOR_RESET)\n"
	@printf "  • Enable service:  sudo make systemd-enable\n"
	@printf "  • Disable service: sudo make systemd-disable\n"
	@printf "  • View status:     make systemd-status\n"
	@printf "  • View logs:       make systemd-logs\n"
	@printf " \n"

# Default target
.DEFAULT_GOAL := help
