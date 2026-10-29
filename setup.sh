#!/usr/bin/env bash
set -e

if [ "$EUID" -ne 0 ]; then
  echo "Error: This script must be run as root (or via sudo)." >&2
  exit 1
fi

TARGET_USER="${SUDO_USER:-$USER}"
USER_HOME=$(eval echo "~$TARGET_USER")

# UPDATE APT & INSTALL REQUIRED SYSTEM DEPENDENCIES
apt-get update
apt-get install -y curl unzip alsa-utils build-essential scons git libglfw3

# INIT SOUNDCARD
CONFIG_FILE="/boot/firmware/config.txt"

if [ -f "$CONFIG_FILE" ]; then
  sed -i 's/^\s*dtparam=audio=on/# dtparam=audio=on/' "$CONFIG_FILE"
  sed -i 's/^\s*dtoverlay=vc4-kms-v3d\s*$/dtoverlay=vc4-kms-v3d,noaudio/' "$CONFIG_FILE"
  grep -qxF "dtoverlay=rpi-codeczero" "$CONFIG_FILE" || echo "dtoverlay=rpi-codeczero" >> "$CONFIG_FILE"
fi

cat << 'EOF' > "$USER_HOME/.asoundrc"
pcm.!default {
        type asym
        playback.pcm {
                type plug
                slave.pcm "dmixer"
        }
        capture.pcm {
                type plug
                slave.pcm "dsnooper"
        }
}
pcm.dmixer {
	type dmix
	ipc_key 1024
	slave {
		pcm "hw:0,0"
		rate 44100
		period_time 0
		period_size 1024
		buffer_size 8192
	}
}
pcm.dsnooper {
	type dsnoop
	ipc_key 816357492
	ipc_key_add_uid 0
	ipc_perm 0666
	slave {
		pcm "hw:0,0"
		channels 1
	}
}
EOF

chown "$TARGET_USER:" "$USER_HOME/.asoundrc"
# SOUNDCARD INIT DONE

# INSTALL PYTHON DEPS
python3 -m pip install streamlit==1.64.0 --break-system-packages --no-input --ignore-installed

# INSTALL .NET 8
curl -sSL https://dot.net/v1/dotnet-install.sh | bash /dev/stdin --channel 8.0 --version latest --install-dir /opt/dotnet --verbose
BASHRC="$USER_HOME/.bashrc"
LINE1='export DOTNET_ROOT=/opt/dotnet'
LINE2='export PATH=$PATH:/opt/dotnet'
grep -qF -- "$LINE1" "$BASHRC" || echo "$LINE1" >> "$BASHRC"
grep -qF -- "$LINE2" "$BASHRC" || echo "$LINE2" >> "$BASHRC"
chown "$TARGET_USER:" "$BASHRC"

# BUILD AND INSTALL RPI_WS281X
BUILD_DIR=$(mktemp -d)
(
  cd "$BUILD_DIR"
  git clone https://github.com/jgarff/rpi_ws281x.git .
  scons
  gcc -shared -o ws2811.so *.o
  cp -f ws2811.so /usr/lib/
)
rm -rf "$BUILD_DIR"

# DOWNLOAD AND INSTALL AUDIO SYSTEM RELEASE
# Stop services if already installed and running to avoid "Text file busy"
systemctl is-active --quiet audio-led-strip.service && systemctl stop audio-led-strip.service || true
systemctl is-active --quiet audio-web-ui.service && systemctl stop audio-web-ui.service || true

mkdir -p /opt/audio-system
APP_ZIP=$(mktemp)
curl -fLk https://github.com/ckriroe/RPI-audio-reactive-led-strip/releases/latest/download/app.zip -o "$APP_ZIP"
TMP_EXTRACT=$(mktemp -d)
unzip -q -o "$APP_ZIP" -d "$TMP_EXTRACT"
rm -f "$APP_ZIP"

# Deploy default settings ONLY if they do not exist already
if [ ! -f "/opt/audio-system/presets.json" ] && [ -f "$TMP_EXTRACT/presets.json" ]; then
  echo "Deploying default presets.json..."
  cp "$TMP_EXTRACT/presets.json" "/opt/audio-system/presets.json"
fi
rm -f "$TMP_EXTRACT/presets.json"

if [ ! -f "/opt/audio-system/static_settings.json" ] && [ -f "$TMP_EXTRACT/static_settings.json" ]; then
  echo "Deploying default static_settings.json..."
  cp "$TMP_EXTRACT/static_settings.json" "/opt/audio-system/static_settings.json"
fi
rm -f "$TMP_EXTRACT/static_settings.json"

# Delete entire existing web_ui to remove any deleted/stale python files
rm -rf /opt/audio-system/web_ui

# Copy updated files into /opt/audio-system
cp -r "$TMP_EXTRACT"/* /opt/audio-system/
rm -rf "$TMP_EXTRACT"

chmod +x /opt/audio-system/Application
if [ -f /opt/audio-system/update.sh ]; then
  sed -i -e 's/\r$//' /opt/audio-system/update.sh 2>/dev/null || true
  chmod +x /opt/audio-system/update.sh
fi

# CREATE SYSTEMD SERVICES

# 1. Audio Setup Service (ALSA restore only)
cat << 'EOF' > /etc/systemd/system/audio-setup.service
[Unit]
Description=Restore ALSA state for audio system
After=sound.target

[Service]
Type=oneshot
WorkingDirectory=/opt/audio-system
ExecStart=/usr/sbin/alsactl restore -f /opt/audio-system/aux_in_alsa_state.state
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

# 2. Audio LED Strip Service
cat << 'EOF' > /etc/systemd/system/audio-led-strip.service
[Unit]
Description=Audio led strip service
After=graphical-session.target audio-web-ui.service
Requires=audio-web-ui.service

[Service]
Type=simple
WorkingDirectory=/opt/audio-system
ExecStart=/opt/audio-system/Application

Environment=WAYLAND_DISPLAY=wayland-0
Environment=XDG_RUNTIME_DIR=/run/user/1000
Environment=DOTNET_ROOT=/opt/dotnet
Environment="PATH=/opt/dotnet:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment=DOTNET_USE_POLLING_FILE_WATCHER=1

Restart=always
RestartSec=2
User=root

[Install]
WantedBy=multi-user.target
EOF

# 3. Audio Web UI Service
cat << 'EOF' > /etc/systemd/system/audio-web-ui.service
[Unit]
Description=Audio web UI service
After=audio-setup.service
Requires=audio-setup.service

[Service]
Type=simple
WorkingDirectory=/opt/audio-system

ExecStart=/bin/bash -c 'streamlit run audio_parameter_web_ui.py --server.address 0.0.0.0 --server.port 8501'

Restart=always
RestartSec=5

User=root
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

# 4. Audio Update Service (triggered on-demand via Web UI or systemctl)
cat << 'EOF' > /etc/systemd/system/audio-update.service
[Unit]
Description=Update service for Audio LED Strip and Web UI
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/opt/audio-system
ExecStart=/bin/bash /opt/audio-system/update.sh
User=root
StandardOutput=journal
StandardError=journal
EOF

# SYSTEMD RELOAD & ENABLE SERVICES
systemctl daemon-reload
systemctl enable audio-setup.service
systemctl enable audio-web-ui.service
systemctl enable audio-led-strip.service

echo "Setup completed successfully. Rebooting system now..."
reboot
