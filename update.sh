#!/usr/bin/env bash
set -e

if [ "$EUID" -ne 0 ]; then
  echo "Error: This script must be run as root (or via sudo)." >&2
  exit 1
fi

INSTALL_DIR="/opt/audio-system"
TMP_DIR="/tmp/audio-system-update"
APP_ZIP="$TMP_DIR/app.zip"
EXTRACT_DIR="$TMP_DIR/extracted"
BACKUP_DIR="/opt/audio-system-backup"
RELEASE_URL="https://github.com/ckriroe/RPI-audio-reactive-led-strip/releases/latest/download/app.zip"

echo "=== Starting RPISC Update ==="

# 1. Download if not already provided or if corrupted
mkdir -p "$TMP_DIR"
if [ ! -f "$APP_ZIP" ] || ! unzip -t "$APP_ZIP" > /dev/null 2>&1; then
  echo "Downloading latest release package..."
  rm -f "$APP_ZIP"
  curl -fLk "$RELEASE_URL" -o "$APP_ZIP"
fi

# Verify archive
echo "Verifying update package..."
if ! unzip -t "$APP_ZIP" > /dev/null 2>&1; then
  echo "Error: Downloaded package is not a valid zip file." >&2
  rm -f "$APP_ZIP"
  exit 1
fi

# 2. Extract into staging directory
rm -rf "$EXTRACT_DIR"
mkdir -p "$EXTRACT_DIR"
unzip -q -o "$APP_ZIP" -d "$EXTRACT_DIR"

# 3. Stop services cleanly before modifying files
echo "Stopping audio system services..."
systemctl stop audio-led-strip.service 2>/dev/null || true
systemctl stop audio-web-ui.service 2>/dev/null || true

# 4. Create local safety rollback snapshot
if [ -d "$INSTALL_DIR" ]; then
  echo "Creating rollback backup in $BACKUP_DIR..."
  rm -rf "$BACKUP_DIR"
  mkdir -p "$BACKUP_DIR"
  cp -r "$INSTALL_DIR"/* "$BACKUP_DIR"/ 2>/dev/null || true
fi

# 5. Handle user settings files: NEVER overwrite existing settings
if [ ! -f "$INSTALL_DIR/presets.json" ] && [ -f "$EXTRACT_DIR/presets.json" ]; then
  echo "Deploying default presets.json..."
  cp "$EXTRACT_DIR/presets.json" "$INSTALL_DIR/presets.json"
fi
rm -f "$EXTRACT_DIR/presets.json"

if [ ! -f "$INSTALL_DIR/static_settings.json" ] && [ -f "$EXTRACT_DIR/static_settings.json" ]; then
  echo "Deploying default static_settings.json..."
  cp "$EXTRACT_DIR/static_settings.json" "$INSTALL_DIR/static_settings.json"
fi
rm -f "$EXTRACT_DIR/static_settings.json"

# 6. Delete entire existing web_ui directory to prevent stale Python files
echo "Cleaning existing web_ui directory..."
rm -rf "$INSTALL_DIR/web_ui"

# 7. Copy all updated files into target directory
echo "Copying updated files to $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR" "$INSTALL_DIR/backups"
cp -r "$EXTRACT_DIR"/* "$INSTALL_DIR"/

# 8. Ensure proper execution permissions
chmod +x "$INSTALL_DIR/Application"
if [ -f "$INSTALL_DIR/update.sh" ]; then
  sed -i -e 's/\r$//' "$INSTALL_DIR/update.sh" 2>/dev/null || true
  chmod +x "$INSTALL_DIR/update.sh"
fi

# 9. Restore ALSA soundcard state if file exists
if [ -f "$INSTALL_DIR/aux_in_alsa_state.state" ]; then
  /usr/sbin/alsactl restore -f "$INSTALL_DIR/aux_in_alsa_state.state" 2>/dev/null || true
fi

# 10. Clean up staging
rm -rf "$TMP_DIR"

# 11. Restart services
echo "Restarting audio system services..."
systemctl start audio-web-ui.service
systemctl start audio-led-strip.service

echo "=== RPISC Update completed successfully! ==="
