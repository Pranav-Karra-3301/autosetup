#!/bin/bash
# Universal install script for autosetup

set -e

REPO="Pranav-Karra-3301/autosetup"
INSTALL_DIR="/usr/local/bin"
BINARY_NAME="autosetup"

# Detect OS and architecture
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case "$OS" in
    linux)
        PLATFORM="linux"
        ;;
    darwin)
        PLATFORM="macos"
        ;;
    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac

case "$ARCH" in
    x86_64|amd64)
        ARCH="x86_64"
        ;;
    arm64|aarch64)
        ARCH="aarch64"
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

echo "🚀 Installing autosetup for $PLATFORM-$ARCH..."

# Get latest release
LATEST_RELEASE=$(curl -s "https://api.github.com/repos/$REPO/releases/latest" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')

if [ -z "$LATEST_RELEASE" ]; then
    echo "❌ Failed to get latest release"
    exit 1
fi

echo "📦 Downloading version $LATEST_RELEASE..."

# Download binary
DOWNLOAD_URL="https://github.com/$REPO/releases/download/$LATEST_RELEASE/autosetup-$PLATFORM-$ARCH"
TMP_FILE="/tmp/autosetup-$$"

curl -L -o "$TMP_FILE" "$DOWNLOAD_URL"

# Make executable
chmod +x "$TMP_FILE"

# Move to install directory (may require sudo)
if [ -w "$INSTALL_DIR" ]; then
    mv "$TMP_FILE" "$INSTALL_DIR/$BINARY_NAME"
else
    echo "📝 Administrator privileges required to install to $INSTALL_DIR"
    sudo mv "$TMP_FILE" "$INSTALL_DIR/$BINARY_NAME"
fi

# Verify installation
if command -v autosetup &> /dev/null; then
    echo "✅ autosetup installed successfully!"
    autosetup --version
else
    echo "⚠️ Installation completed but autosetup not found in PATH"
    echo "Add $INSTALL_DIR to your PATH if needed"
fi