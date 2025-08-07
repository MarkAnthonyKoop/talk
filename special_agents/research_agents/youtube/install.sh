#!/bin/bash

# YouTube Research CLI Installation Script

echo "Installing YouTube Research CLI..."

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create symlink in user's local bin directory
LOCAL_BIN="$HOME/.local/bin"
mkdir -p "$LOCAL_BIN"

# Create symlink for youtube-research
ln -sf "$SCRIPT_DIR/youtube-research" "$LOCAL_BIN/youtube-research"

# Check if ~/.local/bin is in PATH
if [[ ":$PATH:" != *":$LOCAL_BIN:"* ]]; then
    echo ""
    echo "⚠️  Note: $LOCAL_BIN is not in your PATH"
    echo "Add this line to your ~/.bashrc or ~/.zshrc:"
    echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

echo "✅ Installation complete!"
echo ""
echo "Usage:"
echo "  youtube-research --help"
echo "  youtube-research research-video https://youtube.com/watch?v=VIDEO_ID"
echo "  youtube-research research-topic 'AI agents'"
echo "  youtube-research analyze 'What should I learn next?'"