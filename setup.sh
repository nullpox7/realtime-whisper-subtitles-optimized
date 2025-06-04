#!/bin/bash
# Real-time Whisper Subtitles - Setup Script (Link to scripts/setup.sh)
# This file links to the main setup script in the scripts directory
# Encoding: UTF-8

# Check if scripts/setup.sh exists
if [ -f "scripts/setup.sh" ]; then
    echo "? Executing main setup script..."
    exec bash scripts/setup.sh "$@"
else
    echo "? Error: scripts/setup.sh not found"
    echo "Please make sure you are in the project root directory"
    exit 1
fi
