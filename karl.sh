#!/usr/bin/env bash
# Program: karl-offline-lab - karl.sh
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Launcher script: creates the Python virtual environment and installs dependencies on first run, downloads the base model if missing, then starts the app.
# Stop on the first failing command.
set -e

# Change directory to the script's location to support running from anywhere
cd "$(dirname "$0")"
# Relative paths below (venv, data) now resolve against the repository folder.

echo "=================================================="
echo "⚡ Starting Karl Introspection Environment..."
echo "=================================================="

# Create Python virtual environment if it doesn't exist
# First run only: build an isolated environment so system Python is untouched.
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment in venv..."
    python3 -m venv venv
    echo "Upgrading pip..."
    venv/bin/pip install --upgrade pip
    echo "Installing required Python dependencies..."
    venv/bin/pip install -r requirements.txt
fi

# Ensure models directory exists
# Make sure the folder for model files exists.
mkdir -p data/models

# Download base model if it doesn't exist
# Fetch the default small model only when it is not already present.
if [ ! -f "data/models/deepseek-r1-1.5b.gguf" ]; then
    echo "Base model not found. Starting automatic download..."
    venv/bin/python download_test_model.py
fi

# Launch the PyQt app
echo "Launching Karl..."
# Start the desktop app using the environment's Python.
venv/bin/python main.py
