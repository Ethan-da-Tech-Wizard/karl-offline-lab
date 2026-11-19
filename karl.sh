#!/usr/bin/env bash
# ==============================================================================
# Program: karl.sh
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Primary Unix startup script for Karl Offline Lab.
#          Provisions the Python virtual environment, installs dependencies from
#          requirements.txt, validates baseline model availability, and launches
#          the main PyQt6 desktop application.
# ==============================================================================

# Abort on the first failing command so a half-provisioned venv is not launched.
set -e

# Change directory to the script's location to support running from anywhere
# Resolve all relative paths (venv/, data/, main.py) against the repo root.
cd "$(dirname "$0")"

echo "=================================================="
echo "⚡ Starting Karl Introspection Environment..."
echo "=================================================="

# Create Python virtual environment if it doesn't exist
# First-run provisioning only: an existing venv is trusted as-is, so later dependency changes need a
# manual pip install or deleting venv/.
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment in venv..."
    python3 -m venv venv
    echo "Upgrading pip..."
    venv/bin/pip install --upgrade pip
    echo "Installing required Python dependencies..."
    venv/bin/pip install -r requirements.txt
fi

# Ensure models directory exists
# The model download and the app both expect this directory to exist.
mkdir -p data/models

# Download base model if it doesn't exist
# Default Tier-1 model; fetched once (requires network) so the app can start offline afterwards.
if [ ! -f "data/models/deepseek-r1-1.5b.gguf" ]; then
    echo "Base model not found. Starting automatic download..."
    venv/bin/python download_test_model.py
fi

# Launch the PyQt app
echo "Launching Karl..."
# Run through the venv interpreter explicitly; no need to 'activate' the environment.
venv/bin/python main.py
