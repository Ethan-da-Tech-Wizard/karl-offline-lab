# Program: karl-offline-lab - __init__.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Test package marker that forces offline mode and a headless Qt application before any test module is imported.
"""Test package initializer: sets offline environment flags and creates one shared QApplication."""

import os
import sys

# Keep Hugging Face libraries offline so tests never touch the network
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# Qt is imported after the environment is configured
from PyQt6.QtWidgets import QApplication

# Set offscreen platform plugin to run PyQt tests headlessly in all environments
if "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Initialize a global QApplication instance before any test modules are imported
if QApplication.instance() is None:
    _global_test_app = QApplication(sys.argv)
