# Program: karl-offline-lab - qt_test_helper.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Helper that creates a headless QApplication flagged as a test instance.
"""Importing this module guarantees a headless QApplication exists and is marked as running under test."""

import os
import sys
from PyQt6.QtWidgets import QApplication

# Set offscreen platform plugin to run PyQt tests headlessly in all environments
# Use the offscreen platform so no display is required
if "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Initialize a global QApplication instance before any test modules are imported
# Create the single shared application object only once
if QApplication.instance() is None:
    _global_test_app = QApplication(sys.argv)

if QApplication.instance() is not None:
    # Mark the app as a test instance so application code can adapt
    QApplication.instance().setProperty("is_test", True)
