# ==============================================================================
# Program: qt_test_helper.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Comprehensive test suite exercising qt helper functionality.
#          Verifies state transitions, error boundaries, regression resistance,
#          and contract compliance under tests.
# ==============================================================================

import os
import sys
from PyQt6.QtWidgets import QApplication

# Set offscreen platform plugin to run PyQt tests headlessly in all environments
if "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Initialize a global QApplication instance before any test modules are imported
if QApplication.instance() is None:
    _global_test_app = QApplication(sys.argv)

if QApplication.instance() is not None:
    QApplication.instance().setProperty("is_test", True)
