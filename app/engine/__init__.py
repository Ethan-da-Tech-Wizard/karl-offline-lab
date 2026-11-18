# ==============================================================================
# Program: __init__.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Karl backend package initialization.
#          Coordinates with adjacent subsystems within the Karl Offline Lab
#          architecture.
# ==============================================================================

"""Karl backend package initialization."""

from app.engine.offline_guard import install as _install_offline_guard


_install_offline_guard()

