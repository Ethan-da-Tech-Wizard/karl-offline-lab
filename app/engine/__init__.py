# Program: karl-offline-lab - __init__.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Package initializer for the Karl engine that installs the offline network guard as soon as the package is imported.
"""Karl backend package initialization."""

from app.engine.offline_guard import install as _install_offline_guard


# Importing any engine module patches the HTTP clients so external network calls fail closed.
_install_offline_guard()

