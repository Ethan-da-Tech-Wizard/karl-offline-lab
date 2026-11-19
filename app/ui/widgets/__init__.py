# ==============================================================================
# Program: __init__.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Reusable desktop widget component implementing init .
#          Enforces styling standards, event handling, and visual telemetry
#          across Karl offline workspaces.
# ==============================================================================

# Re-export the shared widgets so callers can write
# `from app.ui.widgets import StatusBar` instead of reaching into submodules.
from .status_bar import StatusBar
from .tracing_panel import TracingPanel
from .glow_panel import GlowPanel

# Public API of the package (controls `from app.ui.widgets import *`).
__all__ = ["StatusBar", "TracingPanel", "GlowPanel"]
