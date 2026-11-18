# Program: karl-offline-lab - __init__.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Package entry point that re-exports the reusable StatusBar, TracingPanel and GlowPanel widgets.

"""Reusable UI widgets shared across the application workspaces."""

# Public widget classes importable straight from app.ui.widgets
from .status_bar import StatusBar
from .tracing_panel import TracingPanel
from .glow_panel import GlowPanel

__all__ = ["StatusBar", "TracingPanel", "GlowPanel"]
