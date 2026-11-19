# ==============================================================================
# Program: __init__.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Workbench package — primary interaction space.
#          Coordinates with adjacent subsystems within the Karl Offline Lab
#          architecture.
# ==============================================================================
# Package facade: re-export the three public names so callers can import them
# from app.ui.workspaces.workbench without knowing the module layout.

"""Workbench package — primary interaction space.

Split into focused modules:
- workspace.py  — WorkbenchWorkspace (layout, sessions, model selection, threads)
- chat_view.py  — ChatView streaming conversation widget
- profiles.py   — agent profile definitions

Importing WorkbenchWorkspace from app.ui.workspaces.workbench keeps working.
"""

from app.ui.workspaces.workbench.chat_view import ChatView
from app.ui.workspaces.workbench.profiles import AGENT_PROFILES
from app.ui.workspaces.workbench.workspace import WorkbenchWorkspace

__all__ = ["AGENT_PROFILES", "ChatView", "WorkbenchWorkspace"]
