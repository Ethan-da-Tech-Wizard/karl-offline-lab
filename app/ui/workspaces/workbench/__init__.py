# Program: karl-offline-lab - __init__.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Package entry point for the Workbench workspace that re-exports ChatView, AGENT_PROFILES and WorkbenchWorkspace.
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
