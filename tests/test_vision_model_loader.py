# Program: karl-offline-lab - test_vision_model_loader.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Tests that the vision model registry can be read and that installed vision model rows report file paths and install state.
"""Tests for reading the vision model registry and reporting installed models."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vision.vision_model_loader import installed_vision_models, read_vision_registry


def test_vision_registry_is_readable():
    """Scenario: the registry is read.
    Expected: it has entries, each with an id and model and projector file names ending in .gguf.
    """
    entries = read_vision_registry()

    assert entries
    assert all(entry.id for entry in entries)
    assert all(entry.model_filename.endswith(".gguf") for entry in entries)
    assert all(entry.projector_filename.endswith(".gguf") for entry in entries)


def test_installed_vision_models_reports_paths_and_install_state():
    """Scenario: installed vision models are listed.
    Expected: rows exist and each includes model and projector paths and their installed flags.
    """
    rows = installed_vision_models()

    assert rows
    first = rows[0]
    assert "model_path" in first
    assert "projector_path" in first
    assert "model_installed" in first
    assert "projector_installed" in first
