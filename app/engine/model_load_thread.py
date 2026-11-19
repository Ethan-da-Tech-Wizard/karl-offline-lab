# ==============================================================================
# Program: model_load_thread.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Asynchronous worker thread handling GGUF model weight ingestion.
#          Loads weights into RAM/VRAM, monitors allocation safety, and reports
#          loading progress percentages.
# ==============================================================================

"""
Qt thread that loads (or reloads) a GGUF model without freezing the GUI.

The heavy work is done by ModelLoader; this thread only sequences it, records
the active model in config, and reports success or failure via signals.
"""

from __future__ import annotations

import os

from PyQt6.QtCore import QThread, pyqtSignal


class ModelLoadThread(QThread):
    """Load or reload a GGUF model without blocking the Qt GUI thread."""

    # Emitted with the loaded file name, the adapter name (or None) and draft model file name (or None).
    loaded = pyqtSignal(str, object, object)  # filename, adapter_name, draft_filename
    error = pyqtSignal(str)

    def __init__(
        self,
        model_path: str | None = None,
        *,
        adapter_name: str | None = None,
        draft_model_path: str | None = None,
        persist_active: bool = True,
        parent=None,
    ):
        # Step 1: Validate input parameters, operational preconditions, and state
        """Keyword-only options select an adapter, a speculative-decoding draft model and
        whether the choice should be saved as the active model for the next launch.
        """
        super().__init__(parent)
        self.model_path = model_path
        self.adapter_name = adapter_name
        self.draft_model_path = draft_model_path
        self.persist_active = persist_active

    def run(self) -> None:
        # Step 1: Validate input parameters, operational preconditions, and state
        # Imported inside run() to keep module import light and avoid import cycles.
        """Drop any previous model, load the requested one, optionally persist it, then emit ``loaded``."""
        from app.engine import config_store
        from app.engine.model_loader import ModelLoader

        try:
            # Execute guarded operations and establish recovery boundaries
            # Release the old model first so two large models are never in memory at once.
            ModelLoader.reset_instance()
            ModelLoader.get_instance(
                model_path=self.model_path,
                adapter_name=self.adapter_name,
                draft_model_path=self.draft_model_path,
            )

            # With no explicit path, ask the loader which model it ended up using (default model).
            filename = os.path.basename(self.model_path) if self.model_path else ModelLoader.model_name()
            draft_filename = os.path.basename(self.draft_model_path) if self.draft_model_path else None

            # Failing to save the choice is treated as an error so the UI does not claim a switch that will be lost on restart.
            if self.persist_active:
                if not config_store.set_active_model(filename, self.adapter_name):
                    raise OSError("Failed to persist data/active_model.json")
                if draft_filename:
                    config_store.set_active_draft_model(draft_filename, enabled=True)

            self.loaded.emit(filename, self.adapter_name, draft_filename)
        except Exception as exc:
            self.error.emit(str(exc))
