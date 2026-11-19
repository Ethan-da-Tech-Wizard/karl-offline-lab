# ==============================================================================
# Program: quantization_panel.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Desktop workspace implementation for Quantization Panel.
#          Provides responsive layout controls, parameter adjustment surfaces,
#          and visual feedback components within Karl's PyQt6 shell.
# ==============================================================================
# Slots for the quantize form on the Model tab. Real quantization is done by
# QuantizerThread (app/engine), which drives the local llama-quantize binary;
# this mixin validates input, starts/cancels the job and updates the progress UI.
# The completion slot (_on_quant_done) lives in registry_panel.py.

from __future__ import annotations

import logging
import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QMessageBox,
)

from app.engine.quantizer_thread import QuantizerThread

logger = logging.getLogger("karl.system_config")

class QuantizationPanelMixin:
    """Mixin with the browse/start/cancel/progress/error slots for GGUF quantization."""
    def _browse_quant_source(self):
        """Pick the source GGUF and suggest an output name of the form base-FORMAT.gguf if none is set."""
        # Step 1: Validate input parameters, operational preconditions, and state
        path, _ = QFileDialog.getOpenFileName(
            self, "Select source GGUF", "data/models", "GGUF (*.gguf);;All Files (*)"
        )
        if path:
            self._quant_src_input.setText(path)
            # Never overwrite a name the user already typed.
            # Auto-suggest output filename if field is empty
            if not self._quant_out_input.text().strip():
                base = os.path.splitext(os.path.basename(path))[0]
                fmt  = self._quant_format_combo.currentText()
                self._quant_out_input.setText(f"{base}-{fmt}.gguf")



    def _start_quantize(self):
        """Validate inputs and launch a QuantizerThread, wiring its signals to the progress UI."""
        # Step 1: Validate input parameters, operational preconditions, and state
        # Only one quantization job at a time.
        if self._quantizer_thread and self._quantizer_thread.isRunning():
            QMessageBox.warning(self, "Quantization Running", "A quantization job is already in progress.")
            return

        src = self._quant_src_input.text().strip()
        out_name = self._quant_out_input.text().strip()
        fmt = self._quant_format_combo.currentText()

        if not src:
            QMessageBox.warning(self, "Missing Input", "Select a source GGUF file first.")
            return
        if not os.path.isfile(src):
            QMessageBox.warning(self, "File Not Found", f"Source file not found:\n{src}")
            return
        if not out_name:
            QMessageBox.warning(self, "Missing Output", "Enter an output filename.")
            return

        # A bare filename is placed in data/models so the result shows up in the model scan.
        # Resolve output to data/models/ if a bare filename was given
        if not os.path.dirname(out_name):
            out_path = os.path.join("data", "models", out_name)
        else:
            out_path = out_name

        self._quant_status_lbl.setText(f"Starting {fmt} quantization…")
        self._quant_progress_bar.setValue(0)
        self._quant_progress_bar.setVisible(True)
        self._quant_btn.setEnabled(False)
        self._quant_cancel_btn.setEnabled(True)

        self._quantizer_thread = QuantizerThread(
            input_path=src,
            output_path=out_path,
            target_format=fmt,
        )
        # Track the thread in _active_threads so it is not garbage collected while running;
        # it is discarded again on finished, and deleteLater releases the Qt object.
        self._active_threads.add(self._quantizer_thread)
        self._quantizer_thread.finished.connect(
            lambda t=self._quantizer_thread: self._active_threads.discard(t)
        )
        self._quantizer_thread.finished.connect(self._quantizer_thread.deleteLater)
        self._quantizer_thread.progress.connect(self._on_quant_progress)
        self._quantizer_thread.done.connect(self._on_quant_done)
        self._quantizer_thread.error.connect(self._on_quant_error)
        self._quantizer_thread.start()



    def _cancel_quantize(self):
        """After confirmation, ask the running quantize thread to cancel."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if self._quantizer_thread and self._quantizer_thread.isRunning():
            reply = QMessageBox.question(
                self, "Cancel Quantization",
                "Cancel the running quantization job?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                # Cancellation is cooperative; the thread stops at its next check, hence "requested".
                self._quantizer_thread.cancel()
                self._quant_status_lbl.setText("Cancellation requested…")



    def _on_quant_progress(self, pct: int):
        """Update the progress bar and status text with the percent complete."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._quant_progress_bar.setValue(pct)
        self._quant_status_lbl.setText(f"Quantizing… {pct}%")



    def _on_quant_error(self, msg: str):
        """Reset the quantize controls and report the error (inline, truncated, and in a dialog)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._quant_progress_bar.setVisible(False)
        self._quant_btn.setEnabled(True)
        self._quant_cancel_btn.setEnabled(False)
        # Drop the reference so a new job can start.
        self._quantizer_thread = None
        self._quant_status_lbl.setText(
            f"<span style='color:#FF5C7A;'>Error: {msg[:120]}</span>"
        )
        self._quant_status_lbl.setTextFormat(Qt.TextFormat.RichText)
        QMessageBox.critical(self, "Quantization Failed", msg)

    # ── registry tab ──────────────────────────────────────────────────────────


