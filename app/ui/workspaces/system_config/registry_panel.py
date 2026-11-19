# ==============================================================================
# Program: registry_panel.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Desktop workspace implementation for Registry Panel.
#          Provides responsive layout controls, parameter adjustment surfaces,
#          and visual feedback components within Karl's PyQt6 shell.
# ==============================================================================
# Registry tab of the System workspace: shows each model tier from the model
# registry as a card with Download / Activate / Active (and Quantize for FP16)
# buttons, plus the shared progress strip used for downloads, activation and
# registry-initiated quantization.

from __future__ import annotations

import logging
import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, QMessageBox, QScrollArea, QProgressBar, QDialog,
)

from app.engine import config_store
from app.ui.themes import MONO
from .download_threads import DownloadThread, QuantizationDialog, QuantizationThread

logger = logging.getLogger("karl.system_config")

# Mixin for the System workspace. The registry (self._registry) is a list of tier
# dicts from config_store.get_model_registry() (data/model_registry.json); this
# mixin only renders it and starts jobs. State the rest of the tab relies on:
# self._active_threads (set of live QThreads), self._download_thread (the single
# DownloadThread or None) and the shared progress strip widgets created below.
class RegistryPanelMixin:
    """Mixin that renders the model registry and runs downloads and activations."""
    # Completion slot for quantize jobs started from the Model tab (the registry's
    # own quantize flow ends in _on_registry_quant_done further down). It lives in
    # this file because both flows share the progress/cleanup helpers here.
    def _on_quant_done(self, out_path: str):
        """Finish a Model-tab quantization job: reset controls, rescan models and tell the user."""
        self._quant_progress_bar.setValue(100)
        self._quant_progress_bar.setVisible(False)
        self._quant_btn.setEnabled(True)
        self._quant_cancel_btn.setEnabled(False)
        # Dropping the reference lets the finished thread be garbage-collected; the
        # QThread itself was already scheduled for deleteLater by the starter.
        self._quantizer_thread = None
        self._scan_models(force=True)
        self._quant_status_lbl.setText(
            f"<span style='color:#2DD4A0;'>Done — {os.path.basename(out_path)} saved to data/models/</span>"
        )
        self._quant_status_lbl.setTextFormat(Qt.TextFormat.RichText)
        QMessageBox.information(
            self, "Quantization Complete",
            f"Quantized model saved:\n{out_path}"
        )


    def _build_registry_tab(self) -> QWidget:
        """Build the Registry tab: a scrolling list of tier cards and a hidden progress strip."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        # Object names let the global QSS style this area transparently so the cards
        # (objectName "panel") are the only visible surfaces.
        scroll.setObjectName("registry-scroll")
        
        scroll_content = QWidget()
        scroll_content.setObjectName("registry-content")
        # Kept as an attribute because _populate_registry clears and refills it on
        # every rescan, download or activation.
        self._registry_layout = QVBoxLayout(scroll_content)
        self._registry_layout.setContentsMargins(0, 0, 0, 0)
        self._registry_layout.setSpacing(10)
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        # Progress strip is hidden until a download, quantize or activation is running.
        self._progress_panel = QWidget()
        self._progress_panel.setObjectName("panel")
        self._progress_panel.setVisible(False)
        p_layout = QHBoxLayout(self._progress_panel)
        p_layout.setContentsMargins(12, 8, 12, 8)
        p_layout.setSpacing(10)
        
        # One progress strip serves three different jobs (download, quantize,
        # activation); each starter rewrites this label, so the text is never trusted
        # across jobs. Stretch 1:2 gives the bar twice the label's share of the row.
        self._download_status_lbl = QLabel("Downloading...")
        p_layout.addWidget(self._download_status_lbl, 1)
        
        self._download_bar = QProgressBar()
        self._download_bar.setRange(0, 100)
        self._download_bar.setValue(0)
        # Range 0-100 is percent for downloads/quantize. Activation switches the range to
        # (0, 0) for a busy indicator and restores 0-100 afterwards (see
        # _activate_registry_model), so any code that sets a value must not assume the
        # range survived.
        self._download_bar.setFixedHeight(12)
        p_layout.addWidget(self._download_bar, 2)
        
        self._cancel_download_btn = QPushButton("Cancel")
        # The cancel button is shared too: it is only meaningful for downloads, and
        # _cancel_download is expected to be a no-op when no download thread runs.
        self._cancel_download_btn.setObjectName("btn-danger")
        self._cancel_download_btn.clicked.connect(self._cancel_download)
        p_layout.addWidget(self._cancel_download_btn)
        
        layout.addWidget(self._progress_panel)
        
        self._populate_registry()
        return w


    # Called from the workspace constructor before the tab is built, so _registry
    # always exists by the time _populate_registry runs.
    def _load_registry(self):
        """Load the registry from the config store, falling back to a built-in tier list if it is empty."""
        # list() copies, so later mutation of self._registry (the fallback below)
        # cannot alias the config store's cached object.
        self._registry = list(config_store.get_model_registry())

        # Fallback tiers mirror the five DeepSeek-R1 distill tiers shipped in data/model_registry.json
        # so the tab is still usable if the file is missing or unreadable.
        if not self._registry:
            self._registry = [
                {
                    "tier": 1,
                    "name": "DeepSeek-R1-Distill-Qwen-1.5B Q4_K_M",
                    "filename": "deepseek-r1-1.5b.gguf",
                    "min_ram_gb": 3.0,
                    "min_vram_gb": 0.0,
                    "min_storage_gb": 1.5,
                    "url": "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
                    "n_ctx": 4096
                },
                {
                    "tier": 2,
                    "name": "DeepSeek-R1-Distill-Qwen-7B Q4_K_M",
                    "filename": "deepseek-r1-7b.gguf",
                    "min_ram_gb": 8.0,
                    "min_vram_gb": 0.0,
                    "min_storage_gb": 5.0,
                    "url": "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
                    "n_ctx": 8192
                },
                {
                    "tier": 2,
                    "name": "DeepSeek-R1-Distill-Llama-8B Q4_K_M",
                    "filename": "deepseek-r1-llama-8b.gguf",
                    "min_ram_gb": 8.0,
                    "min_vram_gb": 0.0,
                    "min_storage_gb": 5.5,
                    "url": "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
                    "n_ctx": 8192
                },
                {
                    "tier": 3,
                    "name": "DeepSeek-R1-Distill-Qwen-14B Q4_K_M",
                    "filename": "deepseek-r1-14b.gguf",
                    "min_ram_gb": 16.0,
                    "min_vram_gb": 0.0,
                    "min_storage_gb": 10.0,
                    "url": "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
                    "n_ctx": 16384
                },
                {
                    "tier": 4,
                    "name": "DeepSeek-R1-Distill-Llama-70B Q4_K_M",
                    "filename": "deepseek-r1-70b.gguf",
                    "min_ram_gb": 48.0,
                    "min_vram_gb": 0.0,
                    "min_storage_gb": 42.0,
                    "url": "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf",
                    "n_ctx": 32768
                }
            ]


    def _get_active_model_name(self) -> str:
        """Return the active model filename from state, else from active_model.json, else "none"."""
        # state.model_name is None before any load this session; the persisted
        # active_model.json is then the source of truth, which keeps the ACTIVE badge
        # correct right after launch.
        active_name = self.state.model_name or "none"
        if active_name == "none":
            from app.engine import config_store
            data = config_store.read_json(config_store.ACTIVE_MODEL_PATH, default=None)
            if isinstance(data, dict):
                active_name = data.get("filename") or "none"
        return active_name


    def _populate_registry(self):
        """Rebuild the tier cards from self._registry, marking active and downloaded models."""
        # Clear the old cards first; deleteLater avoids touching widgets still in use.
        while self._registry_layout.count():
            item = self._registry_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
                
        active_name = self._get_active_model_name()
        # Relative to the working directory; the app always runs from the project root.
        models_dir = "data/models"
        
        for item in self._registry:
            # Defaults keep a malformed registry entry renderable (as tier 1 with the
            # smallest requirements) instead of crashing the whole tab; a card with a missing
            # filename simply shows Download that cannot succeed.
            tier = item.get("tier", 1)
            name = item.get("name", "Unknown")
            filename = item.get("filename", "")
            n_ctx = item.get("n_ctx", 4096)
            min_ram = item.get("min_ram_gb", 3.0)
            min_storage = item.get("min_storage_gb", 1.5)
            url = item.get("url", "")
            
            card = QWidget()
            card.setObjectName("panel")
            c_layout = QVBoxLayout(card)
            c_layout.setContentsMargins(12, 12, 12, 12)
            c_layout.setSpacing(6)
            
            header = QWidget()
            h_layout = QHBoxLayout(header)
            h_layout.setContentsMargins(0, 0, 0, 0)
            lbl_title = QLabel(f"Tier {tier}: {name}")
            lbl_title.setObjectName("lbl-accent")
            lbl_title.setStyleSheet("font-weight: bold; font-size: 10.5pt;")
            h_layout.addWidget(lbl_title, 1)
            
            # "Downloaded" means the target file exists. DownloadThread writes to a temporary
            # file and renames it only on success, so a registry download never leaves a
            # partial target; a hand-copied file could still be bad, which is what the GGUF
            # header check in model_preflight catches.
            file_path = os.path.join(models_dir, filename)
            is_downloaded = os.path.exists(file_path)
            # Compared by filename, the registry's identity for a model; two tiers pointing at
            # the same file would both show Active.
            is_active = (filename == active_name)
            
            # Three states per card: active (disabled badge), downloaded (Activate), missing (Download).
            btn = QPushButton()
            quant_btn = None
            
            if is_active:
                btn.setText("Active")
                btn.setEnabled(False)
                btn.setStyleSheet("background-color: rgba(45, 212, 160, 0.15); color: #2DD4A0; border: 1px solid rgba(45, 212, 160, 0.4); border-radius: 4px; font-weight: bold; padding: 5px 14px;")
            elif is_downloaded:
                btn.setText("Activate")
                # Default arguments (f=filename) bind the loop variable at definition time; a bare closure
                # would make every button act on the last card.
                btn.clicked.connect(lambda checked, f=filename: self._activate_registry_model(f))
                btn.setStyleSheet("background-color: rgba(0, 194, 255, 0.1); color: #00C2FF; border: 1px solid rgba(0, 194, 255, 0.35); border-radius: 4px; padding: 5px 14px;")
                
                # FP16 models get an extra Quantize button; the check is by name, filename or precision field.
                # If downloaded and precision is FP16, add Quantize button
                # We'll infer FP16 if 'FP16' is in the name or filename
                if "FP16" in name.upper() or "FP16" in filename.upper() or item.get("precision") == "FP16":
                    quant_btn = QPushButton("Quantize")
                    quant_btn.setStyleSheet("background-color: rgba(240, 176, 48, 0.1); color: #F0B030; border: 1px solid rgba(240, 176, 48, 0.35); border-radius: 4px; padding: 5px 14px;")
                    quant_btn.clicked.connect(lambda checked, f=filename, n=name: self._on_quantize_clicked(f, n))
            else:
                btn.setText("Download")
                btn.setObjectName("btn-primary")
                btn.setStyleSheet("padding: 5px 14px;")
                btn.clicked.connect(lambda checked, u=url, f=filename: self._start_download(u, f))
                
            # Quantize is placed before the main button so the primary action stays at the far
            # right edge of every card, aligned across cards.
            if quant_btn:
                h_layout.addWidget(quant_btn)
            h_layout.addWidget(btn)
            c_layout.addWidget(header)
            
            # Requirements come straight from the registry: min_ram_gb and min_storage_gb are
            # lower bounds the preflight compares with the hardware profile, and n_ctx is the
            # context length the model is loaded with.
            meta = QLabel(
                f"Context size: <b>{n_ctx:,}</b> tokens &middot; "
                f"RAM: <b>&ge; {min_ram} GB</b> &middot; "
                f"Storage: <b>&ge; {min_storage} GB</b> &middot; "
                f"Filename: <span style='font-family:{MONO}; font-size:8.5pt;'>{filename}</span>"
            )
            meta.setObjectName("lbl-muted")
            meta.setWordWrap(True)
            meta.setTextFormat(Qt.TextFormat.RichText)
            c_layout.addWidget(meta)
            
            self._registry_layout.addWidget(card)
            
        # Stretch after the last card keeps cards at natural height at the top instead
        # of being stretched to fill a tall window.
        self._registry_layout.addStretch(1)


    def _on_quantize_clicked(self, filename: str, model_name: str):
        """Ask for a target format and start a registry-initiated quantization of an FP16 model."""
        # Source and output stay in data/models so the new file shows up in the model
        # list after the rescan in _on_registry_quant_done.
        src_path = os.path.join("data", "models", filename)
        
        dialog = QuantizationDialog(model_name, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Returned as the llama.cpp format name (for example Q4_K_M) and used verbatim in
            # the output filename.
            fmt = dialog.selected_format()
            
            # Output is named <base>-<FORMAT>.gguf next to the source.
            # Suggest output name: model-name-Q4_K_M.gguf
            base = os.path.splitext(filename)[0]
            out_filename = f"{base}-{fmt}.gguf"
            out_path = os.path.join("data", "models", out_filename)
            
            # Show progress overlay in the registry tab
            self._progress_panel.setVisible(True)
            self._download_status_lbl.setText("Compiling custom GGUF weights...")
            self._download_bar.setValue(0)
            self._set_ui_enabled_for_download(False)
            
            # This uses the simulated QuantizationThread from download_threads (it reuses the
            # download progress strip for feedback).
            thread = QuantizationThread(src_path, out_path, fmt)
            self._active_threads.add(thread)
            # Registering in _active_threads before start() ensures the app-wide shutdown
            # wait sees the thread; discard on finished prevents the set from growing.
            thread.finished.connect(lambda: self._active_threads.discard(thread))
            thread.finished.connect(thread.deleteLater)
            
            thread.progress.connect(self._on_download_progress)
            thread.log.connect(self._on_download_log)
            # Progress and error slots are the download ones, reused so the strip behaves the
            # same for every job type; only "done" differs because it must rescan models.
            thread.done.connect(lambda p: self._on_registry_quant_done(p))
            thread.error.connect(self._on_download_error)
            
            thread.start()


    def _on_registry_quant_done(self, output_path: str):
        """Hide the progress strip, re-enable the cards and refresh the model views after quantization."""
        self._progress_panel.setVisible(False)
        self._set_ui_enabled_for_download(True)
        # Basename only in the dialog: the full relative path adds noise for what the user
        # needs to recognise.
        filename = os.path.basename(output_path)
        
        # Refresh UI
        self._scan_models(force=True)
        self._populate_registry()
        
        QMessageBox.information(
            self, "Quantization Complete",
            f"Successfully compiled weights to:\n{filename}\n\nYou can now activate this optimized model."
        )


    def _activate_registry_model(self, filename: str):
        """Load the chosen registry model in the background and show a busy bar while it activates."""
        from app.engine.model_loader import ModelLoader
        # Cannot swap models while a generation holds the loader.
        if ModelLoader.is_instance_locked():
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Karl — Inference Active",
                "Cannot switch model while inference is active.\n"
                "Stop the current generation first.",
            )
            return

        # Activation is just a model load: it reuses _start_model_load (model_panel.py), so
        # it gets the same single-load guard, control disabling and preflight refresh.
        path = os.path.join("data", "models", filename)
        self._progress_panel.setVisible(True)
        self._download_status_lbl.setText(f"Activating {filename}...")
        # setRange(0, 0) turns the bar into an indeterminate busy indicator.
        self._download_bar.setRange(0, 0)
        self._set_ui_enabled_for_download(False)

        # Common cleanup shared by the success, error and finished callbacks.
        # May run twice (on_error then on_finished), so it must stay idempotent: it only
        # hides, resets and re-enables.
        def finish_activation_ui():
            self._progress_panel.setVisible(False)
            self._download_bar.setRange(0, 100)
            self._download_bar.setValue(0)
            self._set_ui_enabled_for_download(True)

        def on_loaded(_filename, _adapter, _draft):
            finish_activation_ui()
            QMessageBox.information(
                self, "Model Activated",
                f"Model '{filename}' has been activated successfully!"
            )

        # Callbacks are layered on top of the Model tab's own handlers: on_loaded shows
        # the success dialog, while on_error/on_finished restore this tab's progress
        # strip. status_label is the Model tab status line (or None if that tab was not
        # built).
        started = self._start_model_load(
            path,
            status_label=getattr(self, "_model_status", None),
            on_loaded=on_loaded,
            on_error=lambda _msg: finish_activation_ui(),
            on_finished=finish_activation_ui,
        )
        # _start_model_load returns False when another load is running; none of the
        # callbacks will fire then, so the strip must be reset here.
        if not started:
            self._progress_panel.setVisible(False)
            self._download_bar.setRange(0, 100)
            self._set_ui_enabled_for_download(True)


    def _start_download(self, url: str, filename: str):
        """Start downloading a registry model to data/models, wiring progress, speed and completion."""
        # One download at a time.
        if self._download_thread and self._download_thread.isRunning():
            QMessageBox.warning(self, "Download In Progress", "A download is already running.")
            return
            
        # Downloads write straight into data/models so a finished download is visible to
        # the scanner and the Activate button without any move step.
        target_path = os.path.join("data", "models", filename)
        
        self._progress_panel.setVisible(True)
        self._download_status_lbl.setText(f"Downloading {filename}...")
        self._download_bar.setValue(0)
        self._set_ui_enabled_for_download(False)
        
        # The single self._download_thread attribute is what enforces "one download at a
        # time" (see the isRunning check above). It is reset to None in the done and
        # error slots; after a cancel neither fires, so the stale reference is harmless
        # only because isRunning() is then False.
        self._download_thread = DownloadThread(url, target_path)
        self._active_threads.add(self._download_thread)
        self._download_thread.finished.connect(
            lambda t=self._download_thread: self._active_threads.discard(t)
        )
        self._download_thread.finished.connect(self._download_thread.deleteLater)
        self._download_thread.progress.connect(self._on_download_progress)
        # Signals (emitted from the worker thread, delivered on the GUI thread):
        # progress(int percent), speed(str such as "3.2 MiB/s"), log(str status line),
        # done(), error(str). A cancel emits only log("Download cancelled.") and
        # none of done/error, so no activation is attempted, but note that nothing in
        # this panel then hides the progress strip or re-enables the card buttons.
        self._download_thread.speed.connect(self._on_download_speed)
        self._download_thread.done.connect(lambda: self._on_download_done(filename))
        self._download_thread.error.connect(self._on_download_error)
        self._download_thread.log.connect(self._on_download_log)
        self._download_thread.start()


    def _cancel_download(self):
        """Ask for confirmation and then cancel the running download."""
        if self._download_thread and self._download_thread.isRunning():
            # Confirmation first: a multi-GB download is expensive to restart. cancel() only
            # sets a flag the worker polls, so the strip stays visible until it stops.
            reply = QMessageBox.question(
                self, "Cancel Download", "Are you sure you want to cancel the download?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._download_thread.cancel()


    def _set_ui_enabled_for_download(self, enabled: bool):
        """Enable or disable every card button (except the Active badge) while a job is running."""
        # Walks the cards by layout position rather than keeping a button list, because
        # _populate_registry rebuilds the cards and a stored list would go stale.
        for i in range(self._registry_layout.count()):
            item = self._registry_layout.itemAt(i)
            if item and item.widget():
                card = item.widget()
                for btn in card.findChildren(QPushButton):
                    # The Active badge is permanently disabled, so enabling "everything" would
                    # wrongly make it clickable; it is identified by its text.
                    if btn.text() != "Active":
                        btn.setEnabled(enabled)


    def _on_download_progress(self, percent: int):
        """Move the progress bar to the given percent."""
        self._download_bar.setValue(percent)


    def _on_download_speed(self, speed_str: str):
        """Append the current speed to the status text, replacing any previous speed suffix."""
        # Speed arrives many times per second, so the old suffix is stripped by splitting
        # on " (" (the filename itself never contains that sequence) instead of
        # accumulating "(1 MB/s) (2 MB/s)...".
        status_text = self._download_status_lbl.text().split(" (")[0]
        self._download_status_lbl.setText(f"{status_text} ({speed_str})")


    def _on_download_log(self, text: str):
        """Show a status line from the worker as the progress text."""
        # Worker log lines replace the whole text (including any speed suffix), which the
        # next speed signal re-appends.
        self._download_status_lbl.setText(f"{text}")


    def _on_download_error(self, err_msg: str):
        """Hide the progress strip, re-enable the cards and show the download error."""
        self._progress_panel.setVisible(False)
        self._set_ui_enabled_for_download(True)
        # Modal on purpose: a failed model download needs the user to notice the cause
        # (network, disk space) before retrying.
        QMessageBox.critical(self, "Download Error", f"An error occurred during download:\n{err_msg}")
        self._download_thread = None


    def _on_download_done(self, filename: str):
        """After a successful download, hide the strip and activate the new model right away."""
        self._progress_panel.setVisible(False)
        self._set_ui_enabled_for_download(True)
        # A finished download is activated immediately: the user clicked Download
        # expecting to use the model. If inference is running, the activation shows its
        # own "inference active" warning and the model simply stays downloaded.
        self._activate_registry_model(filename)
        self._download_thread = None

    # ── defaults tab ──────────────────────────────────────────────────────────
