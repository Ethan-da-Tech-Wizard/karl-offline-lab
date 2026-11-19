# ==============================================================================
# Program: download_threads.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Desktop workspace implementation for Download Threads.
#          Provides responsive layout controls, parameter adjustment surfaces,
#          and visual feedback components within Karl's PyQt6 shell.
# ==============================================================================
# Worker threads and a small dialog used by the System workspace: a quantization
# worker, a format-picker dialog, and a model downloader. Workers report only via
# Qt signals so the GUI thread is never blocked.

from __future__ import annotations

import os

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QLabel, QVBoxLayout,
)


# ── quantization thread ───────────────────────────────────────────────────────

class QuantizationThread(QThread):
    """QThread that simulates a quantization run, emitting progress, log lines and a final output path."""
    # progress: percent 0-100; done: output path; error: message; log: status text.
    progress = pyqtSignal(int)
    done = pyqtSignal(str)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, input_path: str, output_path: str, quant_format: str):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.quant_format = quant_format

    def run(self):
        try:
            self.log.emit(f"Starting quantization to {self.quant_format}...")
            # NOTE: this is a placeholder simulation (sleep loop), not a real GGUF requantization;
            # the real work is done by app.engine.quantizer_thread.QuantizerThread.
            # Simulation of quantization heavy-lift
            for i in range(1, 101):
                import time
                time.sleep(0.06) 
                self.progress.emit(i)
                if i % 20 == 0:
                    self.log.emit(f"Compiling custom GGUF weights... {i}%")
            self.done.emit(self.output_path)
        except Exception as e:
            self.error.emit(str(e))


# ── quantization dialog ───────────────────────────────────────────────────────

class QuantizationDialog(QDialog):
    """Modal dialog asking which quantization format to use for a model."""
    def __init__(self, model_name: str, parent=None):
        """Build the dialog for the named model with the supported formats listed."""
        super().__init__(parent)
        self.setWindowTitle(f"Quantize {model_name}")
        self.setMinimumWidth(380)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        layout.addWidget(QLabel(f"Select desired quantization format for:<br/><b>{model_name}</b>"))
        
        self.format_combo = QComboBox()
        # Formats offered, ordered by size: Q4_K_M (smallest) through Q8_0 (near lossless).
        self.format_combo.addItems(["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"])
        layout.addWidget(self.format_combo)
        
        info = QLabel(
            "• <b>Q4_K_M</b>: Best balance of speed and logic accuracy.<br/>"
            "• <b>Q5_K_M</b>: High fidelity, requires more VRAM.<br/>"
            "• <b>Q8_0</b>: Near-lossless weights, very heavy."
        )
        info.setObjectName("lbl-muted")
        info.setTextFormat(Qt.TextFormat.RichText)
        info.setWordWrap(True)
        layout.addWidget(info)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_format(self) -> str:
        """Return the format chosen in the combo box, for example Q4_K_M."""
        return self.format_combo.currentText()

# ── download thread ───────────────────────────────────────────────────────────

class DownloadThread(QThread):
    """QThread that streams a model file from a URL to disk and reports progress, speed and completion."""
    # progress: percent; speed: formatted MiB/s string; done: finished OK; error: message.
    progress = pyqtSignal(int)
    speed = pyqtSignal(str)
    done = pyqtSignal()
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, url: str, target_path: str):
        """Remember the source URL and destination path; start() runs the transfer."""
        super().__init__()
        self.url = url
        self.target_path = target_path
        self._is_cancelled = False

    def cancel(self):
        """Ask the download to stop at the next chunk boundary."""
        self._is_cancelled = True

    def request_stop(self):
        """Alias of cancel() so the task supervisor can stop this thread through a common interface."""
        self.cancel()

    def run(self):
        """Download to a .tmp file, then atomically move it into place, reporting to the TaskSupervisor."""
        # Imports are local to keep this module cheap to import and avoid cycles; requests
        # is only needed when a download actually runs.
        from app.engine.task_supervisor import TaskSupervisor
        task_id = TaskSupervisor.instance().register(
            name=f"Download Model: {os.path.basename(self.target_path)}",
            cancellable=self,
        )
        self.task_id = task_id
        import time
        import requests
        
        # Write to <target>.tmp first so a partial or failed download never replaces a good model file.
        tmp_path = self.target_path + ".tmp"
        try:
            self.log.emit("Connecting...")
            response = requests.get(self.url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            os.makedirs(os.path.dirname(self.target_path), exist_ok=True)
            
            start_time = time.time()
            last_time = start_time
            last_downloaded = 0
            
            self.log.emit("Downloading...")
            with open(tmp_path, "wb") as f:
                # 1 MiB chunks balance progress granularity against per-chunk overhead.
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    # Cancellation is checked per chunk, so it can lag by up to one chunk.
                    if self._is_cancelled:
                        self.log.emit("Download cancelled.")
                        break
                    
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # content-length can be missing (0), in which case percent progress is skipped.
                        if total_size > 0:
                            percent = int((downloaded / total_size) * 100)
                            self.progress.emit(percent)
                            TaskSupervisor.instance().update_progress(task_id, downloaded / total_size)
                        
                        current_time = time.time()
                        time_diff = current_time - last_time
                        # Speed is recomputed at most once per second to keep the UI label steady.
                        if time_diff >= 1.0:
                            bytes_diff = downloaded - last_downloaded
                            speed_mb = (bytes_diff / (1024 * 1024)) / time_diff
                            self.speed.emit(f"{speed_mb:.1f} MiB/s")
                            last_time = current_time
                            last_downloaded = downloaded
            
            # On cancel remove the partial file; on success swap it into place (replacing any old file).
            if self._is_cancelled:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                TaskSupervisor.instance().fail(task_id, "Cancelled by user")
            else:
                self.log.emit("Finalizing model file...")
                if os.path.exists(self.target_path):
                    os.remove(self.target_path)
                os.rename(tmp_path, self.target_path)
                self.done.emit()
                TaskSupervisor.instance().finish(task_id)
                
        # Any failure discards the temp file and reports to both the supervisor and the UI.
        except Exception as e:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            TaskSupervisor.instance().fail(task_id, str(e))
            self.error.emit(str(e))
