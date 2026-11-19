# ==============================================================================
# Program: quantizer_thread.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Background worker running local model quantization pipelines.
#          Converts full-precision FP16 weights into compact GGUF quantization
#          formats (Q4_K_M, Q8_0, etc.).
# ==============================================================================

"""
QuantizerThread — background GGUF quantization via llama-quantize CLI
======================================================================
Wraps the `llama-quantize` binary in a QThread so Karl's UI stays
responsive during potentially long quantization runs.

Signals
-------
progress(int)  — completion percentage [0–100] derived from layer counters
done(str)      — output_path on success (exit code 0)
error(str)     — human-readable failure message
"""

import os
import re
import shutil
import subprocess

from PyQt6.QtCore import QThread, pyqtSignal

import logging

logger = logging.getLogger("karl.quantizer_thread")

# ── Binary search ─────────────────────────────────────────────────────────────

# Likely locations of the llama-quantize binary (build output, install dirs).
# Relative entries are resolved against the current working directory.
_CANDIDATE_DIRS = [
    "build/bin",          # cmake out-of-source build (relative to project root)
    "build",
    "llama.cpp/build/bin",
    "llama.cpp/build",
    "/usr/local/bin",
    "/usr/bin",
    "/opt/homebrew/bin",  # macOS Homebrew
]

# Executable name (no .exe handling here).
_BINARY_NAME = "llama-quantize"


def _locate_llama_quantize(input_path: str | None = None) -> str | None:
    """
    Return the absolute path to llama-quantize, or None if not found.

    Search order:
      1. Same directory as *input_path* (models may live next to llama.cpp tools)
      2. Candidate build/install directories relative to the working directory
      3. System PATH via shutil.which
    """
    candidates: list[str] = []

    if input_path:
        model_dir = os.path.dirname(os.path.abspath(input_path))
        candidates.append(os.path.join(model_dir, _BINARY_NAME))

    # Candidates are tried in order; the model's own folder was added first so it wins.
    for d in _CANDIDATE_DIRS:
        candidates.append(os.path.join(os.getcwd(), d, _BINARY_NAME))

    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    # Last resort: whatever shutil.which finds on PATH.
    # Fall through to system PATH
    return shutil.which(_BINARY_NAME)


# ── Progress regex ─────────────────────────────────────────────────────────────
# llama-quantize emits layer progress as: [  1/ 291] blk.0.attn_norm ...
# Capture both the bracketed fraction and bare "NN%" patterns.

# Layer counter like "[  1/ 291]" gives a precise fraction.
_RE_BRACKET = re.compile(r"\[\s*(\d+)\s*/\s*(\d+)\s*\]")
# Fallback for lines that print a bare percentage.
_RE_PERCENT = re.compile(r"\b(\d{1,3})\s*%")


def _parse_progress(line: str) -> int | None:
    """
    Extract a 0-100 integer progress value from a single llama-quantize output line.
    Returns None if the line carries no recognisable progress information.
    """
    m = _RE_BRACKET.search(line)
    if m:
        current, total = int(m.group(1)), int(m.group(2))
        if total > 0:
            return min(100, int(current / total * 100))

    m = _RE_PERCENT.search(line)
    if m:
        return min(100, int(m.group(1)))

    return None


# ── Thread ─────────────────────────────────────────────────────────────────────

class QuantizerThread(QThread):
    """
    Background thread that runs llama-quantize and emits real-time progress.

    Parameters
    ----------
    input_path   : str   — path to the source GGUF file
    output_path  : str   — destination path for the quantized GGUF
    target_format: str   — quantization type token (e.g. "Q5_K_M", "Q4_K_M")
    """

    progress = pyqtSignal(int)   # 0–100
    done     = pyqtSignal(str)   # output_path
    error    = pyqtSignal(str)   # failure message

    def __init__(
        self,
        input_path: str,
        output_path: str,
        target_format: str,
    ) -> None:
        """Remember the source/destination GGUF paths and quantization type; no work happens until start()."""
        super().__init__()
        self.input_path    = input_path
        self.output_path   = output_path
        self.target_format = target_format
        # Handle to the child process so cancel() can terminate it from another thread.
        self._proc: subprocess.Popen | None = None

    # ── public API ────────────────────────────────────────────────────────────

    def cancel(self) -> None:
        """Request cancellation by terminating the child process."""
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except OSError:
                pass

    # ── thread entry point ────────────────────────────────────────────────────

    def run(self) -> None:
        """Thread body: locate the binary, validate input, launch it, stream progress, report the result."""
        # ── locate binary ─────────────────────────────────────────────────────
        binary = _locate_llama_quantize(self.input_path)
        if not binary:
            self.error.emit(
                "llama-quantize binary not found.\n\n"
                "Build llama.cpp (cmake -B build && cmake --build build -t llama-quantize) "
                "and ensure build/bin/llama-quantize is present, "
                "or install it on PATH."
            )
            return

        # ── validate input ────────────────────────────────────────────────────
        if not os.path.isfile(self.input_path):
            self.error.emit(f"Input file not found: {self.input_path}")
            return

        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)

        # ── build command ─────────────────────────────────────────────────────
        cmd = [binary, self.input_path, self.output_path, self.target_format]
        logger.info("QuantizerThread: launching %s", " ".join(cmd))

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                # Merge stderr so progress lines printed there are also parsed.
                stderr=subprocess.STDOUT,  # merge stderr into stdout
                text=True,
                bufsize=1,                 # line-buffered
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except OSError as exc:
            self.error.emit(f"Failed to start llama-quantize: {exc}")
            return

        # ── stream stdout, parse progress ─────────────────────────────────────
        # Only emit when the percentage changes, to avoid flooding the UI with duplicate updates.
        last_pct = -1
        try:
            for raw_line in self._proc.stdout:
                line = raw_line.rstrip()
                if not line:
                    continue
                logger.debug("quantize | %s", line)

                pct = _parse_progress(line)
                if pct is not None and pct != last_pct:
                    last_pct = pct
                    self.progress.emit(pct)
        except Exception as exc:
            logger.warning("Stream read error: %s", exc)

        # ── wait for process to exit ──────────────────────────────────────────
        # stdout is exhausted, so the process has exited or is about to.
        exit_code = self._proc.wait()
        self._proc = None

        if exit_code == 0:
            self.progress.emit(100)
            self.done.emit(self.output_path)
        else:
            self.error.emit(
                f"llama-quantize exited with code {exit_code}.\n"
                f"Check that the source GGUF is valid and that '{self.target_format}' "
                f"is a supported quantization type for this model."
            )
