# ==============================================================================
# Program: adapter_sandbox.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Adapter Sandbox Mixin Panel =========================== Provides UI controls to list, load, and benchmark trained LoRA adapters.
#          Coordinates with adjacent subsystems within the Karl Offline Lab
#          architecture.
# ==============================================================================
# This module is a mixin: SystemConfigWorkspace inherits it, so `self` here is
# the workspace (it supplies the adapter_changed signal). Both slow operations
# (benchmark, generation) run in QThread workers so the GUI never blocks.

"""
Adapter Sandbox Mixin Panel
===========================
Provides UI controls to list, load, and benchmark trained LoRA adapters.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QTextEdit, QFormLayout, QGroupBox, QTableWidget,
    QTableWidgetItem, QHeaderView
)

from app.engine.model_loader import ModelLoader

logger = logging.getLogger("karl.adapter_sandbox")


# The benchmark is delegated to a separate script/process rather than run
# in-process: it loads its own model copies and would otherwise contend with
# the model the UI already holds in memory.
class BenchmarkWorker(QThread):
    """Runs tools/evaluate_adapters.py in a background thread."""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, adapter_name: str, parent=None):
        """Store the adapter name to benchmark; the thread starts later via start()."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(parent)
        self.adapter_name = adapter_name

    def run(self):
        """Run the evaluation script, then emit the parsed JSON report (or an error message)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        try:
            # sys.executable keeps the child in the same virtualenv; --limit 3 keeps the run short.
            # Execute benchmark subprocess
            cmd = [sys.executable, "tools/evaluate_adapters.py", "--adapter", self.adapter_name, "--limit", "3"]
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            # The script writes its result to disk; read it back instead of parsing stdout.
            # Parse output json report
            report_path = Path("data/adapters") / f"eval_report_{self.adapter_name}.json"
            if report_path.exists():
                with report_path.open("r", encoding="utf-8") as fh:
                    report = json.load(fh)
                self.finished.emit(report)
            else:
                self.error.emit("Benchmark completed but report file was not written.")
        except Exception as exc:
            self.error.emit(str(exc))


class GenWorker(QThread):
    """Runs a quick generation for speed testing in a background thread."""
    finished = pyqtSignal(str, float, int)  # text, elapsed, tokens
    error = pyqtSignal(str)

    def __init__(self, prompt: str, adapter_name: str | None, parent=None):
        """Store the prompt and adapter name (None means the baseline model)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(parent)
        self.prompt = prompt
        self.adapter_name = adapter_name

    def run(self):
        """Stream a short completion from the loaded model and emit text, elapsed seconds and token count."""
        # Step 1: Validate input parameters, operational preconditions, and state
        try:
            # get_instance returns the shared singleton, reloading only if the adapter differs.
            # Load model handle
            llm = ModelLoader.get_instance(adapter_name=self.adapter_name)
            
            # ChatML-style template with an opened <think> block, matching the model's training format.
            # Format prompt template
            full_prompt = f"<|im_start|>user\n{self.prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
            
            import time
            start = time.perf_counter()
            
            # Chunks are collected and joined so elapsed time covers pure generation only.
            chunks = []
            for chunk in llm(
                full_prompt,
                max_tokens=128,
                temperature=0.2,
                top_p=0.95,
                stream=True,
                stop=["<|im_end|>", "<|endoftext|>"]
            ):
                # Iterate over items, evaluate loop invariant, and update progress
                if "choices" in chunk and chunk["choices"]:
                    chunks.append(chunk["choices"][0].get("text", ""))
                    
            # Timing stops before tokenization so the TPS figure is not inflated by token counting.
            elapsed = time.perf_counter() - start
            text = "".join(chunks)
            
            # Prefer the real tokenizer count; fall back to a whitespace word count if unavailable.
            # Count tokens using the tokenizer
            try:
                # Execute guarded operations and establish recovery boundaries
                tokens = llm.tokenize(text.encode("utf-8"))
                tok_len = len(tokens)
            except Exception:
                tok_len = len(text.split())
                
            self.finished.emit(text, elapsed, tok_len)
        except Exception as exc:
            self.error.emit(str(exc))


class AdapterSandboxMixin:
    """Mixin that builds the visual tab for LoRA hot-reloading and benchmarks."""

    def _build_adapter_sandbox_tab(self) -> QWidget:
        """Build the Adapter Sandbox tab: adapter picker, speed test, and benchmark table."""
        # Step 1: Validate input parameters, operational preconditions, and state
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Group 1: choose an adapter and activate it or reset to the base model.
        # 1. Config Group
        cfg_group = QGroupBox("Trained LoRA Configuration")
        cfg_layout = QFormLayout(cfg_group)

        self._sandbox_adapter_combo = QComboBox()
        cfg_layout.addRow("Select Adapter:", self._sandbox_adapter_combo)

        btn_bar = QHBoxLayout()
        self._btn_sandbox_load = QPushButton("Activate Adapter")
        self._btn_sandbox_load.clicked.connect(self._on_sandbox_load)
        btn_bar.addWidget(self._btn_sandbox_load)

        self._btn_sandbox_unload = QPushButton("Unload Adapter (Reset Baseline)")
        self._btn_sandbox_unload.clicked.connect(self._on_sandbox_unload)
        btn_bar.addWidget(self._btn_sandbox_unload)

        cfg_layout.addRow("", btn_bar)
        layout.addWidget(cfg_group)

        # Group 2: run a short generation and display tokens-per-second.
        # 2. Speculative Test Run Group
        test_group = QGroupBox("Speculative Generation Test (Speed & TPS)")
        test_layout = QVBoxLayout(test_group)

        self._sandbox_prompt_input = QTextEdit()
        self._sandbox_prompt_input.setPlaceholderText("Enter test prompt to measure speed (e.g. Write a quick python binary search)...")
        self._sandbox_prompt_input.setMaximumHeight(80)
        test_layout.addWidget(self._sandbox_prompt_input)

        self._btn_sandbox_run = QPushButton("Run Generation Test")
        self._btn_sandbox_run.clicked.connect(self._on_sandbox_run)
        test_layout.addWidget(self._btn_sandbox_run)

        self._sandbox_output_area = QTextEdit()
        self._sandbox_output_area.setReadOnly(True)
        self._sandbox_output_area.setPlaceholderText("Generation output will appear here...")
        self._sandbox_output_area.setMaximumHeight(150)
        test_layout.addWidget(self._sandbox_output_area)

        self._lbl_sandbox_stats = QLabel("Speed Stats: Not tested.")
        self._lbl_sandbox_stats.setStyleSheet("font-weight: bold;")
        test_layout.addWidget(self._lbl_sandbox_stats)

        layout.addWidget(test_group)

        # Group 3: baseline-vs-adapted benchmark results table.
        # 3. Benchmarks Group
        bench_group = QGroupBox("Accuracy & Performance Benchmarks")
        bench_layout = QVBoxLayout(bench_group)

        self._btn_sandbox_benchmark = QPushButton("Run Adapter Speculative Benchmark")
        self._btn_sandbox_benchmark.clicked.connect(self._on_sandbox_benchmark)
        bench_layout.addWidget(self._btn_sandbox_benchmark)

        self._table_sandbox_results = QTableWidget()
        self._table_sandbox_results.setColumnCount(4)
        self._table_sandbox_results.setHorizontalHeaderLabels([
            "Configuration", "TPS (Speed)", "Syntax Accuracy", "VRAM Footprint"
        ])
        self._table_sandbox_results.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table_sandbox_results.setMaximumHeight(120)
        bench_layout.addWidget(self._table_sandbox_results)

        self._lbl_bench_improvement = QLabel("Improvement Stats: Run benchmark to measure.")
        self._lbl_bench_improvement.setStyleSheet("font-weight: bold; color: #2DD4A0;")
        bench_layout.addWidget(self._lbl_bench_improvement)

        layout.addWidget(bench_group)
        
        # Populate adapters list
        self._scan_sandbox_adapters()
        return tab

    # Called from the tab build and again each time the workspace is shown.
    def _scan_sandbox_adapters(self):
        """Scrapes data/adapters/ directories and updates combo."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._sandbox_adapter_combo.clear()
        adapter_dir = Path("data/adapters")
        if not adapter_dir.exists():
            return
            
        adapters = [d.name for d in adapter_dir.iterdir() if d.is_dir()]
        self._sandbox_adapter_combo.addItems(sorted(adapters))

    def _on_sandbox_load(self):
        """Hot-load the selected LoRA adapter into the shared ModelLoader and announce the change."""
        # Step 1: Validate input parameters, operational preconditions, and state
        adapter = self._sandbox_adapter_combo.currentText()
        if not adapter:
            return
            
        logger.info("Activating LoRA adapter '%s'...", adapter)
        self._btn_sandbox_load.setEnabled(False)
        self._btn_sandbox_load.setText("Loading...")
        
        try:
            # Execute guarded operations and establish recovery boundaries
            # Reset first so the loader rebuilds the model with the new adapter attached.
            ModelLoader.reset_instance()
            ModelLoader.get_instance(adapter_name=adapter)
            # Propagate signals to update UI status bars and state
            self.adapter_changed.emit(adapter)
            logger.info("Successfully activated adapter '%s'", adapter)
        except Exception as exc:
            logger.error("Failed to load adapter: %s", exc)
        finally:
            self._btn_sandbox_load.setEnabled(True)
            self._btn_sandbox_load.setText("Activate Adapter")

    def _on_sandbox_unload(self):
        """Drop any active adapter and reload the plain base model."""
        # Step 1: Validate input parameters, operational preconditions, and state
        logger.info("Unloading active LoRA adapter...")
        try:
            # Execute guarded operations and establish recovery boundaries
            ModelLoader.reset_instance()
            ModelLoader.get_instance(adapter_name=None)
            self.adapter_changed.emit("")
            logger.info("Successfully reset baseline model.")
        except Exception as exc:
            logger.error("Failed to unload adapter: %s", exc)

    def _on_sandbox_run(self):
        """Start a background generation test with the entered prompt and selected adapter."""
        # Step 1: Validate input parameters, operational preconditions, and state
        prompt = self._sandbox_prompt_input.toPlainText().strip()
        if not prompt:
            return
            
        adapter = self._sandbox_adapter_combo.currentText()
        self._btn_sandbox_run.setEnabled(False)
        self._btn_sandbox_run.setText("Generating...")
        self._sandbox_output_area.clear()
        
        self._gen_worker = GenWorker(prompt, adapter, self)
        self._gen_worker.finished.connect(self._on_gen_finished)
        self._gen_worker.error.connect(self._on_gen_error)
        self._gen_worker.start()

    def _on_gen_finished(self, text: str, elapsed: float, tokens: int):
        """Show generated text and compute tokens per second (guarding against zero elapsed)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._btn_sandbox_run.setEnabled(True)
        self._btn_sandbox_run.setText("Run Generation Test")
        self._sandbox_output_area.setText(text)
        tps = tokens / max(elapsed, 0.001)
        self._lbl_sandbox_stats.setText(
            f"Speed Stats: Generated {tokens} tokens in {elapsed:.2f}s | Speed: {tps:.2f} tokens/sec (TPS)"
        )

    def _on_gen_error(self, err_msg: str):
        """Re-enable the run button and display the generation error."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._btn_sandbox_run.setEnabled(True)
        self._btn_sandbox_run.setText("Run Generation Test")
        self._sandbox_output_area.setText(f"Error occurred: {err_msg}")

    def _on_sandbox_benchmark(self):
        """Start the adapter benchmark worker for the selected adapter."""
        # Step 1: Validate input parameters, operational preconditions, and state
        adapter = self._sandbox_adapter_combo.currentText()
        if not adapter:
            return
            
        self._btn_sandbox_benchmark.setEnabled(False)
        self._btn_sandbox_benchmark.setText("Benchmarking...")
        
        self._bench_worker = BenchmarkWorker(adapter, self)
        self._bench_worker.finished.connect(self._on_bench_finished)
        self._bench_worker.error.connect(self._on_bench_error)
        self._bench_worker.start()

    def _on_bench_finished(self, report: dict):
        """Fill the two-row results table (baseline vs adapted) from the benchmark report."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._btn_sandbox_benchmark.setEnabled(True)
        self._btn_sandbox_benchmark.setText("Run Adapter Speculative Benchmark")
        
        # Populate results table
        self._table_sandbox_results.setRowCount(2)
        
        # Row 1: Baseline
        # Report keys come from tools/evaluate_adapters.py; baseline has no VRAM delta, hence 0.00 MB.
        base = report["baseline"]
        self._table_sandbox_results.setItem(0, 0, QTableWidgetItem("Baseline Model"))
        self._table_sandbox_results.setItem(0, 1, QTableWidgetItem(f"{base['avg_tps']} TPS"))
        self._table_sandbox_results.setItem(0, 2, QTableWidgetItem(f"{base['syntax_accuracy']}%"))
        self._table_sandbox_results.setItem(0, 3, QTableWidgetItem("0.00 MB"))
        
        # Row 2: Adapted
        adap = report["adapted"]
        self._table_sandbox_results.setItem(1, 0, QTableWidgetItem(f"Adapted ({report['adapter_name']})"))
        self._table_sandbox_results.setItem(1, 1, QTableWidgetItem(f"{adap['avg_tps']} TPS"))
        self._table_sandbox_results.setItem(1, 2, QTableWidgetItem(f"{adap['syntax_accuracy']}%"))
        self._table_sandbox_results.setItem(1, 3, QTableWidgetItem(f"{adap['vram_delta_mb']} MB"))
        
        # Update summary label
        self._lbl_bench_improvement.setText(
            f"Improvement Stats: Speed delta: {report['tps_improvement_percent']}% | "
            f"Syntax Accuracy delta: {report['accuracy_improvement_percent']}%"
        )

    def _on_bench_error(self, err_msg: str):
        """Re-enable the benchmark button and show the error in red."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._btn_sandbox_benchmark.setEnabled(True)
        self._btn_sandbox_benchmark.setText("Run Adapter Speculative Benchmark")
        self._lbl_bench_improvement.setText(f"Benchmark error: {err_msg}")
        self._lbl_bench_improvement.setStyleSheet("font-weight: bold; color: #FF4D4D;")
