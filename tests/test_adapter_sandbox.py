# Program: karl-offline-lab - test_adapter_sandbox.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Unit tests for adapter evaluation and the adapter sandbox benchmark and generation workers.
"""
Unit Tests for Adapter Sandbox & Speculative Evaluation
=========================================================
Verifies evaluate_model_performance and visual PyQt6 sandbox components.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from tools.evaluate_adapters import evaluate_model_performance, extract_code_block
from app.ui.workspaces.system_config.adapter_sandbox import BenchmarkWorker, GenWorker


class TestAdapterSandbox(unittest.TestCase):
    
    """Tests for code-block extraction, model performance evaluation and the sandbox worker threads."""
    @classmethod
    def setUpClass(cls):
        # Create QApplication instance for PyQt tests
        """Create (or reuse) the QApplication that the Qt workers need."""
        # Reuse or create the QApplication instance for PyQt tests
        cls.app = QApplication.instance() or QApplication([])

    def test_extract_code_block(self):
        """Scenario: text with and without a fenced code block. Expected: the code inside the fence, or the text itself when no fence exists."""
        text_with_fences = "Some thought\n```python\nprint('hello')\n```\nSome footer"
        self.assertEqual(extract_code_block(text_with_fences), "print('hello')")
        
        text_no_fences = "print('hello')"
        self.assertEqual(extract_code_block(text_no_fences), "print('hello')")

    @patch("app.engine.model_loader.ModelLoader.get_instance")
    @patch("app.engine.model_loader.ModelLoader.reset_instance")
    def test_evaluate_model_performance_tps(self, mock_reset, mock_get_instance):
        """Scenario: a mocked model returns valid Python for one case. Expected: stats report the adapter name, one case, positive tokens per second and 100 percent syntax accuracy."""
        mock_llm = MagicMock()
        mock_llm.return_value = [{"choices": [{"text": "```python\ndef add(a, b): return a + b\n```"}]}]
        mock_llm.tokenize.return_value = [1, 2, 3, 4, 5]
        mock_get_instance.return_value = mock_llm
        
        stats = evaluate_model_performance(adapter_name="math_adapter", dataset_path=None, limit=1)
        
        self.assertEqual(stats["adapter"], "math_adapter")
        self.assertEqual(stats["eval_cases"], 1)
        self.assertTrue(stats["avg_tps"] > 0.0)
        self.assertEqual(stats["syntax_accuracy"], 100.0)

    @patch("subprocess.run")
    def test_benchmark_worker_success(self, mock_run):
        """Scenario: the benchmark subprocess succeeds and a report file exists. Expected: the worker emits the report contents once through its finished signal."""
        mock_run.return_value = MagicMock(returncode=0)
        
        # Mock writing of report file
        report_data = {
            "adapter_name": "math_adapter",
            "baseline": {"avg_tps": 10.0, "syntax_accuracy": 80.0},
            "adapted": {"avg_tps": 15.0, "syntax_accuracy": 90.0, "vram_delta_mb": 50.0},
            "tps_improvement_percent": 50.0,
            "accuracy_improvement_percent": 10.0
        }
        
        # Write a fake report where the worker expects to find it
        report_path = Path("data/adapters") / "eval_report_math_adapter.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(report_data, fh)
            
        try:
            worker = BenchmarkWorker("math_adapter")
            
            # Setup signal handlers
            results = []
            worker.finished.connect(results.append)
            
            worker.run()
            
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["adapter_name"], "math_adapter")
            self.assertEqual(results[0]["tps_improvement_percent"], 50.0)
        finally:
            if report_path.exists():
                # Remove the fake report so later runs start clean
                report_path.unlink()

    @patch("app.engine.model_loader.ModelLoader.get_instance")
    def test_gen_worker_tps(self, mock_get_instance):
        """Scenario: a mocked model generates text for a prompt. Expected: finished emits once with the text, the token count and a non-negative elapsed time."""
        mock_llm = MagicMock()
        mock_llm.return_value = [{"choices": [{"text": "print('speed test')"}]}]
        mock_llm.tokenize.return_value = [1, 2, 3]
        mock_get_instance.return_value = mock_llm
        
        worker = GenWorker("hello", "test_adapter")
        
        finished_signals = []
        worker.finished.connect(lambda txt, elap, tok: finished_signals.append((txt, elap, tok)))
        
        worker.run()
        
        self.assertEqual(len(finished_signals), 1)
        txt, elap, tok = finished_signals[0]
        self.assertEqual(txt, "print('speed test')")
        self.assertEqual(tok, 3)
        self.assertTrue(elap >= 0.0)
