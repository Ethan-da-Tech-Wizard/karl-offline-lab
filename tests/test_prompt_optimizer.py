# ==============================================================================
# Program: test_prompt_optimizer.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Tests for core/prompt_optimizer.py — mutation and hill-climbing optimize_loop.
#          Coordinates with adjacent subsystems within the Karl Offline Lab
#          architecture.
# ==============================================================================

"""Tests for core/prompt_optimizer.py — mutation and hill-climbing optimize_loop."""

# Test scope and how it is run
#   * core/prompt_optimizer.py implements a small hill-climbing loop: evaluate the current prompt on a mini
#     validation split, collect failures on a mini training split, ask the local model to rewrite the prompt, and keep the
#     rewrite only if it scores better on validation.
#   * No real model or dataset is used. EvalHarness.run is replaced with scripted EvalReport objects and
#     ModelLoader.get_instance is mocked, so the whole file runs offline in milliseconds.
#   * Run: python -m pytest tests/test_prompt_optimizer.py -q
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.default_prompts import DEFAULT_SYSTEM_PROMPT, load_prompt_presets
from core.prompt_optimizer import PromptOptimizer
from eval.harness import CaseResult, EvalHarness, EvalReport


# Fixture builder: the optimizer only reads avg_score and the per-case grade, so one fabricated case is enough. pass_rate
# is left at 0.0 because the loop never uses it.
def _report(score: float, has_failure: bool) -> EvalReport:
    """Build a one-case EvalReport with the given score, either passing or failing, to script EvalHarness.run results."""
    # grade["detail"] is what mutate_prompt later shows to the model as the failure explanation.
    case = CaseResult(
        case_id="c0", prompt="q", workflow="general_chat", template="t",
        output="out", grader="keyword_hit",
        grade={"passed": not has_failure, "score": score, "detail": "bad" if has_failure else ""},
        latency_s=0.1, context_used=[],
    )
    return EvalReport(
        workflow="general_chat", template="t", dataset="d", total=1,
        passed=0 if has_failure else 1, failed=1 if has_failure else 0, errors=0,
        pass_rate=0.0, avg_latency_s=0.1, avg_score=score, timestamp="t", cases=[case],
    )


def _write_dataset(path: Path, n: int = 4) -> None:
    """Write n trivial JSONL eval cases to path."""
    # JSONL (one JSON object per line) matches the real eval dataset format that EvalHarness loads.
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps({"id": f"c{i}", "prompt": f"q{i}", "expected": "ok"}) + "\n")


class TestMutatePrompt(unittest.TestCase):
    """Tests for PromptOptimizer.mutate_prompt (asking the local model to rewrite a prompt from failures)."""
    def test_returns_current_prompt_when_no_failures(self):
        """With nothing to fix, the prompt is returned unchanged and the model is never needed."""
        # No failures means no signal to learn from, so the short-circuit must happen before any model call.
        optimizer = PromptOptimizer()
        self.assertEqual(optimizer.mutate_prompt("base", []), "base")

    @patch("core.prompt_optimizer.ModelLoader.get_instance")
    def test_mutates_prompt_using_failures(self, mock_get_instance):
        """The model reply is cleaned (markdown fences stripped) and returned as the new prompt, and the model is called exactly once."""
        # The mocked llama-cpp call returns the completion-style dict {"choices": [{"text": ...}]}; the fenced reply checks the
        # fence-stripping clean-up.
        mock_llm = MagicMock(return_value={"choices": [{"text": "```\nRefined prompt text.\n```"}]})
        mock_get_instance.return_value = mock_llm

        optimizer = PromptOptimizer()

        # Failure records have exactly the keys the optimizer formats into its rewrite instruction: prompt, output, detail.
        failures = [{"prompt": "2+2?", "output": "5", "detail": "wrong answer"}]
        result = optimizer.mutate_prompt("Old prompt", failures)

        self.assertEqual(result, "Refined prompt text.")

        # A single rewrite call per mutation keeps the loop's cost predictable on slow local models.
        mock_llm.assert_called_once()

    @patch("core.prompt_optimizer.ModelLoader.get_instance", side_effect=RuntimeError("no model"))
    def test_falls_back_to_current_prompt_on_load_failure(self, _mock):
        """If the model cannot be loaded the original prompt is kept rather than raising."""
        optimizer = PromptOptimizer()
        result = optimizer.mutate_prompt("Old prompt", [{"prompt": "x", "output": "y", "detail": "z"}])
        self.assertEqual(result, "Old prompt")


class TestOptimizeLoop(unittest.TestCase):
    """Tests for the hill-climbing loop PromptOptimizer.optimize_loop; EvalHarness.run is mocked with scripted scores in a fixed call order (baseline on mini-val, mini-train eval, candidate on mini-val)."""
    # The loop splits cases into mini-train and mini-val, so it needs at least one case for each.
    def test_requires_at_least_two_cases(self):
        """A dataset with one case cannot be split into train and validation sets, so ValueError is raised."""
        optimizer = PromptOptimizer()
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "cases.jsonl"

            # n=1 is the boundary just below the minimum.
            _write_dataset(dataset_path, n=1)
            with self.assertRaises(ValueError):
                optimizer.optimize_loop("_karl_default", str(dataset_path), iterations=1)

    def test_unknown_prompt_key_falls_back_to_default(self):
        """An unregistered prompt key starts from DEFAULT_SYSTEM_PROMPT instead of failing (iterations=0 returns the starting prompt)."""
        optimizer = PromptOptimizer()

        # iterations=0 means only the baseline evaluation runs; the same report is returned for every call, which is fine here.
        with patch.object(EvalHarness, "run", return_value=_report(0.5, has_failure=False)):
            with tempfile.TemporaryDirectory() as tmp:
                dataset_path = Path(tmp) / "cases.jsonl"
                _write_dataset(dataset_path)
                result = optimizer.optimize_loop("not_a_real_key", str(dataset_path), iterations=0)
        self.assertEqual(result, DEFAULT_SYSTEM_PROMPT)

    @patch.object(EvalHarness, "run")
    def test_keeps_better_candidate(self, mock_run):
        """A candidate that scores higher on validation replaces the best prompt, and progress is reported once for the baseline and once for the iteration."""
        # side_effect is consumed in call order, so the three reports below encode the exact evaluation sequence of one iteration.
        mock_run.side_effect = [
            _report(0.5, has_failure=True),   # baseline on mini-val
            _report(0.2, has_failure=True),   # mini-train eval -> has failures, triggers mutation

            # Scores are chosen so the candidate (0.9) clearly beats the baseline (0.5): the test does not depend on tie handling.
            _report(0.9, has_failure=False),  # candidate on mini-val, better than baseline
        ]
        optimizer = PromptOptimizer()

        # Replacing mutate_prompt isolates the accept/reject logic from model behaviour.
        optimizer.mutate_prompt = MagicMock(return_value="mutated prompt")

        progress_calls = []
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "cases.jsonl"
            _write_dataset(dataset_path)
            result = optimizer.optimize_loop(
                "_karl_default", str(dataset_path), iterations=1,
                progress_callback=lambda *args: progress_calls.append(args),
            )

        self.assertEqual(result, "mutated prompt")
        optimizer.mutate_prompt.assert_called_once()

        # progress_callback is invoked for the baseline (iteration 0) and once per iteration, which is what the UI progress bar relies on.
        self.assertEqual(len(progress_calls), 2)  # baseline (iter 0) + iteration 1
        self.assertEqual(progress_calls[-1][2], 0.9)  # best_score after keeping candidate

    @patch.object(EvalHarness, "run")
    def test_discards_worse_candidate(self, mock_run):
        """A lower-scoring candidate is discarded and the original preset prompt is returned (hill climbing never regresses)."""
        mock_run.side_effect = [
            _report(0.8, has_failure=True),   # baseline on mini-val
            _report(0.2, has_failure=True),   # mini-train eval -> triggers mutation
            _report(0.3, has_failure=False),  # candidate on mini-val, worse than baseline
        ]
        optimizer = PromptOptimizer()
        optimizer.mutate_prompt = MagicMock(return_value="worse mutated prompt")

        # Reads the real preset so the expected value follows the shipped default prompt instead of a hard-coded copy.
        baseline_prompt = load_prompt_presets()["_karl_default"]["prompt"]

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "cases.jsonl"
            _write_dataset(dataset_path)
            result = optimizer.optimize_loop("_karl_default", str(dataset_path), iterations=1)

        # Hill climbing must never return something worse than what it started with.
        self.assertEqual(result, baseline_prompt)

    @patch.object(EvalHarness, "run")
    def test_skips_mutation_when_no_train_failures(self, mock_run):
        """If the current prompt has no mini-train failures there is nothing to learn from, so mutate_prompt is not called."""
        mock_run.side_effect = [
            _report(0.9, has_failure=False),  # baseline on mini-val
            _report(1.0, has_failure=False),  # mini-train eval -> no failures, nothing to mutate
        ]
        optimizer = PromptOptimizer()

        # No return value is configured: if the code wrongly called it, assert_not_called below would catch it.
        optimizer.mutate_prompt = MagicMock()

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "cases.jsonl"
            _write_dataset(dataset_path)
            optimizer.optimize_loop("_karl_default", str(dataset_path), iterations=1)

        optimizer.mutate_prompt.assert_not_called()


if __name__ == "__main__":
    unittest.main()
