# Program: karl-offline-lab - test_swarm_replay.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Unit tests for SwarmReplayManager, checking run listing, ordering, malformed-file handling and the step timeline with diffs built from saved run traces.
"""Tests for app/utils/swarm_replay.py — SwarmReplayManager."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.utils.swarm_replay import SwarmReplayManager


def _write_run(cognition_dir: Path, run_id: str, nodes: list[dict]) -> Path:
    """Write a list of trace nodes to a run file named after the run id and return its path."""
    path = cognition_dir / f"run_{run_id}.json"
    path.write_text(json.dumps(nodes), encoding="utf-8")
    return path


class TestListPastRuns(unittest.TestCase):
    """Tests for summarizing every saved run in the trace directory."""
    def setUp(self):
        """Create a temporary trace directory and a replay manager that reads it."""
        self._tmp = tempfile.TemporaryDirectory()
        self.cognition_dir = Path(self._tmp.name)
        self.manager = SwarmReplayManager(trace_dir=str(self.cognition_dir))

    def tearDown(self):
        """Remove the temporary directory after each test."""
        self._tmp.cleanup()

    def test_returns_empty_list_when_dir_missing(self):
        """Scenario: the trace directory does not exist.
        Expected: an empty list of runs.
        """
        manager = SwarmReplayManager(trace_dir=str(self.cognition_dir / "does_not_exist"))
        self.assertEqual(manager.list_past_runs(), [])

    def test_summarizes_run_started_and_swarm_finished(self):
        """Scenario: a run with start, plan and finish nodes.
        Expected: one summary with the run id, objective, success flag and a step count of 3.
        """
        _write_run(self.cognition_dir, "abc123", [
            {"type": "run_started", "timestamp": 1000.0, "objective": "Add math_utils.add", "workspace_path": "/ws"},
            {"type": "architect_plan", "timestamp": 1001.0, "explanation": "plan", "tasks": []},
            {"type": "swarm_finished", "timestamp": 1010.0, "success": True, "summary": "Modified files: math_utils.py"},
        ])

        runs = self.manager.list_past_runs()

        self.assertEqual(len(runs), 1)
        run = runs[0]
        self.assertEqual(run["run_id"], "abc123")
        self.assertEqual(run["objective"], "Add math_utils.add")
        self.assertTrue(run["success"])
        self.assertEqual(run["steps_count"], 3)

    def test_run_without_finished_node_reports_not_successful(self):
        """Scenario: a run that never wrote a finish node.
        Expected: the run is reported as not successful.
        """
        _write_run(self.cognition_dir, "unfinished", [
            {"type": "run_started", "timestamp": 1000.0, "objective": "x", "workspace_path": "/ws"},
        ])

        runs = self.manager.list_past_runs()

        self.assertEqual(runs[0]["success"], False)

    def test_sorted_newest_first(self):
        """Scenario: two runs with different start timestamps.
        Expected: the run with the later timestamp is listed first.
        """
        _write_run(self.cognition_dir, "older", [
            {"type": "run_started", "timestamp": 1000.0, "objective": "old run", "workspace_path": "/ws"},
        ])
        _write_run(self.cognition_dir, "newer", [
            {"type": "run_started", "timestamp": 2000.0, "objective": "new run", "workspace_path": "/ws"},
        ])

        runs = self.manager.list_past_runs()

        self.assertEqual([r["run_id"] for r in runs], ["newer", "older"])

    def test_skips_malformed_json(self):
        """Scenario: one run file holds invalid JSON next to a valid one.
        Expected: only the valid run is listed.
        """
        bad_path = self.cognition_dir / "run_broken.json"
        bad_path.write_text("{not valid json", encoding="utf-8")
        _write_run(self.cognition_dir, "good", [
            {"type": "run_started", "timestamp": 1000.0, "objective": "ok", "workspace_path": "/ws"},
        ])

        runs = self.manager.list_past_runs()

        self.assertEqual([r["run_id"] for r in runs], ["good"])


class TestGetRunDetails(unittest.TestCase):
    """Tests for rebuilding a single run as a timeline of steps with diffs."""
    def setUp(self):
        """Create temporary trace and workspace directories and a replay manager."""
        self._tmp = tempfile.TemporaryDirectory()
        self.cognition_dir = Path(self._tmp.name) / "cognition"
        self.cognition_dir.mkdir()
        self.workspace_dir = Path(self._tmp.name) / "workspace"
        self.workspace_dir.mkdir()
        self.manager = SwarmReplayManager(trace_dir=str(self.cognition_dir))

    def tearDown(self):
        """Remove the temporary directories after each test."""
        self._tmp.cleanup()

    def test_missing_run_returns_empty_steps(self):
        """Scenario: the requested run id has no saved file.
        Expected: a result with the id, an empty objective and no steps.
        """
        details = self.manager.get_run_details("nope")
        self.assertEqual(details, {"run_id": "nope", "objective": "", "steps": []})

    def test_builds_step_timeline_with_diff_and_drift_flag(self):
        """Scenario: a run with a plan, a drift event, a file write with a backup, and a test result.
        Expected: steps in order architect, coder, coder, tester; coder steps flagged as drift; the write step carries a unified diff; the tester step carries the test output.
        """
        # Current on-disk content (the "after" state)
        # Current on-disk content: this is the "after" state of the edit.
        current_file = self.workspace_dir / "math_utils.py"
        current_file.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

        # Pre-edit backup (the "before" state)
        # Pre-edit backup: this is the "before" state used to build the diff.
        backup_file = Path(self._tmp.name) / "backup" / "math_utils.py"
        backup_file.parent.mkdir(parents=True)
        backup_file.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")

        _write_run(self.cognition_dir, "run1", [
            {"type": "run_started", "timestamp": 1000.0, "objective": "Fix add()", "workspace_path": str(self.workspace_dir)},
            {"type": "architect_plan", "timestamp": 1001.0, "explanation": "Fix add()", "tasks": [{"filepath": "math_utils.py"}]},
            {"type": "drift_detected", "timestamp": 1002.0, "filepath": "math_utils.py", "reason": "identical failure trace"},
            {
                "type": "file_written", "timestamp": 1003.0, "layer_index": 1,
                "filepath": "math_utils.py", "backup_path": str(backup_file),
            },
            {"type": "test_result", "timestamp": 1004.0, "layer_index": 1, "passed": True, "trace": "TESTS PASSED"},
            {"type": "swarm_finished", "timestamp": 1005.0, "success": True, "summary": "Modified files: math_utils.py"},
        ])

        details = self.manager.get_run_details("run1")

        self.assertEqual(details["run_id"], "run1")
        self.assertEqual(details["objective"], "Fix add()")

        # The drift node is shown as a coder step, so two coder steps appear between architect and tester.
        step_types = [s["type"] for s in details["steps"]]
        self.assertEqual(step_types, ["architect", "coder", "coder", "tester"])

        coder_steps = [s for s in details["steps"] if s["filepath"] == "math_utils.py"]
        # Both the drift_detected step and the file_written step reference this file,
        # and both should be flagged as drifted since drift touched this filepath.
        self.assertTrue(all(s["is_drift"] for s in coder_steps))

        write_step = next(s for s in details["steps"] if s["diff"])
        self.assertIn("-    return a - b", write_step["diff"])
        self.assertIn("+    return a + b", write_step["diff"])

        tester_step = next(s for s in details["steps"] if s["type"] == "tester")
        self.assertEqual(tester_step["test_output"], "TESTS PASSED")

    def test_new_file_diff_shows_full_content_as_added(self):
        """Scenario: a file was written whose backup does not exist (a brand-new file).
        Expected: the diff shows the whole file as added, compared against /dev/null.
        """
        current_file = self.workspace_dir / "new_module.py"
        current_file.write_text("def hello():\n    return 'hi'\n", encoding="utf-8")

        _write_run(self.cognition_dir, "run2", [
            {"type": "run_started", "timestamp": 1000.0, "objective": "Add hello()", "workspace_path": str(self.workspace_dir)},
            {
                "type": "file_written", "timestamp": 1001.0, "layer_index": 1,
                "filepath": "new_module.py", "backup_path": str(Path(self._tmp.name) / "backup" / "new_module.py.missing"),
            },
        ])

        details = self.manager.get_run_details("run2")
        write_step = next(s for s in details["steps"] if s["filepath"] == "new_module.py")
        self.assertIn("+def hello():", write_step["diff"])
        self.assertIn("/dev/null", write_step["diff"])


# Allow running this file directly with the unittest runner.
if __name__ == "__main__":
    unittest.main()
