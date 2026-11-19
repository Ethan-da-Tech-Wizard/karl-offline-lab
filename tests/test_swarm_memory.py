# ==============================================================================
# Program: test_swarm_memory.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Tests for SwarmMemory — cross-run failure/fix learning.
#          Coordinates with adjacent subsystems within the Karl Offline Lab
#          architecture.
# ==============================================================================

"""Tests for SwarmMemory — cross-run failure/fix learning."""

import json
import os
import tempfile
import unittest

from app.engine.swarm_memory import SwarmMemory, fingerprint_failure


class TestFingerprintFailure(unittest.TestCase):
    """fingerprint_failure reduces a failure trace to a stable key (error class plus normalised message) so recurring failures match across runs."""
    def test_extracts_error_class_and_normalizes_message(self):
        """The fingerprint starts with the exception class and volatile values such as memory addresses are normalised away."""
        trace = (
            "Traceback (most recent call last):\n"
            '  File "math_utils.py", line 42, in add\n'
            "AssertionError: expected 5, got 'foo' at 0x7f0000000000\n"
        )
        fp = fingerprint_failure(trace)
        self.assertTrue(fp.startswith("AssertionError:"))
        self.assertNotIn("0x7f0000000000", fp)

    def test_empty_trace_is_stable(self):
        """An empty trace yields the constant 'unknown_error'."""
        self.assertEqual(fingerprint_failure(""), "unknown_error")

    def test_same_class_different_details_fingerprint_identically(self):
        """The same error class with different details (numbers, addresses) gives the same fingerprint."""
        fp1 = fingerprint_failure("TypeError: unsupported operand type(s) for +: 'int' and 'str'")
        fp2 = fingerprint_failure("TypeError: unsupported operand type(s) for +: 'int' and 'str'")
        self.assertEqual(fp1, fp2)


class TestSwarmMemory(unittest.TestCase):
    """SwarmMemory is the cross-run store that remembers failures and what fixed them, and recalls relevant lessons for similar tasks."""
    def setUp(self):
        """Create a temp database path for each test."""
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "swarm_memory.json")

    def tearDown(self):
        """Remove the temp store."""
        self.tmp.cleanup()

    def test_recall_empty_store_returns_empty_string(self):
        """With nothing recorded, recall returns '' (nothing is injected into prompts)."""
        mem = SwarmMemory("ws", db_path=self.db_path)
        self.assertEqual(mem.recall("math_utils.py", "add two numbers"), "")

    def test_record_failure_then_recall_matches_by_keyword(self):
        """A recorded failure is recalled for a task whose file and description share keywords."""
        mem = SwarmMemory("ws", db_path=self.db_path)
        mem.record_failure("math_utils.py", "define add function", "AssertionError: expected 5 got -1")
        recalled = mem.recall("math_utils.py", "add function returns wrong value")
        self.assertIn("AssertionError", recalled)
        self.assertIn("math_utils.py", recalled)

    def test_recall_does_not_match_unrelated_task(self):
        """An unrelated task recalls nothing, avoiding noise in the prompt."""
        mem = SwarmMemory("ws", db_path=self.db_path)
        mem.record_failure("math_utils.py", "define add function", "AssertionError: expected 5 got -1")
        recalled = mem.recall("network_client.py", "open a socket connection")
        self.assertEqual(recalled, "")

    def test_record_success_links_fix_to_failure_and_surfaces_on_recall(self):
        """Recording a later success links the fix to the earlier failure and recall then shows 'Previously fixed by ...'."""
        mem = SwarmMemory("ws", db_path=self.db_path)
        fp = mem.record_failure("math_utils.py", "define add function", "AssertionError: expected 5 got -1")
        mem.record_success("math_utils.py", "used + instead of - operator", [fp])
        recalled = mem.recall("math_utils.py", "add function implementation")
        self.assertIn("Previously fixed by", recalled)
        self.assertIn("+ instead of -", recalled)

    def test_persists_across_instances(self):
        """The store is written to disk and a new instance can recall earlier entries."""
        mem1 = SwarmMemory("ws", db_path=self.db_path)
        mem1.record_failure("a.py", "task instructions", "ValueError: bad input")
        self.assertTrue(os.path.exists(self.db_path))

        mem2 = SwarmMemory("ws", db_path=self.db_path)
        recalled = mem2.recall("a.py", "task instructions")
        self.assertIn("ValueError", recalled)

    def test_corrupted_store_recovers_gracefully(self):
        """A corrupt store file is treated as empty and can be written again instead of crashing the swarm."""
        with open(self.db_path, "w") as f:
            f.write("{not valid json")
        mem = SwarmMemory("ws", db_path=self.db_path)
        self.assertEqual(mem.recall("a.py", "anything"), "")
        # Should still be able to write after recovering from corruption.
        mem.record_failure("a.py", "task", "TypeError: boom")
        with open(self.db_path, encoding="utf-8") as f:
            data = json.loads(f.read())
        self.assertEqual(len(data), 1)


if __name__ == "__main__":
    unittest.main()
