# Program: karl-offline-lab - test_swarm_memory.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Unit tests for fingerprint_failure and SwarmMemory, covering failure fingerprinting, recall by keyword, fix linking, persistence and recovery from a corrupted store.
"""Tests for SwarmMemory — cross-run failure/fix learning."""

import json
import os
import tempfile
import unittest

from app.engine.swarm_memory import SwarmMemory, fingerprint_failure


class TestFingerprintFailure(unittest.TestCase):
    """Tests for turning a raw traceback into a short, stable fingerprint string."""
    def test_extracts_error_class_and_normalizes_message(self):
        """Scenario: a traceback containing a memory address.
        Expected: the fingerprint starts with the error class and the address is removed.
        """
        trace = (
            "Traceback (most recent call last):\n"
            '  File "math_utils.py", line 42, in add\n'
            "AssertionError: expected 5, got 'foo' at 0x7f0000000000\n"
        )
        fp = fingerprint_failure(trace)
        self.assertTrue(fp.startswith("AssertionError:"))
        self.assertNotIn("0x7f0000000000", fp)

    def test_empty_trace_is_stable(self):
        """Scenario: an empty trace string.
        Expected: the fixed fallback fingerprint "unknown_error".
        """
        self.assertEqual(fingerprint_failure(""), "unknown_error")

    def test_same_class_different_details_fingerprint_identically(self):
        """Scenario: the same error text is fingerprinted twice.
        Expected: both fingerprints are identical (stable and repeatable).
        """
        fp1 = fingerprint_failure("TypeError: unsupported operand type(s) for +: 'int' and 'str'")
        fp2 = fingerprint_failure("TypeError: unsupported operand type(s) for +: 'int' and 'str'")
        self.assertEqual(fp1, fp2)


class TestSwarmMemory(unittest.TestCase):
    """Tests for the persistent cross-run failure and fix store, using a temporary database file."""
    def setUp(self):
        """Create a temporary directory and a database path inside it for each test."""
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "swarm_memory.json")

    def tearDown(self):
        """Remove the temporary directory after each test."""
        self.tmp.cleanup()

    def test_recall_empty_store_returns_empty_string(self):
        """Scenario: recall on a store with no records.
        Expected: an empty string.
        """
        mem = SwarmMemory("ws", db_path=self.db_path)
        self.assertEqual(mem.recall("math_utils.py", "add two numbers"), "")

    def test_record_failure_then_recall_matches_by_keyword(self):
        """Scenario: a failure is recorded, then a related task is recalled.
        Expected: the recalled text mentions the error class and the file name.
        """
        mem = SwarmMemory("ws", db_path=self.db_path)
        mem.record_failure("math_utils.py", "define add function", "AssertionError: expected 5 got -1")
        recalled = mem.recall("math_utils.py", "add function returns wrong value")
        self.assertIn("AssertionError", recalled)
        self.assertIn("math_utils.py", recalled)

    def test_recall_does_not_match_unrelated_task(self):
        """Scenario: a failure is recorded for one file, recall is asked about an unrelated file and task.
        Expected: an empty string.
        """
        mem = SwarmMemory("ws", db_path=self.db_path)
        mem.record_failure("math_utils.py", "define add function", "AssertionError: expected 5 got -1")
        recalled = mem.recall("network_client.py", "open a socket connection")
        self.assertEqual(recalled, "")

    def test_record_success_links_fix_to_failure_and_surfaces_on_recall(self):
        """Scenario: a failure is recorded and then a fix is linked to its fingerprint.
        Expected: recall includes the "Previously fixed by" note with the fix text.
        """
        mem = SwarmMemory("ws", db_path=self.db_path)
        fp = mem.record_failure("math_utils.py", "define add function", "AssertionError: expected 5 got -1")
        mem.record_success("math_utils.py", "used + instead of - operator", [fp])
        recalled = mem.recall("math_utils.py", "add function implementation")
        self.assertIn("Previously fixed by", recalled)
        self.assertIn("+ instead of -", recalled)

    def test_persists_across_instances(self):
        """Scenario: a record is written by one instance and read by a new instance on the same file.
        Expected: the file exists and the second instance recalls the record.
        """
        mem1 = SwarmMemory("ws", db_path=self.db_path)
        mem1.record_failure("a.py", "task instructions", "ValueError: bad input")
        self.assertTrue(os.path.exists(self.db_path))

        mem2 = SwarmMemory("ws", db_path=self.db_path)
        recalled = mem2.recall("a.py", "task instructions")
        self.assertIn("ValueError", recalled)

    def test_corrupted_store_recovers_gracefully(self):
        # Write deliberately invalid JSON to simulate a corrupted store.
        """Scenario: the database file holds invalid JSON.
        Expected: recall returns an empty string and a later write still succeeds with one entry.
        """
        with open(self.db_path, "w") as f:
            f.write("{not valid json")
        mem = SwarmMemory("ws", db_path=self.db_path)
        self.assertEqual(mem.recall("a.py", "anything"), "")
        # Should still be able to write after recovering from corruption.
        mem.record_failure("a.py", "task", "TypeError: boom")
        with open(self.db_path, encoding="utf-8") as f:
            data = json.loads(f.read())
        self.assertEqual(len(data), 1)


# Allow running this file directly with the unittest runner.
if __name__ == "__main__":
    unittest.main()
