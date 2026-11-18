# Program: karl-offline-lab - test_swarm_judge.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Tests SwarmJudge heuristic candidate scoring and winner selection.

"""Tests for SwarmJudge — heuristic multiverse candidate scoring."""

import unittest

from app.engine.swarm_judge import score_candidate, select_winner


class TestScoreCandidate(unittest.TestCase):
    """Test group or helper class: test score candidate."""
    def test_syntax_error_is_heavily_penalized(self):
        """Scenario: syntax error is heavily penalized. Expected: the assertions in this test hold."""
        broken = score_candidate("x.py", "def f(\n    pass\n")
        working = score_candidate("x.py", "def f():\n    pass\n")
        self.assertFalse(broken["syntax_ok"])
        self.assertTrue(working["syntax_ok"])
        self.assertLess(broken["total_score"], working["total_score"])

    def test_json_candidate_checks_json_validity_not_python_syntax(self):
        """Scenario: json candidate checks json validity not python syntax. Expected: the assertions in this test hold."""
        good = score_candidate("config.json", '{"a": 1}')
        bad = score_candidate("config.json", '{"a": 1,}')
        self.assertTrue(good["syntax_ok"])
        self.assertFalse(bad["syntax_ok"])

    def test_non_code_file_always_syntax_ok(self):
        """Scenario: non code file always syntax ok. Expected: the assertions in this test hold."""
        result = score_candidate("README.md", "# Title\nAnything goes here.")
        self.assertTrue(result["syntax_ok"])

    def test_diff_size_zero_when_no_original(self):
        """Scenario: diff size zero when no original. Expected: the assertions in this test hold."""
        result = score_candidate("new_file.py", "def f():\n    pass\n", original_content="")
        self.assertEqual(result["diff_size"], 0)

    def test_diff_size_grows_with_change_amount(self):
        """Scenario: diff size grows with change amount. Expected: the assertions in this test hold."""
        original = "def f():\n    return 1\n"
        small_change = score_candidate("x.py", "def f():\n    return 2\n", original_content=original)
        big_rewrite = score_candidate(
            "x.py",
            "def f():\n    return 2\n\n\ndef g():\n    return 3\n\n\ndef h():\n    return 4\n",
            original_content=original,
        )
        self.assertLess(small_change["diff_size"], big_rewrite["diff_size"])

    def test_signature_alignment_neutral_when_no_known_names(self):
        """Scenario: signature alignment neutral when no known names. Expected: the assertions in this test hold."""
        result = score_candidate("x.py", "def f():\n    return made_up_call()\n", known_names=set())
        self.assertEqual(result["signature_alignment"], 1.0)

    def test_signature_alignment_rewards_known_calls(self):
        """Scenario: signature alignment rewards known calls. Expected: the assertions in this test hold."""
        content = "def f():\n    return existing_helper()\n"
        aligned = score_candidate("x.py", content, known_names={"existing_helper"})
        misaligned = score_candidate("x.py", content, known_names={"totally_unrelated_name"})
        self.assertGreater(aligned["signature_alignment"], misaligned["signature_alignment"])


class TestSelectWinner(unittest.TestCase):
    """Test group or helper class: test select winner."""
    def test_requires_at_least_one_candidate(self):
        """Scenario: requires at least one candidate. Expected: the assertions in this test hold."""
        with self.assertRaises(ValueError):
            select_winner("x.py", [])

    def test_picks_the_syntactically_valid_candidate(self):
        """Scenario: picks the syntactically valid candidate. Expected: the assertions in this test hold."""
        candidates = [
            "def add(a, b)\n    return a - b\n",   # broken syntax
            "def add(a, b):\n    return a + b\n",  # valid
        ]
        idx, winner, all_scores = select_winner("math_utils.py", candidates)
        self.assertEqual(idx, 1)
        self.assertTrue(winner["syntax_ok"])
        self.assertEqual(len(all_scores), 2)

    def test_ties_break_toward_earlier_candidate(self):
        """Scenario: ties break toward earlier candidate. Expected: the assertions in this test hold."""
        identical = ["def f():\n    pass\n", "def f():\n    pass\n"]
        idx, _winner, _scores = select_winner("x.py", identical)
        self.assertEqual(idx, 0)

    def test_single_candidate_always_wins(self):
        """Scenario: single candidate always wins. Expected: the assertions in this test hold."""
        idx, winner, all_scores = select_winner("x.py", ["def f():\n    pass\n"])
        self.assertEqual(idx, 0)
        self.assertEqual(len(all_scores), 1)


if __name__ == "__main__":
    unittest.main()
