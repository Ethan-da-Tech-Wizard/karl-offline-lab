# ==============================================================================
# Program: test_swarm_judge.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Tests for SwarmJudge — heuristic multiverse candidate scoring.
#          Coordinates with adjacent subsystems within the Karl Offline Lab
#          architecture.
# ==============================================================================

"""Tests for SwarmJudge — heuristic multiverse candidate scoring."""

import unittest

from app.engine.swarm_judge import score_candidate, select_winner


class TestScoreCandidate(unittest.TestCase):
    """SwarmJudge.score_candidate scores one candidate file by syntax validity, diff size versus the original and alignment with known signatures."""
    def test_syntax_error_is_heavily_penalized(self):
        """A candidate with a Python syntax error has syntax_ok False and scores below a working candidate."""
        # Step 1: Validate input parameters, operational preconditions, and state
        broken = score_candidate("x.py", "def f(\n    pass\n")
        working = score_candidate("x.py", "def f():\n    pass\n")
        self.assertFalse(broken["syntax_ok"])
        self.assertTrue(working["syntax_ok"])
        self.assertLess(broken["total_score"], working["total_score"])

    def test_json_candidate_checks_json_validity_not_python_syntax(self):
        """For .json files validity means parseable JSON, not Python syntax."""
        # Step 1: Validate input parameters, operational preconditions, and state
        good = score_candidate("config.json", '{"a": 1}')
        bad = score_candidate("config.json", '{"a": 1,}')
        self.assertTrue(good["syntax_ok"])
        self.assertFalse(bad["syntax_ok"])

    def test_non_code_file_always_syntax_ok(self):
        """Non-code files (such as docs) are always considered syntactically fine."""
        # Step 1: Validate input parameters, operational preconditions, and state
        result = score_candidate("README.md", "# Title\nAnything goes here.")
        self.assertTrue(result["syntax_ok"])

    def test_diff_size_zero_when_no_original(self):
        """With no original file there is nothing to diff against, so diff_size is 0."""
        # Step 1: Validate input parameters, operational preconditions, and state
        result = score_candidate("new_file.py", "def f():\n    pass\n", original_content="")
        self.assertEqual(result["diff_size"], 0)

    def test_diff_size_grows_with_change_amount(self):
        """A larger rewrite has a larger diff_size than a small edit; smaller diffs are preferred."""
        # Step 1: Validate input parameters, operational preconditions, and state
        original = "def f():\n    return 1\n"
        small_change = score_candidate("x.py", "def f():\n    return 2\n", original_content=original)
        big_rewrite = score_candidate(
            "x.py",
            "def f():\n    return 2\n\n\ndef g():\n    return 3\n\n\ndef h():\n    return 4\n",
            original_content=original,
        )
        self.assertLess(small_change["diff_size"], big_rewrite["diff_size"])

    def test_signature_alignment_neutral_when_no_known_names(self):
        """With no known function names the alignment score is neutral (1.0)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        result = score_candidate("x.py", "def f():\n    return made_up_call()\n", known_names=set())
        self.assertEqual(result["signature_alignment"], 1.0)

    def test_signature_alignment_rewards_known_calls(self):
        """A candidate that calls the names that exist elsewhere in the project scores higher than one that invents others."""
        # Step 1: Validate input parameters, operational preconditions, and state
        content = "def f():\n    return existing_helper()\n"
        aligned = score_candidate("x.py", content, known_names={"existing_helper"})
        misaligned = score_candidate("x.py", content, known_names={"totally_unrelated_name"})
        self.assertGreater(aligned["signature_alignment"], misaligned["signature_alignment"])


class TestSelectWinner(unittest.TestCase):
    """SwarmJudge.select_winner chooses the best of N candidates."""
    def test_requires_at_least_one_candidate(self):
        """An empty candidate list raises ValueError."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with self.assertRaises(ValueError):
            select_winner("x.py", [])

    def test_picks_the_syntactically_valid_candidate(self):
        """The valid candidate beats the broken one, and scores for all candidates are returned."""
        # Step 1: Validate input parameters, operational preconditions, and state
        candidates = [
            "def add(a, b)\n    return a - b\n",   # broken syntax
            "def add(a, b):\n    return a + b\n",  # valid
        ]
        idx, winner, all_scores = select_winner("math_utils.py", candidates)
        self.assertEqual(idx, 1)
        self.assertTrue(winner["syntax_ok"])
        self.assertEqual(len(all_scores), 2)

    def test_ties_break_toward_earlier_candidate(self):
        """Equal scores pick the earliest candidate, making the result deterministic."""
        # Step 1: Validate input parameters, operational preconditions, and state
        identical = ["def f():\n    pass\n", "def f():\n    pass\n"]
        idx, _winner, _scores = select_winner("x.py", identical)
        self.assertEqual(idx, 0)

    def test_single_candidate_always_wins(self):
        """A lone candidate wins with index 0."""
        # Step 1: Validate input parameters, operational preconditions, and state
        idx, winner, all_scores = select_winner("x.py", ["def f():\n    pass\n"])
        self.assertEqual(idx, 0)
        self.assertEqual(len(all_scores), 1)


if __name__ == "__main__":
    unittest.main()
