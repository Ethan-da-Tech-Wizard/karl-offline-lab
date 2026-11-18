# Program: karl-offline-lab - test_swarm_specialists.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Unit tests for classify_task and the security, performance and critic review agents that statically inspect code.
"""Tests for swarm_specialists — task classification and static-analysis auditors."""

import unittest

from app.engine.swarm_specialists import (
    classify_task,
    SecurityAuditorAgent,
    PerformanceAuditorAgent,
    CriticAgent,
)


class TestClassifyTask(unittest.TestCase):
    """Tests for tagging a task as security and/or performance related."""
    def test_flags_security_from_filepath(self):
        """Scenario: the file path contains an auth directory.
        Expected: the "security" tag is returned.
        """
        tags = classify_task({"filepath": "app/auth/login.py", "instructions": "add a new endpoint"})
        self.assertIn("security", tags)

    def test_flags_security_from_instructions(self):
        """Scenario: the instructions mention hashing a password.
        Expected: the "security" tag is returned.
        """
        tags = classify_task({"filepath": "utils.py", "instructions": "hash the password with a stronger algorithm"})
        self.assertIn("security", tags)

    def test_flags_performance_from_instructions(self):
        """Scenario: the instructions ask to optimize a loop for throughput.
        Expected: the "performance" tag is returned.
        """
        tags = classify_task({"filepath": "utils.py", "instructions": "optimize the loop for better throughput"})
        self.assertIn("performance", tags)

    def test_unrelated_task_gets_no_tags(self):
        """Scenario: a documentation typo fix.
        Expected: an empty tag list.
        """
        tags = classify_task({"filepath": "README.md", "instructions": "fix a typo in the docs"})
        self.assertEqual(tags, [])

    def test_can_flag_both(self):
        """Scenario: instructions mention both optimizing and a login session token.
        Expected: both "security" and "performance" tags are returned.
        """
        tags = classify_task({
            "filepath": "auth.py",
            "instructions": "optimize the login loop and cache the session token",
        })
        self.assertIn("security", tags)
        self.assertIn("performance", tags)


class TestSecurityAuditorAgent(unittest.TestCase):
    """Tests for the static security reviewer."""
    def setUp(self):
        """Create a fresh security auditor for each test."""
        self.auditor = SecurityAuditorAgent()

    def test_flags_eval(self):
        """Scenario: code calls eval on user input.
        Expected: verdict "revise" with at least one concern.
        """
        result = self.auditor.review("x.py", "result = eval(user_input)\n")
        self.assertEqual(result["verdict"], "revise")
        self.assertTrue(result["concerns"])

    def test_flags_shell_true(self):
        """Scenario: a subprocess call with shell enabled.
        Expected: verdict "revise".
        """
        result = self.auditor.review("x.py", "subprocess.run(cmd, shell=True)\n")
        self.assertEqual(result["verdict"], "revise")

    def test_flags_sql_string_concat(self):
        """Scenario: an SQL query built by string concatenation.
        Expected: verdict "revise".
        """
        result = self.auditor.review("x.py", 'query = "SELECT * FROM users WHERE id=" + user_id\n')
        self.assertEqual(result["verdict"], "revise")

    def test_flags_hardcoded_credential(self):
        """Scenario: an API key assigned as a string literal.
        Expected: verdict "revise".
        """
        result = self.auditor.review("x.py", 'api_key = "sk-abcdef1234567890"\n')
        self.assertEqual(result["verdict"], "revise")

    def test_clean_code_approves(self):
        """Scenario: a simple, safe function.
        Expected: verdict "approve", no concerns and a risk score of 0.0.
        """
        result = self.auditor.review("x.py", "def add(a, b):\n    return a + b\n")
        self.assertEqual(result["verdict"], "approve")
        self.assertEqual(result["concerns"], [])
        self.assertEqual(result["risk_score"], 0.0)


class TestPerformanceAuditorAgent(unittest.TestCase):
    """Tests for the advisory performance reviewer."""
    def setUp(self):
        """Create a fresh performance auditor for each test."""
        self.auditor = PerformanceAuditorAgent()

    def test_flags_nested_loops(self):
        """Scenario: a function with a loop nested inside a loop.
        Expected: a "Nested loop" concern and the advisory verdict.
        """
        code = "def f(items):\n    for i in items:\n        for j in items:\n            pass\n"
        result = self.auditor.review("x.py", code)
        self.assertTrue(any("Nested loop" in c for c in result["concerns"]))
        self.assertEqual(result["verdict"], "advisory")

    def test_never_gates_even_when_concerns_found(self):
        """Scenario: code with concerns is reviewed.
        Expected: the verdict stays "advisory", so this reviewer never blocks a change.
        """
        code = "def f(items):\n    for i in items:\n        for j in items:\n            pass\n"
        result = self.auditor.review("x.py", code)
        self.assertEqual(result["verdict"], "advisory")

    def test_syntax_error_produces_no_concerns_not_a_crash(self):
        """Scenario: the source code has a syntax error.
        Expected: no concerns and no exception.
        """
        result = self.auditor.review("x.py", "def f(\n    broken")
        self.assertEqual(result["concerns"], [])

    def test_non_python_file_is_a_no_op(self):
        """Scenario: a non-Python file path.
        Expected: no concerns are produced.
        """
        result = PerformanceAuditorAgent().review("README.md", "for for for" * 100)
        self.assertEqual(result["concerns"], [])


class TestCriticAgent(unittest.TestCase):
    """Tests for the general code critic."""
    def setUp(self):
        """Create a fresh critic for each test."""
        self.critic = CriticAgent()

    def test_flags_todo_markers(self):
        """Scenario: code contains a TODO comment.
        Expected: a concern mentioning TODO.
        """
        result = self.critic.review("x.py", "def f():\n    # TODO: fix this\n    pass\n")
        self.assertTrue(any("TODO" in c for c in result["concerns"]))

    def test_flags_bare_except_pass(self):
        """Scenario: a bare except block that only passes.
        Expected: a concern saying the error is swallowed.
        """
        code = "def f():\n    try:\n        risky()\n    except:\n        pass\n"
        result = self.critic.review("x.py", code)
        self.assertTrue(any("swallows" in c for c in result["concerns"]))

    def test_does_not_flag_except_with_handling(self):
        """Scenario: an except block that logs a warning.
        Expected: no "swallows" concern.
        """
        code = "def f():\n    try:\n        risky()\n    except ValueError:\n        logger.warning('oops')\n"
        result = self.critic.review("x.py", code)
        self.assertFalse(any("swallows" in c for c in result["concerns"]))

    def test_clean_code_has_no_concerns(self):
        """Scenario: a simple clean function.
        Expected: no concerns.
        """
        result = self.critic.review("x.py", "def add(a, b):\n    return a + b\n")
        self.assertEqual(result["concerns"], [])


# Allow running this file directly with the unittest runner.
if __name__ == "__main__":
    unittest.main()
