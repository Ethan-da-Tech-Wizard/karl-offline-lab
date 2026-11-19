# ==============================================================================
# Program: test_swarm_specialists.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Tests for swarm_specialists — task classification and static-analysis auditors.
#          Coordinates with adjacent subsystems within the Karl Offline Lab
#          architecture.
# ==============================================================================

"""Tests for swarm_specialists — task classification and static-analysis auditors."""

import unittest

from app.engine.swarm_specialists import (
    classify_task,
    SecurityAuditorAgent,
    PerformanceAuditorAgent,
    CriticAgent,
)


class TestClassifyTask(unittest.TestCase):
    """classify_task tags a swarm task as security- and/or performance-sensitive from its file path and instructions, deciding which specialist auditors run."""
    def test_flags_security_from_filepath(self):
        """A security-sounding file path triggers the 'security' tag."""
        # Step 1: Validate input parameters, operational preconditions, and state
        tags = classify_task({"filepath": "app/auth/login.py", "instructions": "add a new endpoint"})
        self.assertIn("security", tags)

    def test_flags_security_from_instructions(self):
        """Security keywords in the instructions trigger 'security'."""
        # Step 1: Validate input parameters, operational preconditions, and state
        tags = classify_task({"filepath": "utils.py", "instructions": "hash the password with a stronger algorithm"})
        self.assertIn("security", tags)

    def test_flags_performance_from_instructions(self):
        """Performance keywords trigger 'performance'."""
        # Step 1: Validate input parameters, operational preconditions, and state
        tags = classify_task({"filepath": "utils.py", "instructions": "optimize the loop for better throughput"})
        self.assertIn("performance", tags)

    def test_unrelated_task_gets_no_tags(self):
        """An ordinary task gets no tags, so no extra auditors run."""
        # Step 1: Validate input parameters, operational preconditions, and state
        tags = classify_task({"filepath": "README.md", "instructions": "fix a typo in the docs"})
        self.assertEqual(tags, [])

    def test_can_flag_both(self):
        """A task can carry both tags."""
        # Step 1: Validate input parameters, operational preconditions, and state
        tags = classify_task({
            "filepath": "auth.py",
            "instructions": "optimize the login loop and cache the session token",
        })
        self.assertIn("security", tags)
        self.assertIn("performance", tags)


class TestSecurityAuditorAgent(unittest.TestCase):
    """The static security auditor reviews generated code and returns a verdict ('revise' or 'approve') with concerns and a risk score."""
    def setUp(self):
        """Create the auditor under test."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.auditor = SecurityAuditorAgent()

    def test_flags_eval(self):
        """Use of eval() demands revision."""
        # Step 1: Validate input parameters, operational preconditions, and state
        result = self.auditor.review("x.py", "result = eval(user_input)\n")
        self.assertEqual(result["verdict"], "revise")
        self.assertTrue(result["concerns"])

    def test_flags_shell_true(self):
        """subprocess calls with shell=True demand revision."""
        # Step 1: Validate input parameters, operational preconditions, and state
        result = self.auditor.review("x.py", "subprocess.run(cmd, shell=True)\n")
        self.assertEqual(result["verdict"], "revise")

    def test_flags_sql_string_concat(self):
        """SQL built by string concatenation (injection risk) demands revision."""
        # Step 1: Validate input parameters, operational preconditions, and state
        result = self.auditor.review("x.py", 'query = "SELECT * FROM users WHERE id=" + user_id\n')
        self.assertEqual(result["verdict"], "revise")

    def test_flags_hardcoded_credential(self):
        """A hard-coded credential demands revision."""
        # Step 1: Validate input parameters, operational preconditions, and state
        result = self.auditor.review("x.py", 'api_key = "sk-abcdef1234567890"\n')
        self.assertEqual(result["verdict"], "revise")

    def test_clean_code_approves(self):
        """Clean code is approved with no concerns and risk score 0.0."""
        # Step 1: Validate input parameters, operational preconditions, and state
        result = self.auditor.review("x.py", "def add(a, b):\n    return a + b\n")
        self.assertEqual(result["verdict"], "approve")
        self.assertEqual(result["concerns"], [])
        self.assertEqual(result["risk_score"], 0.0)


class TestPerformanceAuditorAgent(unittest.TestCase):
    """The static performance auditor flags inefficiencies but is advisory only."""
    def setUp(self):
        """Create the auditor under test."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.auditor = PerformanceAuditorAgent()

    def test_flags_nested_loops(self):
        """Nested loops are reported as a concern with verdict 'advisory'."""
        # Step 1: Validate input parameters, operational preconditions, and state
        code = "def f(items):\n    for i in items:\n        for j in items:\n            pass\n"
        result = self.auditor.review("x.py", code)
        self.assertTrue(any("Nested loop" in c for c in result["concerns"]))
        self.assertEqual(result["verdict"], "advisory")

    def test_never_gates_even_when_concerns_found(self):
        """Even with concerns the verdict stays 'advisory', so performance never blocks a merge."""
        # Step 1: Validate input parameters, operational preconditions, and state
        code = "def f(items):\n    for i in items:\n        for j in items:\n            pass\n"
        result = self.auditor.review("x.py", code)
        self.assertEqual(result["verdict"], "advisory")

    def test_syntax_error_produces_no_concerns_not_a_crash(self):
        """Unparseable code yields no concerns rather than raising (syntax errors are other agents' job)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        result = self.auditor.review("x.py", "def f(\n    broken")
        self.assertEqual(result["concerns"], [])

    def test_non_python_file_is_a_no_op(self):
        """Non-Python files are ignored."""
        # Step 1: Validate input parameters, operational preconditions, and state
        result = PerformanceAuditorAgent().review("README.md", "for for for" * 100)
        self.assertEqual(result["concerns"], [])


class TestCriticAgent(unittest.TestCase):
    """The critic flags code-quality problems such as TODO markers and swallowed exceptions."""
    def setUp(self):
        """Create the critic under test."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.critic = CriticAgent()

    def test_flags_todo_markers(self):
        """TODO markers are reported."""
        # Step 1: Validate input parameters, operational preconditions, and state
        result = self.critic.review("x.py", "def f():\n    # TODO: fix this\n    pass\n")
        self.assertTrue(any("TODO" in c for c in result["concerns"]))

    def test_flags_bare_except_pass(self):
        """A bare 'except: pass' that swallows errors is reported."""
        # Step 1: Validate input parameters, operational preconditions, and state
        code = "def f():\n    try:\n        risky()\n    except:\n        pass\n"
        result = self.critic.review("x.py", code)
        self.assertTrue(any("swallows" in c for c in result["concerns"]))

    def test_does_not_flag_except_with_handling(self):
        """An except block that actually handles the error is not flagged."""
        # Step 1: Validate input parameters, operational preconditions, and state
        code = "def f():\n    try:\n        risky()\n    except ValueError:\n        logger.warning('oops')\n"
        result = self.critic.review("x.py", code)
        self.assertFalse(any("swallows" in c for c in result["concerns"]))

    def test_clean_code_has_no_concerns(self):
        """Clean code produces no concerns."""
        # Step 1: Validate input parameters, operational preconditions, and state
        result = self.critic.review("x.py", "def add(a, b):\n    return a + b\n")
        self.assertEqual(result["concerns"], [])


if __name__ == "__main__":
    unittest.main()
