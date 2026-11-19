# ==============================================================================
# Program: test_self_correction_dataset.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Comprehensive test suite exercising self correction dataset functionality.
#          Verifies state transitions, error boundaries, regression resistance,
#          and contract compliance under tests.
# ==============================================================================

import json
import ast

from tools.generate_self_correction_dataset import (
    CodingProblem,
    build_correction_prompt,
    build_self_correction_row,
    code_diff,
    extract_code_block,
    extract_thought_block,
    normalize_problem_record,
)
from tools.generate_code_sft_dataset import DefinitionVisitor, build_user_prompt


def test_normalize_problem_record_accepts_verifiable_task():
    """A record with a verification_script is turned into a CodingProblem: the id is kept, and supplied code is appended to the prompt as 'Context/code'. Pins the normalisation contract the STaR generator relies on."""
    # Step 1: Validate input parameters, operational preconditions, and state
    record = {
        "id": "task-1",
        "problem_statement": "Write solve(x).",
        "verification_script": "def verify(response): return True",
        "code": "def solve(x): pass",
    }

    problem = normalize_problem_record(record)

    assert problem is not None
    assert problem.problem_id == "task-1"
    assert "Write solve" in problem.prompt
    assert "Context/code" in problem.prompt
    assert problem.verification_script.startswith("def verify")


def test_normalize_problem_record_rejects_missing_verifier():
    """Without a verification_script the record is rejected (None); an unverifiable problem could never produce trustworthy training data."""
    # Step 1: Validate input parameters, operational preconditions, and state
    assert normalize_problem_record({"prompt": "Write code"}) is None


def test_extract_code_and_thought_blocks():
    """The <think> body and the fenced python code are extracted separately and stripped."""
    # Step 1: Validate input parameters, operational preconditions, and state
    text = "<think>\nfix off by one\n</think>\n```python\ndef solve():\n    return 1\n```"

    assert extract_thought_block(text) == "fix off by one"
    assert extract_code_block(text) == "def solve():\n    return 1"


def test_code_diff_tracks_correction():
    """The unified diff between failed and corrected code uses the fixed filenames failed_attempt.py/corrected_attempt.py and marks the changed line with -/+."""
    # Step 1: Validate input parameters, operational preconditions, and state
    before = "```python\ndef solve():\n    return 1\n```"
    after = "```python\ndef solve():\n    return 2\n```"

    diff = code_diff(before, after)

    assert "--- failed_attempt.py" in diff
    assert "+++ corrected_attempt.py" in diff
    assert "-    return 1" in diff
    assert "+    return 2" in diff


def test_build_self_correction_row_format():
    """A failed-then-fixed trajectory becomes a system/user/assistant row tagged origin=self_correction_star with the iteration count; the assistant turn carries <think>, the diff, and a JSON transition block that must itself parse as valid JSON."""
    # Step 1: Validate input parameters, operational preconditions, and state
    problem = CodingProblem(
        problem_id="p1",
        prompt="Return 2.",
        verification_script="def verify(response): return '2' in response",
    )
    attempts = [
        {
            "iteration": 1,
            "thought": "try one",
            "response": "```python\ndef solve():\n    return 1\n```",
            "raw": "<think>try one</think>",
            "passed": False,
            "trace": "AssertionError",
        },
        {
            "iteration": 2,
            "thought": "fix constant",
            "response": "```python\ndef solve():\n    return 2\n```",
            "raw": "<think>fix constant</think>",
            "passed": True,
            "trace": "",
        },
    ]

    row = build_self_correction_row(problem, attempts)

    assert row["origin"] == "self_correction_star"
    assert row["iterations"] == 2
    assert row["messages"][0]["role"] == "system"
    assert row["messages"][1]["role"] == "user"
    assert row["messages"][2]["role"] == "assistant"
    assistant = row["messages"][2]["content"]
    assert "<think>" in assistant
    assert "correction_diff" in assistant
    assert "+    return 2" in assistant
    json.loads(assistant.split("```json\n", 1)[1].split("\n```", 1)[0])


def test_build_correction_prompt_includes_traceback():
    """The retry prompt must show the model both its bad response and the verifier traceback, and ask for <think> reasoning."""
    # Step 1: Validate input parameters, operational preconditions, and state
    problem = CodingProblem("p1", "Write solve().", "def verify(response): return False")

    prompt = build_correction_prompt(problem, "bad response", "NameError: solve")

    assert "bad response" in prompt
    assert "NameError: solve" in prompt
    assert "<think>" in prompt


def test_code_scrape_docstring_parsing_feeds_prompt_formatting():
    """The AST scraper in generate_code_sft_dataset captures docstring, argument string and return annotation for a function, and build_user_prompt then embeds the description and file path (cross-tool integration of the two generators)."""
    # Step 1: Validate input parameters, operational preconditions, and state
    source = '''
def add(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b
'''
    tree = ast.parse(source)
    visitor = DefinitionVisitor(source, "pkg/math_utils.py")
    visitor.visit(tree)

    assert len(visitor.definitions) == 1
    entry = visitor.definitions[0]
    assert entry["docstring"] == "Return the sum of two integers."
    assert entry["args"] == "a: int, b: int"
    assert entry["returns"] == "int"
    prompt = build_user_prompt(entry)
    assert "Return the sum of two integers." in prompt
    assert "pkg/math_utils.py" in prompt
