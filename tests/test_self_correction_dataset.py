# ==============================================================================
# Program: test_self_correction_dataset.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Comprehensive test suite exercising self correction dataset functionality.
#          Verifies state transitions, error boundaries, regression resistance,
#          and contract compliance under tests.
# ==============================================================================

# json is used to prove the transition block embedded in an answer is valid JSON; ast feeds the
# AST scraper test at the bottom.
import json
import ast

# The "STaR" (self-taught reasoner) generator turns a problem that a model first fails and then
# solves after seeing the verifier's traceback into one training row. These imports are its public
# helpers; the tests below pin each helper's contract on tiny hand-made inputs, with no model run.
from tools.generate_self_correction_dataset import (
    CodingProblem,
    build_correction_prompt,
    build_self_correction_row,
    code_diff,
    extract_code_block,
    extract_thought_block,
    normalize_problem_record,
)
# The code-SFT generator is imported to test how its AST scraper output feeds the shared prompt builder.
from tools.generate_code_sft_dataset import DefinitionVisitor, build_user_prompt


# Normalisation: raw dataset records (several shapes) become a CodingProblem. Only records that
# carry a verification_script are accepted, since correctness must be machine-checkable.
def test_normalize_problem_record_accepts_verifiable_task():
    """A record with a verification_script is turned into a CodingProblem: the id is kept, and supplied code is appended to the prompt as 'Context/code'. Pins the normalisation contract the STaR generator relies on."""
    # A minimal executable problem: id, statement, verifier and the starting code.
    # A minimal executable problem: id, statement, verifier and the starting code.
    record = {
        "id": "task-1",
        "problem_statement": "Write solve(x).",
        "verification_script": "def verify(response): return True",
        "code": "def solve(x): pass",
    }

    # The statement and the starting code are merged into one prompt, with the code labelled "Context/code".
    problem = normalize_problem_record(record)

    assert problem is not None
    assert problem.problem_id == "task-1"
    assert "Write solve" in problem.prompt
    assert "Context/code" in problem.prompt
    assert problem.verification_script.startswith("def verify")


# A record without a verifier is unusable: the STaR loop could not tell success from failure.
def test_normalize_problem_record_rejects_missing_verifier():
    """Without a verification_script the record is rejected (None); an unverifiable problem could never produce trustworthy training data."""
    # Returns None (not an exception) so a dataset loader can simply skip bad lines.
    assert normalize_problem_record({"prompt": "Write code"}) is None


# Parsing of model output: reasoning lives in <think>, the answer in a fenced ```python block.
def test_extract_code_and_thought_blocks():
    """The <think> body and the fenced python code are extracted separately and stripped."""
    # Typical model output: reasoning block followed by a fenced solution.
    # Typical model output: reasoning block followed by a fenced solution.
    text = "<think>\nfix off by one\n</think>\n```python\ndef solve():\n    return 1\n```"

    assert extract_thought_block(text) == "fix off by one"
    assert extract_code_block(text) == "def solve():\n    return 1"


# The correction diff is part of the training target: it shows what changed between the failing and
# the passing attempt. Fixed pseudo-filenames keep rows reproducible across runs.
def test_code_diff_tracks_correction():
    """The unified diff between failed and corrected code uses the fixed filenames failed_attempt.py/corrected_attempt.py and marks the changed line with -/+."""
    # Both inputs are fenced blocks like real model responses; code_diff extracts the code itself.
    before = "```python\ndef solve():\n    return 1\n```"
    after = "```python\ndef solve():\n    return 2\n```"

    diff = code_diff(before, after)

    assert "--- failed_attempt.py" in diff
    assert "+++ corrected_attempt.py" in diff
    # unified-diff markers: "-" lines were removed from the failed attempt, "+" lines added in the fix.
    assert "-    return 1" in diff
    assert "+    return 2" in diff


# Output format of a training row: chat messages plus metadata, ready for SFT.
def test_build_self_correction_row_format():
    """A failed-then-fixed trajectory becomes a system/user/assistant row tagged origin=self_correction_star with the iteration count; the assistant turn carries <think>, the diff, and a JSON transition block that must itself parse as valid JSON."""
    # CodingProblem is a simple dataclass: id, prompt, verification script.
    problem = CodingProblem(
        problem_id="p1",
        prompt="Return 2.",
        verification_script="def verify(response): return '2' in response",
    )
    # Two attempts: iteration 1 fails the verifier, iteration 2 fixes it; only such a trajectory
    # becomes training data.
    # Two attempts: iteration 1 fails the verifier, iteration 2 fixes it; only such a trajectory
    # becomes training data.
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

    # Fails then passes -> builds one row.
    row = build_self_correction_row(problem, attempts)

    # origin tags the data source so mixed datasets can be audited or filtered later.
    assert row["origin"] == "self_correction_star"
    assert row["iterations"] == 2
    # system/user/assistant order is what chat-format SFT trainers expect.
    assert row["messages"][0]["role"] == "system"
    assert row["messages"][1]["role"] == "user"
    assert row["messages"][2]["role"] == "assistant"
    assistant = row["messages"][2]["content"]
    assert "<think>" in assistant
    assert "correction_diff" in assistant
    assert "+    return 2" in assistant
    # The transition block embedded in the answer must itself be valid JSON.
    # The transition block embedded in the answer must itself be valid JSON.
    json.loads(assistant.split("```json\n", 1)[1].split("\n```", 1)[0])


# The second-chance prompt: shows the model what it answered and why that failed.
def test_build_correction_prompt_includes_traceback():
    """The retry prompt must show the model both its bad response and the verifier traceback, and ask for <think> reasoning."""
    problem = CodingProblem("p1", "Write solve().", "def verify(response): return False")

    prompt = build_correction_prompt(problem, "bad response", "NameError: solve")

    assert "bad response" in prompt
    assert "NameError: solve" in prompt
    assert "<think>" in prompt


# Integration across two tools: the code-SFT scraper extracts documented definitions via ast, and
# build_user_prompt turns each entry into the question that SFT row is trained on.
def test_code_scrape_docstring_parsing_feeds_prompt_formatting():
    """The AST scraper in generate_code_sft_dataset captures docstring, argument string and return annotation for a function, and build_user_prompt then embeds the description and file path (cross-tool integration of the two generators)."""
    # Source text for the AST scraper, written with a docstring and annotations so all fields are exercised.
    # Source text for the AST scraper, written with a docstring and annotations so all fields are
    # exercised.
    source = '''
def add(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b
'''
    # The visitor records functions it finds; the second argument is the file path shown in the prompt.
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
