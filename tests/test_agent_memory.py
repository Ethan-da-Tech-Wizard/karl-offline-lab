# ==============================================================================
# Program: test_agent_memory.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Comprehensive test suite exercising agent memory functionality.
#          Verifies state transitions, error boundaries, regression resistance,
#          and contract compliance under tests.
# ==============================================================================

import json
from unittest.mock import patch

from app.engine.agent_memory import CodebaseMemory


def test_codebase_memory_indexes_functions_classes_and_methods(tmp_path):
    """build_index() scans a workspace and records functions (name, args, docstring) and classes with their methods; the JSON persisted at db_path must equal the returned index so memory survives restarts."""
    # The workspace is a throwaway source tree: build_index walks it for .py files and parses them, so the test controls exactly what symbols exist.
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "math_utils.py").write_text(
        '''
def add(a, b):
    """Add two numbers."""
    return a + b

class Calculator:
    """Arithmetic helper."""

    def multiply(self, x, y):
        """Multiply two numbers."""
        return x * y
''',
        encoding="utf-8",
    )
    # The index is persisted as JSON; the test checks the file equals the in-memory index.
    db_path = tmp_path / "agent_memory.json"

    memory = CodebaseMemory(str(workspace), db_path=db_path)
    # The index is built explicitly before querying so the query runs against known contents.
    index = memory.build_index()

    assert db_path.exists()
    persisted = json.loads(db_path.read_text(encoding="utf-8"))
    # Reading the JSON back and comparing it with the returned dict checks that nothing is lost in serialisation (tuples/sets would not round-trip).
    assert persisted == index
    assert index["math_utils.py"]["functions"][0] == {
        "name": "add",
        "args": ["a", "b"],
        "doc": "Add two numbers.",
    }
    # Class entries nest their methods, each with its own args.
    cls = index["math_utils.py"]["classes"][0]
    assert cls["name"] == "Calculator"
    assert cls["methods"][0]["name"] == "multiply"
    assert cls["methods"][0]["args"] == ["self", "x", "y"]


def test_query_memory_returns_matching_signature_text(tmp_path):
    """query_memory(keywords) returns a compact '- file: def name(args) -> doc' line for matching symbols, the text format later injected into prompts."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "math_utils.py").write_text(
        'def add(a, b):\n    """Add two numbers."""\n    return a + b\n',
        encoding="utf-8",
    )

    # Query by symbol name; the result is the compact signature line used in prompts.
    memory = CodebaseMemory(str(workspace), db_path=tmp_path / "agent_memory.json")
    memory.build_index()

    result = memory.query_memory(["add"])

    # The one-line signature+docstring form is deliberately compact: it is pasted into the coder prompt, where every token counts.
    assert "- math_utils.py: def add(a, b) -> Add two numbers." in result


def test_coder_agent_injects_codebase_signature_reference(tmp_path):
    """CoderAgent.generate must include a 'Codebase Interfaces & Signatures Reference' section built from the workspace index in the prompt it sends to the model, so generated code calls existing helpers with correct signatures. The model is replaced by a fake that records prompts."""
    from app.engine.swarm_agents import CoderAgent

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "math_utils.py").write_text(
        'def add(a, b):\n    """Add two numbers."""\n    return a + b\n',
        encoding="utf-8",
    )
    prompts = []

    # Capture every prompt so the test can see what context the Coder was given.
    def fake_llm(prompt, **kwargs):
        """Record each prompt and return a canned completion that calls add()."""
        prompts.append(prompt)
        return {"choices": [{"text": "def use_add(a, b):\n    return add(a, b)\n"}]}

    # No real model: the agent receives the recording fake.
    # Patching get_instance as seen from swarm_agents means CoderAgent runs its normal prompt-building path without any real model.
    with patch("app.engine.swarm_agents.ModelLoader.get_instance", return_value=fake_llm):
        agent = CoderAgent()
        result = agent.generate(
            {"filepath": "consumer.py", "instructions": "Call add from math_utils"},
            {"consumer.py": ""},
            workspace_path=str(workspace),
        )

    assert "use_add" in result
    assert prompts
    # Only the FIRST prompt is checked: the interface reference must be present in the initial request, not added after a failed attempt.
    assert "Codebase Interfaces & Signatures Reference" in prompts[0]
    assert "def add(a, b) -> Add two numbers." in prompts[0]
