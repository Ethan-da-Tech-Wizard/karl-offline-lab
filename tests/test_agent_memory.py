# Program: karl-offline-lab - test_agent_memory.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Tests for the codebase memory index and its use by the coder agent prompt.
"""Tests for app.engine.agent_memory.CodebaseMemory and its injection into the coder agent."""

import json
from unittest.mock import patch

from app.engine.agent_memory import CodebaseMemory


def test_codebase_memory_indexes_functions_classes_and_methods(tmp_path):
    """Scenario: a workspace with one function and one class method. Expected: the index lists both with names, args and docstrings and is saved to disk."""
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
    db_path = tmp_path / "agent_memory.json"

    memory = CodebaseMemory(str(workspace), db_path=db_path)
    index = memory.build_index()

    assert db_path.exists()
    # The saved JSON file must equal the in-memory index
    persisted = json.loads(db_path.read_text(encoding="utf-8"))
    assert persisted == index
    assert index["math_utils.py"]["functions"][0] == {
        "name": "add",
        "args": ["a", "b"],
        "doc": "Add two numbers.",
    }
    cls = index["math_utils.py"]["classes"][0]
    assert cls["name"] == "Calculator"
    assert cls["methods"][0]["name"] == "multiply"
    assert cls["methods"][0]["args"] == ["self", "x", "y"]


def test_query_memory_returns_matching_signature_text(tmp_path):
    """Scenario: query the built index for a function name. Expected: the result contains that function signature and summary line."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "math_utils.py").write_text(
        'def add(a, b):\n    """Add two numbers."""\n    return a + b\n',
        encoding="utf-8",
    )

    memory = CodebaseMemory(str(workspace), db_path=tmp_path / "agent_memory.json")
    memory.build_index()

    result = memory.query_memory(["add"])

    assert "- math_utils.py: def add(a, b) -> Add two numbers." in result


def test_coder_agent_injects_codebase_signature_reference(tmp_path):
    """Scenario: CoderAgent generates code with a workspace path and a fake model. Expected: the prompt sent to the model includes the signature reference for existing functions."""
    from app.engine.swarm_agents import CoderAgent

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "math_utils.py").write_text(
        'def add(a, b):\n    """Add two numbers."""\n    return a + b\n',
        encoding="utf-8",
    )
    prompts = []

    def fake_llm(prompt, **kwargs):
        """Fake model that records the prompt and returns canned code."""
        prompts.append(prompt)
        return {"choices": [{"text": "def use_add(a, b):\n    return add(a, b)\n"}]}

    # Replace the model loader so no real model is needed
    with patch("app.engine.swarm_agents.ModelLoader.get_instance", return_value=fake_llm):
        agent = CoderAgent()
        result = agent.generate(
            {"filepath": "consumer.py", "instructions": "Call add from math_utils"},
            {"consumer.py": ""},
            workspace_path=str(workspace),
        )

    assert "use_add" in result
    assert prompts
    assert "Codebase Interfaces & Signatures Reference" in prompts[0]
    assert "def add(a, b) -> Add two numbers." in prompts[0]
