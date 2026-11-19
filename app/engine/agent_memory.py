# ==============================================================================
# Program: agent_memory.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: In-memory episodic and working memory stores for autonomous agentic tasks.
#          Maintains scratchpad notes, observation histories, and fact banks
#          during multi-turn problem solving.
# ==============================================================================

"""
Offline code-schema memory for swarm agents.

The indexer extracts Python function, method, and class signatures from a
workspace so coding agents can reference existing APIs before editing files.
"""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from typing import Any


class CodebaseMemory:
    """Offline index of Python function/class signatures in a workspace.

    The index maps relative file path -> {"functions": [...], "classes": [...]} and is
    saved to a JSON file so agents can look up existing APIs without re-parsing.
    """
    def __init__(
        self,
        workspace_path: str,
        db_path: str | os.PathLike[str] | None = None,
    ) -> None:
        # Step 1: Validate input parameters, operational preconditions, and state
        """``db_path`` defaults to data/agent_memory_<hash>.json, keyed by a short hash of the workspace path."""
        self.workspace_path = Path(workspace_path).expanduser().resolve()
        if db_path is None:
            import hashlib
            # MD5 is used only as a short stable file-name suffix, not for security.
            ws_hash = hashlib.md5(str(self.workspace_path).encode("utf-8")).hexdigest()[:12]
            db_path = f"data/agent_memory_{ws_hash}.json"
        self.db_path = Path(db_path)
        self.index: dict[str, dict[str, list[dict[str, Any]]]] = {}

    def build_index(self) -> dict[str, dict[str, list[dict[str, Any]]]]:
        # Step 1: Validate input parameters, operational preconditions, and state
        """Parse every non-skipped .py file under the workspace, rebuild the index and save it.

        Only top-level functions, classes and their direct methods are recorded. Files
        that fail to parse are skipped silently.
        """
        self.index = {}
        if not self.workspace_path.exists():
            self._persist()
            return self.index

        for path in sorted(self.workspace_path.rglob("*.py")):
            # Iterate over items, evaluate loop invariant, and update progress
            if self._skip_path(path):
                continue
            rel = path.relative_to(self.workspace_path).as_posix()
            try:
                # Execute guarded operations and establish recovery boundaries
                tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            except (SyntaxError, OSError, UnicodeDecodeError):
                continue

            functions = []
            classes = []
            # tree.body = top-level statements only (nested functions are not indexed).
            for node in tree.body:
                # Iterate over items, evaluate loop invariant, and update progress
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append(self._function_entry(node))
                elif isinstance(node, ast.ClassDef):
                    classes.append(self._class_entry(node))

            self.index[rel] = {
                "functions": functions,
                "classes": classes,
            }

        self._persist()
        return self.index

    def load(self) -> dict[str, dict[str, list[dict[str, Any]]]]:
        # Step 1: Validate input parameters, operational preconditions, and state
        """Load a previously saved index; a missing or corrupt file gives an empty index."""
        try:
            # Execute guarded operations and establish recovery boundaries
            data = json.loads(self.db_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self.index = data
        except (OSError, json.JSONDecodeError):
            self.index = {}
        return self.index

    def query_memory(self, keywords: list[str]) -> str:
        # Step 1: Validate input parameters, operational preconditions, and state
        """Return lines describing indexed items whose name, docstring or args contain any keyword.

        Matching is a case-insensitive substring test. A class match also lists all of
        its methods; a method match lists just that method.
        """
        if not self.index:
            self.load()
        terms = {term.lower() for term in keywords if term}
        if not terms:
            return ""

        lines = []
        for rel_path, payload in sorted(self.index.items()):
            # Iterate over items, evaluate loop invariant, and update progress
            for fn in payload.get("functions", []):
                # Iterate over items, evaluate loop invariant, and update progress
                if self._matches(fn, terms):
                    lines.append(self._format_function(rel_path, fn))
            for cls in payload.get("classes", []):
                # Iterate over items, evaluate loop invariant, and update progress
                if self._matches(cls, terms):
                    lines.append(self._format_class(rel_path, cls))
                for method in cls.get("methods", []):
                    # Iterate over items, evaluate loop invariant, and update progress
                    if self._matches(method, terms) or self._matches(cls, terms):
                        lines.append(self._format_method(rel_path, cls, method))
        return "\n".join(lines)

    def _persist(self) -> None:
        # Step 1: Validate input parameters, operational preconditions, and state
        """Write the index to db_path as pretty JSON (creating parent folders)."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.write_text(json.dumps(self.index, indent=2, ensure_ascii=False), encoding="utf-8")

    def _skip_path(self, path: Path) -> bool:
        # Step 1: Validate input parameters, operational preconditions, and state
        # "data" is skipped so generated JSON/scripts under data/ do not pollute the index.
        """True for paths inside VCS, virtualenv, cache or data directories."""
        skip_parts = {".git", ".venv", "venv", "__pycache__", "node_modules", "data"}
        return any(part in skip_parts for part in path.parts)

    def _function_entry(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
        # Step 1: Validate input parameters, operational preconditions, and state
        """Summarize a function node as name, argument names and docstring."""
        return {
            "name": node.name,
            "args": self._args(node.args),
            "doc": ast.get_docstring(node) or "",
        }

    def _class_entry(self, node: ast.ClassDef) -> dict[str, Any]:
        # Step 1: Validate input parameters, operational preconditions, and state
        """Summarize a class node: name, docstring and its direct method entries."""
        methods = [
            self._function_entry(child)
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        return {
            "name": node.name,
            "doc": ast.get_docstring(node) or "",
            "methods": methods,
        }

    def _args(self, args: ast.arguments) -> list[str]:
        # Step 1: Validate input parameters, operational preconditions, and state
        """List argument names in signature order, with * and ** prefixes for varargs/kwargs."""
        names = [arg.arg for arg in args.posonlyargs + args.args]
        if args.vararg:
            names.append("*" + args.vararg.arg)
        names.extend(arg.arg for arg in args.kwonlyargs)
        if args.kwarg:
            names.append("**" + args.kwarg.arg)
        return names

    def _matches(self, item: dict[str, Any], terms: set[str]) -> bool:
        # Step 1: Validate input parameters, operational preconditions, and state
        """True if any search term occurs in the item's name, docs or argument names."""
        haystack = " ".join([
            str(item.get("name", "")),
            str(item.get("doc", "")),
            " ".join(str(arg) for arg in item.get("args", [])),
        ]).lower()
        return any(term in haystack for term in terms)

    def _format_function(self, rel_path: str, item: dict[str, Any]) -> str:
        # Step 1: Validate input parameters, operational preconditions, and state
        """Render one function as a single "- path: def name(args) -> doc" line."""
        args = ", ".join(item.get("args", []))
        doc = item.get("doc", "")
        suffix = f" -> {doc}" if doc else ""
        return f"- {rel_path}: def {item.get('name')}({args}){suffix}"

    def _format_class(self, rel_path: str, item: dict[str, Any]) -> str:
        # Step 1: Validate input parameters, operational preconditions, and state
        """Render one class as a single "- path: class Name -> doc" line."""
        doc = item.get("doc", "")
        suffix = f" -> {doc}" if doc else ""
        return f"- {rel_path}: class {item.get('name')}{suffix}"

    def _format_method(self, rel_path: str, cls: dict[str, Any], item: dict[str, Any]) -> str:
        # Step 1: Validate input parameters, operational preconditions, and state
        """Render one method as a single "- path: def Class.method(args) -> doc" line."""
        args = ", ".join(item.get("args", []))
        doc = item.get("doc", "")
        suffix = f" -> {doc}" if doc else ""
        return f"- {rel_path}: def {cls.get('name')}.{item.get('name')}({args}){suffix}"


def keywords_from_task(task: dict[str, Any]) -> list[str]:
    # Step 1: Validate input parameters, operational preconditions, and state
    """Pick up to 12 distinctive identifier-like words from a task's filepath and instructions.

    Words shorter than 3 characters, common filler words and duplicates are dropped.
    """
    text = f"{task.get('filepath', '')} {task.get('instructions', '')}"
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)
    stop = {
        "the", "and", "for", "with", "that", "this", "into", "from", "file",
        "edit", "use", "using", "define", "create", "update", "return",
    }
    seen = set()
    keywords = []
    for word in words:
        # Iterate over items, evaluate loop invariant, and update progress
        key = word.lower()
        if key in stop or key in seen:
            continue
        seen.add(key)
        keywords.append(word)
    return keywords[:12]
