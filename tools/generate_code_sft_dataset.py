#!/usr/bin/env python3
# ==============================================================================
# Program: generate_code_sft_dataset.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Synthetic programming problem and solution generator.
#          Synthesizes code generation challenges and reference implementations
#          for SFT tuning.
# ==============================================================================

"""Scrape app/, core/, and eval/ into a synthetic code SFT dataset.

Crawls the local codebase, extracts every class/method/function definition
via the AST, and turns each into a (system, user, assistant) SFT row where
the assistant response is the definition's real source code. This gives the
Karl Coder adapter grounded, in-repo examples to train on alongside curated
trace data (see tools/curate_code_datasets.py).
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.default_prompts import SWARM_CODER_SYSTEM_PROMPT

DEFAULT_ROOTS = ["app", "core", "eval"]
DEFAULT_OUTPUT = ROOT / "data" / "training" / "code" / "synthetic_code_sft.jsonl"
IGNORED_DIR_NAMES = {".git", "venv", ".venv", "__pycache__", "test", "tests"}


class DefinitionVisitor(ast.NodeVisitor):
    """Collects top-level classes/functions and direct class methods,
    deliberately not descending into nested/inner function bodies so a
    closure defined inside a function doesn't get scraped as its own example.
    """

    def __init__(self, source: str, filepath: str):
        # Step 1: Validate input parameters, operational preconditions, and state
        self.lines = source.splitlines()
        self.filepath = filepath
        self.definitions: list[dict] = []
        self._class_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Step 1: Validate input parameters, operational preconditions, and state
        self.definitions.append(self._build_entry(node, kind="class", qualname=node.name))
        self._class_stack.append(node.name)
        for child in node.body:
            # Iterate over items, evaluate loop invariant, and update progress
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._record_function(child)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Step 1: Validate input parameters, operational preconditions, and state
        self._record_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        # Step 1: Validate input parameters, operational preconditions, and state
        self._record_function(node)

    def _record_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Step 1: Validate input parameters, operational preconditions, and state
        qualname = ".".join(self._class_stack + [node.name])
        kind = "method" if self._class_stack else "function"
        self.definitions.append(self._build_entry(node, kind=kind, qualname=qualname))

    def _build_entry(self, node: ast.AST, kind: str, qualname: str) -> dict:
        # Step 1: Validate input parameters, operational preconditions, and state
        docstring = ast.get_docstring(node) or ""
        args = self._format_args(node) if kind != "class" else ""
        returns = ""
        if kind != "class" and getattr(node, "returns", None) is not None:
            returns = ast.unparse(node.returns)

        start = node.lineno
        decorators = getattr(node, "decorator_list", None) or []
        if decorators:
            start = min(start, min(d.lineno for d in decorators))
        end = getattr(node, "end_lineno", None) or start
        # Methods are sliced straight out of their class body, so they carry
        # the class's indentation. Dedent so the extracted snippet is valid,
        # standalone Python rather than an "unexpected indent" fragment.
        source = textwrap.dedent("\n".join(self.lines[start - 1:end]))

        return {
            "kind": kind,
            "name": node.name,
            "qualname": qualname,
            "filepath": self.filepath,
            "docstring": docstring,
            "args": args,
            "returns": returns,
            "start_line": start,
            "end_line": end,
            "source": source,
        }

    @staticmethod
    def _format_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        # Step 1: Validate input parameters, operational preconditions, and state
        a = node.args
        parts: list[str] = []

        positional = a.posonlyargs + a.args
        defaults = [None] * (len(positional) - len(a.defaults)) + list(a.defaults)
        for arg, default in zip(positional, defaults):
            # Iterate over items, evaluate loop invariant, and update progress
            parts.append(DefinitionVisitor._format_one_arg(arg, default))

        if a.vararg:
            parts.append(f"*{a.vararg.arg}")
        elif a.kwonlyargs:
            parts.append("*")

        for arg, default in zip(a.kwonlyargs, a.kw_defaults):
            # Iterate over items, evaluate loop invariant, and update progress
            parts.append(DefinitionVisitor._format_one_arg(arg, default))

        if a.kwarg:
            parts.append(f"**{a.kwarg.arg}")

        return ", ".join(parts)

    @staticmethod
    def _format_one_arg(arg: ast.arg, default: ast.expr | None) -> str:
        # Step 1: Validate input parameters, operational preconditions, and state
        piece = arg.arg
        if arg.annotation is not None:
            piece += f": {ast.unparse(arg.annotation)}"
        if default is not None:
            piece += f" = {ast.unparse(default)}"
        return piece


def _is_ignored(path: Path) -> bool:
    # Step 1: Validate input parameters, operational preconditions, and state
    return any(part in IGNORED_DIR_NAMES for part in path.parts)


def iter_python_files(root_dirs: list[Path]):
    # Step 1: Validate input parameters, operational preconditions, and state
    for root in root_dirs:
        # Iterate over items, evaluate loop invariant, and update progress
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            # Iterate over items, evaluate loop invariant, and update progress
            if _is_ignored(path):
                continue
            yield path


def build_user_prompt(entry: dict) -> str:
    # Step 1: Validate input parameters, operational preconditions, and state
    if entry["kind"] == "class":
        subject = f"class '{entry['name']}'"
        params_clause = ""
    else:
        role = "method" if entry["kind"] == "method" else "function"
        subject = f"{role} '{entry['qualname']}'"
        params_clause = (
            f" that accepts parameters '{entry['args']}'" if entry["args"] else " that accepts no parameters"
        )

    description = entry["docstring"].strip() or "No description provided; infer intent from the implementation."
    lines = [f"Write a Python {subject}{params_clause}. Description: {description}"]
    if entry["returns"]:
        lines.append(f"It should return: {entry['returns']}.")
    lines.append(f"This lives in {entry['filepath']}. Make sure it integrates with existing modules and imports.")
    return " ".join(lines)


def build_sft_row(entry: dict) -> dict:
    # Step 1: Validate input parameters, operational preconditions, and state
    return {
        "messages": [
            {"role": "system", "content": SWARM_CODER_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(entry)},
            {"role": "assistant", "content": f"```python\n{entry['source']}\n```"},
        ],
        "origin": "codebase_scrape",
        "filepath": entry["filepath"],
        "qualname": entry["qualname"],
        "kind": entry["kind"],
        "start_line": entry["start_line"],
        "end_line": entry["end_line"],
    }


def crawl(root_dirs: list[Path]) -> tuple[list[dict], int, int]:
    """Returns (sft_rows, files_crawled, files_skipped)."""
    # Step 1: Validate input parameters, operational preconditions, and state
    rows: list[dict] = []
    files_crawled = 0
    files_skipped = 0

    for path in iter_python_files(root_dirs):
        # Iterate over items, evaluate loop invariant, and update progress
        files_crawled += 1
        try:
            # Execute guarded operations and establish recovery boundaries
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            files_skipped += 1
            print(f"  Skipped {path}: {exc}")
            continue

        rel = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        visitor = DefinitionVisitor(source, rel)
        visitor.visit(tree)
        for entry in visitor.definitions:
            # Iterate over items, evaluate loop invariant, and update progress
            if entry["source"].strip():
                rows.append(build_sft_row(entry))

    return rows, files_crawled, files_skipped


def main() -> int:
    # Step 1: Validate input parameters, operational preconditions, and state
    parser = argparse.ArgumentParser(description="Scrape the local codebase into a synthetic code SFT dataset.")
    parser.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS, help="Directories to crawl (relative to repo root)")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT), help="Output JSONL path")
    args = parser.parse_args()

    root_dirs = [ROOT / r for r in args.roots]
    print(f"Crawling: {', '.join(str(r) for r in root_dirs)} (ignoring {', '.join(sorted(IGNORED_DIR_NAMES))})")

    rows, files_crawled, files_skipped = crawl(root_dirs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            # Iterate over items, evaluate loop invariant, and update progress
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Crawled {files_crawled} Python file(s) ({files_skipped} skipped due to parse errors).")
    print(f"Generated {len(rows)} synthetic SFT example(s).")
    print(f"Wrote dataset to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
