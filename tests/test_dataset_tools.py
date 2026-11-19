# ==============================================================================
# Program: test_dataset_tools.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Tests for the codebase-scrape SFT generator (tools/generate_code_sft_dataset.py) and the dataset curation/merge/quality-validation pipeline (tools/curate_code_datasets.py).
#          Coordinates with adjacent subsystems within the Karl Offline Lab
#          architecture.
# ==============================================================================

"""Tests for the codebase-scrape SFT generator (tools/generate_code_sft_dataset.py)
and the dataset curation/merge/quality-validation pipeline (tools/curate_code_datasets.py).
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
for _path in (str(ROOT), str(TOOLS_DIR)):
    # Iterate over items, evaluate loop invariant, and update progress
    if _path not in sys.path:
        sys.path.insert(0, _path)

import generate_code_sft_dataset as gen  # noqa: E402
import curate_code_datasets as curate  # noqa: E402


class TestGenerateCodeSftDataset(unittest.TestCase):
    """Tests for the codebase-scrape generator tools/generate_code_sft_dataset.py using throwaway directory trees."""
    def setUp(self):
        """Create a temp directory to hold fake source trees."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        """Delete the temp directory."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._tmp.cleanup()

    def _write(self, relpath: str, content: str) -> Path:
        """Write a dedented source file at a relative path (creating parent folders) and return its path."""
        # Step 1: Validate input parameters, operational preconditions, and state
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content), encoding="utf-8")
        return path

    def test_crawl_extracts_function_class_and_method(self):
        """One module yields function, class and method rows; every assistant code block parses on its own (regression check for method dedenting) and the user prompt contains typed args and the return annotation."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._write("pkg/mod.py", '''
            def add(a: int, b: int = 0) -> int:
                """Add two numbers."""
                return a + b

            class Greeter:
                """Greets people."""

                def greet(self, name: str) -> str:
                    """Say hello."""
                    return f"Hello, {name}!"
        ''')

        rows, files_crawled, files_skipped = gen.crawl([self.root])

        self.assertEqual(files_crawled, 1)
        self.assertEqual(files_skipped, 0)
        self.assertEqual({row["kind"] for row in rows}, {"function", "class", "method"})

        # Every generated assistant code block must be independently valid,
        # standalone Python — regression check for the method-dedent fix.
        for row in rows:
            # Iterate over items, evaluate loop invariant, and update progress
            code = row["messages"][2]["content"].split("```python\n", 1)[1].rsplit("```", 1)[0]
            ast.parse(code)

        method_row = next(r for r in rows if r["kind"] == "method")
        self.assertEqual(method_row["qualname"], "Greeter.greet")
        method_code = method_row["messages"][2]["content"].split("```python\n", 1)[1]
        self.assertTrue(method_code.startswith("def greet"), method_code)

        function_row = next(r for r in rows if r["kind"] == "function")
        user_prompt = function_row["messages"][1]["content"]
        self.assertIn("a: int, b: int = 0", user_prompt)
        self.assertIn("It should return: int.", user_prompt)

    def test_ignores_venv_and_test_directories(self):
        """venv and test directories are not crawled, so third-party or test code never becomes training data."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._write("pkg/real.py", "def keep():\n    return 1\n")
        self._write("pkg/venv/skip.py", "def skip():\n    return 2\n")
        self._write("pkg/tests/skip_too.py", "def skip_too():\n    return 3\n")

        rows, files_crawled, _ = gen.crawl([self.root])

        self.assertEqual(files_crawled, 1)
        self.assertEqual({row["qualname"] for row in rows}, {"keep"})

    def test_skips_files_with_syntax_errors(self):
        """A file that fails to parse is counted as skipped and produces no rows; the crawl continues."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._write("pkg/broken.py", "def broken(:\n    pass\n")

        rows, files_crawled, files_skipped = gen.crawl([self.root])

        self.assertEqual(files_crawled, 1)
        self.assertEqual(files_skipped, 1)
        self.assertEqual(rows, [])


class TestCurateAndMergeDatasets(unittest.TestCase):
    """Tests for the quality gates and merging in tools/curate_code_datasets.py."""
    def test_is_syntactically_valid_accepts_fenced_valid_code(self):
        """Valid Python inside a ```python fence passes the syntax gate."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertTrue(curate.is_syntactically_valid("```python\ndef ok():\n    return 1\n```"))

    def test_is_syntactically_valid_rejects_broken_code(self):
        """A fenced snippet with a syntax error is rejected."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertFalse(curate.is_syntactically_valid("```python\ndef broken(:\n```"))

    def test_is_syntactically_valid_rejects_non_code_response(self):
        """Prose with no code is not valid Python and is rejected; only code answers belong in the code dataset."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertFalse(curate.is_syntactically_valid("Here is the code:\n"))

    def test_estimate_tokens_uses_char_length_heuristic(self):
        """Token estimate is len/4 (400 chars -> 100) so no tokenizer is needed."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertEqual(curate.estimate_tokens("a" * 400), 100)

    def test_passes_quality_checks_rejects_oversized_prompt(self):
        """A row whose estimated tokens exceed max_tokens is rejected."""
        # Step 1: Validate input parameters, operational preconditions, and state
        row = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u" * 20000},
                {"role": "assistant", "content": "```python\ndef ok():\n    return 1\n```"},
            ]
        }
        self.assertFalse(curate.passes_quality_checks(row, max_tokens=100))

    def test_passes_quality_checks_accepts_small_valid_row(self):
        """A small row with valid assistant code passes."""
        # Step 1: Validate input parameters, operational preconditions, and state
        row = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "write add()"},
                {"role": "assistant", "content": "```python\ndef add(a, b):\n    return a + b\n```"},
            ]
        }
        self.assertTrue(curate.passes_quality_checks(row, max_tokens=100))

    def test_merge_datasets_filters_invalid_and_oversized_then_shuffles(self):
        """merge_datasets counts rows per source, drops invalid-syntax and oversized rows, writes only the 'messages' key, and keeps the valid rows (order is shuffled)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            curated_path = tmp_path / "sft.jsonl"
            synthetic_path = tmp_path / "synthetic.jsonl"
            out_path = tmp_path / "merged.jsonl"

            def make_row(name: str, valid: bool = True, oversized: bool = False) -> dict:
                # Step 1: Validate input parameters, operational preconditions, and state
                assistant = f"```python\ndef {name}():\n    return 1\n```" if valid else "def broken(:\n    pass"
                user_content = ("x" * 20000) if oversized else "short prompt"
                return {
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant},
                    ]
                }

            curated_rows = [make_row("a"), make_row("bad", valid=False)]
            synthetic_rows = [make_row("b"), make_row("huge", oversized=True)]

            with curated_path.open("w", encoding="utf-8") as fh:
                for r in curated_rows:
                    # Iterate over items, evaluate loop invariant, and update progress
                    fh.write(json.dumps(r) + "\n")
            with synthetic_path.open("w", encoding="utf-8") as fh:
                for r in synthetic_rows:
                    # Iterate over items, evaluate loop invariant, and update progress
                    fh.write(json.dumps(r) + "\n")

            stats = curate.merge_datasets(curated_path, synthetic_path, out_path, max_tokens=100, seed=42)

            self.assertEqual(stats["curated_total"], 2)
            self.assertEqual(stats["synthetic_total"], 2)
            self.assertEqual(stats["combined_total"], 4)
            self.assertEqual(stats["valid_total"], 2)
            self.assertEqual(stats["discarded_total"], 2)

            self.assertTrue(out_path.exists())
            with out_path.open("r", encoding="utf-8") as fh:
                out_rows = [json.loads(line) for line in fh]
            self.assertEqual(len(out_rows), 2)
            bodies = {r["messages"][2]["content"] for r in out_rows}
            self.assertTrue(any("def a()" in b for b in bodies))
            self.assertTrue(any("def b()" in b for b in bodies))
            # only the "messages" key should survive into the merged output
            self.assertEqual(set(out_rows[0].keys()), {"messages"})

    def test_merge_datasets_includes_scraped_library_sft_when_present(self):
        """A scraped_library_sft.jsonl next to the curated file is merged in automatically and reported as scraped_library_total."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            curated_path = tmp_path / "sft.jsonl"
            synthetic_path = tmp_path / "synthetic.jsonl"
            scraped_library_path = tmp_path / "scraped_library_sft.jsonl"
            out_path = tmp_path / "merged.jsonl"

            def make_row(name: str) -> dict:
                # Step 1: Validate input parameters, operational preconditions, and state
                return {
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": f"How do I use {name}?"},
                        {"role": "assistant", "content": f"```python\ndef {name}():\n    return 1\n```"},
                    ]
                }

            with curated_path.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps(make_row("a")) + "\n")
            with synthetic_path.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps(make_row("b")) + "\n")
            with scraped_library_path.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps(make_row("numpy_arange")) + "\n")
                fh.write(json.dumps(make_row("pandas_read_csv")) + "\n")

            stats = curate.merge_datasets(curated_path, synthetic_path, out_path, seed=1)

            self.assertEqual(stats["curated_total"], 1)
            self.assertEqual(stats["synthetic_total"], 1)
            self.assertEqual(stats["scraped_library_total"], 2)
            self.assertEqual(stats["combined_total"], 4)
            self.assertEqual(stats["valid_total"], 4)

            with out_path.open("r", encoding="utf-8") as fh:
                out_rows = [json.loads(line) for line in fh]
            bodies = {r["messages"][2]["content"] for r in out_rows}
            self.assertTrue(any("def numpy_arange()" in b for b in bodies))
            self.assertTrue(any("def pandas_read_csv()" in b for b in bodies))

    def test_merge_datasets_scraped_library_sft_optional(self):
        """The scraped-library file is optional: when absent, merging works and reports zero for it."""
        # No scraped_library_sft.jsonl on disk at all — merge should proceed
        # exactly as before, just with scraped_library_total == 0.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            curated_path = tmp_path / "sft.jsonl"
            synthetic_path = tmp_path / "synthetic.jsonl"
            out_path = tmp_path / "merged.jsonl"
            curated_path.write_text("", encoding="utf-8")
            synthetic_path.write_text("", encoding="utf-8")

            stats = curate.merge_datasets(curated_path, synthetic_path, out_path)

            self.assertEqual(stats["scraped_library_total"], 0)
            self.assertEqual(stats["combined_total"], 0)

    def test_merge_datasets_missing_files_produce_empty_output(self):
        """If no input files exist the merge succeeds with zero rows instead of raising."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stats = curate.merge_datasets(
                tmp_path / "missing_curated.jsonl",
                tmp_path / "missing_synthetic.jsonl",
                tmp_path / "merged.jsonl",
            )
            self.assertEqual(stats["combined_total"], 0)
            self.assertEqual(stats["valid_total"], 0)


if __name__ == "__main__":
    unittest.main()
