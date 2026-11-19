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
    if _path not in sys.path:
        sys.path.insert(0, _path)

import generate_code_sft_dataset as gen  # noqa: E402
import curate_code_datasets as curate  # noqa: E402


class TestGenerateCodeSftDataset(unittest.TestCase):
    """Tests for the codebase-scrape generator tools/generate_code_sft_dataset.py using throwaway directory trees."""
    def setUp(self):
        """Create a temp directory to hold fake source trees."""
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        """Delete the temp directory."""
        self._tmp.cleanup()

    def _write(self, relpath: str, content: str) -> Path:
        """Write a dedented source file at a relative path (creating parent folders) and return its path."""
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content), encoding="utf-8")
        return path

    def test_crawl_extracts_function_class_and_method(self):
        """One module yields function, class and method rows; every assistant code block parses on its own (regression check for method dedenting) and the user prompt contains typed args and the return annotation."""
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
            # Row layout is [system, user, assistant]; index 2 is the generated code answer, which is what must parse standalone.
            code = row["messages"][2]["content"].split("```python\n", 1)[1].rsplit("```", 1)[0]
            ast.parse(code)

        method_row = next(r for r in rows if r["kind"] == "method")
        self.assertEqual(method_row["qualname"], "Greeter.greet")
        method_code = method_row["messages"][2]["content"].split("```python\n", 1)[1]
        # Methods are extracted from inside a class body, so their source is indented; the generator must dedent it, otherwise the snippet starts with spaces and is not valid top-level Python.
        self.assertTrue(method_code.startswith("def greet"), method_code)

        function_row = next(r for r in rows if r["kind"] == "function")
        user_prompt = function_row["messages"][1]["content"]
        self.assertIn("a: int, b: int = 0", user_prompt)
        self.assertIn("It should return: int.", user_prompt)

    def test_ignores_venv_and_test_directories(self):
        """venv and test directories are not crawled, so third-party or test code never becomes training data."""
        self._write("pkg/real.py", "def keep():\n    return 1\n")
        # Virtualenv and tests folders would flood the dataset with vendored library code and assertions; the crawler prunes them by directory name.
        self._write("pkg/venv/skip.py", "def skip():\n    return 2\n")
        self._write("pkg/tests/skip_too.py", "def skip_too():\n    return 3\n")

        rows, files_crawled, _ = gen.crawl([self.root])

        self.assertEqual(files_crawled, 1)
        self.assertEqual({row["qualname"] for row in rows}, {"keep"})

    def test_skips_files_with_syntax_errors(self):
        """A file that fails to parse is counted as skipped and produces no rows; the crawl continues."""
        # The file still counts as crawled (it was opened) but also as skipped; a single unparsable file must never abort the whole crawl.
        self._write("pkg/broken.py", "def broken(:\n    pass\n")

        rows, files_crawled, files_skipped = gen.crawl([self.root])

        self.assertEqual(files_crawled, 1)
        self.assertEqual(files_skipped, 1)
        self.assertEqual(rows, [])


class TestCurateAndMergeDatasets(unittest.TestCase):
    """Tests for the quality gates and merging in tools/curate_code_datasets.py."""
    def test_is_syntactically_valid_accepts_fenced_valid_code(self):
        """Valid Python inside a ```python fence passes the syntax gate."""
        self.assertTrue(curate.is_syntactically_valid("```python\ndef ok():\n    return 1\n```"))

    def test_is_syntactically_valid_rejects_broken_code(self):
        """A fenced snippet with a syntax error is rejected."""
        self.assertFalse(curate.is_syntactically_valid("```python\ndef broken(:\n```"))

    def test_is_syntactically_valid_rejects_non_code_response(self):
        """Prose with no code is not valid Python and is rejected; only code answers belong in the code dataset."""
        self.assertFalse(curate.is_syntactically_valid("Here is the code:\n"))

    def test_estimate_tokens_uses_char_length_heuristic(self):
        """Token estimate is len/4 (400 chars -> 100) so no tokenizer is needed."""
        # Pins the len/4 heuristic: a cheap approximation that avoids loading a tokenizer inside a data tool.
        self.assertEqual(curate.estimate_tokens("a" * 400), 100)

    def test_passes_quality_checks_rejects_oversized_prompt(self):
        """A row whose estimated tokens exceed max_tokens is rejected."""
        row = {
            "messages": [
                {"role": "system", "content": "sys"},
                # 20000 chars is about 5000 estimated tokens, far above the max_tokens=100 budget, so rejection can only come from the size gate (the code itself is valid).
                {"role": "user", "content": "u" * 20000},
                {"role": "assistant", "content": "```python\ndef ok():\n    return 1\n```"},
            ]
        }
        self.assertFalse(curate.passes_quality_checks(row, max_tokens=100))

    def test_passes_quality_checks_accepts_small_valid_row(self):
        """A small row with valid assistant code passes."""
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
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            curated_path = tmp_path / "sft.jsonl"
            synthetic_path = tmp_path / "synthetic.jsonl"
            out_path = tmp_path / "merged.jsonl"

            def make_row(name: str, valid: bool = True, oversized: bool = False) -> dict:
                assistant = f"```python\ndef {name}():\n    return 1\n```" if valid else "def broken(:\n    pass"
                user_content = ("x" * 20000) if oversized else "short prompt"
                return {
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant},
                    ]
                }

            # Each source gets one good and one bad row so the test can distinguish per-source counting from filtering: totals are counted before the gates, valid_total after.
            curated_rows = [make_row("a"), make_row("bad", valid=False)]
            synthetic_rows = [make_row("b"), make_row("huge", oversized=True)]

            with curated_path.open("w", encoding="utf-8") as fh:
                for r in curated_rows:
                    fh.write(json.dumps(r) + "\n")
            with synthetic_path.open("w", encoding="utf-8") as fh:
                for r in synthetic_rows:
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
            # Training loaders expect exactly the chat format; extra metadata such as kind or qualname must be stripped on merge.
            self.assertEqual(set(out_rows[0].keys()), {"messages"})

    def test_merge_datasets_includes_scraped_library_sft_when_present(self):
        """A scraped_library_sft.jsonl next to the curated file is merged in automatically and reported as scraped_library_total."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            curated_path = tmp_path / "sft.jsonl"
            synthetic_path = tmp_path / "synthetic.jsonl"
            # The scraped-library file is optional and found by name beside the curated file, so no extra CLI argument is needed when it exists.
            scraped_library_path = tmp_path / "scraped_library_sft.jsonl"
            out_path = tmp_path / "merged.jsonl"

            def make_row(name: str) -> dict:
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
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stats = curate.merge_datasets(
                tmp_path / "missing_curated.jsonl",
                tmp_path / "missing_synthetic.jsonl",
                tmp_path / "merged.jsonl",
            )
            self.assertEqual(stats["combined_total"], 0)
            self.assertEqual(stats["valid_total"], 0)

    def test_json_fence_does_not_break_following_python_block(self):
        """A ```json block before a ```python block must not make the python body unparseable.

        The old regex only matched python/untagged openers, so the json block's closing fence
        paired with the python opener and the snippet became garbage.
        """
        reply = "Config:\n```json\n{\"a\": 1}\n```\nCode:\n```python\nx = 1\n```\n"
        self.assertEqual(curate._code_snippets(reply), ["x = 1\n"])
        self.assertTrue(curate.is_syntactically_valid(reply))


if __name__ == "__main__":
    unittest.main()
