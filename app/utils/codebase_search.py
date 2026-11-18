# Program: karl-offline-lab - codebase_search.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Searches a workspace folder with the ripgrep command-line tool and returns the matches as structured records.

"""Codebase text search helper that runs ripgrep and returns structured matches."""

# Standard-library imports and typing helpers used by the search function.
import logging
import subprocess
import os
from typing import List, Dict, Any

# Logger for this module; warnings are emitted if ripgrep cannot be run.
logger = logging.getLogger("karl.codebase_search")


def codebase_search(query: str, workspace_path: str) -> List[Dict[str, Any]]:
    """
    Invokes ripgrep (rg) from sub-process to search the codebase.
    Returns a list of matching lines with filepath, line number, column, and content.
    """
    # Nothing to search for, so skip launching a subprocess.
    if not query:
        return []

    try:
        # The 10 second timeout stops a huge workspace from hanging the caller.
        # Run ripgrep with --vimgrep for precise parser-friendly output:
        # filepath:line:col:content
        res = subprocess.run(
            ["rg", "--vimgrep", "--no-config", "--follow", query, workspace_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        results = []
        # Each output line looks like path:line:column:text; split into at most 4 parts so colons in the text survive.
        for line in res.stdout.splitlines():
            parts = line.split(":", 3)
            if len(parts) >= 4:
                abs_path = parts[0]
                # Report paths relative to the workspace so callers do not see absolute locations.
                rel_path = os.path.relpath(abs_path, workspace_path)
                try:
                    line_num = int(parts[1])
                    col_num = int(parts[2])
                # Skip any line whose line/column are not numbers (malformed output).
                except ValueError:
                    continue
                content = parts[3]
                results.append({
                    "filepath": rel_path,
                    "line": line_num,
                    "column": col_num,
                    "content": content
                })
        return results
    # Any failure (rg missing, timeout, etc.) degrades to an empty result instead of crashing the caller.
    except Exception as e:
        logger.warning(f"Error calling ripgrep: {e}")
        return []
