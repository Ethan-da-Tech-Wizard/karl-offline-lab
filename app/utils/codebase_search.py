# ==============================================================================
# Program: codebase_search.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: High-speed codebase search and indexing utility.
#          Executes ripgrep and regex queries across project workspaces for
#          context retrieval.
# ==============================================================================

import logging
import subprocess
import os
from typing import List, Dict, Any

logger = logging.getLogger("karl.codebase_search")


def codebase_search(query: str, workspace_path: str) -> List[Dict[str, Any]]:
    """
    Invokes ripgrep (rg) from sub-process to search the codebase.
    Returns a list of matching lines with filepath, line number, column, and content.
    """
    # Step 1: Validate input parameters, operational preconditions, and state
    if not query:
        return []

    try:
        # Run ripgrep with --vimgrep for precise parser-friendly output:
        # filepath:line:col:content
        res = subprocess.run(
            ["rg", "--vimgrep", "--no-config", "--follow", query, workspace_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        results = []
        for line in res.stdout.splitlines():
            # Iterate over items, evaluate loop invariant, and update progress
            parts = line.split(":", 3)
            if len(parts) >= 4:
                abs_path = parts[0]
                rel_path = os.path.relpath(abs_path, workspace_path)
                try:
                    # Execute guarded operations and establish recovery boundaries
                    line_num = int(parts[1])
                    col_num = int(parts[2])
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
    except Exception as e:
        logger.warning(f"Error calling ripgrep: {e}")
        return []
