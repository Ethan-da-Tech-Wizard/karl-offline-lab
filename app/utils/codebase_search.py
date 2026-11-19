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

# Namespaced logger so search failures can be filtered separately from the rest of karl.
logger = logging.getLogger("karl.codebase_search")


# Never raises: any failure (rg missing, timeout, bad output) degrades to an empty list
# so callers can treat "no results" and "search unavailable" the same way.
def codebase_search(query: str, workspace_path: str) -> List[Dict[str, Any]]:
    """
    Invokes ripgrep (rg) from sub-process to search the codebase.
    Returns a list of matching lines with filepath, line number, column, and content.
    """
    # Step 1: Validate input parameters, operational preconditions, and state
    if not query:
        return []

    try:
        # --no-config ignores the user's ~/.ripgreprc so results are reproducible;
        # --follow traverses symlinks; the 10 s timeout keeps a huge workspace from stalling the caller.
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
            # Split on at most 3 colons so colons inside the matched text stay in `content`.
            # (Caveat: a path containing ":" would be mis-split; such lines fail the int() check below.)
            parts = line.split(":", 3)
            if len(parts) >= 4:
                abs_path = parts[0]
                # Report paths relative to the workspace so output does not leak absolute locations.
                rel_path = os.path.relpath(abs_path, workspace_path)
                try:
                    # Execute guarded operations and establish recovery boundaries
                    line_num = int(parts[1])
                    col_num = int(parts[2])
                # Not a "path:line:col:text" record (e.g. a path with a colon); skip it.
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
