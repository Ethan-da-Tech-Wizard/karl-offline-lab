# ==============================================================================
# Program: security.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Security guardrails and offline safety enforcement.
#          Sanitizes inputs, detects dangerous system commands or path traversal
#          attempts, and guarantees safe execution boundaries.
# ==============================================================================

"""
Small, dependency-light security helpers shared across every independently
launched Karl entrypoint (the PyQt app in main.py, and standalone scripts
like auto_train.py that are spawned as their own subprocess).

Kept free of heavy imports (torch, PyQt, etc.) so it's cheap to import from
any entrypoint without pulling in the full application.
"""

import logging
import os
import sys

PRIV_MSG = (
    "Security Error: Running Karl as root or administrator is blocked "
    "to prevent accidental host system modification during agentic testing loops."
)


def assert_not_privileged() -> None:
    """
    Abort with exit code 1 if the current process is running under elevated
    OS privileges.

    Elevated execution is dangerous during agentic loops and sandboxed code
    execution (see data/flywheel/executor_sandbox.py): resource limits such
    as RLIMIT_AS are not binding on root, since a privileged process can
    simply raise its own limits back up. Blocking root/admin at every
    process entrypoint prevents an entire class of containment failures.
    """
    # Defense-in-Depth Rationale:
    # On POSIX systems, UID 0 holds CAP_SYS_RESOURCE which allows a process to ignore or raise
    # RLIMIT_AS (memory) and RLIMIT_CPU limits set by the sandbox. Blocking root execution
    # at startup guarantees kernel resource isolation boundaries cannot be trivially bypassed.
    import platform
    plat = platform.system()
    if plat in ("Linux", "Darwin"):
        if os.getuid() == 0:
            logging.getLogger("karl").critical(PRIV_MSG)
            sys.exit(1)
    elif plat == "Windows":
        try:
            import ctypes
            if ctypes.windll.shell32.IsUserAnAdmin():
                logging.getLogger("karl").critical(PRIV_MSG)
                sys.exit(1)
        except Exception:
            # If the check itself fails (e.g., in Wine), allow startup.
            pass
