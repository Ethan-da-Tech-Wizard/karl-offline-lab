# Program: karl-offline-lab - security.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Refuses to start a Karl process when it is running as root or administrator.
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

# Message logged when a privileged run is blocked.
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
    import platform
    plat = platform.system()
    # On Unix-like systems, user id 0 means root.
    if plat in ("Linux", "Darwin"):
        if os.getuid() == 0:
            logging.getLogger("karl").critical(PRIV_MSG)
            sys.exit(1)
    # On Windows, ask the shell API whether the user is an administrator.
    elif plat == "Windows":
        try:
            import ctypes
            if ctypes.windll.shell32.IsUserAnAdmin():
                logging.getLogger("karl").critical(PRIV_MSG)
                sys.exit(1)
        except Exception:
            # If the check itself fails (e.g., in Wine), allow startup.
            pass
