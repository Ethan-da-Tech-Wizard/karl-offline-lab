# ==============================================================================
# Program: main.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Master entry point and runtime orchestrator for the Karl Offline Lab.
#          Configures environment variables, suppresses OS and library-level
#          noise (CUDA, PyTorch, HuggingFace), enforces strict offline isolation
#          guarantees, initializes the PyQt6 desktop interface, applies styling
#          themes, and manages global lifecycle shutdown.
# ==============================================================================

"""
Karl -- entry point.
All library noise is suppressed here, before any third-party import runs.
"""

# ── 1. Environment variables ──────────────────────────────────────────────────
# Ordering matters in this file: environment variables and warning filters must be
# set BEFORE torch / faiss / sentence-transformers are imported, because those
# libraries read them at import time. Do not reorder the sections below.
# PYTORCH_ALLOC_CONF is set first because the CUDA allocator reads it on first use.
import os

# Harden CUDA memory management against fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import sys
import io
import multiprocessing

# Force fully offline operation: no model downloads, no telemetry, no progress bars.
# TOKENIZERS_PARALLELISM=false avoids fork-related deadlock warnings from HF tokenizers.
# Silence HuggingFace network calls and telemetry
os.environ["HF_HUB_OFFLINE"]               = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"]    = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_OFFLINE"]         = "1"
os.environ["TOKENIZERS_PARALLELISM"]       = "false"
os.environ["TRITON_SUPPRESS_WARNINGS"]     = "1"

# ── 2. Python warnings ────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

# ── 3. Redirect the OS-level stderr during noisy library imports ──────────────
# Some C-extensions (triton, torch cpp) write directly to fd=2 before Python
# warning filters run. We swap stderr for /dev/null at the file-descriptor level
# for the duration of the heavy imports, then restore it.

# Used as `with _NullFD():` around the heavy imports below.
class _NullFD:
    """Context manager: redirects fd 2 (stderr) to os.devnull at C level."""
    # dup the real stderr so it can be restored, then point fd 2 at /dev/null.
    def __enter__(self):
        self._orig_fd2 = os.dup(2)          # save real stderr fd
        self._devnull  = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, 2)            # point fd 2 at /dev/null
        return self

    def __exit__(self, *_):
        # Restore the original stderr and close both temporary descriptors.
        os.dup2(self._orig_fd2, 2)           # restore real stderr
        os.close(self._orig_fd2)
        os.close(self._devnull)


with _NullFD():
    # All noisy imports go here -- their C-level stderr goes to /dev/null
    # (imported only for their side effects: warm the import cache and trigger one-time init quietly).
    # All noisy imports go here -- their C-level stderr goes to /dev/null
    import torch          # noqa: F401  (triggers triton + cpp warnings)
    import faiss          # noqa: F401
    from sentence_transformers import SentenceTransformer  # noqa: F401

# ── 4. Python-level logging: mute library loggers ─────────────────────────────
import logging

# Raise third-party loggers to CRITICAL and stop propagation so only Karl's own
# log records reach the console.
for _name in [
    "torch", "torch.distributed",
    "torch.distributed.elastic.multiprocessing.redirects",
    "triton", "transformers", "sentence_transformers",
    "huggingface_hub", "huggingface_hub.utils",
    "filelock", "faiss", "urllib3", "requests",
]:
    _log = logging.getLogger(_name)
    _log.setLevel(logging.CRITICAL)
    _log.propagate = False

# Silence absl (used by torch internally for the "W0523..." lines)
# (absl is optional, hence the try/except.)
# Silence absl (used by torch internally for the "W0523..." lines)
try:
    import absl.logging as _absl_log
    _absl_log.set_verbosity(_absl_log.FATAL)
    logging.getLogger("absl").setLevel(logging.CRITICAL)
except Exception:
    pass

# ── 4b. Karl's own logger ──────────────────────────────────────────────────────
# Every module logs through a child of the "karl" logger
# (logging.getLogger("karl.<module>")). Set KARL_DEBUG=1 for verbose output.
# CorrelationFilter stamps each record with a per-request correlation id (see
# app/utils/correlation_logger.py); the handler format below prints it.
from app.utils.correlation_logger import CorrelationFilter

_karl_log = logging.getLogger("karl")
_karl_log.setLevel(logging.DEBUG if os.environ.get("KARL_DEBUG") else logging.INFO)
_karl_handler = logging.StreamHandler()
_karl_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] [Corr: %(correlation_id)s] %(name)s: %(message)s"
))
_karl_handler.addFilter(CorrelationFilter())
_karl_log.addHandler(_karl_handler)
_karl_log.propagate = False

# ── 5. Application ────────────────────────────────────────────────────────────
# Qt and the main window are imported only after logging/env setup above.
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from app.ui.main_window import MainWindow

# Re-exported for backwards compatibility (tests and other entrypoints import
# these names from main). The actual implementation lives in core/security.py
# so lightweight standalone scripts (e.g. auto_train.py) can reuse the same
# check without pulling in main.py's heavy PyQt/torch imports.
from core.security import PRIV_MSG as _PRIV_MSG
from core.security import assert_not_privileged as _assert_not_privileged


def _run_headless() -> int:
    """
    Start Karl's local WebSocket bridge without constructing the PyQt window.

    This mode is intended for containerized editor integrations where the VS Code
    extension talks to Karl through WSS and no desktop UI should be spawned.
    """
    # Headless mode: signal handlers set an Event so SIGINT/SIGTERM shut the bridge
    # down cleanly (used by Docker/Kubernetes, which send SIGTERM).
    import signal
    import threading

    from app.engine.websocket_server import WebSocketServerManager

    log = logging.getLogger("karl")
    port = int(os.environ.get("KARL_WS_PORT", "8080"))
    stop_event = threading.Event()

    def stop(_signum=None, _frame=None):
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    # Start the bridge and wait (<= 15 s) for the bind attempt to finish; the manager
    # sets started_event even on failure, so check `server` to detect a failed bind.
    manager = WebSocketServerManager.get_instance(port=port)
    manager.started_event.wait(timeout=15.0)

    if manager.server is None:
        log.error("Headless WebSocket bridge failed to start.")
        WebSocketServerManager.reset_instance()
        return 1

    log.info("Karl headless bridge listening on port %d.", manager.port)
    try:
        stop_event.wait()
    finally:
        WebSocketServerManager.reset_instance()
    return 0


def main():
    # Refuse to run as root/admin (see core/security.py) before doing anything else.
    _assert_not_privileged()

    # ── Feature flags & boot guard ────────────────────────────────────────────
    # CLI maintenance commands below (--diagnose, --generate-key, --revoke, --safe-mode,
    # --enable-flag, --headless) are handled before any window is created; the
    # first four exit the process when done.
    # ── Feature flags & boot guard ────────────────────────────────────────────
    # Imported here (inside main) so the module-level noise-suppression above
    # runs before any Karl sub-module is imported.
    from app.engine.feature_flags import FeatureFlagStore, run_boot_guard
    _flags = FeatureFlagStore()

    if "--diagnose" in sys.argv:
        from app.utils.diagnostics import run_diagnostics
        sys.exit(run_diagnostics())

    # --generate-key --scope <role>: mint a scoped bridge token (roles map to scope
    # lists in ROLE_SCOPES), store it, print it once, and exit.
    if "--generate-key" in sys.argv:
        from app.utils.keychain_manager import ROLE_SCOPES, add_scoped_token
        import uuid as _uuid

        scope_val: str = ""
        try:
            scope_val = sys.argv[sys.argv.index("--scope") + 1]
        except (ValueError, IndexError):
            pass

        if not scope_val or scope_val not in ROLE_SCOPES:
            print(
                "[Karl --generate-key] --scope must be one of: "
                + ", ".join(ROLE_SCOPES)
            )
            sys.exit(1)

        scopes = ROLE_SCOPES[scope_val]
        token = _uuid.uuid4().hex
        add_scoped_token(token, scopes)
        print(f"[Karl --generate-key] Token:  {token}")
        print(f"[Karl --generate-key] Scopes: {scopes}")
        sys.exit(0)

    # --revoke: delete the on-disk token file and keychain entry, and force-close
    # live connections if a server exists in this process.
    if "--revoke" in sys.argv:
        token_path = "data/bridge_token.json"
        if os.path.exists(token_path):
            os.remove(token_path)
            print(f"[Karl --revoke] Removed {token_path}")
        else:
            print(f"[Karl --revoke] {token_path} not present — nothing to wipe")
        try:
            from app.utils.keychain_manager import revoke_tokens
            revoke_tokens()
            print("[Karl --revoke] Bridge token revoked from OS keychain.")
        except Exception as exc:
            print(f"[Karl --revoke] Keychain revocation failed: {exc}")
        # Force-close connections if this process owns a running server instance.
        try:
            from app.engine.websocket_server import WebSocketServerManager
            inst = WebSocketServerManager._instance
            if inst is not None:
                inst.force_revoke()
                print("[Karl --revoke] Active WebSocket connections force-closed.")
        except Exception:
            pass
        sys.exit(0)

    if "--safe-mode" in sys.argv:
        _flags.enter_safe_mode()
        logging.getLogger("karl").info("Safe Mode active (--safe-mode flag).")

    if "--enable-flag" in sys.argv:
        try:
            _flag_name = sys.argv[sys.argv.index("--enable-flag") + 1]
            _flags.set_flag(_flag_name, True)
            print(f"[Karl --enable-flag] Enabled feature flag: {_flag_name}")
        except (ValueError, IndexError):
            print("[Karl --enable-flag] Usage: --enable-flag <flag_name>")
            sys.exit(1)
        except KeyError as exc:
            print(f"[Karl --enable-flag] Unknown flag: {exc}")
            sys.exit(1)

    # Headless mode: bridge only, no Qt window.
    if "--headless" in sys.argv or os.environ.get("KARL_HEADLESS") == "1":
        sys.exit(_run_headless())

    # ── Boot guard (Qt path only) ─────────────────────────────────────────────
    # Skip if --safe-mode was passed (user is already aware of degraded state).
    if "--safe-mode" not in sys.argv:
        run_boot_guard(_flags)

    # Normal desktop startup: create the Qt app, icon and base stylesheet, then the main window.
    app = QApplication(sys.argv)
    app.setApplicationName("Karl")
    icon_path = os.path.join(os.path.dirname(__file__), "oss/vss_extension/media/icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # Minimal inline base stylesheet (pure ASCII, no file read).
    # Full theme applied by MainWindow after init.
    # Imported lazily so themes (and Qt widgets) load after the noise suppression above.
    from app.ui.themes import stylesheet
    app.setStyleSheet(stylesheet())

    window = MainWindow()
    window.show()
    
    # app.exec() blocks until the window closes; the bridge token is then revoked so a
    # stale token cannot be reused after the app exits.
    code = app.exec()
    
    try:
        from app.utils.keychain_manager import revoke_tokens
        revoke_tokens()
    except Exception:
        pass
        
    sys.exit(code)


if __name__ == "__main__":
    main()
