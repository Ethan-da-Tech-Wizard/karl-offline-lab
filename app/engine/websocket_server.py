# ==============================================================================
# Program: websocket_server.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Encrypted local WebSocket (WSS) server exposing Karl's bridge API.
#          Enables IDE extensions (Neovim, VS Code) to submit prompts and stream
#          responses securely.
# ==============================================================================

"""
WebSocket Server Manager — Karl Workbench
==========================================
Hosts a local WebSocket server to bridge communication between Karl's Multi-Agent Swarm
and editor extensions (such as VS Code/Code OSS).
"""

# Who uses this module
#   * main_window.py creates the singleton (WebSocketServerManager.get_instance) on the GUI thread and
#     calls stop() on shutdown; main.py's headless entrypoint does the same without a window.
#   * Clients: the VS Code extension (oss/vss_extension/media/karl_socket.js), neovim/karl.lua and
#     remote_rpc_client.py all speak the protocol implemented here (docs/08_vscode_extension.md).
# Wire protocol summary
#   * Requests: {"jsonrpc": "2.0", "id": <any>, "method": <name in _RPC_METHODS>, "params": {...}}.
#   * Replies: {"jsonrpc": "2.0", "id": <same>, "result": ...} or {"error": {"code", "message", "data"}}.
#   * Server pushes (no "id"): "notifications" such as chat_response_token, file_edited, status_update;
#     they are broadcast to every connected client by _send_notification (see the known-limits note below).
# Known limits (documented, not yet fixed)
#   * The bearer token is passed in the connection URL (?token=...), which can end up in proxy and
#     access logs; browser WebSocket APIs cannot set an Authorization header, which is why it is done.
#   * Notifications are broadcast to ALL clients regardless of their scopes, so a read-only client also
#     receives swarm file contents and another client's chat tokens.
# How to test: tests/test_websocket_bridge.py, test_websocket_contract.py, test_rbac_scopes.py.
# RPC method reference (scope in brackets; "-" means any authenticated connection)
#   authenticate [-]            params {token}                       -> {authenticated: true}; legacy no-op after the handshake
#   refresh_token [-]           params {token?}                      -> {token, expires_at}; extends the 12 h lease
#   get_runtime_status [read:telemetry]  -> {bridge, runtime, model, adapter, system}
#   list_models [-]             -> {active, models[]} (registry + local GGUF files)
#   list_adapters [read:telemetry]       -> {adapters[]}
#   set_active_model [admin:execute]     params {filename, adapter?} -> {active, loaded: false, message}
#   list_prompt_pairs [-]       -> {pairs[]}
#   get_prompt_pair [-]         params {name}                        -> the saved pair JSON
#   save_prompt_pair [admin:execute]     params {name, system_a, user_a, ...} -> {name, saved}
#   delete_prompt_pair [admin:execute]   params {name}               -> {name, deleted}
#   list_kb_sources [read:kb]   -> knowledge-base snapshot
#   ingest_path [write:kb]      params {path, recursive?, chunk_size?, overlap?} -> {status: started, task_id}
#   search_kb [read:kb]         params {query, top_k?, threshold?, source_filter?} -> {results[], snapshot}
#   submit_task [admin:execute] params {objective, workspace_path, test_command?, hyperparams?} -> {status: started}
#   swarm_commit_edits [admin:execute]   params {selected_filepaths[]} -> {status: committed, count}
#   rollback_swarm_run [admin:execute]   params {workspace_path, run_id?} -> restore report
#   swarm_inject_guidance / swarm_pause / swarm_resume / swarm_step / swarm_override_task [admin:execute]
#       steer the active swarm run (see the notes at each handler)
#   swarm_get_history [read:telemetry]   params {run_id?} -> run details or {runs[]}
#   submit_chat [admin:execute] params {message, workspace_path?, hyperparams?, agent?} -> {status: started}
#   stop_task [admin:execute]   -> {status: idle | stopping}
#   get_agent_profiles [read:telemetry]  -> all agent profiles
#   save_agent_profile [admin:execute]   params {profile} -> {status: saved, id, profile}
#   create_custom_agent [admin:execute]  params {name, label?, prompt?, base_model?, adapter?, ...} -> {status, name}
#   list_custom_agents [-]      -> user-defined profiles only
#   list_codex_topics / get_codex_content [-]  browse the bundled reference library
#   compute_diff [-]            params {text_a, text_b}              -> {diff_html}
#   start_auto_train [admin:execute]     params {topic, adapter_name, count?, epochs?, lr?} -> {status: started}
#   fit_vectorizer [write:kb]   params {documents[]}                 -> {vocabulary[], vectors[][]}
#   start_mini_train [admin:execute]     params {lr?, max_iters?, batch_size?} -> {status: started, task_id}
# Server notifications (no id; broadcast to all clients)
#   status_update {message}; chat_thought_token / chat_response_token {token}; chat_finished {error?};
#   task_plan_created {plan}; file_edited {filepath, content}; test_result {passed, error_trace};
#   proposal_verification_finished {layer, passed, trace}; edits_proposed {proposals}; finished_swarm {success, summary, run_id};
#   swarm_* telemetry (candidates_generated, winner_selected, memory_recalled, specialist_review,
#   concurrency_adjusted, guidance_injected, drift_detected, paused, telemetry);
#   kb_ingest_progress / kb_ingest_finished; auto_train_log / auto_train_finished; agent_profiles_updated.
# Design notes:
#   * Transport: JSON-RPC 2.0 over WebSocket. Loopback binds use plain WS (so
#     editor webviews accept it); any non-loopback bind requires TLS and fails
#     closed without a certificate (see _start_server).
#   * Auth: bearer tokens persisted in data/bridge_token.json, each mapped to a
#     list of scopes; METHOD_SCOPES decides which scope each RPC method needs.
#   * Threading: an asyncio loop runs on a dedicated daemon thread, while Qt
#     objects (orchestrator, chat threads) live on the GUI thread. Shared state
#     is guarded by _threads_lock / _clients_lock.

# --- standard library and third-party imports ---
import logging
import os
import json
import asyncio
import threading
import re
import time
import uuid
import ssl
import subprocess
import pathlib
from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qs
import websockets
from websockets.datastructures import Headers
from websockets.http11 import Response
from typing import Set, Optional, Any
from PyQt6.QtCore import Qt

# --- Karl internals: engine, RAG, UI helpers reused by RPC handlers, key storage ---
from app.engine import config_store
from app.engine.swarm_orchestrator import SwarmOrchestratorThread, rollback_swarm_run
from app.engine.inference_service import InferenceService
from app.engine.model_loader import CircuitBreakerOpenException, ModelLoader
from app.utils.rag_pipeline import RAGPipeline
from app.ui.workspaces.docs_data import DEFAULT_LIBRARY
from app.ui.workspaces.prompt_lab import generate_char_diff_html
from app.utils.keychain_manager import save_cached_token
from app.utils.correlation_logger import new_correlation_id, set_correlation_id
from app.utils.swarm_agent_profiles import load_agent_profiles, save_agent_profile


# Every log line from the bridge uses the "karl.websocket" logger so operators can filter or raise its
# level independently of the rest of the application.
# Module logger; handlers and levels are configured by the application entrypoint.
logger = logging.getLogger("karl.websocket")

# Adapter/artifact names are used to build filesystem paths under data/adapters/
# (see auto_train.py, model_loader.py). Restrict to a safe charset so a
# malicious "adapter_name" RPC param can't path-traverse outside that
# directory (e.g. "../../../etc/cron.d/x").
# The pattern is anchored on both ends and capped at 64 chars: re.match alone would accept a trailing
# newline-free prefix, and an unbounded name could exceed filesystem name limits.
_SAFE_ADAPTER_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def _is_plain_filename(value) -> bool:
    """True when value is a bare file/dir name (no separators or traversal), safe for os.path.join.

    Used for the base_model/adapter fields of create_custom_agent. Rejects non-strings, "", ".", "..",
    any value whose basename differs from itself (contains a path separator), Windows backslashes and
    NUL bytes (which truncate paths in C APIs).

    Args:
        value: Untrusted client-supplied value of any type.

    Returns:
        True only for a plain, non-special file name.
    """
    return (
        isinstance(value, str)
        and value not in ("", ".", "..")
        and os.path.basename(value) == value
        and "\\" not in value
        and "\x00" not in value
    )


def _coerce_auto_train_numbers(count, epochs, lr):
    """Validate/clamp auto-train numeric params before they reach a subprocess argv.

    JSON numbers may arrive as strings, floats or bools; they end up as command-line arguments for
    auto_train.py, so they are normalised to bounded numbers here instead of being trusted.

    Args:
        count: Requested number of synthetic training cases (clamped to 1..1000).
        epochs: Requested SFT epochs (clamped to 1..100).
        lr: Requested learning rate (clamped to 1e-8..1.0; NaN/inf are rejected).

    Returns:
        (count, epochs, lr) with the clamped values, or None when any value is not a finite number
        (bools are rejected explicitly because bool is an int subclass and True would pass as 1).
    """
    # Local import: math is only needed on this rarely used path, so it is not paid at module import time.
    import math
    try:
        # Reject bools first: int(True) == 1 would silently turn a JSON `true` into a valid-looking number.
        if isinstance(count, bool) or isinstance(epochs, bool) or isinstance(lr, bool):
            return None

        # Non-numeric strings raise ValueError, None/lists raise TypeError, huge floats raise OverflowError;
        # all three mean "not a usable number" for the caller.
        count_i, epochs_i, lr_f = int(count), int(epochs), float(lr)
    except (TypeError, ValueError, OverflowError):
        return None

    # float("nan") and float("inf") parse fine but would produce a meaningless (or crashing) training run.
    if not math.isfinite(lr_f):
        return None
    return (
        # Upper bounds keep one RPC from queuing an unbounded GPU job; lower bounds avoid zero/negative values
        # that argparse in auto_train.py would reject.
        max(1, min(count_i, 1000)),
        max(1, min(epochs_i, 100)),
        max(1e-8, min(lr_f, 1.0)),
    )


# Rate limiter used to throttle per-client request rates (classic token bucket:
# a bucket of `capacity` tokens refilled continuously at `fill_rate` tokens/s).
class TokenBucket:
    """Thread-safe Token Bucket rate limiter.

    A bucket holds up to `capacity` tokens and is refilled continuously at `fill_rate` tokens per
    second. Each request consumes tokens; when the bucket is empty the request is rejected. This allows
    short bursts (capacity) while enforcing a long-run average rate (fill_rate). One lock makes
    consume() safe to call from several threads, although in practice the asyncio loop thread is the
    only caller.
    """
    def __init__(self, capacity: float, fill_rate: float):
        """Create a full bucket.

        Args:
            capacity: Maximum burst size (tokens the bucket can hold).
            fill_rate: Tokens added back per second.
        """
        self.capacity = capacity
        self.fill_rate = fill_rate
        self.tokens = capacity

        # Wall clock is used (not monotonic) for simplicity; a backwards system clock change only delays the
        # next refill because `elapsed` can then be negative and the min() below caps the result.
        self.last_update = time.time()
        self._lock = threading.Lock()

    def consume(self, amount: float = 1.0) -> bool:
        """Try to take `amount` tokens; return True if allowed, False if rate-limited.

        Refill is computed lazily from elapsed wall time, so no timer thread is needed.
        """
        with self._lock:
            # Lazy refill: credit the tokens earned since the last call, capped at capacity.
            # Credit tokens earned since the last call, then clamp at capacity so an idle client cannot bank an
            # unlimited burst.
            now = time.time()
            elapsed = now - self.last_update
            self.last_update = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
            if self.tokens >= amount:
                self.tokens -= amount
                return True
            return False


class WebSocketServerManager:
    """Secure JSON-RPC 2.0 WebSocket bridge for editor integrations.

    Owns the asyncio loop thread, the listening socket, the per-client registry, the token store and
    every RPC handler. It is a process-wide singleton (see get_instance) because the desktop UI and the
    headless entrypoint both need to share one listening port and one token store.

    Thread model: the asyncio loop runs on its own daemon thread; handlers that touch Qt objects
    (SwarmOrchestratorThread, chat threads) connect signals with DirectConnection so they can be
    re-published without a Qt event loop on the loop thread. _clients_lock guards the client
    registry and _threads_lock guards the orchestrator/chat/mini-train thread hand-off.
    """

    # Singleton slot, protected by the class-level lock below; reset_instance() clears it (used by tests
    # and by the settings UI when it needs a fresh token store).
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, port: int = 8080, state=None) -> "WebSocketServerManager":
        """Return the process-wide bridge manager, creating it if necessary.

        Args:
            port: Preferred listening port; only used when the instance is first created.
            state: Optional AppState (shares its RAG pipeline); ignored after the first call.

        Returns:
            The singleton WebSocketServerManager. Thread-safe via the class lock.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(port, state)
            return cls._instance

    @classmethod
    def reset_instance(cls):
        """Forces recreation of the singleton on the next call.

        Stops the current instance (closing sockets and joining the loop thread) before dropping it. Used by
        tests so each test gets a fresh port, token store and client registry.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance.stop()
                cls._instance = None

    def __init__(self, port: int = 8080, state=None):
        """Initialize bridge state, token security, SSL files, and event loop.

        Args:
            port: Preferred local WSS port. Startup may use the next available
                port in the configured range.
            state: Optional AppState. When provided, its RAG pipeline is reused.
        """
        self.port = port
        self.state = state
        self.clients: Set[Any] = set()

        # Per-connection record: id, ip, connected_at, session_start, scopes and the two TokenBuckets.
        # client_histories below holds each connection's private chat history for submit_chat.
        self.client_metadata = {} # client -> {id, ip, latency}
        self.client_histories = {}
        self._clients_lock = threading.Lock()

        # Seed values served at /metrics before any generation has finished, so Prometheus scrapes always see
        # all six gauges instead of missing series.
        self.last_generation_metrics = {
            "prefill_duration_seconds": 0.0,
            "tokens_per_second": 0.0,
            "kv_cache_hit_rate": 0.0,
            "vram_delta_mb": 0.0,
        }
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.loop_thread: Optional[threading.Thread] = None
        self.orchestrator: Optional[SwarmOrchestratorThread] = None
        self.orchestrator_owner = None
        self.chat_thread = None
        self.chat_thread_owner = None
        self.mini_train_thread = None

        # Background task that closes sessions whose lease expired (see _audit_session_leases).
        self._lease_audit_task: asyncio.Task | None = None
        # Guards orchestrator/chat_thread hand-off: the asyncio handler
        # (loop thread) and stop() (Qt thread) both stop/replace them.
        self._threads_lock = threading.Lock()
        self.kb_ingest_thread: Optional[threading.Thread] = None

        # Reuse the UI's RAG pipeline when available so the bridge and the Knowledge Base tab share one index
        # and one embedding model instead of loading two copies into RAM.
        if state is not None:
            self.rag = state.rag
        else:
            self.rag = RAGPipeline()

        # Shared generation service: same model singleton, prompt building and trace logging as the desktop
        # Workbench, so a chat from the editor behaves exactly like a local one.
        self._inference_service = InferenceService(state)

        # Order matters: codex files and certificates must exist, and the token store must be loaded, before
        # the loop thread binds the socket and accepts the first connection.
        self._seed_codex()
        self._init_security()
        self._ensure_ssl_certs()
        self.server = None

        # Set (success or failure) when _start_server finishes so tests and the UI can wait for startup.
        self.started_event = threading.Event()
        self._start_loop_thread()

    # --- Security / protocol constants ---
    # Token lifetime is also the session lease: both are checked against 12 hours (43,200 s) so an
    # unattended editor cannot keep an admin session forever; refresh_token extends it.
    _TOKEN_LIFETIME = 43_200      # 12 hours in seconds
    _LEASE_AUDIT_INTERVAL = 60    # seconds between session-lease sweeps
    # Application-defined WebSocket close codes (4000-4999 range) telling the
    # client why the server dropped it: bad/missing auth, session lease timed
    # out, or its token was revoked.
    _CLOSE_CODE_UNAUTHORIZED = 4001
    _CLOSE_CODE_LEASE_EXPIRED = 4002
    _CLOSE_CODE_REVOKED = 4003
    _TOKEN_PATH = "data/bridge_token.json"

    # Scope vocabulary: read:telemetry (status, adapters, history), read:kb (list/search the knowledge
    # base), write:kb (ingest, fit_vectorizer), admin:execute (anything that runs work or changes state).
    # The full scope set granted to the admin token.
    # Ordered from least to most privileged — also written to bridge_token.json.
    _FULL_SCOPES: list[str] = [
        "read:telemetry",
        "read:kb",
        "write:kb",
        "admin:execute",
    ]

    # Scope enforcement is deny-by-default only for methods listed here; the rest (list_models,
    # get_prompt_pair, compute_diff, codex reads, authenticate, refresh_token, list_custom_agents) are
    # considered harmless and need just an authenticated connection.
    # Maps each guarded RPC method to the single scope it requires.
    # Methods absent from this dict are accessible to any authenticated client.
    METHOD_SCOPES: dict[str, str] = {
        "get_runtime_status": "read:telemetry",
        "list_kb_sources":    "read:kb",
        "list_adapters":      "read:telemetry",
        "search_kb":          "read:kb",
        "ingest_path":        "write:kb",
        "submit_task":        "admin:execute",
        "swarm_commit_edits": "admin:execute",
        "rollback_swarm_run": "admin:execute",
        "submit_chat":        "admin:execute",
        "swarm_inject_guidance": "admin:execute",
        "swarm_pause":        "admin:execute",
        "swarm_resume":       "admin:execute",
        "swarm_step":         "admin:execute",
        "swarm_override_task": "admin:execute",
        "swarm_get_history":  "read:telemetry",
        "get_agent_profiles": "read:telemetry",
        "save_agent_profile": "admin:execute",
        "set_active_model":   "admin:execute",
        "save_prompt_pair":   "admin:execute",
        "delete_prompt_pair": "admin:execute",
        "start_auto_train":   "admin:execute",
        "create_custom_agent": "admin:execute",
        "start_mini_train":   "admin:execute",
        "stop_task":          "admin:execute",
        "fit_vectorizer":     "write:kb",
    }

    # Only used when KARL_WS_HOST is not loopback; loopback binds use plain WS (see _start_server).
    # Self-signed localhost TLS material, generated on first run by _ensure_ssl_certs.
    _SSL_CERT_PATH = "data/ssl/localhost.crt"
    _SSL_KEY_PATH = "data/ssl/localhost.key"
    # JSON-RPC protocol version string required in every request/response envelope.
    _JSONRPC_VERSION = "2.0"
    # Standard JSON-RPC 2.0 error codes mapped to their canonical messages.
    _RPC_ERROR_MESSAGES = {
        -32700: "Parse error",
        -32600: "Invalid Request",
        -32601: "Method not found",
        -32602: "Invalid params",
        -32603: "Internal error",
    }
    # Whitelist of every RPC method the dispatcher will route; anything else yields
    # a "Method not found" (-32601) error before any handler runs.
    _RPC_METHODS = {
        "authenticate",
        "refresh_token",
        "get_runtime_status",
        "list_models",
        "list_adapters",
        "set_active_model",
        "list_prompt_pairs",
        "get_prompt_pair",
        "save_prompt_pair",
        "delete_prompt_pair",
        "list_kb_sources",
        "ingest_path",
        "search_kb",
        "submit_task",
        "swarm_commit_edits",
        "rollback_swarm_run",
        "stop_task",
        "submit_chat",
        "swarm_inject_guidance",
        "swarm_pause",
        "swarm_resume",
        "swarm_step",
        "swarm_override_task",
        "swarm_get_history",
        "get_agent_profiles",
        "save_agent_profile",
        "list_codex_topics",
        "get_codex_content",
        "compute_diff",
        "start_auto_train",
        "fit_vectorizer",
        "start_mini_train",
        "create_custom_agent",
        "list_custom_agents",
    }

    def _ensure_ssl_certs(self):
        """Generates self-signed localhost SSL certificates if missing.

        Never overwrites an existing pair. Failure is logged, not raised: the loopback path does not need
        the certificate, and the non-loopback path fails closed later in _build_ssl_context.
        """
        # Make sure the certificate directory exists before openssl writes into it.
        ssl_dir = os.path.dirname(self._SSL_CERT_PATH)
        os.makedirs(ssl_dir, exist_ok=True)
        
        # Only generate when either half of the cert/key pair is missing, so existing
        # (possibly operator-provisioned) certificates are never overwritten.
        if not os.path.exists(self._SSL_CERT_PATH) or not os.path.exists(self._SSL_KEY_PATH):
            logger.info("Generating self-signed SSL certificates for WSS...")
            try:
                # Shell out to openssl: avoids a Python crypto dependency for a one-off
                # 10-year self-signed cert valid for CN=localhost.
                # Use openssl command if available
                cmd = [
                    "openssl", "req", "-x509", "-newkey", "rsa:2048",
                    "-keyout", self._SSL_KEY_PATH,
                    "-out", self._SSL_CERT_PATH,
                    "-sha256", "-days", "3650", "-nodes",
                    "-subj", "/C=XX/ST=State/L=City/O=Karl/OU=Engine/CN=localhost"
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                logger.info(f"SSL certificates generated at {self._SSL_CERT_PATH}")
            except Exception as e:
                logger.error(f"Failed to generate SSL certificates: {e}. WSS server will fail to start.")

    def _init_security(self):
        """Initialise the multi-token store and safe-path rules."""
        # bridge_token: the admin token; _token_store: token -> scopes for all tokens.
        # Three pieces of state: the admin token (full scopes), its creation time (drives expiry and the
        # 12 hour lease) and the map of every valid token to its scope list.
        self.bridge_token: str = ""
        self._token_created_at: float = 0.0
        self._token_store: dict[str, list[str]] = {}

        # Token source precedence: KARL_BRIDGE_TOKEN env var (operator-supplied, e.g.
        # in containers) > persisted token file > freshly generated token.
        env_token = os.environ.get("KARL_BRIDGE_TOKEN", "").strip()

        # Container deployments inject KARL_BRIDGE_TOKEN so the token is stable across restarts; it is
        # persisted so the extension's token file discovery keeps working.
        if env_token:
            self.bridge_token = env_token
            self._token_created_at = time.time()
            self._token_store = {env_token: list(self._FULL_SCOPES)}
            self._persist_token_store()
            self._save_token_safe(self.bridge_token)
        elif os.path.exists(self._TOKEN_PATH):
            try:
                with open(self._TOKEN_PATH, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                # Two on-disk formats exist: the current {"tokens": {...}} multi-token file
                # and the legacy single {"token": ...} file, which is migrated in place.
                if "tokens" in payload:
                    # New multi-token format
                    self._token_store = {k: list(v) for k, v in payload["tokens"].items()}
                    self._token_created_at = float(payload.get("created_at", 0.0))
                else:
                    # Legacy single-token format — migrate in place
                    old_tok = payload.get("token", "")
                    self._token_created_at = float(payload.get("created_at", 0.0))
                    if old_tok:
                        self._token_store = {old_tok: list(self._FULL_SCOPES)}
                # Identify the admin token (first one carrying admin:execute scope)
                self.bridge_token = next(
                    (tok for tok, sc in self._token_store.items() if "admin:execute" in sc),
                    "",
                )
                # Rotate when the store has expired or contains no admin-capable token;
                # otherwise just refresh the keychain copy.
                if time.time() - self._token_created_at > self._TOKEN_LIFETIME or not self.bridge_token:
                    self._rotate_token()
                else:
                    self._save_token_safe(self.bridge_token)

            # Failing closed (new random token) is safer than trusting a half-parsed file.
            # Corrupt/unreadable token file: fail safe by issuing a brand-new token.
            except Exception:
                self._rotate_token()
        else:
            self._rotate_token()

        # Sensitive directories that should never be targeted by agents or RAG.
        #
        # This is a blocklist, not an allowlist: paths outside it (e.g. a
        # mounted USB drive under /media, or /tmp scratch space) are
        # permitted by design, so this list needs to name every sensitive
        # directory explicitly rather than relying on a catch-all. Note "/"
        # itself only blocks the literal path "/" (via the `==` branch in
        # _is_safe_path below) — a bare "/" entry can't act as "block every
        # path under it" because real_path.startswith(real_blocked + os.sep)
        # would require the path to start with "//", which no normalized
        # absolute path does. List concrete directories instead of relying
        # on that.
        # Blocklist of top-level system directories (Linux layout; the macOS equivalents are covered by the
        # home-directory entries below). Compared by realpath in _is_safe_path.
        self.blocked_paths = {
            "/", "/etc", "/bin", "/sbin", "/usr/bin", "/usr/sbin", "/usr/lib",
            "/usr/local", "/var", "/boot", "/dev", "/proc", "/sys", "/root",
            "/opt", "/srv", "/lib", "/lib64",
        }

        # Personal directories that routinely contain secrets or unrelated private files are blocked too, so
        # an editor client cannot ask Karl to ingest or modify them.
        user_home = os.path.expanduser("~")
        self.blocked_paths.add(os.path.join(user_home, "Desktop"))
        self.blocked_paths.add(os.path.join(user_home, "Documents"))
        self.blocked_paths.add(os.path.join(user_home, "Downloads"))
        self.blocked_paths.add(os.path.join(user_home, ".ssh"))
        self.blocked_paths.add(os.path.join(user_home, ".gnupg"))
        self.blocked_paths.add(os.path.join(user_home, ".aws"))
        self.blocked_paths.add(os.path.join(user_home, ".kube"))
        self.blocked_paths.add(os.path.join(user_home, ".docker"))
        self.blocked_paths.add(os.path.join(user_home, ".config"))
        self.blocked_paths.add(os.path.join(user_home, ".local"))
        self.blocked_paths.add(os.path.join(user_home, ".karl"))

        # Block every other local user's home directory too, not just this
        # process's own $HOME — on a shared/multi-user host, ingest_path
        # previously had no protection against reading another user's files.
        try:
            homes_root = os.path.dirname(user_home) or "/home"
            # If $HOME is /root, its parent is "/" -- not a homes directory.
            # Skipping it avoids blocking every top-level path (e.g. /tmp).
            if homes_root != os.sep and os.path.isdir(homes_root):
                for entry in os.listdir(homes_root):
                    candidate = os.path.join(homes_root, entry)
                    if os.path.isdir(candidate) and candidate != user_home:
                        self.blocked_paths.add(candidate)
        except OSError:
            pass

    def _save_token_safe(self, token: str) -> None:
        """Saves the token to the OS keychain without blocking the caller.

        save_cached_token() can hit an OS-keychain backend (SecretService/
        D-Bus) that blocks for several seconds — or, before
        keychain_manager's own timeout guard, indefinitely — waiting on an
        unlock prompt nothing will answer. This method is called from
        _init_security() during WebSocketServerManager.__init__(), which
        itself runs on the GUI thread via MainWindow's
        QTimer.singleShot(0, self._init_websocket_server) — i.e. right as
        the Qt event loop starts, before the main window has painted.
        Always run it on a plain background thread rather than only when
        self.loop happens to already be running (it typically isn't yet at
        this point in startup, which made that condition a no-op here).
        """
        threading.Thread(
            target=save_cached_token, args=(token,), daemon=True,
            name="karl-save-bridge-token",
        ).start()

    def _generate_token(self) -> str:
        """Return a new random 128-bit token as a 32-char hex string (uuid4).

        uuid4 draws from os.urandom, so tokens are unguessable; they are bearer credentials, so never log them.
        """
        # hex form is URL-safe, so the token can sit in a ?token= query string without escaping.
        return uuid.uuid4().hex

    def _rotate_token(self) -> None:
        """Generate a fresh admin token.

        Non-admin scoped tokens in _token_store are preserved so that
        previously issued read-only/KB keys stay valid after rotation.
        """
        # Only the old admin token is removed; scoped tokens handed out earlier (read-only editor keys) stay
        # valid because they were issued independently.
        # Drop the previous admin token only
        if not hasattr(self, "_token_store"):
            self._token_store = {}
        self._token_store.pop(self.bridge_token, None)
        self.bridge_token = self._generate_token()
        self._token_created_at = time.time()

        # Copy the list so later edits to one token's scopes can never alias the class-level constant.
        self._token_store[self.bridge_token] = list(self._FULL_SCOPES)

        # Persist before keychain save so a crash between the two never leaves the keychain holding a token the file does not know.
        self._persist_token_store()
        self._save_token_safe(self.bridge_token)

    def _persist_token_store(self) -> None:
        """Write the in-memory token store to data/bridge_token.json.

        Bearer tokens (including the admin:execute-scoped bridge token) are
        stored here in plaintext, so the file is created owner-only (0600)
        rather than relying on the process umask — a world-readable token
        store lets any other local user connect to the RPC bridge with full
        admin scope.
        """
        try:
            # O_CREAT with mode 0o600 gives the file owner-only permission atomically at creation; the explicit
            # chmod below also fixes files created earlier by an older version with a looser mode.
            os.makedirs("data", exist_ok=True)
            fd = os.open(self._TOKEN_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "token": self.bridge_token,
                        "tokens": self._token_store,
                        "created_at": self._token_created_at,
                    },
                    f,
                    indent=2,
                )
            os.chmod(self._TOKEN_PATH, 0o600)

        # Persistence failure is non-fatal: the in-memory store still works for this run, only restarts lose the tokens.
        except Exception as exc:
            logger.warning("Could not persist token store: %s", exc)

    def add_scoped_token(self, scopes: list[str]) -> str:
        """Generate and register a new token with *scopes*. Returns the token hex.

        Args:
            scopes: Subset of _FULL_SCOPES to grant (e.g. ["read:telemetry"] for a read-only editor key).

        Returns:
            The new 32-char hex bearer token; it is persisted to data/bridge_token.json (0600).

        Side effects: writes the token file and logs the granted scopes (never the token itself).
        """
        token = self._generate_token()

        # Defensive: tests build the manager with __new__ and skip __init__, so the attribute may be missing.
        if not hasattr(self, "_token_store"):
            self._token_store = {}
        self._token_store[token] = list(scopes)
        self._persist_token_store()
        logger.info("Scoped token added: scopes=%s", scopes)
        return token

    def _get_token_scopes(self, token: str) -> list[str] | None:
        """Return scopes for *token* if valid and unexpired, else None.

        Side effect: when the whole store has outlived _TOKEN_LIFETIME the admin token is rotated (which
        invalidates the supplied token) and None is returned, so expiry is enforced lazily on use.
        """
        if not token:
            return None
        if not hasattr(self, "_token_store"):
            return None

        # Single creation time for the whole store: when it is old, rotate (new admin token, scoped tokens kept) and treat this lookup as a failure.
        if time.time() - self._token_created_at > self._TOKEN_LIFETIME:
            logger.info("Token store expired (12h). Rotating.")
            self._rotate_token()
            return None
        return self._token_store.get(token)

    def _validate_token(self, token: str) -> bool:
        """Returns True if token is present in the store and unexpired.

        Args:
            token: Bearer token string supplied by the client (may be empty or None).
        """
        return self._get_token_scopes(token) is not None

    def _is_safe_path(self, path: str) -> bool:
        """Checks if a path is outside sensitive system directories.

        Uses os.path.realpath so that symlinks pointing into blocked directories are caught before any
        filesystem operation is performed on them. The check is a blocklist (see _init_security), not an
        allowlist.

        Args:
            path: Client-supplied file or directory path ("~" is expanded).

        Returns:
            True when the path may be read or written; False for empty paths, the home directory itself,
            and anything under a blocked directory unless it is inside the project working directory.
        """
        if not path:
            return False

        # realpath resolves symlinks, so a link inside an allowed folder that points at /etc is judged by its
        # real target.
        real_path = os.path.realpath(os.path.expanduser(path))
        user_home = os.path.realpath(os.path.expanduser("~"))

        # The bare home directory is refused (too broad); specific subdirectories not listed in blocked_paths
        # remain allowed.
        if real_path == user_home:
            return False

        # A path under a blocked directory is still allowed when it lies inside the project working
        # directory (e.g. Karl itself checked out under ~/Documents); anything else under it is denied.
        # Containment test appends os.sep so "/var2" is not mistaken for a child of "/var".
        for blocked in self.blocked_paths:
            real_blocked = os.path.realpath(blocked)
            if real_path == real_blocked or real_path.startswith(real_blocked + os.sep):
                project_root = os.path.realpath(os.getcwd())
                if real_path == project_root or real_path.startswith(project_root + os.sep):
                    return True
                return False
        return True

    def _seed_codex(self):
        """Install/refresh the bundled "Codex" reference library on disk.

        Writes each DEFAULT_LIBRARY topic as an .html file under data/codex_library and stamps a .version
        file, so the docs workspace and the bridge's codex RPCs have content on first run. Re-seeds only
        when the stamped version differs (or the directory has no html/md files), so user edits survive
        normal restarts. Failures to write individual files are logged and skipped.
        """
        library_dir = "data/codex_library"
        os.makedirs(library_dir, exist_ok=True)
        version_filepath = os.path.join(library_dir, ".version")
        # Bump this when DEFAULT_LIBRARY content changes to force a re-seed.
        current_version = "5.0"

        needs_upgrade = True
        if os.path.exists(version_filepath):
            try:
                with open(version_filepath, "r", encoding="utf-8") as vf:
                    if vf.read().strip() == current_version:
                        needs_upgrade = False
            except OSError as exc:
                logger.warning("could not read codex version file %s: %s", version_filepath, exc)

        # Re-seed if the version stamp is stale OR the library was emptied.
        if needs_upgrade or not [f for f in os.listdir(library_dir) if f.endswith((".html", ".md"))]:
            for topic, content in DEFAULT_LIBRARY.items():
                # Sanitize the topic title into a filesystem-safe filename (no separators/dots).
                safe_name = "".join(c for c in topic if c.isalnum() or c in (" ", "_", "-")).strip()
                filepath = os.path.join(library_dir, f"{safe_name}.html")
                try:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(content)
                except OSError as exc:
                    logger.warning("could not seed codex topic %r to %s: %s", topic, filepath, exc)
            try:
                with open(version_filepath, "w", encoding="utf-8") as vf:
                    vf.write(current_version)
            except OSError as exc:
                logger.warning("could not write codex version file %s: %s", version_filepath, exc)

    def _active_model_config(self) -> dict:
        """Return the persisted active-model selection ({"filename", "adapter"}).

        Thin wrapper over config_store so tests can monkeypatch one method.
        """
        return config_store.get_active_model()

    def _read_model_registry(self) -> list[dict]:
        """Return the model registry entries (tiers shipped in data/model_registry.json)."""
        return config_store.get_model_registry()

    def _list_models(self) -> dict:
        """Build the list_models RPC payload.

        Merges two sources: registry entries (marked installed if the GGUF exists in data/models) and stray
        local .gguf files not in the registry. Each item carries installed/active flags and on-disk size in
        GiB so the client can render a picker.

        Returns:
            {"active": {"filename", "adapter"?}, "models": [{name, filename, tier, n_ctx, min_ram_gb,
            min_vram_gb, min_storage_gb, installed, active, size_gb, source}]} where source is "registry"
            or "local". Unreadable files only get size_gb None; the method never raises for I/O errors.
        """
        models_dir = os.path.join("data", "models")
        active = self._active_model_config()
        registry = self._read_model_registry()

        # `registered` remembers which filenames the registry already covers so the second pass below only
        # adds GGUF files the registry does not know about.
        registered = {}
        models = []

        # First pass: every registry entry is listed, installed or not, so the picker can offer downloads.
        for entry in registry:
            filename = entry.get("filename")
            if not filename:
                continue
            registered[filename] = True
            path = os.path.join(models_dir, filename)
            installed = os.path.exists(path)

            # 1024**3 converts bytes to GiB; two decimals are enough for a model picker.
            size_gb = None
            if installed:
                try:
                    size_gb = round(os.path.getsize(path) / (1024 ** 3), 2)
                except OSError as exc:
                    logger.debug("could not stat model file %s: %s", path, exc)
            models.append({
                "name": entry.get("name", filename),
                "filename": filename,
                "tier": entry.get("tier"),
                "n_ctx": entry.get("n_ctx", 4096),
                "min_ram_gb": entry.get("min_ram_gb"),
                "min_vram_gb": entry.get("min_vram_gb"),
                "min_storage_gb": entry.get("min_storage_gb"),
                "installed": installed,
                # Exactly one model should match the persisted selection; the flag drives the radio-button state in clients.
                "active": filename == active["filename"],
                "size_gb": size_gb,
                "source": "registry",
            })

        # Second pass: surface any local GGUF files that the registry doesn't know about.
        if os.path.exists(models_dir):
            try:
                for filename in sorted(os.listdir(models_dir)):
                    # Skip non-model files and anything already listed from the registry.
                    if not filename.endswith(".gguf") or registered.get(filename):
                        continue
                    path = os.path.join(models_dir, filename)
                    try:
                        size_gb = round(os.path.getsize(path) / (1024 ** 3), 2)
                    except OSError as exc:
                        logger.debug("could not stat model file %s: %s", path, exc)
                        size_gb = None
                    models.append({
                        "name": filename,
                        "filename": filename,
                        "tier": None,
                        # Unregistered models fall back to the loader's default context length.
                        "n_ctx": ModelLoader._read_registry_n_ctx(filename),
                        "min_ram_gb": None,
                        "min_vram_gb": None,
                        "min_storage_gb": None,
                        "installed": True,
                        "active": filename == active["filename"],
                        "size_gb": size_gb,
                        "source": "local",
                    })
            except OSError as exc:
                logger.warning("could not scan models directory %s: %s", models_dir, exc)

        # Reply shape: {"active": {...}, "models": [...]}.
        return {
            "active": active,
            "models": models,
        }

    def _list_adapters(self) -> dict:
        """Return the names of LoRA adapter directories under data/adapters.

        Only directories count (an adapter is a folder with adapter_config.json and weights).
        """
        adapters_dir = os.path.join("data", "adapters")
        adapters = []
        if os.path.exists(adapters_dir):
            try:
                for entry in sorted(os.listdir(adapters_dir)):
                    path = os.path.join(adapters_dir, entry)
                    if os.path.isdir(path):
                        adapters.append(entry)
            except OSError as exc:
                logger.warning("could not scan adapters directory %s: %s", adapters_dir, exc)
        return {"adapters": adapters}

    def _set_active_model(self, filename: str, adapter: str | None = None) -> dict:
        """Persist a new active model (and optional adapter) and drop the loaded model.

        Both names are reduced with os.path.basename and rejected if that changes them, so a caller cannot
        smuggle path separators (traversal) into data/models paths. The model is only unloaded here; it
        reloads lazily on the next generation.

        Args:
            filename: GGUF file name inside data/models.
            adapter: Optional LoRA adapter directory name inside data/adapters (no dots allowed).

        Returns:
            {"active": ..., "loaded": False, "message": str} describing the new selection.

        Raises:
            ValueError: invalid filename/adapter name.
            FileNotFoundError: the GGUF isn't installed.
            OSError: the active_model.json write failed.
        """
        # basename-equality check: any separator or ".." changes the string, so path traversal is rejected
        # rather than silently stripped.
        # filename may be None when a caller bypasses validation; `or ""` keeps basename from raising.
        safe_filename = os.path.basename(filename or "")
        if not safe_filename or safe_filename != filename:
            raise ValueError("Invalid model filename.")

        model_path = os.path.join("data", "models", safe_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file is not installed: {safe_filename}")

        # Mirrors the structure stored in data/active_model.json.
        active = {"filename": safe_filename}

        # Adapter names are directory names under data/adapters; a dot is rejected so ".." or hidden/extension
        # tricks cannot appear.
        if adapter:
            safe_adapter = os.path.basename(adapter)
            if not safe_adapter or safe_adapter != adapter or "." in adapter:
                raise ValueError("Invalid adapter name.")
            active["adapter"] = adapter

        # config_store writes atomically; False means the write failed.
        if not config_store.set_active_model(safe_filename, adapter):
            raise OSError("Failed to persist data/active_model.json")

        # Unload now so the next generation loads the new file/adapter; loading here would block the loop thread.
        ModelLoader.reset_instance()
        return {
            "active": active,
            # Always False: the swap is lazy (see ModelLoader.reset_instance above).
            "loaded": False,
            "message": f"Active model set to {safe_filename}. It will load on the next generation.",
        }

    def _prompt_pairs_dir(self) -> str:
        """Return (and create if needed) the directory holding saved A/B prompt pairs.

        Same directory the Prompt Lab workspace uses, so pairs saved from the editor show up in the desktop app.
        """
        path = os.path.join("data", "prompt_pairs")
        os.makedirs(path, exist_ok=True)
        return path

    def _safe_prompt_pair_name(self, name: str) -> str:
        """Sanitize a client-supplied pair name into [A-Za-z0-9_-] so it is a safe file stem.

        Raises:
            ValueError: when nothing is left after trimming (empty name).
        """
        safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", (name or "").strip())
        if not safe_name:
            raise ValueError("Prompt pair name is required.")
        return safe_name

    def _prompt_pair_path(self, name: str) -> str:
        """Absolute-ish JSON file path for a prompt pair, derived only from the sanitized name."""
        return os.path.join(self._prompt_pairs_dir(), f"{self._safe_prompt_pair_name(name)}.json")

    def _list_prompt_pairs(self) -> dict:
        """List all saved prompt pairs with their system/user prompts and model names.

        Unreadable or corrupt files are logged and reported with empty fields rather than failing the whole
        listing.

        Returns:
            {"pairs": [{name, system_a, system_b, user_a, user_b, model_a, model_b}]} sorted by file name.
        """
        # Unreadable files still appear (with empty fields) so the user can see and delete them.
        pairs = []
        for filename in sorted(os.listdir(self._prompt_pairs_dir())):
            if not filename.endswith(".json"):
                continue
            name = os.path.splitext(filename)[0]
            path = os.path.join(self._prompt_pairs_dir(), filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    # Whole file is small (prompts and outputs); no streaming needed.
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("could not read prompt pair %s: %s", path, exc)
                data = {}
            pairs.append({
                "name": name,
                "system_a": data.get("system_a", ""),
                "system_b": data.get("system_b", ""),
                "user_a": data.get("user_a", ""),
                "user_b": data.get("user_b", ""),
                # Models are None when the pair was saved without a model choice (use whatever is active).
                "model_a": data.get("model_a"),
                "model_b": data.get("model_b"),
            })
        return {"pairs": pairs}

    def _get_prompt_pair(self, name: str) -> dict:
        """Load one saved prompt pair by name; raises FileNotFoundError if absent.

        The returned dict is the raw saved JSON plus the sanitized `name`.
        """
        safe_name = self._safe_prompt_pair_name(name)
        path = self._prompt_pair_path(safe_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompt pair not found: {safe_name}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["name"] = safe_name
        return data

    def _save_prompt_pair(self, params: dict) -> dict:
        """Persist a Prompt Lab A/B pair (prompts, models, toggles, captured outputs) as JSON.

        Field names match prompt_lab.py's own save format, so the desktop and editor UIs can load each
        other's pairs. Missing fields default to empty strings / False.

        Args:
            params: RPC params; "name" is required and sanitized, everything else is optional.

        Returns:
            {"name": sanitized_name, "saved": True}.

        Raises:
            ValueError: when the name is empty after sanitizing.
            OSError: when the file cannot be written.
        """
        name = self._safe_prompt_pair_name(params.get("name", ""))
        data = {
            "name": name,
            "system_a": params.get("system_a", ""),
            "user_a": params.get("user_a", ""),
            "system_b": params.get("system_b", ""),
            "user_b": params.get("user_b", ""),
            "model_a": params.get("model_a"),
            "adapter_a": params.get("adapter_a"),
            "model_b": params.get("model_b"),
            "adapter_b": params.get("adapter_b"),
            # Booleans are coerced so a JSON null or 0/1 from another client never stores a non-bool.
            "rag_a": bool(params.get("rag_a", False)),
            "loop_a": bool(params.get("loop_a", False)),
            "rag_b": bool(params.get("rag_b", False)),
            "loop_b": bool(params.get("loop_b", False)),
            "output_a_raw": params.get("output_a_raw", ""),
            "output_b_raw": params.get("output_b_raw", ""),
            "output_a_display": params.get("output_a_display", ""),
            "output_b_display": params.get("output_b_display", ""),
        }

        # ensure_ascii=False below keeps non-English prompts readable in the file.
        with open(self._prompt_pair_path(name), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return {"name": name, "saved": True}

    def _delete_prompt_pair(self, name: str) -> dict:
        """Delete a saved prompt pair; idempotent if the file is already gone."""
        safe_name = self._safe_prompt_pair_name(name)
        path = self._prompt_pair_path(safe_name)
        if os.path.exists(path):
            os.remove(path)
        return {"name": safe_name, "deleted": True}

    def _kb_supported_extensions(self) -> set[str]:
        """File extensions the RAG ingester can parse (must match RAGPipeline.ingest_file)."""
        return {".pdf", ".docx", ".txt", ".md", ".py", ".csv"}

    def _kb_snapshot(self) -> dict:
        """Summarize the knowledge base: per-source chunk counts, totals, ingest-in-progress flag.

        Reloads the index first so chunks ingested by another process or the desktop UI are visible.
        """
        # Cheap if already loaded; picks up changes written by another process.
        self.rag._load_index()
        sources = []
        counts = {}

        # First-seen timestamp per source: every chunk of a file carries the same ingested_at.
        ingested_at = {}

        # documents is the in-memory metadata list, one dict per chunk.
        for doc in self.rag.documents:
            source = doc.get("source_file", "unknown")
            counts[source] = counts.get(source, 0) + 1
            if source not in ingested_at:
                ingested_at[source] = doc.get("ingested_at")

        for source in sorted(counts):
            sources.append({
                "name": source,
                "chunks": counts[source],
                "ingested_at": ingested_at.get(source),
            })

        # Reply shape: per-source rows plus global totals and the supported extension list so UIs can filter drops.
        return {
            "sources": sources,
            "total_sources": len(sources),
            "total_chunks": self.rag.total_chunks,
            "supported_extensions": sorted(self._kb_supported_extensions()),
            "ingesting": bool(self.kb_ingest_thread and self.kb_ingest_thread.is_alive()),
        }

    def _collect_kb_files(self, path: str, recursive: bool) -> list[str]:
        """Resolve `path` (file or directory) into a sorted list of ingestible files.

        Raises ValueError for unsupported single files, FileNotFoundError for bad paths. Recursive walks
        prune VCS/venv/cache directories to avoid indexing junk.

        Args:
            path: File or directory (with "~" allowed). Safety (_is_safe_path) is checked by the caller.
            recursive: Walk sub-directories when True, otherwise only the directory's direct children.

        Returns:
            Absolute file paths whose extension is in _kb_supported_extensions(), sorted.
        """
        if not path:
            raise ValueError("Path is required.")

        # abspath+expanduser so "~/notes" works and relative paths are resolved against the server's working directory.
        expanded = os.path.abspath(os.path.expanduser(path))

        # A single file is accepted only if its extension is supported, so the error message is specific.
        if os.path.isfile(expanded):
            ext = os.path.splitext(expanded)[1].lower()
            if ext not in self._kb_supported_extensions():
                raise ValueError(f"Unsupported file type: {ext or 'none'}")
            return [expanded]

        if not os.path.isdir(expanded):
            raise FileNotFoundError(f"Path not found: {path}")

        # Both branches below only collect candidates; nothing is read or parsed here.
        files = []
        if recursive:
            # Recursive mode follows the directory tree but not symlinked directories (os.walk default), avoiding cycles.
            for root, dirs, names in os.walk(expanded):
                # Prune in place (os.walk honors edits to `dirs`) so noisy directories are skipped.
                # Slice-assign (not rebind) so os.walk sees the pruned list and never descends into those folders.
                dirs[:] = [d for d in dirs if d not in {".git", "venv", ".venv", "__pycache__", "node_modules"}]
                for name in names:
                    ext = os.path.splitext(name)[1].lower()
                    if ext in self._kb_supported_extensions():
                        files.append(os.path.join(root, name))

        # Non-recursive mode: only the files directly inside the directory.
        else:
            for name in os.listdir(expanded):
                candidate = os.path.join(expanded, name)
                ext = os.path.splitext(name)[1].lower()
                if os.path.isfile(candidate) and ext in self._kb_supported_extensions():
                    files.append(candidate)

        # Sorted so ingestion order, and therefore chunk ids, is deterministic.
        return sorted(files)

    def _start_kb_ingest(self, params: dict) -> dict:
        """Validate params and start background KB ingestion; return immediately.

        Only one ingest may run at a time. Progress is streamed to clients as `kb_ingest_progress`
        notifications and completion as `kb_ingest_finished` (carrying a fresh KB snapshot), all tagged with
        the returned task_id.

        Args:
            params: {"path", "recursive"=True, "chunk_size"=200 (50..2000), "overlap"=50 (0..chunk_size-1)}.

        Returns:
            {"status": "started", "task_id", "file_count", "path"} (absolute path).

        Raises:
            RuntimeError: another ingest is still running.
            ValueError: bad chunking parameters or no supported files.
            FileNotFoundError: the path does not exist.
        """
        # One ingest at a time: concurrent writers would race on the FAISS index and SQLite metadata.
        if self.kb_ingest_thread and self.kb_ingest_thread.is_alive():
            raise RuntimeError("Knowledge Base ingestion is already running.")

        path = params.get("path", "")
        recursive = bool(params.get("recursive", True))

        # Defaults mirror RAGPipeline.ingest_file (200-unit chunks, 50 overlap); int() may raise ValueError,
        # which the dispatcher reports as an internal error.
        chunk_size = int(params.get("chunk_size", 200))
        overlap = int(params.get("overlap", 50))
        # Bounds keep a bad client from requesting degenerate chunking (tiny chunks
        # explode the index; overlap >= size would never advance).
        # Tiny chunks explode the index size and huge chunks exceed the embedder's input window.
        if chunk_size < 50 or chunk_size > 2000:
            raise ValueError("chunk_size must be between 50 and 2000.")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and lower than chunk_size.")

        # Collect (and validate) synchronously so errors are reported in the RPC reply, not as a later notification.
        files = self._collect_kb_files(path, recursive)
        if not files:
            raise ValueError("No supported files found for ingestion.")

        # Correlates the progress/finished notifications with the RPC reply, since several clients may be listening.
        task_id = str(uuid.uuid4())

        # Runs on a plain thread so a long ingest never blocks the asyncio loop; it reports only through
        # notifications because the original RPC reply has already been sent.
        def worker():
            # Captured separately from the batch API's own totals to report `queued_file_count` at the end.
            total = len(files)

            # Forwarded as kb_ingest_progress; `event` comes from RAGPipeline.ingest_files (status "parsed", "embedded"...).
            def progress_cb(current: int, total_files: int, event: dict):
                self._send_notification("kb_ingest_progress", {
                    "task_id": task_id,
                    "current": current,
                    "total": total_files,
                    "filename": event.get("filename", ""),
                    "chunks": int(event.get("chunks", 0) or 0),
                    "status": event.get("status", "parsed"),
                })

            # batch_size=32 embeds 32 chunks per forward pass, a speed/memory trade-off that fits CPU laptops.
            # Prefer the batched ingest_files API; fall back to per-file ingestion on
            # older RAG pipelines that lack it.
            if hasattr(self.rag, "ingest_files"):
                # The batched path embeds chunks across files in groups, which is much faster than one file at a time.
                result = self.rag.ingest_files(
                    files,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    batch_size=32,
                    progress_cb=progress_cb,
                )
            else:
                # Fallback accumulators mimic the result dict that ingest_files returns.
                total_chunks = 0
                per_file = []
                errors = []
                for index, filepath in enumerate(files, 1):
                    filename = os.path.basename(filepath)

                    # Emitted before ingestion so the client shows which file is in progress.
                    progress_cb(index, total, {"filename": filename, "status": "parsed"})
                    try:
                        chunks = self.rag.ingest_file(
                            filepath,
                            chunk_size=chunk_size,
                            overlap=overlap,
                        )
                        total_chunks += chunks
                        per_file.append({
                            "path": filepath,
                            "filename": filename,
                            "chunks": chunks,
                        })
                    # One bad file must not abort the batch; record it and continue.
                    except Exception as exc:
                        errors.append({
                            "path": filepath,
                            "filename": filename,
                            "error": str(exc),
                        })

                # Rebuild the same result structure so the finished notification below has one code path.
                result = {
                    "files": per_file,
                    "errors": errors,
                    "file_count": len(per_file),
                    "error_count": len(errors),
                    "chunks_added": total_chunks,
                }

            # Final notification carries the snapshot so clients refresh their source list without another RPC.
            self._send_notification("kb_ingest_finished", {
                "task_id": task_id,
                "files": result["files"],
                "errors": result["errors"],
                "file_count": result["file_count"],
                "error_count": result["error_count"],
                "chunks_added": result["chunks_added"],
                "queued_file_count": total,
                "snapshot": self._kb_snapshot(),
            })

        # Daemon thread so a long ingest can't block process shutdown.
        # Stored on self so _kb_snapshot can report `ingesting` and a second ingest can be refused.
        self.kb_ingest_thread = threading.Thread(target=worker, daemon=True)
        self.kb_ingest_thread.start()

        # Immediate reply: the actual outcome arrives later as kb_ingest_finished with this task_id.
        return {
            "status": "started",
            "task_id": task_id,
            "file_count": len(files),
            "path": os.path.abspath(os.path.expanduser(path)),
        }

    def _search_kb(self, params: dict) -> dict:
        """Run a dense KB search for the search_kb RPC.

        top_k is clamped to [1, 25]. `threshold` is a maximum L2 distance (smaller is closer); 0 disables
        filtering. Returns hits plus a KB snapshot for the UI.

        Args:
            params: {"query" (required), "top_k"=3, "threshold"=0.0, "source_filter"=None}.

        Returns:
            {"query", "top_k", "threshold", "source_filter", "results": [{text, source_file, chunk_id, rank,
            distance, ingested_at}], "snapshot"}.

        Raises:
            ValueError: query missing or blank.
        """
        # Whitespace-only queries are rejected because embedding an empty string returns arbitrary neighbours.
        query = (params.get("query") or "").strip()
        if not query:
            raise ValueError("query is required.")

        top_k = int(params.get("top_k", 3))
        threshold = float(params.get("threshold", 0.0))

        # An empty string means "no filter".
        source_filter = params.get("source_filter") or None

        # Clamp so a client cannot ask for the entire index in one reply.
        top_k = max(1, min(top_k, 25))
        threshold = max(0.0, threshold)

        self.rag._load_index()

        # Returns dicts with text, source_file, chunk_id, rank and L2 distance (smaller means more similar).
        results = self.rag.retrieve_with_metadata(
            query,
            top_k=top_k,
            source_filter=source_filter,
        )

        # Filter after retrieval because retrieve_with_metadata has no distance cutoff of its own.
        if threshold > 0.0:
            results = [r for r in results if float(r.get("distance", 0.0)) <= threshold]

        return {
            "query": query,
            "top_k": top_k,
            "threshold": threshold,
            "source_filter": source_filter,
            "results": [
                {
                    "text": r.get("text", ""),
                    "source_file": r.get("source_file", "unknown"),
                    "chunk_id": r.get("chunk_id"),
                    "rank": r.get("rank"),
                    "distance": float(r.get("distance", 0.0)),
                    "ingested_at": r.get("ingested_at"),
                }
                for r in results
            ],
            "snapshot": self._kb_snapshot(),
        }

    def _runtime_status(self) -> dict:
        """Build the get_runtime_status payload: bridge, activity, model, adapter, RSS memory.

        When no model is loaded the persisted active selection is reported instead, so the editor can show
        "deepseek... (not loaded)" before the first generation.
        """
        active_model = self._active_model_config()
        loaded = ModelLoader.is_loaded()

        # Report the loaded model when one is resident, otherwise what would be loaded next.
        model_name = ModelLoader.model_name() if loaded else active_model["filename"]
        adapter = getattr(ModelLoader, "_active_adapter", None) if loaded else active_model["adapter"]
        n_ctx = ModelLoader.n_ctx() if loaded else ModelLoader._read_registry_n_ctx(model_name)

        # Both flags come from QThread.isRunning(); `state` collapses them into running/idle for a status light.
        swarm_active = bool(self.orchestrator and self.orchestrator.isRunning())
        chat_active = bool(self.chat_thread and self.chat_thread.isRunning())

        # psutil is optional; report RAM as None when unavailable.
        # Resident memory of this process only (not the whole machine), in MiB.
        ram_mb = None
        try:
            import psutil
            ram_mb = round(psutil.Process(os.getpid()).memory_info().rss / 1_048_576, 1)
        except Exception:
            pass

        # Reply is grouped into bridge / runtime / model / adapter / system so clients can render each part independently.
        return {
            "bridge": {
                "port": self.port,
                "clients": len(self.clients),
                "listening": self.server is not None,
            },
            "runtime": {
                "state": "running" if (swarm_active or chat_active) else "idle",
                "swarm_active": swarm_active,
                "chat_active": chat_active,
            },
            "model": {
                "name": model_name,
                "loaded": loaded,
                "n_ctx": n_ctx,
            },
            "adapter": {
                "name": adapter,
                "loaded": bool(adapter),
            },
            "system": {
                "ram_mb": ram_mb,
            },
        }

    def _rpc_error_response(
        self,
        code: int,
        req_id: Any = None,
        message: str | None = None,
        data: Any = None,
    ) -> dict:
        """Build a JSON-RPC 2.0 error envelope.

        Falls back to the canonical message for `code` when `message` is not given; `data` carries optional
        human-readable detail (e.g. the parse error text).

        Args:
            code: JSON-RPC error code (-327xx standard, -320xx server-defined).
            req_id: Id of the failing request, or None when it could not be determined.
            message: Overrides the canonical message.
            data: Extra detail placed in error["data"].
        """
        # Per JSON-RPC 2.0, `error.data` is optional; it is omitted entirely when there is no detail.
        error = {
            "code": code,
            "message": message or self._RPC_ERROR_MESSAGES.get(code, "Error"),
        }
        if data is not None:
            error["data"] = data
        return {
            "jsonrpc": self._JSONRPC_VERSION,
            "error": error,
            "id": req_id,
        }

    def _rpc_result_response(self, req_id: Any, result: Any) -> dict:
        """Build a JSON-RPC 2.0 success envelope for request `req_id`."""
        return {
            "jsonrpc": self._JSONRPC_VERSION,
            "id": req_id,
            "result": result,
        }

    async def _send_rpc_error(
        self,
        websocket,
        code: int,
        req_id: Any = None,
        message: str | None = None,
        data: Any = None,
    ) -> None:
        """Serialize and send an error response over `websocket`."""
        await websocket.send(json.dumps(self._rpc_error_response(code, req_id, message, data)))

    async def _send_rpc_result(self, websocket, req_id: Any, result: Any) -> None:
        """Serialize and send a success response over `websocket`."""
        await websocket.send(json.dumps(self._rpc_result_response(req_id, result)))

    def _parse_json_rpc_request(self, message: str) -> tuple[dict | None, dict | None]:
        """Parse and structurally validate one raw JSON-RPC request string.

        Returns (request, None) on success, or (None, error_response) where the error uses the matching
        standard code: -32700 bad JSON, -32600 not a valid request object, -32601 unknown method (not in
        _RPC_METHODS), -32602 when `params` is present but not an object. A missing/null `params` is
        normalized to {} so handlers can always call params.get(...).

        Args:
            message: Raw text frame received from the client.

        Returns:
            A (request_dict, None) or (None, error_envelope) tuple; exactly one element is None.
        """
        # A top-level non-object (list, number) is valid JSON but an invalid request, handled just below.
        try:
            data = json.loads(message)
        except json.JSONDecodeError as exc:
            return None, self._rpc_error_response(-32700, None, data=str(exc))

        # The id is extracted before validation so even malformed requests get an error that carries it back.
        req_id = data.get("id") if isinstance(data, dict) else None
        if not isinstance(data, dict):
            return None, self._rpc_error_response(-32600, req_id)

        # Strict version check: a request without "jsonrpc": "2.0" is rejected as -32600.
        if data.get("jsonrpc") != self._JSONRPC_VERSION:
            return None, self._rpc_error_response(-32600, req_id)
        method = data.get("method")
        if not isinstance(method, str) or not method:
            return None, self._rpc_error_response(-32600, req_id)

        # Whitelist check happens before scope or param checks so unknown methods never reach a handler.
        if method not in self._RPC_METHODS:
            return None, self._rpc_error_response(
                -32601,
                req_id,
                data=f"Method not found: {method}",
            )

        # Per JSON-RPC, params may be omitted or an object (arrays are not supported by this bridge); null is
        # treated as "no params".
        params = data.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return None, self._rpc_error_response(
                -32602,
                req_id,
                data="params must be an object when provided.",
            )
        data["params"] = params
        return data, None

    def _validate_rpc_params(self, method: str, params: dict) -> str | None:
        """Check per-method required parameters before dispatch.

        Returns an error string describing the first problem found, or None when the params are acceptable.
        This is shape validation only; semantic checks (path safety, existence) happen inside the individual
        handlers. Methods not mentioned here take optional-only params.

        Args:
            method: RPC method name (already known to be in _RPC_METHODS).
            params: The normalized params object.
        """
        # The error text is returned to the client in error["data"], so it names the offending parameter.
        def require_string(name: str) -> str | None:
            # Local helper: a required, non-blank string parameter.
            value = params.get(name)
            if not isinstance(value, str) or not value.strip():
                return f"{name} is required and must be a string."
            return None

        # Methods with one mandatory string share the same helper; the first failing field wins.
        if method == "set_active_model":
            return require_string("filename")
        if method == "get_prompt_pair":
            return require_string("name")
        if method == "save_prompt_pair":
            return require_string("name")
        if method == "delete_prompt_pair":
            return require_string("name")
        if method == "ingest_path":
            return require_string("path")
        if method == "search_kb":
            return require_string("query")
        if method == "submit_chat":
            return require_string("message")

        # Note the loop returns only on the first error; falling out of the loop continues to later checks.
        if method == "submit_task":
            for name in ("objective", "workspace_path"):
                error = require_string(name)
                if error:
                    return error
        # List-of-paths check: every element must be a string, otherwise the later path handling could crash.
        if method == "swarm_commit_edits":
            selected = params.get("selected_filepaths")
            if not isinstance(selected, list) or any(not isinstance(p, str) for p in selected):
                return "selected_filepaths is required and must be a list of strings."
        if method == "rollback_swarm_run":
            error = require_string("workspace_path")
            if error:
                return error
            run_id = params.get("run_id")
            if run_id is not None and not isinstance(run_id, str):
                return "run_id must be a string when provided."
        if method == "swarm_inject_guidance":
            for name in ("filepath", "message"):
                error = require_string(name)
                if error:
                    return error
        if method in ("swarm_pause", "swarm_resume", "swarm_step"):
            error = require_string("run_id")
            if error:
                return error
        if method == "swarm_override_task":
            for name in ("run_id", "filepath", "instructions"):
                error = require_string(name)
                if error:
                    return error
        if method == "swarm_get_history":
            run_id = params.get("run_id")
            if run_id is not None and not isinstance(run_id, str):
                return "run_id must be a string when provided."
        if method == "save_agent_profile":
            profile = params.get("profile")
            if not isinstance(profile, dict):
                return "profile is required and must be an object."

        # topic and adapter_name are mandatory; numeric parameters are validated separately (they may be omitted).
        if method == "start_auto_train":
            for name in ("topic", "adapter_name"):
                error = require_string(name)
                if error:
                    return error
        # A non-empty list is required: TF-IDF cannot be fitted on zero documents.
        if method == "fit_vectorizer":
            documents = params.get("documents")
            if not isinstance(documents, list) or not documents:
                return "documents is required and must be a non-empty list."
        return None

    def _record_generation_metrics(self, diagnostics: dict | None) -> None:
        """Normalize generation diagnostics into the metrics served at /metrics.

        Diagnostics come from different backends with different key names, so each value is read under its
        primary and alternate names. If no explicit KV-cache hit rate is given it is derived as
        cached_tokens / total_prefill_tokens.
        """
        diagnostics = diagnostics or {}

        # Key aliases: different generation backends name the same measurement differently.
        prefill = diagnostics.get("prefill_duration_sec", diagnostics.get("prefill_time", 0.0))
        tps = diagnostics.get("tokens_per_second", diagnostics.get("generation_tps", 0.0))
        vram_delta = diagnostics.get("vram_usage_mb_delta", diagnostics.get("vram_delta_mb", 0.0))

        # Derive the hit rate from token counts when the backend didn't report one.
        kv_hit_rate = diagnostics.get("kv_cache_hit_rate")
        if kv_hit_rate is None:
            kv_cache = diagnostics.get("kv_cache") or {}

            # Prefix-cache hits: tokens whose KV entries were reused instead of recomputed.
            cached = kv_cache.get("tokens_from_cache", diagnostics.get("kv_cache_hits", 0))
            total = (
                diagnostics.get("prefill_tokens_count")
                or cached + kv_cache.get("tokens_to_eval", 0)
            )

            # Guard against division by zero when no prefill tokens were reported.
            kv_hit_rate = (float(cached) / float(total)) if total else 0.0

        # The hit rate is clamped to [0, 1] because a prefix-cache accounting glitch must never emit an
        # out-of-range gauge to Prometheus.
        # Gauge values in a name -> value map; specs below supplies the help text for each.
        self.last_generation_metrics = {
            "prefill_duration_seconds": self._metric_float(prefill),
            "tokens_per_second": self._metric_float(tps),
            "kv_cache_hit_rate": max(0.0, min(1.0, self._metric_float(kv_hit_rate))),
            "vram_delta_mb": self._metric_float(vram_delta),
        }

    def _metric_float(self, value: Any) -> float:
        """Coerce to float, treating None/garbage as 0.0 so metrics never raise."""
        try:
            if value is None:
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _process_rss_bytes(self) -> int:
        """Return this process's resident set size in bytes (0 if psutil is missing)."""
        try:
            import psutil
            return int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            return 0

    def _prometheus_metrics(self) -> str:
        """Render last-generation and connection gauges in Prometheus text exposition format.

        The "# HELP"/"# TYPE" lines are required by the text format version 0.0.4; every series is a gauge
        because each value is a point-in-time sample, not a counter.
        """
        metrics = {
            "karl_active_connections": len(self.clients),
            "karl_system_memory_usage_bytes": self._process_rss_bytes(),
            "karl_prefill_duration_seconds": self.last_generation_metrics.get(
                "prefill_duration_seconds", 0.0
            ),
            "karl_tokens_per_second": self.last_generation_metrics.get(
                "tokens_per_second", 0.0
            ),
            "karl_kv_cache_hit_rate": self.last_generation_metrics.get(
                "kv_cache_hit_rate", 0.0
            ),
            "karl_vram_delta_mb": self.last_generation_metrics.get("vram_delta_mb", 0.0),
        }
        specs = [
            ("karl_active_connections", "Number of active WebSocket clients"),
            ("karl_system_memory_usage_bytes", "Resident memory usage of the Karl process in bytes"),
            ("karl_prefill_duration_seconds", "Prefill latency of the last generation in seconds"),
            ("karl_tokens_per_second", "Token generation throughput of the last generation"),
            ("karl_kv_cache_hit_rate", "KV cache hit rate of the last generation from 0.0 to 1.0"),
            ("karl_vram_delta_mb", "VRAM consumption delta of the last generation in megabytes"),
        ]

        # Format per metric: HELP line, TYPE line, then "name value".
        lines = []
        for name, help_text in specs:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {metrics[name]}")
        return "\n".join(lines) + "\n"

    def _http_response(self, status_code: int, reason: str, content_type: str, body: str) -> Response:
        """Build a websockets Response with an accurate Content-Length for plain HTTP replies.

        Content-Length is computed on the encoded bytes (not len(str)) so multi-byte characters are counted correctly.
        """
        # Status line, headers (type and exact length) and body bytes make a complete HTTP/1.1 reply.
        return Response(
            status_code,
            reason,
            Headers([
                ("Content-Type", content_type),
                ("Content-Length", str(len(body.encode("utf-8")))),
            ]),
            body.encode("utf-8"),
        )

    def _is_websocket_upgrade(self, request) -> bool:
        """True when the request carries WebSocket upgrade headers."""
        headers = getattr(request, "headers", {}) or {}
        try:
            upgrade = headers.get("Upgrade", "")
            connection = headers.get("Connection", "")
        except Exception:
            return False
        return upgrade.lower() == "websocket" or "upgrade" in connection.lower()

    def _process_http_request(self, connection, request) -> Response | None:
        """websockets `process_request` hook: answer plain HTTP before the WS handshake.

        Serves GET /metrics (Prometheus scrape, unauthenticated) and 404s anything that is not a WebSocket
        upgrade. Returning None lets the handshake proceed.

        Args:
            connection: The websockets connection object (unused).
            request: The parsed HTTP request (only its path and headers are read).

        Returns:
            A Response to send instead of upgrading, or None to continue with the WebSocket handshake.
        """
        # Different websockets versions expose the request differently, so read defensively.
        path = getattr(request, "path", "") or ""
        parsed = urlparse(path)

        # Intentionally unauthenticated, like most Prometheus exporters; it exposes only coarse gauges, no content.
        if parsed.path == "/metrics":
            return self._http_response(
                200,
                "OK",
                "text/plain; version=0.0.4; charset=utf-8",
                self._prometheus_metrics(),
            )

        # Returning None tells websockets to continue with the normal WebSocket handshake.
        if self._is_websocket_upgrade(request):
            return None
        return self._http_response(404, "Not Found", "text/plain; charset=utf-8", "Not Found\n")

    async def _audit_session_leases(self) -> None:
        """Periodically sweeps active connections and closes sessions whose lease has exceeded
        _TOKEN_LIFETIME without a refresh.

        Runs forever on the loop thread (cancelled by _stop_server). Close code 4002 tells the client to
        reconnect with a fresh token.
        """
        while True:
            # Sweep every minute; precision beyond that is unnecessary for a 12 hour lease.
            await asyncio.sleep(self._LEASE_AUDIT_INTERVAL)
            now = time.time()
            with self._clients_lock:
                # Snapshot under the lock, close outside it: ws.close awaits, and holding a threading lock across an
                # await would block every other thread that needs the registry.
                metadata_items = list(self.client_metadata.items())

            # session_start is reset by refresh_token, so a client that refreshes regularly is never closed.
            expired = [
                ws
                for ws, meta in metadata_items
                if now - meta.get("session_start", now) > self._TOKEN_LIFETIME
            ]
            for ws in expired:
                with self._clients_lock:
                    cid = self.client_metadata.get(ws, {}).get("id", "?")
                logger.info("Session lease expired for client %s — closing 4002.", cid)
                try:
                    # Close code 4002 lets clients tell lease expiry apart from revocation (4003) and auth failure (4001).
                    await ws.close(self._CLOSE_CODE_LEASE_EXPIRED, "Session lease expired")
                except Exception as exc:
                    logger.debug("Error closing expired session %s: %s", cid, exc)

    async def _close_all_clients(self, code: int, reason: str) -> None:
        """Close every active WebSocket connection with *code* and *reason*.

        return_exceptions=True so one broken socket cannot prevent the others from being closed.
        """
        with self._clients_lock:
            clients = list(self.clients)
        if clients:
            # Concurrent closes so shutdown time does not grow with the number of clients.
            await asyncio.gather(
                *[ws.close(code, reason) for ws in clients],
                return_exceptions=True,
            )

    def force_revoke(self) -> None:
        """Revoke the bridge token and force-close all active connections.

        Safe to call from any thread (e.g. management UI or CLI signal handler). Order: clear memory first
        so no new authentication can succeed, then remove the disk and keychain copies, then drop sockets.
        """
        # Invalidate all in-memory tokens so no new auth can succeed
        # Order matters: wipe memory, then disk, then keychain, then the live sockets.
        self.bridge_token = ""
        self._token_created_at = 0.0
        self._token_store = {}

        # Wipe the on-disk token
        try:
            # Each step has its own try/except so one failure never stops the remaining revocation steps.
            if os.path.exists(self._TOKEN_PATH):
                os.remove(self._TOKEN_PATH)
        except Exception as exc:
            logger.warning("force_revoke: could not remove token file: %s", exc)

        # Remove from OS keychain
        try:
            # Imported lazily; the keychain backend can be slow or unavailable.
            from app.utils.keychain_manager import revoke_tokens
            revoke_tokens()
        except Exception as exc:
            logger.warning("force_revoke: keychain revocation failed: %s", exc)

        # Close active WebSocket sessions
        if self.loop and self.loop.is_running():
            # The coroutine is built first so it can be closed explicitly if scheduling fails (avoids an un-awaited warning).
            close_coro = self._close_all_clients(self._CLOSE_CODE_REVOKED, "Token revoked")
            try:
                asyncio.run_coroutine_threadsafe(close_coro, self.loop)
                logger.info("force_revoke: all connections scheduled for closure.")
            except Exception as exc:
                close_coro.close()
                logger.warning("force_revoke: could not schedule client closure: %s", exc)

    def _start_loop_thread(self):
        """Starts a background daemon thread running an asyncio event loop.

        Daemon so an unclean exit never hangs on this thread.
        """
        # Dedicated loop so asyncio never competes with the Qt event loop.
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.loop_thread.start()

    def _run_loop(self):
        """Thread body: bind the server, start the lease auditor, then run the loop forever.

        On exit all pending tasks are cancelled and awaited before the loop is closed so no coroutine is
        destroyed mid-flight.
        """
        # Each thread needs its own current loop; this thread owns self.loop.
        asyncio.set_event_loop(self.loop)

        # Bind synchronously on this thread so started_event is set before the loop starts serving.
        self.loop.run_until_complete(self._start_server())

        # The auditor only makes sense when a server is actually listening.
        if self.server is not None:
            self._lease_audit_task = self.loop.create_task(self._audit_session_leases())
        try:
            self.loop.run_forever()
        finally:
            try:
                # Cancel leftovers (lease auditor, in-flight handlers) so shutdown does not log "Task was destroyed but it is pending".
                pending = asyncio.all_tasks(self.loop)
                if pending:
                    # Let the loop run to cancel tasks
                    for task in pending:
                        task.cancel()
                    self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            self.loop.close()

    # Clients (Neovim, tooling) read this JSON to find the active port and token without configuration.
    # Discovery file clients read to find the active port/token without config.
    _DISCOVERY_PATH = pathlib.Path.home() / ".karl" / "service_discovery.json"
    # Number of consecutive ports tried, starting at the configured one.
    _PORT_RANGE = 10  # try self.port through self.port + 9

    def _build_ssl_context(self):
        """Return an SSL context from the project cert files, or raise RuntimeError.

        Raises:
            RuntimeError: certificate or key file missing (the caller fails closed).
        """
        if os.path.exists(self._SSL_CERT_PATH) and os.path.exists(self._SSL_KEY_PATH):
            # Server-side TLS context; the default minimum protocol version of the Python build applies.
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(self._SSL_CERT_PATH, self._SSL_KEY_PATH)
            logger.info("WSS Secure WebSockets enabled.")
            return ctx
        logger.error("WSS SSL certificates missing. Server will NOT start without WSS.")
        raise RuntimeError("SSL context required for WSS enforcement.")

    async def _start_server(self):
        """Bind the WebSocket server (called once on the loop thread).

        Loopback hosts use plain WS; any other KARL_WS_HOST requires TLS and fails closed (no plaintext
        fallback). Walks _PORT_RANGE ports to survive conflicts. Failures are logged, not raised, and
        started_event is always set so callers waiting on startup never hang.
        """
        try:
            # Default is loopback-only; deployments (Docker, k8s) set this to 0.0.0.0 together with a certificate.
            host = os.environ.get("KARL_WS_HOST", "localhost")
            ssl_context = None

            # Standard WS (no SSL) bypasses self-signed certificate validation errors
            # in editor webviews, but is only safe for a strictly loopback bind.
            # "0.0.0.0" (and any other host) binds a network-reachable interface, so
            # TLS is mandatory there -- never fall back to plaintext on that path.
            # Only these three are treated as loopback; anything else, including hostnames, takes the TLS-required path.
            if host in ("localhost", "127.0.0.1", "::1"):
                logger.info("Using standard WS (no SSL) for loopback-only server.")
            else:
                ssl_context = self._build_ssl_context()

            # Try self.port through self.port + _PORT_RANGE - 1 to survive conflicts.
            server = None
            last_exc: Exception | None = None

            # Port fallback so two Karl instances (or another app on 8080) do not prevent start-up; the chosen port is published through service discovery.
            for candidate in range(self.port, self.port + self._PORT_RANGE):
                try:
                    server = await websockets.serve(
                        self._handler,
                        host,
                        candidate,
                        ssl=ssl_context,
                        # Hook lets plain HTTP GET /metrics be answered on the same port before the WebSocket handshake.
                        process_request=self._process_http_request,
                    )
                    self.port = candidate
                    break
                except OSError as exc:
                    logger.info("Port %d in use (%s). Trying %d.", candidate, exc, candidate + 1)
                    last_exc = exc

            if server is None:
                raise last_exc or OSError(
                    f"No port available in {self.port}–{self.port + self._PORT_RANGE - 1}"
                )

            # Kept for _stop_server and the /status checks; None means the bridge is not listening.
            self.server = server
            logger.info("Server running on wss://%s:%d", host, self.port)

            # Written only after a successful bind so a stale file never advertises a port nobody listens on.
            self._write_service_discovery()
        except Exception as e:
            logger.warning("Failed to start server: %s", e)
        finally:
            self.started_event.set()

    def _write_service_discovery(self) -> None:
        """Write ~/.karl/service_discovery.json so clients can locate this instance.

        Contains the bearer token, so it lives in the user's home directory; failures are logged only.
        """
        try:
            self._DISCOVERY_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._DISCOVERY_PATH.write_text(
                json.dumps({
                    "active_port": self.port,
                    "token": self.bridge_token,
                    # UTC ISO timestamp lets clients detect a stale file left by a crashed process.
                    "bound_at": datetime.now(timezone.utc).isoformat(),
                }),
                encoding="utf-8",
            )
            logger.info("Service discovery written: %s", self._DISCOVERY_PATH)
        except Exception as exc:
            logger.warning("Could not write service discovery: %s", exc)

    def _remove_service_discovery(self) -> None:
        """Remove ~/.karl/service_discovery.json on shutdown."""
        try:
            # missing_ok: removing an already-absent file is not an error.
            self._DISCOVERY_PATH.unlink(missing_ok=True)
            logger.info("Service discovery file removed.")
        except Exception as exc:
            logger.warning("Could not remove service discovery: %s", exc)

    def stop(self):
        """Synchronously shuts down the server and joins the background thread.

        Order: stop work first (orchestrator, chat, mini-train) so nothing emits notifications into a
        closing server, then close the sockets on the loop, then stop the loop and join its thread.
        Called from the Qt thread; timeouts (5 s server close, 2 s join) prevent shutdown from hanging.
        """
        if self.loop and self.loop.is_running():
            with self._threads_lock:
                # Stop the orchestrator if running
                # request_stop sets a flag; wait() then blocks until the worker thread actually exits.
                if self.orchestrator and self.orchestrator.isRunning():
                    self.orchestrator.request_stop()
                    self.orchestrator.wait()
                self.orchestrator_owner = None

                # Stop chat thread if running
                if self.chat_thread and self.chat_thread.isRunning():
                    if hasattr(self.chat_thread, "request_stop"):
                        self.chat_thread.request_stop()
                    self.chat_thread.wait()
                self.chat_thread_owner = None

                # Stop mini train thread if running
                if hasattr(self, "mini_train_thread") and self.mini_train_thread and self.mini_train_thread.isRunning():
                    self.mini_train_thread.stop()
                    self.mini_train_thread.wait()

            # run_coroutine_threadsafe hands the coroutine to the loop thread; blocking on .result() with a timeout
            # is safe because the loop is a different thread from this caller.
            # Stop websockets server
            future = asyncio.run_coroutine_threadsafe(self._stop_server(), self.loop)
            try:
                future.result(timeout=5.0)

            # Server cleanup is bounded by 5 s so a client that never acknowledges the close cannot hang application exit.
            except TimeoutError:
                future.cancel()
                logger.debug("Timed out waiting for WebSocket server cleanup; cancelling close task.")
            except Exception as e:
                logger.warning(f"Error closing server connection: {e}")

            # Stop the loop and join thread
            # Stopping from another thread must go through call_soon_threadsafe; loop.stop() is not thread-safe.
            self.loop.call_soon_threadsafe(self.loop.stop)
            if self.loop_thread:
                self.loop_thread.join(timeout=2.0)

        # Drop references to closed sockets so they can be garbage collected.
        self.clients.clear()

    async def _stop_server(self):
        """Coroutine run on the loop to stop lease auditing and close the server and clients."""
        audit_task = getattr(self, "_lease_audit_task", None)
        if audit_task is not None:
            # Cancel and await the auditor so it cannot wake up after the server has gone.
            audit_task.cancel()
            try:
                await audit_task
            except asyncio.CancelledError:
                pass
            self._lease_audit_task = None

        if self.clients:
            await asyncio.gather(
                *[client.close() for client in list(self.clients)],
                return_exceptions=True,
            )
            self.clients.clear()

        # close() stops accepting; wait_closed() waits for existing handlers to finish.
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
            logger.info("Server stopped.")

        self._remove_service_discovery()

    async def _handler(self, websocket, path=None):
        """Per-connection coroutine: authenticate, register, then serve JSON-RPC frames until close.

        Lifecycle: (1) read ?token= from the URL and close with 4001 if it is not a valid token; (2) register
        the client with its scopes and two rate-limit buckets; (3) for every incoming frame apply, in order,
        the global rate limit, JSON-RPC parsing, the heavy-method rate limit, parameter validation, the
        authenticate/refresh_token special cases, scope enforcement and finally the method dispatch chain;
        (4) in `finally`, stop work owned by this client and forget its state.

        Error codes: -32700/-32600/-32601/-32602/-32603 are the JSON-RPC standard codes; -32000 is a
        generic application error, -32001 a permission/security denial, -32005 rate limiting.

        Args:
            websocket: The websockets connection object.
            path: Request path on legacy websockets versions that pass it as an argument.
        """
        # Per-connection coroutine: authenticates the handshake, registers the client,
        # then loops over incoming JSON-RPC frames until the socket closes.
        #
        # Custom JSON-RPC error codes used below (server-defined range):
        #   -32000 generic application error / bad token for authenticate or refresh
        #   -32001 permission or security-policy denial (missing scope, blocked path)
        #   -32005 rate limited (global or heavy-method bucket)
        # Per-frame order: rate limit -> parse -> param validation -> auth/refresh
        # methods -> scope check -> dispatch.
        # ── Connection-stage token authentication ─────────────────────────────
        # websockets versions expose the request path differently (legacy handler arg,
        # `.request.path`, or `.path`); try each so the token can be read from the URL.
        # Different websockets releases expose the request path in different places, so probe each.
        raw_path = path or ""
        if not raw_path:
            # websockets >= 12 exposes websocket.request.path; older versions use websocket.path.
            try:
                if hasattr(websocket, "request"):
                    raw_path = websocket.request.path
                else:
                    raw_path = getattr(websocket, "path", "") or ""
            except Exception:
                pass
        
        # parse_qs returns lists; the first ?token= value is used.
        parsed = urlparse(raw_path)
        qs = parse_qs(parsed.query)
        # The token travels in the connection URL (?token=...) because browser/webview
        # WebSocket APIs cannot set custom headers on the handshake.
        # Empty string (no token) simply fails validation below.
        url_token = qs.get("token", [""])[0]

        token_scopes = self._get_token_scopes(url_token)
        # Defensive fallback: a token that validates but yields no scope list gets the
        # full scope set (legacy single-token behavior).
        if token_scopes is None and self._validate_token(url_token):
            token_scopes = list(self._FULL_SCOPES)

        # Note: the rejected path (which contains the attempted token) is logged here; treat logs as sensitive.
        if token_scopes is None:
            # Used only for the log line; unknown if the transport hides it.
            remote = getattr(websocket, "remote_address", "unknown")
            logger.warning(
                "Rejected connection from %s: invalid or expired token (path: %s)",
                remote, raw_path,
            )

            # Custom close code 4001 so clients can distinguish "bad token" from a network failure.
            await websocket.close(self._CLOSE_CODE_UNAUTHORIZED, "Unauthorized: invalid or expired token")
            return
        # ─────────────────────────────────────────────────────────────────────

        # Short random id used only to label this client in logs and the client list.
        # Eight hex chars are enough to tell a handful of local clients apart in logs.
        client_id = str(uuid.uuid4())[:8]

        # IPv4/IPv6 address of the peer; used only for display in the connections panel.
        ip = "unknown"
        try:
            ip = websocket.remote_address[0]
        except Exception:
            pass

        # Register the connection; each client gets two independent token buckets:
        # a general one for all frames and a stricter one for expensive operations.
        # Registry mutation under the lock because other threads (get_client_info, UI) read it concurrently.
        with self._clients_lock:
            self.clients.add(websocket)
            self.client_metadata[websocket] = {
                "id": client_id,
                "ip": ip,
                "connected_at": datetime.now(timezone.utc).isoformat(),
                # Wall-clock start of the current lease; refresh_token resets it.
                "session_start": time.time(),  # reset by refresh_token to extend lease
                "authenticated": True,
                "scopes": token_scopes,         # RBAC scope list for this connection

                # Global bucket: bursts of 20 frames, sustained 10 frames/s per client; generous for UI use, tight against floods.
                "rate_limit_global": TokenBucket(capacity=20.0, fill_rate=10.0),
                # Heavy bucket: 3 burst, one more every 2 s; protects the GPU from rapid-fire chat/train submissions.
                "rate_limit_heavy": TokenBucket(capacity=3.0, fill_rate=0.5),
            }

            # Chat history list used by submit_chat for this connection only.
            self.client_histories[websocket] = []
        logger.info(f"Client connected: {websocket.remote_address} (ID: {client_id})")
        try:
            # One frame = one JSON-RPC request. The loop ends when the client disconnects (ConnectionClosed).
            async for message in websocket:
                # Rate Limiting DoS Protection (checked first before parsing to prevent parser exhaustion)
                # Rate Limiting DoS Protection (checked first before parsing to prevent parser exhaustion)
                with self._clients_lock:
                    client_meta = self.client_metadata.get(websocket)
                
                # Both limiters live in the client's metadata; a missing record means the connection is already being torn down.
                if client_meta:
                    # Checked before parsing JSON so a flood of huge or malformed frames costs one token each, not a parse.
                    global_limiter = client_meta.get("rate_limit_global")
                    if global_limiter and not global_limiter.consume():
                        # Echo back the request id (when the frame is parseable) so the client can
                        # correlate the throttle error with its request.
                        # Try a lightweight peek to extract req_id if valid JSON-RPC
                        req_id = None
                        try:
                            if message.strip().startswith("{"):
                                temp = json.loads(message)
                                if isinstance(temp, dict):
                                    req_id = temp.get("id")
                        except Exception:
                            pass

                        # Reply with -32005 and keep the connection: clients should back off and retry, not reconnect.
                        await self._send_rpc_error(
                            websocket,
                            -32005,
                            req_id,
                            message="Too Many Requests: global rate limit exceeded."
                        )
                        continue

                # Parse errors reply immediately with the standard code and keep the connection open.
                data, rpc_error = self._parse_json_rpc_request(message)
                if rpc_error is not None:
                    await websocket.send(json.dumps(rpc_error))
                    continue

                method = data.get("method")
                params = data.get("params", {})
                req_id = data.get("id")

                # Only used by authenticate/refresh_token; the connection-level scopes were fixed at handshake time.
                token = params.get("token")

                # Heavy Method Rate Limiting
                # Heavy Method Rate Limiting
                # Heavy-method check follows parsing because the method name is needed to pick the bucket.
                if client_meta:
                    heavy_limiter = client_meta.get("rate_limit_heavy")
                    # Methods that start long-running or resource-heavy work get a tighter bucket
                    # (burst 3, refill 0.5/s) than ordinary calls.
                    # Fixed set rebuilt per frame; cheap, and it keeps the list next to the check that uses it.
                    heavy_methods = {
                        "submit_chat", "submit_task", "set_active_model",
                        "ingest_path", "start_auto_train", "start_mini_train"
                    }
                    if method in heavy_methods:
                        if heavy_limiter and not heavy_limiter.consume():
                            await self._send_rpc_error(
                                websocket,
                                -32005,
                                req_id,
                                message=f"Too Many Requests: rate limit exceeded for heavy method '{method}'."
                            )
                            continue

                # Bind a per-frame correlation ID so every log line emitted
                # during this request is tagged with the client and request ID.
                # Correlation id in a context variable: logs from this request (including helpers) can be traced back to the client.
                set_correlation_id(
                    f"ws:{client_id}:{req_id if req_id is not None else new_correlation_id()}"
                )

                # Validation happens before the scope check on purpose: malformed requests get -32602 regardless of privileges.
                params_error = self._validate_rpc_params(method, params)
                if params_error:
                    # -32602 "Invalid params" with the specific problem in `data`.
                    await self._send_rpc_error(websocket, -32602, req_id, data=params_error)
                    continue

                # Re-read the metadata: it is the same dict object normally, but may be {} if the client is closing.
                with self._clients_lock:
                    client_meta = self.client_metadata.get(websocket, {})

                # `authenticate` is kept for protocol compatibility; the handshake token already
                # authenticated the connection.
                # Kept so older clients that send an explicit authenticate frame still work.
                if method == "authenticate":
                    # This method is now redundant for clients who passed handshake
                    # Reaching here means the handshake token was already valid; this only re-checks the token the client sent in params.
                    is_valid = self._validate_token(token)
                    if is_valid:
                        client_meta["authenticated"] = True
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {"authenticated": True}
                        }))
                    else:
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": {"code": -32000, "message": "Invalid token."}
                        }))
                    continue

                # Lease renewal: extends the connection's session_start and issues a replacement token so long-lived
                # editors never have to reconnect.
                if method == "refresh_token":
                    client_token = params.get("token", "")
                    # Accept if the supplied token is current OR the connection is
                    # already authenticated (token was valid at handshake).
                    if self._validate_token(client_token) or client_meta.get("authenticated"):
                        # Fallback to full scopes mirrors the legacy single-token behaviour for connections without metadata.
                        old_scopes = list(client_meta.get("scopes", self._FULL_SCOPES))
                        # Issue a fresh token scoped to what this connection already
                        # held -- never the global admin bridge token. Minting the
                        # admin token here would let any authenticated (even
                        # read-only) client hand itself full admin scope on a new
                        # connection.
                        new_token = self.add_scoped_token(old_scopes)
                        client_meta["scopes"] = old_scopes
                        client_meta["session_start"] = time.time()

                        # Absolute epoch seconds, so clients can schedule the next refresh without clock arithmetic.
                        expires_at = client_meta["session_start"] + self._TOKEN_LIFETIME
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {
                                "token": new_token,
                                "expires_at": expires_at,
                            },
                        }))
                        logger.info(
                            "Token refreshed for client %s. New expiry in %ds.",
                            client_meta.get("id", "?"),
                            int(self._TOKEN_LIFETIME),
                        )
                    else:
                        await self._send_rpc_error(
                            websocket, -32000, req_id, message="Invalid token for refresh."
                        )
                    continue

                # Scope enforcement — verify the client's token carries the
                # required scope for this method before dispatching.
                # Scope enforcement — look up the single scope this method requires.
                # Methods absent from METHOD_SCOPES need only an authenticated connection.
                # Authorization point: every guarded method is checked here, once, before any handler runs.
                required_scope = self.METHOD_SCOPES.get(method)
                if required_scope is not None:
                    # Missing scopes means no privileges (deny), unlike the refresh fallback above.
                    client_scopes = client_meta.get("scopes", [])
                    if required_scope not in client_scopes:
                        await self._send_rpc_error(
                            websocket,
                            -32001,
                            req_id,
                            # -32001 reply names the missing scope so the operator knows which token to issue.
                            message=f"Permission Denied: Missing scope {required_scope}",
                        )
                        continue

                # ── Method dispatch: each branch validates its params, calls a handler, and
                # replies with a JSON-RPC result or error. Unhandled exceptions are caught by
                # the except clause that closes this try block.
                try:
                    # Dispatch chain. Each branch below handles exactly one RPC method named in the branch condition. The
                    # common shape is: read params, call a self._xxx helper or manager, reply with {"result": ...}; anything
                    # raised propagates to the except clause that follows the chain.
                    if method == "get_runtime_status":
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": self._runtime_status()
                        }))

                    # list_models (no scope): registry plus local GGUF files with installed/active flags.
                    elif method == "list_models":
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": self._list_models()
                        }))

                    # list_adapters (read:telemetry): LoRA adapter folder names.
                    elif method == "list_adapters":
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": self._list_adapters()
                        }))

                    # set_active_model (admin:execute): persists the selection and unloads the model; it reloads lazily.
                    elif method == "set_active_model":
                        filename = params.get("filename")
                        adapter = params.get("adapter")
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": self._set_active_model(filename, adapter)
                        }))

                    # list_prompt_pairs (no scope): Prompt Lab pair summaries.
                    elif method == "list_prompt_pairs":
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": self._list_prompt_pairs()
                        }))

                    # get_prompt_pair (no scope): one full pair by sanitized name.
                    elif method == "get_prompt_pair":
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": self._get_prompt_pair(params.get("name", ""))
                        }))

                    # save_prompt_pair (admin:execute): create or overwrite a pair.
                    elif method == "save_prompt_pair":
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": self._save_prompt_pair(params)
                        }))

                    # delete_prompt_pair (admin:execute): idempotent delete.
                    elif method == "delete_prompt_pair":
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": self._delete_prompt_pair(params.get("name", ""))
                        }))

                    # list_kb_sources (read:kb): knowledge-base snapshot.
                    elif method == "list_kb_sources":
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": self._kb_snapshot()
                        }))

                    # Path-taking RPCs must clear _is_safe_path (blocklist + symlink resolution)
                    # before anything touches the filesystem.
                    elif method == "ingest_path":
                        path = params.get("path", "")

                        # The check runs before _start_kb_ingest touches the filesystem; it blocks system and personal directories.
                        if not self._is_safe_path(path):
                            await websocket.send(json.dumps({
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "error": {
                                    "code": -32001,
                                    # Echoing the path helps the user fix the request; it is the caller's own input.
                                    "message": f"Security Error: Ingesting a sensitive system path is blocked: {path}"
                                }
                            }))
                            continue
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": self._start_kb_ingest(params)
                        }))

                    # search_kb (read:kb): dense retrieval with optional distance threshold.
                    elif method == "search_kb":
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": self._search_kb(params)
                        }))

                    # get_agent_profiles (read:telemetry): all saved agent profiles from data/agent_profiles.json.
                    elif method == "get_agent_profiles":
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": load_agent_profiles()
                        }))

                    # save_agent_profile (admin:execute): validates the profile, persists it and notifies every client.
                    elif method == "save_agent_profile":
                        # Defensive duplicate of _validate_rpc_params: the id may arrive as "id" or legacy "profile_id".
                        profile_dict = params.get("profile") if isinstance(params, dict) else None

                        # Redundant safety net; normal requests are rejected earlier by _validate_rpc_params.
                        if not isinstance(profile_dict, dict):
                            await websocket.send(json.dumps(self._rpc_error_response(
                                -32602, req_id, message="Invalid params: profile object is required."
                            )))
                            continue
                        profile_id = str(profile_dict.get("id") or profile_dict.get("profile_id") or "").strip()
                        if not profile_id:
                            await websocket.send(json.dumps(self._rpc_error_response(
                                -32602, req_id, message="Invalid params: profile.id is required."
                            )))
                            continue
                        try:
                            # save_agent_profile validates and normalizes; its exception text becomes the -32602 message.
                            saved = save_agent_profile(profile_id, profile_dict)
                        except Exception as exc:
                            await websocket.send(json.dumps(self._rpc_error_response(
                                -32602, req_id, message=f"Invalid profile: {exc}"
                            )))
                            continue

                        # Returns the normalized profile so the client displays what was actually stored.
                        result = {"status": "saved", "id": profile_id, "profile": saved}
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": result,
                        }))

                        # Broadcast so every connected editor refreshes its agent picker.
                        self._send_notification("agent_profiles_updated", {
                            "profiles": load_agent_profiles()
                        })

                    # submit_task: start (or restart) the Architect/Coder/Tester swarm on a
                    # workspace. Any run already in flight is stopped first so only one
                    # orchestrator exists at a time.
                    elif method == "submit_task":
                        objective = params.get("objective")
                        workspace_path = params.get("workspace_path")

                        # Default verification command for the Tester stage; the orchestrator runs it inside the workspace.
                        test_command = params.get("test_command", "python run_tests.py")

                        if not objective or not workspace_path:
                            await websocket.send(json.dumps({
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "error": {
                                    "code": -32602,
                                    "message": "Invalid params: objective and workspace_path are required."
                                }
                            }))
                            continue

                        # The swarm writes files into this folder, so blocked system/personal paths are refused up front.
                        if not self._is_safe_path(workspace_path):
                            await websocket.send(json.dumps({
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "error": {
                                    "code": -32001,
                                    "message": f"Security Error: Targeting a sensitive system path is blocked: {workspace_path}"
                                }
                            }))
                            continue

                        with self._threads_lock:
                            # If there is an active orchestrator, request stop and wait
                            if self.orchestrator and self.orchestrator.isRunning():
                                # Replace semantics: starting a new task cancels the old one rather than queueing a second GPU job.
                                self.orchestrator.request_stop()
                                self.orchestrator.wait()

                            # Start orchestrator QThread
                            # Passed through unvalidated to the orchestrator; it only reads the keys it knows.
                            hyperparams = params.get("hyperparams", {})
                            self.orchestrator = SwarmOrchestratorThread(
                                workspace_path=workspace_path,
                                objective=objective,
                                test_command=test_command,
                                hyperparams=hyperparams
                            )

                            # Remember who started the run so the `finally` block can stop it if that client disconnects.
                            self.orchestrator_owner = websocket

                            # Bind PyQt signals with DirectConnection to bypass thread event loop queues.
                            # Each swarm signal is re-published as a JSON-RPC notification to all clients.
                            # Bind PyQt signals with DirectConnection to bypass thread event loop queues
                            # Signal wiring: each Qt signal of the orchestrator becomes one JSON-RPC notification. DirectConnection
                            # runs the lambda on the emitting (worker) thread, which is required because this loop thread has no Qt
                            # event loop to deliver queued signals. _send_notification is thread-safe (it hops onto the asyncio loop).
                            self.orchestrator.status_update.connect(
                                lambda msg: self._send_notification("status_update", {"message": msg}),
                                Qt.ConnectionType.DirectConnection
                            )

                            # Architect output: ordered file tasks with dependencies.
                            self.orchestrator.task_plan_created.connect(
                                lambda plan: self._send_notification("task_plan_created", {"plan": plan}),
                                Qt.ConnectionType.DirectConnection
                            )

                            # Emitted when a file is written; includes full content so the editor can refresh its buffer.
                            self.orchestrator.file_edited.connect(
                                lambda path, content: self._send_notification(
                                    "file_edited", {"filepath": path, "content": content}
                                ),
                                Qt.ConnectionType.DirectConnection
                            )

                            # Tester outcome after the final verification command.
                            self.orchestrator.test_result.connect(
                                lambda passed, trace: self._send_notification(
                                    "test_result", {"passed": passed, "error_trace": trace}
                                ),
                                Qt.ConnectionType.DirectConnection
                            )

                            # Per-layer sandbox verification of proposed edits before they are shown for cherry-picking.
                            self.orchestrator.proposal_verification_finished.connect(
                                lambda layer, passed, trace: self._send_notification(
                                    "proposal_verification_finished",
                                    {"layer": layer, "passed": passed, "trace": trace}
                                ),
                                Qt.ConnectionType.DirectConnection
                            )

                            # Final event; includes run_id so the client can later call rollback_swarm_run or swarm_get_history.
                            self.orchestrator.finished_swarm.connect(
                                lambda success, summary: self._send_notification(
                                    "finished_swarm",
                                    {
                                        "success": success,
                                        "summary": summary,
                                        "run_id": getattr(self.orchestrator, "_run_id", None),
                                    }
                                ),
                                Qt.ConnectionType.DirectConnection
                            )

                            # Review gate: clients show these diffs and answer with swarm_commit_edits.
                            self.orchestrator.edits_proposed.connect(
                                lambda proposals: self._send_notification(
                                    "edits_proposed", {"proposals": proposals}
                                ),
                                Qt.ConnectionType.DirectConnection
                            )
                            # ── Swarm 2.0 telemetry — forwarded so the editor extension gets
                            # the same multiverse/memory/specialist/steering visibility as the
                            # desktop Swarm Studio UI.
                            # "Multiverse" step: several candidate implementations were generated for a file.
                            self.orchestrator.candidates_generated.connect(
                                lambda fp, count: self._send_notification(
                                    "swarm_candidates_generated", {"filepath": fp, "count": count}
                                ),
                                Qt.ConnectionType.DirectConnection
                            )

                            # The judge picked one candidate; reason explains why.
                            self.orchestrator.winner_selected.connect(
                                lambda fp, index, reason: self._send_notification(
                                    "swarm_winner_selected", {"filepath": fp, "index": index, "reason": reason}
                                ),
                                Qt.ConnectionType.DirectConnection
                            )

                            # Cross-run memory snippets injected into the coder prompt.
                            self.orchestrator.memory_recalled.connect(
                                lambda fp, text: self._send_notification(
                                    "swarm_memory_recalled", {"filepath": fp, "text": text}
                                ),
                                Qt.ConnectionType.DirectConnection
                            )

                            # Result of a specialist reviewer (security, performance...) on a file.
                            self.orchestrator.specialist_review.connect(
                                lambda fp, specialist, result: self._send_notification(
                                    "swarm_specialist_review",
                                    {"filepath": fp, "specialist": specialist, "result": result}
                                ),
                                Qt.ConnectionType.DirectConnection
                            )

                            # Worker count changed because of memory pressure or failures.
                            self.orchestrator.concurrency_adjusted.connect(
                                lambda workers, reason: self._send_notification(
                                    "swarm_concurrency_adjusted", {"workers": workers, "reason": reason}
                                ),
                                Qt.ConnectionType.DirectConnection
                            )

                            # Echo confirming that user guidance reached the task.
                            self.orchestrator.guidance_injected.connect(
                                lambda fp, message: self._send_notification(
                                    "swarm_guidance_injected", {"filepath": fp, "message": message}
                                ),
                                Qt.ConnectionType.DirectConnection
                            )

                            # Repeated identical failures: the run is looping without progress.
                            self.orchestrator.drift_detected.connect(
                                lambda fp, message: self._send_notification(
                                    "swarm_drift_detected", {"filepath": fp, "message": message}
                                ),
                                Qt.ConnectionType.DirectConnection
                            )

                            # Pause acknowledged at a safe checkpoint, with details of the task it stopped before.
                            self.orchestrator.swarm_paused.connect(
                                lambda task_info: self._send_notification(
                                    "swarm_paused",
                                    {"run_id": getattr(self.orchestrator, "_run_id", None), "task": task_info}
                                ),
                                Qt.ConnectionType.DirectConnection
                            )

                            # Free-form metrics (tokens, timings) forwarded verbatim.
                            self.orchestrator.telemetry_update.connect(
                                lambda payload: self._send_notification("swarm_telemetry", payload),
                                Qt.ConnectionType.DirectConnection
                            )

                            # Started last, after every signal is connected, so no early emission is lost.
                            self.orchestrator.start()
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            # Reply immediately; progress arrives through the notifications connected above.
                            "result": {"status": "started"}
                        }))

                    # rollback_swarm_run: restore files from a swarm run's backup snapshot
                    # (latest run when run_id is omitted).
                    elif method == "rollback_swarm_run":
                        workspace_path = params.get("workspace_path")

                        # Empty/missing run_id means "latest run".
                        run_id = params.get("run_id") or None
                        if not self._is_safe_path(workspace_path):
                            await websocket.send(json.dumps({
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "error": {
                                    "code": -32001,
                                    "message": f"Security Error: Targeting a sensitive system path is blocked: {workspace_path}"
                                }
                            }))
                            continue

                        # Reads the manifest written by SwarmOrchestratorThread._record_backup_manifest and restores or deletes files.
                        result = rollback_swarm_run(workspace_path, run_id)
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": result
                        }))

                    # swarm_commit_edits: apply the subset of proposed edits the user cherry-picked.
                    elif method == "swarm_commit_edits":
                        # Cherry-pick step of the review flow: only these proposed files are written to disk.
                        selected = params.get("selected_filepaths", [])
                        with self._threads_lock:
                            if self.orchestrator and self.orchestrator.isRunning():
                                # Applies only the chosen proposals; the rest are skipped and reported as "skipped".
                                self.orchestrator.commit_selected_edits(selected)
                                await websocket.send(json.dumps({
                                    "jsonrpc": "2.0",
                                    "id": req_id,
                                    "result": {"status": "committed", "count": len(selected)}
                                }))
                            else:
                                await websocket.send(json.dumps({
                                    "jsonrpc": "2.0",
                                    "id": req_id,
                                    "error": {
                                        "code": -32000,
                                        # -32000 generic application error: there is nothing to commit.
                                        "message": "No swarm task is currently awaiting edits."
                                    }
                                }))

                    # swarm_inject_guidance: steer an in-flight swarm run with a user message for a file.
                    elif method == "swarm_inject_guidance":
                        # Guidance targets one file's task; the orchestrator queues it for the next attempt.
                        filepath = params.get("filepath")
                        message = params.get("message")
                        with self._threads_lock:
                            if self.orchestrator and self.orchestrator.isRunning():
                                # Queued for the next attempt at that file; it does not interrupt a generation in progress.
                                self.orchestrator.inject_guidance(filepath, message)
                                await websocket.send(json.dumps({
                                    "jsonrpc": "2.0",
                                    "id": req_id,
                                    "result": {"status": "injected"}
                                }))
                            else:
                                await websocket.send(json.dumps({
                                    "jsonrpc": "2.0",
                                    "id": req_id,
                                    "error": {
                                        "code": -32000,
                                        "message": "No swarm task is currently running."
                                    }
                                }))

                    # swarm_pause / swarm_resume / swarm_step: flow control for the active run.
                    # The caller must name the active run_id so a stale client cannot control a
                    # different run that replaced it.
                    # Matching run_id guards against a stale client acting on a newer run that replaced its own.
                    elif method in ("swarm_pause", "swarm_resume", "swarm_step"):
                        run_id = params.get("run_id")
                        with self._threads_lock:
                            orchestrator = self.orchestrator
                            if not orchestrator or not orchestrator.isRunning():
                                await self._send_rpc_error(
                                    websocket, -32000, req_id, "No swarm task is currently running."
                                )

                            # Mismatch error includes both ids to make stale-client bugs obvious.
                            elif orchestrator._run_id != run_id:
                                await self._send_rpc_error(
                                    websocket, -32000, req_id,
                                    f"run_id '{run_id}' does not match the active swarm run '{orchestrator._run_id}'."
                                )
                            else:
                                if method == "swarm_pause":
                                    # pause() takes effect at the next safe point, hence the "pausing" status.
                                    orchestrator.pause()
                                    status = "pausing"
                                elif method == "swarm_resume":
                                    orchestrator.resume()
                                    status = "resumed"
                                else:
                                    orchestrator.step()
                                    status = "stepped"
                                await websocket.send(json.dumps(
                                    self._rpc_result_response(req_id, {"status": status})
                                ))

                    # swarm_override_task: queue replacement instructions for one file's task in the active run.
                    elif method == "swarm_override_task":
                        run_id = params.get("run_id")
                        filepath = params.get("filepath")
                        instructions = params.get("instructions")
                        with self._threads_lock:
                            orchestrator = self.orchestrator
                            if not orchestrator or not orchestrator.isRunning():
                                await self._send_rpc_error(
                                    websocket, -32000, req_id, "No swarm task is currently running."
                                )
                            elif orchestrator._run_id != run_id:
                                await self._send_rpc_error(
                                    websocket, -32000, req_id,
                                    f"run_id '{run_id}' does not match the active swarm run '{orchestrator._run_id}'."
                                )
                            else:
                                # Replaces the instructions for that file's task and restarts it with the new text.
                                orchestrator.override_active_task(filepath, instructions)
                                await websocket.send(json.dumps(
                                    self._rpc_result_response(req_id, {"status": "override_queued"})
                                ))

                    # swarm_get_history: details of one past run, or a list of all runs (imported
                    # lazily to keep bridge startup light).
                    elif method == "swarm_get_history":
                        run_id = params.get("run_id")

                        # Lazy import keeps bridge start-up light and avoids import cycles.
                        from app.utils.swarm_replay import SwarmReplayManager

                        # Reads past run records from disk; no state is kept between calls.
                        replay = SwarmReplayManager()

                        # Detail view for one run versus the list of all runs.
                        if run_id:
                            result = replay.get_run_details(run_id)
                        else:
                            result = {"runs": replay.list_past_runs()}
                        await websocket.send(json.dumps(self._rpc_result_response(req_id, result)))

                    # submit_chat: run a single chat generation on behalf of this client, with
                    # per-connection history, optional RAG context and optional agent-profile overrides.
                    elif method == "submit_chat":
                        message = params.get("message")
                        workspace_path = params.get("workspace_path")
                        hyperparams = params.get("hyperparams", {})

                        # Redundant with _validate_rpc_params; kept so the handler is safe when invoked directly.
                        if not message:
                            await websocket.send(json.dumps({
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "error": {
                                    "code": -32602,
                                    "message": "Invalid params: message is required."
                                }
                            }))
                            continue

                        # Only one generation/swarm at a time: stop whatever is running first.
                        with self._threads_lock:
                            # Stop orchestrator if running
                            if self.orchestrator and self.orchestrator.isRunning():
                                self.orchestrator.request_stop()
                                self.orchestrator.wait()

                            # Stop chat thread if running
                            if self.chat_thread and self.chat_thread.isRunning():
                                if hasattr(self.chat_thread, "request_stop"):
                                    self.chat_thread.request_stop()
                                self.chat_thread.wait()

                        # Get client history
                        # Get client history
                        # History is per connection, so two editors chatting at once do not see each other's context.
                        history = self.client_histories.setdefault(websocket, [])

                        # Appended before generation so the model sees the user turn; the assistant reply is appended by _on_chat_finished.
                        history.append({"role": "user", "content": message})

                        # Retrieve RAG chunks if enabled
                        # RAG is on by default; failures are logged and chat proceeds without context.
                        retrieved_chunks = []
                        if hyperparams.get("rag_enabled", True):
                            try:
                                self.rag._load_index()

                                # Number of chunks to prepend to the prompt (default 3, same as the desktop Workbench).
                                rag_top_k = int(hyperparams.get("rag_top_k", 3))
                                rag_threshold = float(hyperparams.get("rag_threshold", 0.0))

                                # Returns chunk texts; threshold 0.0 disables the distance filter.
                                retrieved_chunks = self.rag.retrieve(
                                    query=message,
                                    top_k=rag_top_k,
                                    threshold=rag_threshold,
                                )
                            except Exception as re:
                                logger.warning(f"RAG retrieval error: {re}")

                        system_prompt = hyperparams.get("system_prompt")
                        if not system_prompt:
                            # Imported lazily so edits to the hackable prompt file are picked up and startup stays cheap.
                            from core.default_prompts import DEFAULT_SYSTEM_PROMPT
                            system_prompt = DEFAULT_SYSTEM_PROMPT

                        # Resolve agent profile overrides (model / adapter / system prompt prefix)
                        # Optional persona: can swap model/adapter and prefix the system prompt.
                        agent_id = params.get("agent")
                        agent_model_name: str | None = None
                        agent_adapter_name: str | None = None
                        if agent_id:
                            from app.ui.workspaces.workbench.profiles import AGENT_PROFILES

                            # Unknown ids silently yield an empty profile (defaults apply).
                            profile = AGENT_PROFILES.get(agent_id, {})
                            agent_model_name = profile.get("base_model") or None
                            agent_adapter_name = profile.get("adapter") or None
                            # Prepend profile system-prompt prefix if present
                            profile_prompt = profile.get("prompt", "")

                            # Avoids stacking the same persona prefix twice when the caller already included it.
                            if profile_prompt and not system_prompt.startswith(profile_prompt):
                                system_prompt = profile_prompt + "\n" + system_prompt

                        # Callbacks below close over this connection's history list (not the
                        # websocket) so the assistant reply is stored even if later lookups fail.
                        # Closure over the list object itself, not a lookup by websocket, because the connection may be gone when the thread finishes.
                        _history_ref = history

                        # Called on the generation thread: records metrics, stores the assistant turn for future context and signals completion.
                        def _on_chat_finished(
                            thought: str, response: str, diagnostics: dict
                        ) -> None:
                            self._record_generation_metrics(diagnostics)
                            if response:
                                _history_ref.append(
                                    {"role": "assistant", "content": response}
                                )

                            # Tells clients the stream is complete so they can re-enable the input box.
                            self._send_notification("chat_finished", {})

                        # Errors are delivered both as status text and as chat_finished{error} so clients stop their spinner.
                        def _on_chat_error(err: str) -> None:
                            message_text = str(err)
                            self._send_notification(
                                "status_update",
                                {"message": message_text},
                            )
                            self._send_notification(
                                "chat_finished",
                                {"error": message_text},
                            )

                        # run_generation starts a worker thread and returns it; a circuit-breaker
                        # error (repeated model-load failures) is surfaced as an internal error.
                        try:
                            # Starts a worker thread; tokens are pushed as chat_thought_token (reasoning) and chat_response_token (answer).
                            chat_thread = self._inference_service.run_generation(
                                prompt=message,
                                system_prompt=system_prompt,
                                chat_history=history,
                                hyperparams=hyperparams,
                                # Reasoning tokens (inside <think>) are separate from the visible answer, matching the desktop UI's two panes.
                                on_thought_token_cb=lambda token: self._send_notification(
                                    "chat_thought_token", {"token": token}
                                ),
                                on_token_cb=lambda token: self._send_notification(
                                    "chat_response_token", {"token": token}
                                ),
                                on_finished_cb=_on_chat_finished,
                                on_error_cb=_on_chat_error,
                                retrieved_chunks=retrieved_chunks,
                                model_name=agent_model_name,
                                adapter_name=agent_adapter_name,
                                connection_type=Qt.ConnectionType.DirectConnection,
                            )

                        # Model loading failed repeatedly and the breaker is open; reply -32603 with the breaker's message and wait out the cool-down.
                        except CircuitBreakerOpenException as exc:
                            await websocket.send(json.dumps(
                                self._rpc_error_response(
                                    -32603,
                                    req_id,
                                    message=str(exc),
                                )
                            ))
                            continue

                        # Publish the new thread under the lock so stop_task and the disconnect cleanup see a consistent pair.
                        with self._threads_lock:
                            # Stored so stop_task and disconnect cleanup can cancel this generation.
                            self.chat_thread = chat_thread
                            self.chat_thread_owner = websocket
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {"status": "started"}
                        }))

                    # create_custom_agent: persist a user-defined agent profile to
                    # data/custom_agents.json and hot-reload the profile registry.
                    elif method == "create_custom_agent":
                        # Validate params
                        # Inputs: name (key), label (display text), prompt (system prefix) plus optional model/adapter/RAG settings.
                        name = params.get("name", "")
                        label = params.get("label", "")
                        prompt_text = params.get("prompt", "")

                        # Aliased because `re` is shadowed by an exception variable elsewhere in this long function.
                        import re as _re

                        # Lowercase snake_case only: the name becomes a profile key in data/custom_agents.json.
                        # Validation failures all use -32602 with a message naming the problem.
                        if not _re.match(r'^[a-z0-9_]+$', name):
                            await websocket.send(json.dumps(self._rpc_error_response(
                                -32602, req_id,
                                message=f"Invalid params: name '{name}' must be alphanumeric lowercase (a-z, 0-9, _)."
                            )))
                            continue
                        from app.ui.workspaces.workbench.profiles import (
                            AGENT_PROFILES, DEFAULT_PROFILES, reload_profiles
                        )

                        # Built-in profiles are immutable so a client cannot hijack "coder" or "reviewer" personas.
                        if name in DEFAULT_PROFILES:
                            await websocket.send(json.dumps(self._rpc_error_response(
                                -32602, req_id,
                                message=f"Invalid params: '{name}' is a default profile name and cannot be overridden."
                            )))
                            continue

                        # Creating twice is rejected instead of overwriting; delete the file entry to replace it.
                        if name in AGENT_PROFILES and name not in DEFAULT_PROFILES:
                            await websocket.send(json.dumps(self._rpc_error_response(
                                -32602, req_id,
                                message=f"Invalid params: custom agent '{name}' already exists."
                            )))
                            continue

                        # Validate optional model/adapter paths
                        # `or None` turns empty strings into "not set".
                        base_model = params.get("base_model") or None
                        adapter = params.get("adapter") or None

                        # for/else trick: `break` after sending an error means a bad value; the else clause resets _val so the
                        # check right after the loop can tell "broke out" from "all fine".
                        for _label, _val in (("base_model", base_model), ("adapter", adapter)):
                            # Reject separators/traversal so the join below stays inside data/.
                            if _val is not None and not _is_plain_filename(_val):
                                await websocket.send(json.dumps(self._rpc_error_response(
                                    -32602, req_id,
                                    message=f"Invalid params: {_label} must be a plain file name."
                                )))
                                break
                        else:
                            _val = None
                        if _val is not None:
                            continue
                        if base_model:
                            # Only files under data/models are allowed (the plain-filename check above prevents traversal).
                            model_path = os.path.join("data", "models", base_model)
                            if not os.path.isfile(model_path):
                                await websocket.send(json.dumps(self._rpc_error_response(
                                    -32602, req_id,
                                    message=f"Invalid params: model file not found: {model_path}"
                                )))
                                continue
                        if adapter:
                            # Adapters are directories under data/adapters.
                            adapter_path = os.path.join("data", "adapters", adapter)
                            if not os.path.isdir(adapter_path):
                                await websocket.send(json.dumps(self._rpc_error_response(
                                    -32602, req_id,
                                    message=f"Invalid params: adapter directory not found: {adapter_path}"
                                )))
                                continue

                        # Build and persist the new profile
                        # Only fields the caller supplied are stored, so defaults can evolve without rewriting saved agents.
                        new_profile: dict = {
                            "label": label or name,
                            "description": params.get("description", ""),
                            "prompt": prompt_text,
                        }
                        if base_model:
                            new_profile["base_model"] = base_model
                        if adapter:
                            new_profile["adapter"] = adapter

                        # Optional per-agent RAG overrides.
                        if "rag_enabled" in params:
                            new_profile["rag_enabled"] = bool(params["rag_enabled"])
                        if "rag_top_k" in params:
                            new_profile["rag_top_k"] = int(params["rag_top_k"])

                        # Load existing custom agents, add new entry, save
                        # Read-modify-write of one JSON file; a corrupt file is treated as empty rather than blocking creation.
                        custom_agents_path = os.path.join("data", "custom_agents.json")
                        existing: dict = {}
                        if os.path.isfile(custom_agents_path):
                            try:
                                with open(custom_agents_path, "r", encoding="utf-8") as _f:
                                    existing = json.load(_f) or {}
                            except (json.JSONDecodeError, OSError):
                                existing = {}

                        # Last writer wins on the file; a crash mid-write could lose other agents, but the file is small and rewritten quickly.
                        existing[name] = new_profile
                        os.makedirs("data", exist_ok=True)
                        with open(custom_agents_path, "w", encoding="utf-8") as _f:
                            json.dump(existing, _f, indent=2)

                        # Make the in-memory AGENT_PROFILES registry pick up the new file immediately.
                        reload_profiles()

                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0", "id": req_id,
                            "result": {"status": "success", "name": name}
                        }))
                        # Broadcast profile update to all connected clients
                        self._send_notification("agent_profiles_updated", {
                            "name": name, "profile": new_profile
                        })

                    # list_custom_agents (no scope): user-defined profiles only (defaults excluded).
                    elif method == "list_custom_agents":
                        from app.ui.workspaces.workbench.profiles import (
                            AGENT_PROFILES, DEFAULT_PROFILES
                        )

                        # Defaults ship in code; only user-created entries are returned.
                        custom_only = {
                            k: v for k, v in AGENT_PROFILES.items()
                            if k not in DEFAULT_PROFILES
                        }
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0", "id": req_id,
                            "result": custom_only
                        }))

                    # list_codex_topics / get_codex_content: browse the seeded reference library.
                    # Topic names are sanitized to a safe filename stem before being opened.
                    elif method == "list_codex_topics":
                        # Relative to the process working directory, the same location _seed_codex writes to.
                        library_dir = "data/codex_library"

                        # Topic names are the file stems of the seeded library files.
                        topics = []
                        if os.path.exists(library_dir):
                            topics = [os.path.splitext(f)[0] for f in sorted(os.listdir(library_dir)) if f.endswith((".html", ".md"))]
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {"topics": topics}
                        }))

                    elif method == "get_codex_content":
                        topic = params.get("topic", "")
                        content = ""
                        if topic:
                            library_dir = "data/codex_library"

                            # Whitelist filter: strips ".", "/" and "\\" so the join below cannot leave the library directory.
                            safe_name = "".join(c for c in topic if c.isalnum() or c in (" ", "_", "-")).strip()
                            filepath = os.path.join(library_dir, f"{safe_name}.html")
                            if os.path.exists(filepath):
                                try:
                                    with open(filepath, "r", encoding="utf-8") as f:
                                        content = f.read()
                                except Exception as e:
                                    content = f"Error reading topic content: {e}"
                            else:
                                # Missing topics return text (not an error) so the UI can show it directly.
                                content = f"Guide not found for: {topic}"
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {"content": content}
                        }))

                    # compute_diff: character-level HTML diff, reusing Prompt Lab's renderer.
                    elif method == "compute_diff":
                        # Both texts are plain strings; the diff is computed server-side so thin clients need no diff library.
                        text_a = params.get("text_a", "")
                        text_b = params.get("text_b", "")

                        # Shared with Prompt Lab so both UIs render identical diffs.
                        diff_html = generate_char_diff_html(text_a, text_b)
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {"diff_html": diff_html}
                        }))

                    # start_auto_train: launch auto_train.py as a subprocess and stream its
                    # output to clients as auto_train_log notifications. The adapter name is
                    # validated against _SAFE_ADAPTER_NAME_RE because it becomes a path under
                    # data/adapters.
                    elif method == "start_auto_train":
                        topic = params.get("topic")
                        adapter_name = params.get("adapter_name")

                        # Defaults mirror auto_train.py's own CLI defaults.
                        count = params.get("count", 15)
                        epochs = params.get("epochs", 3)
                        lr = params.get("lr", 2e-4)

                        # Redundant with _validate_rpc_params but keeps the handler safe when called directly in tests.
                        if not topic or not adapter_name:
                            await websocket.send(json.dumps({
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "error": {
                                    "code": -32602,
                                    "message": "Invalid params: topic and adapter_name are required."
                                }
                            }))
                            continue

                        if not _SAFE_ADAPTER_NAME_RE.match(adapter_name):
                            await websocket.send(json.dumps({
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "error": {
                                    "code": -32602,
                                    # Regex also guarantees the name cannot contain path separators because it becomes data/adapters/<name>.
                                    "message": "Invalid params: adapter_name must match ^[A-Za-z0-9_-]{1,64}$."
                                }
                            }))
                            continue

                        # Numbers become argv entries, so they are clamped before use.
                        _nums = _coerce_auto_train_numbers(count, epochs, lr)
                        if _nums is None or not isinstance(topic, str):
                            await websocket.send(json.dumps(self._rpc_error_response(
                                -32602, req_id,
                                message="Invalid params: topic must be a string; count, epochs and lr must be numbers."
                            )))
                            continue
                        count, epochs, lr = _nums

                        # Local imports: only this branch needs them.
                        import subprocess
                        import sys

                        # Runs auto_train.py with an argv list (never a shell string) so topic text cannot inject shell syntax;
                        # stdout and stderr are merged and streamed line by line to every client.
                        def run_auto_train():
                            cmd = [
                                sys.executable,
                                "auto_train.py",
                                # Each value is its own argv element: no shell parsing, so quotes or semicolons in the topic are harmless.
                                "--topic", topic,
                                "--adapter_name", adapter_name,
                                "--count", str(count),
                                "--epochs", str(epochs),
                                "--lr", str(lr)
                            ]
                            try:
                                # stderr merged into stdout so errors stream into the same log; bufsize=1 with text=True gives line-by-line output.
                                proc = subprocess.Popen(
                                    cmd,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True,
                                    bufsize=1,
                                    universal_newlines=True
                                )
                                for line in proc.stdout:
                                    self._send_notification("auto_train_log", {
                                        "adapter_name": adapter_name,
                                        "message": line.strip()
                                    })

                                # Exit code 0 means success; any other value is reported as a failure.
                                rc = proc.wait()
                                self._send_notification("auto_train_finished", {
                                    "adapter_name": adapter_name,
                                    "success": rc == 0,
                                    "message": f"Process exited with code {rc}"
                                })
                            except Exception as e:
                                self._send_notification("auto_train_finished", {
                                    "adapter_name": adapter_name,
                                    "success": False,
                                    "message": str(e)
                                })
                        
                        # Daemon thread so a hung training process cannot block shutdown.
                        threading.Thread(target=run_auto_train, daemon=True).start()
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {"status": "started"}
                        }))

                    # fit_vectorizer: fit a TF-IDF embedder on the supplied documents and return
                    # the vocabulary and per-document vectors (used by the embedding visualizer).
                    elif method == "fit_vectorizer":
                        documents = params.get("documents", [])

                        # Redundant with param validation; keeps the handler safe when called directly.
                        if not documents:
                            await websocket.send(json.dumps({
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "error": {
                                    "code": -32602,
                                    "message": "Invalid params: documents list is required."
                                }
                            }))
                            continue
                        
                        # Imported lazily; numpy-based and only needed by the embedding visualizer.
                        from app.utils.custom_embeddings import TfidfEmbedder
                        tfidf = TfidfEmbedder()

                        # Learns the vocabulary and inverse document frequencies from the supplied texts.
                        tfidf.fit(documents)
                        
                        vectors = []
                        for doc in documents:
                            v = tfidf.transform(doc)

                            # numpy arrays are converted to lists because json.dumps cannot serialise ndarray.
                            vectors.append(v.tolist())
                            
                        # Sorted by column index so vocabulary[i] labels the i-th dimension of every vector.
                        vocab_sorted = [word for word, idx in sorted(tfidf.vocabulary.items(), key=lambda item: item[1])]
                        
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {
                                "vocabulary": vocab_sorted,
                                "vectors": vectors
                            }
                        }))

                    # start_mini_train: train the tiny educational GPT on a Shakespeare corpus.
                    # If data/tiny_shakespeare.txt is absent, a short built-in excerpt is
                    # repeated 20x and written to that path so later runs reuse it.
                    elif method == "start_mini_train":
                        # Defaults: lr 1e-3, 100 iterations, batch 16: short enough to demo quickly.
                        lr = float(params.get("lr", 0.001))
                        max_iters = int(params.get("max_iters", 100))
                        batch_size = int(params.get("batch_size", 16))
                        
                        # Fallback corpus keeps the demo working offline when the dataset file was never downloaded.
                        shakespeare_path = "data/tiny_shakespeare.txt"
                        if os.path.exists(shakespeare_path):
                            with open(shakespeare_path, "r", encoding="utf-8") as f:
                                # The corpus is read fully into memory; tiny_shakespeare is about 1 MB.
                                dataset_text = f.read()
                        else:
                            dataset_text = (
                                "ROMEO:\n"
                                "But, soft! what light through yonder window breaks?\n"
                                "It is the east, and Juliet is the sun.\n"
                                "Arise, fair sun, and kill the envious moon,\n"
                                "Who is already sick and pale with grief,\n"
                                "That thou her maid art far more fair than she:\n"
                                "Be not her maid, since she is envious;\n"
                                "Her vestal livery is but sick and green\n"
                                "And none but fools do wear it; cast it off.\n"
                                "It is my lady, O, it is my love!\n"
                                "O, that she knew she were!\n"
                                "She speaks yet she says nothing: what of that?\n"
                                "Her eye discourses; I will answer it.\n"
                                "I am too bold, 'tis not to me she speaks:\n"
                                "Two of the fairest stars in all the heaven,\n"
                                "Having some business, do entreat her eyes\n"
                                "To twinkle in their spheres till they return.\n"
                                "What if her eyes were there, they in her head?\n"
                                "The brightness of her cheek would shame those stars,\n"
                                "As daylight doth a lamp; her eyes in heaven\n"
                                "Would through the airy region stream so bright\n"
                                "That birds would sing and think it were not night.\n"
                                "See, how she leans her cheek upon her hand!\n"
                                "O, that I were a glove upon that hand,\n"
                                "That I might touch that cheek!\n\n"
                                "JULIET:\n"
                                "Ay me!\n\n"
                                "ROMEO:\n"
                                "She speaks:\n"
                                "O, speak again, bright angel! for thou art\n"
                                "As glorious to this night, being o'er my head\n"
                                "As is a winged messenger of heaven\n"
                                "Unto the white-upturned wondering eyes\n"
                                "Of mortals that fall back to gaze on him\n"
                                "When he bestrides the lazy-pacing clouds\n"
                                "And sails upon the bosom of the air.\n"

                            # Repetition gives the tiny model enough text to show a falling loss curve.
                            ) * 20

                            # Persist the fallback corpus so later runs and the desktop Mini-GPT tab share it.
                            os.makedirs("data", exist_ok=True)
                            with open(shakespeare_path, "w", encoding="utf-8") as f:
                                f.write(dataset_text)

                        # Fixed small-model hyperparameters; only lr, iterations and batch size are caller-tunable.
                        # block_size 64 / n_embd 128 / 4 heads / 4 layers is a deliberately tiny GPT that trains in minutes on CPU.
                        config = {
                            "batch_size": batch_size,
                            "block_size": 64,
                            "n_embd": 128,
                            "n_heads": 4,
                            "n_layers": 4,
                            "lr": lr,
                            "max_iters": max_iters,
                            "eval_interval": 20,
                            "sample_interval": 50
                        }

                        # Replace any previous mini-train run, and forward its signals as status notifications.
                        with self._threads_lock:
                            if hasattr(self, "mini_train_thread") and self.mini_train_thread and self.mini_train_thread.isRunning():
                                self.mini_train_thread.stop()
                                self.mini_train_thread.wait()
                            
                            from app.engine.mini_train_thread import MiniTrainThread

                            # Training runs in a QThread; its signals (log, loss, progress, done, error) are all mapped to status_update notifications.
                            self.mini_train_thread = MiniTrainThread(
                                dataset_text=dataset_text,
                                config=config
                            )
                            
                            self.mini_train_thread.log.connect(
                                lambda msg: self._send_notification("status_update", {"message": msg}),
                                Qt.ConnectionType.DirectConnection
                            )
                            self.mini_train_thread.loss.connect(
                                lambda step, loss_val: self._send_notification("status_update", {"message": f"Step {step} | Loss: {loss_val:.4f}"}),
                                Qt.ConnectionType.DirectConnection
                            )
                            self.mini_train_thread.progress.connect(
                                lambda step, max_steps, sample_text: self._send_notification("status_update", {"message": f"--- Generation Output (Step {step} / {max_steps}) ---\n{sample_text}"}),
                                Qt.ConnectionType.DirectConnection
                            )
                            self.mini_train_thread.done.connect(
                                lambda save_dir: self._send_notification("status_update", {"message": f"Training completed. Saved to {save_dir}"}),
                                Qt.ConnectionType.DirectConnection
                            )
                            self.mini_train_thread.error.connect(
                                lambda err: self._send_notification("status_update", {"message": f"[Mini-GPT Error] {err}"}),
                                Qt.ConnectionType.DirectConnection
                            )
                            
                            # Started inside the lock so stop() cannot observe a half-built thread.
                            self.mini_train_thread.start()
                            
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {
                                "status": "started",
                                # Fixed id: only one mini training run exists at a time.
                                "task_id": "mini-gpt-train"
                            }
                        }))

                    # stop_task: request a cooperative stop of whichever job (swarm or chat) is running.
                    # stop_task: cooperative stop, so the worker finishes its current step; status reflects what was running.
                    elif method == "stop_task":
                        with self._threads_lock:
                            # Status values: "idle" (nothing to stop) or "stopping" (stop requested; the worker exits at its next check).
                            status = "idle"
                            if self.orchestrator and self.orchestrator.isRunning():
                                self.orchestrator.request_stop()
                                status = "stopping"

                            # Swarm takes precedence; at most one of the two runs because new submissions stop the other.
                            elif self.chat_thread and self.chat_thread.isRunning():
                                if hasattr(self.chat_thread, "request_stop"):
                                    self.chat_thread.request_stop()
                                status = "stopping"
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {"status": status}
                        }))

                    # Unreachable for methods in _RPC_METHODS without a branch; defensive fallback.
                    # Defensive fallback
                    else:
                        await websocket.send(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": {
                                "code": -32601,
                                # Defensive fallback
                                "message": f"Method not found: {method}"
                            }
                        }))

                # Any handler failure becomes an internal error (-32603); the connection stays open.
                # Handler failures reply with -32603 and the traceback goes to the log, not to the client.
                except Exception as inner_e:
                    # logger.exception records the full traceback server-side only.
                    logger.exception("Error handling WebSocket RPC method %r", method)
                    await self._send_rpc_error(websocket, -32603, req_id, data=str(inner_e))

        # Normal client disconnects are not errors.
        except websockets.exceptions.ConnectionClosed:
            pass
        # Cleanup on disconnect: stop work this client started (so orphaned runs don't
        # keep burning the GPU) and drop all per-connection state.
        # Always runs, including after a crash, so stale per-connection state can never accumulate.
        finally:
            with self._threads_lock:
                # Ownership test: only the client that started a run may implicitly cancel it by disconnecting.
                if getattr(self, "orchestrator_owner", None) == websocket:
                    if self.orchestrator and self.orchestrator.isRunning():
                        logger.info("Client disconnected: stopping active orchestrator task.")
                        self.orchestrator.request_stop()
                    self.orchestrator_owner = None
                if getattr(self, "chat_thread_owner", None) == websocket:
                    if self.chat_thread and self.chat_thread.isRunning():
                        logger.info("Client disconnected: stopping active chat generation.")
                        if hasattr(self.chat_thread, "request_stop"):
                            self.chat_thread.request_stop()
                    self.chat_thread_owner = None

            with self._clients_lock:
                self.clients.discard(websocket)
                self.client_metadata.pop(websocket, None)

                # Chat history is dropped with the connection; reconnecting clients start with a fresh context.
                self.client_histories.pop(websocket, None)
            logger.info(f"Client disconnected: {websocket.remote_address}")

    def get_client_info(self) -> list[dict]:
        """Return metadata for all connected clients. Thread-safe snapshot."""
        info = []
        with self._clients_lock:
            metadata_items = list(self.client_metadata.items())
        for ws, meta in metadata_items:
            latency = -1.0
            try:
                # websockets.connection.latency is in seconds
                # websockets.connection.latency is in seconds
                # Round-trip time from websockets' keepalive pings, converted to milliseconds; -1 when unavailable.
                latency = ws.latency * 1000.0 # ms
            except Exception:
                pass
            
            info.append({
                "id": meta.get("id"),
                "ip": meta.get("ip"),
                "latency_ms": latency,
                "connected_at": meta.get("connected_at")
            })
        return info

    def _send_notification(self, method: str, params: dict):
        """Dispatches notification thread-safely into the background event loop.

        Safe to call from Qt worker threads; silently drops the message if the loop is not running.
        """
        # Fire-and-forget: the returned future is intentionally not awaited so worker threads never block on slow clients.
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._broadcast_notification(method, params), self.loop
            )

    def client_count(self) -> int:
        """Return the connected client count under the clients lock."""
        if not hasattr(self, "_clients_lock"):
            self._clients_lock = threading.Lock()
        with self._clients_lock:
            return len(self.clients)

    async def _broadcast_notification(self, method: str, params: dict):
        """Send one notification to all clients concurrently; prune clients whose send failed.

        Notifications have no "id" and expect no reply. NOTE: every client receives every notification,
        whatever its scopes (see the known-limits note in the module header).
        """
        # Serialised once and sent to every client, instead of re-encoding per connection.
        payload = json.dumps({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        })
        if self.clients:
            # Broadcast to all registered websocket connections
            if not hasattr(self, "_clients_lock"):
                self._clients_lock = threading.Lock()
            with self._clients_lock:
                clients = list(self.clients)

            # return_exceptions=True: a dead client yields an Exception result instead of failing the whole broadcast.
            results = await asyncio.gather(
                *[client.send(payload) for client in clients],
                return_exceptions=True,
            )
            failed = [
                client for client, result in zip(clients, results)
                if isinstance(result, Exception)
            ]

            # Prune dead connections now; the handler's finally block would also clean them up, this just speeds it.
            if failed:
                with self._clients_lock:
                    for client in failed:
                        self.clients.discard(client)
                        self.client_metadata.pop(client, None)
                        self.client_histories.pop(client, None)
