# ==============================================================================
# Program: test_security_guards.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Security Guards Tests — Karl Workbench ======================================= Validates path sandboxing, input sanitization, and XSS escaping hardening.
#          Coordinates with adjacent subsystems within the Karl Offline Lab
#          architecture.
# ==============================================================================

"""
Security Guards Tests — Karl Workbench
=======================================
Validates path sandboxing, input sanitization, and XSS escaping hardening.

Tests cover:
- _is_safe_path: symlink resolution, expanded blocklist, project-root allowance
- _collect_kb_files: realpath used, followlinks=False enforced
- create_custom_agent: base_model (.gguf only, in data/models/) and adapter
  (bare dir name, in data/adapters/) path restrictions
- _set_active_model: adapter restricted to data/adapters/
- start_auto_train: topic and adapter_name character allowlists
- HTML XSS escaping helpers: _escape and _escape_pre
"""

from __future__ import annotations

import tests.qt_test_helper  # noqa: F401

import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_server():
    """
    Instantiate WebSocketServerManager with all async/IO side-effects mocked
    out so we can test security logic without starting a real server.
    """
    # Step 1: Validate input parameters, operational preconditions, and state
    from app.engine.websocket_server import WebSocketServerManager

    with (
        patch.object(WebSocketServerManager, '_start_loop_thread', lambda self: None),
        patch.object(WebSocketServerManager, '_seed_codex', lambda self: None),
        patch.object(WebSocketServerManager, '_ensure_ssl_certs', lambda self: None),
        patch('app.engine.websocket_server.RAGPipeline', MagicMock),
        patch('app.engine.websocket_server.save_cached_token', lambda *args, **kwargs: None),
    ):
        return WebSocketServerManager(port=19999)


# ── _is_safe_path ─────────────────────────────────────────────────────────────

class TestIsSafePath(unittest.TestCase):

    """Tests for WebSocketServerManager._is_safe_path, the guard deciding which filesystem paths remote clients may touch: sensitive system and credential locations are blocked, the project tree is allowed, and symlinks are judged by where they resolve."""
    def setUp(self):
        """Build a server object (heavy dependencies mocked) and remember the project root."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.server = _make_server()
        self.project_root = os.path.realpath(os.getcwd())

    # -- blocklist ----------------------------------------------------------------

    def test_blocks_etc(self):
        """/etc files are blocked."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertFalse(self.server._is_safe_path("/etc/passwd"))

    def test_blocks_root_dir(self):
        """The filesystem root itself is blocked."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertFalse(self.server._is_safe_path("/"))

    def test_blocks_proc(self):
        """/proc (e.g. process memory) is blocked."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertFalse(self.server._is_safe_path("/proc/self/mem"))

    def test_blocks_sys(self):
        """/sys is blocked."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertFalse(self.server._is_safe_path("/sys/kernel"))

    def test_blocks_usr_bin(self):
        """System binaries under /usr/bin are blocked."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertFalse(self.server._is_safe_path("/usr/bin/python3"))

    def test_tmp_is_not_blocked(self):
        """/tmp is deliberately allowed so tempfile-based workspaces and tests work; symlink escapes from /tmp are still caught by the realpath check."""
        # /tmp is intentionally NOT in the blocklist so tempfile-based workspaces
        # and test environments remain functional. Symlink attacks from /tmp are
        # caught by the realpath check in _is_safe_path.
        self.assertTrue(self.server._is_safe_path("/tmp"))

    def test_root_home_keeps_tmp_available_and_sensitive_paths_blocked(self):
        """When the home directory is /root, /tmp stays usable while /etc and /root/.ssh remain blocked."""
        # Step 1: Validate input parameters, operational preconditions, and state
        expanduser = os.path.expanduser
        with patch(
            'app.engine.websocket_server.os.path.expanduser',
            side_effect=lambda path: '/root' if path == '~' else expanduser(path),
        ):
            server = _make_server()
            self.assertTrue(server._is_safe_path('/tmp/karl-workspace'))
            self.assertFalse(server._is_safe_path('/etc/passwd'))
            self.assertFalse(server._is_safe_path('/root/.ssh/id_rsa'))

    def test_blocks_home_root(self):
        """The user's whole home directory is blocked; only specific project paths are allowed."""
        # Step 1: Validate input parameters, operational preconditions, and state
        home = os.path.expanduser("~")
        self.assertFalse(self.server._is_safe_path(home))

    def test_blocks_ssh_dir(self):
        """~/.ssh (private keys) is blocked."""
        # Step 1: Validate input parameters, operational preconditions, and state
        ssh = os.path.join(os.path.expanduser("~"), ".ssh", "id_rsa")
        self.assertFalse(self.server._is_safe_path(ssh))

    def test_blocks_aws_credentials(self):
        """~/.aws (cloud credentials) is blocked."""
        # Step 1: Validate input parameters, operational preconditions, and state
        aws = os.path.join(os.path.expanduser("~"), ".aws", "credentials")
        self.assertFalse(self.server._is_safe_path(aws))

    def test_blocks_gnupg(self):
        """~/.gnupg is blocked."""
        # Step 1: Validate input parameters, operational preconditions, and state
        gpg = os.path.join(os.path.expanduser("~"), ".gnupg", "secring.gpg")
        self.assertFalse(self.server._is_safe_path(gpg))

    def test_blocks_kube_config(self):
        """~/.kube (cluster credentials) is blocked."""
        # Step 1: Validate input parameters, operational preconditions, and state
        kube = os.path.join(os.path.expanduser("~"), ".kube", "config")
        self.assertFalse(self.server._is_safe_path(kube))

    # -- allowlist (project root) -------------------------------------------------

    def test_allows_project_root_itself(self):
        """The project root directory is allowed."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertTrue(self.server._is_safe_path(self.project_root))

    def test_allows_file_inside_project(self):
        """Files inside the project are allowed."""
        # Step 1: Validate input parameters, operational preconditions, and state
        target = os.path.join(self.project_root, "app", "engine", "websocket_server.py")
        if os.path.exists(target):
            self.assertTrue(self.server._is_safe_path(target))

    def test_allows_data_dir(self):
        """The project's data directory is allowed."""
        # Step 1: Validate input parameters, operational preconditions, and state
        data = os.path.join(self.project_root, "data")
        if os.path.exists(data):
            self.assertTrue(self.server._is_safe_path(data))

    # -- edge cases ---------------------------------------------------------------

    def test_empty_string_is_unsafe(self):
        """An empty path is rejected."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertFalse(self.server._is_safe_path(""))

    def test_none_is_unsafe(self):
        """None is rejected rather than raising."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertFalse(self.server._is_safe_path(None))  # type: ignore[arg-type]

    # -- symlink resolution -------------------------------------------------------

    def test_symlink_into_blocked_dir_is_blocked(self):
        """A symlink that resolves into /etc/ must be rejected."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with tempfile.TemporaryDirectory() as tmpdir:
            link_path = os.path.join(tmpdir, "evil_link")
            try:
                # Execute guarded operations and establish recovery boundaries
                os.symlink("/etc/passwd", link_path)
            except (OSError, NotImplementedError):
                self.skipTest("Cannot create symlinks in this environment")
            # The symlink itself is inside a temp dir (not in our project root),
            # and its realpath is /etc/passwd — must be blocked.
            self.assertFalse(self.server._is_safe_path(link_path))

    def test_symlink_inside_project_is_allowed(self):
        """A symlink whose realpath stays inside the project root is fine."""
        # Step 1: Validate input parameters, operational preconditions, and state
        data_dir = os.path.join(self.project_root, "data")
        if not os.path.isdir(data_dir):
            self.skipTest("data/ directory does not exist")
        with tempfile.TemporaryDirectory(dir=self.project_root) as tmpdir:
            link_path = os.path.join(tmpdir, "safe_link")
            try:
                # Execute guarded operations and establish recovery boundaries
                os.symlink(data_dir, link_path)
            except (OSError, NotImplementedError):
                self.skipTest("Cannot create symlinks in this environment")
            self.assertTrue(self.server._is_safe_path(link_path))


# ── _collect_kb_files uses realpath ──────────────────────────────────────────

class TestCollectKbFiles(unittest.TestCase):

    """Tests for _collect_kb_files, which gathers knowledge-base files from a path and must return resolved real paths and never follow symlinks out of the tree."""
    def setUp(self):
        """Create a server object and a temp directory for fixture files."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.server = _make_server()
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        """Remove the temp directory."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.tmpdir.cleanup()

    def test_collects_supported_extensions(self):
        """Supported text/code extensions (.txt, .md, .py) are collected and .exe is ignored."""
        # Step 1: Validate input parameters, operational preconditions, and state
        for ext in (".txt", ".md", ".py"):
            # Iterate over items, evaluate loop invariant, and update progress
            Path(self.tmpdir.name, f"file{ext}").write_text("content")
        # .exe should be ignored
        Path(self.tmpdir.name, "ignore.exe").write_text("content")
        results = self.server._collect_kb_files(self.tmpdir.name, recursive=False)
        exts = {os.path.splitext(f)[1] for f in results}
        self.assertIn(".txt", exts)
        self.assertIn(".md", exts)
        self.assertIn(".py", exts)
        self.assertNotIn(".exe", exts)

    def test_returns_realpath_for_each_file(self):
        """Results are os.path.realpath values so later safety checks cannot be fooled by symlinks."""
        # Step 1: Validate input parameters, operational preconditions, and state
        txt = Path(self.tmpdir.name, "sample.txt")
        txt.write_text("hello")
        results = self.server._collect_kb_files(str(txt), recursive=False)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], os.path.realpath(str(txt)))

    def test_raises_on_unsupported_single_file(self):
        """Pointing directly at an unsupported file type raises ValueError."""
        # Step 1: Validate input parameters, operational preconditions, and state
        bad = Path(self.tmpdir.name, "evil.exe")
        bad.write_text("nope")
        with self.assertRaises(ValueError):
            self.server._collect_kb_files(str(bad), recursive=False)

    def test_raises_on_missing_path(self):
        """A non-existent path raises FileNotFoundError."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with self.assertRaises(FileNotFoundError):
            self.server._collect_kb_files("/nonexistent/path/abc123", recursive=False)

    def test_symlink_to_blocked_dir_not_followed_in_recursive_walk(self):
        """symlinks inside a directory walk must NOT escape into /etc/."""
        # Step 1: Validate input parameters, operational preconditions, and state
        try:
            # Execute guarded operations and establish recovery boundaries
            link = os.path.join(self.tmpdir.name, "evil")
            os.symlink("/etc", link)
        except (OSError, NotImplementedError):
            self.skipTest("Cannot create symlinks in this environment")
        # With followlinks=False, the walk never enters the symlinked /etc dir.
        results = self.server._collect_kb_files(self.tmpdir.name, recursive=True)
        for path in results:
            # Iterate over items, evaluate loop invariant, and update progress
            self.assertFalse(
                path.startswith("/etc"),
                f"Symlink to /etc was followed: {path}"
            )


# ── create_custom_agent: base_model / adapter validation ─────────────────────

class TestCreateAgentValidation(unittest.TestCase):
    """
    The validation logic lives inside an async handler; we test the _equivalent_
    helper logic that was extracted into the handler to keep tests synchronous.
    """

    def setUp(self):
        """Prepare a server object and the models/adapters directories used by the checks."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.server = _make_server()

    def _validate_base_model(self, base_model):
        """Mirror the validation logic from create_custom_agent handler."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if base_model is None:
            return None
        safe_bm = os.path.basename(str(base_model).strip())
        if (
            not safe_bm
            or not safe_bm.endswith(".gguf")
            or safe_bm != str(base_model).strip()
        ):
            raise ValueError(f"Invalid base_model: {base_model!r}")
        model_path = os.path.join("data", "models", safe_bm)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {safe_bm}")
        return safe_bm

    def _validate_adapter(self, adapter):
        """Mirror the validation logic from create_custom_agent handler."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if adapter is None:
            return None
        safe_adapter = os.path.basename(str(adapter).strip())
        if (
            not safe_adapter
            or safe_adapter != str(adapter).strip()
            or os.sep in str(adapter)
        ):
            raise ValueError(f"Invalid adapter: {adapter!r}")
        adapter_path = os.path.join("data", "adapters", safe_adapter)
        if not os.path.isdir(adapter_path):
            raise FileNotFoundError(f"Adapter not found: {safe_adapter}")
        return safe_adapter

    # base_model checks --------------------------------------------------------

    def test_base_model_none_is_allowed(self):
        """No base model means 'use the default' and is valid."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertIsNone(self._validate_base_model(None))

    def test_base_model_path_separator_rejected(self):
        """A base model name with a path separator is rejected (path traversal)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with self.assertRaises(ValueError):
            self._validate_base_model("../../etc/shadow")

    def test_base_model_path_separator_slash_rejected(self):
        """A forward-slash path in the base model name is rejected."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with self.assertRaises(ValueError):
            self._validate_base_model("data/models/legit.gguf")

    def test_base_model_non_gguf_extension_rejected(self):
        """Only .gguf model files are accepted."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with self.assertRaises(ValueError):
            self._validate_base_model("model.bin")

    def test_base_model_empty_string_rejected(self):
        """An empty string is not the same as None and is rejected."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with self.assertRaises(ValueError):
            self._validate_base_model("   ")

    def test_base_model_valid_requires_file_to_exist(self):
        """A well-formed name for a model that is not installed raises FileNotFoundError."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with self.assertRaises(FileNotFoundError):
            self._validate_base_model("nonexistent_model.gguf")

    def test_base_model_valid_when_file_exists(self):
        """A well-formed name for an existing model file is accepted and returned unchanged."""
        # Step 1: Validate input parameters, operational preconditions, and state
        model_dir = os.path.join("data", "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "_test_security_model.gguf")
        try:
            # Execute guarded operations and establish recovery boundaries
            Path(model_path).touch()
            result = self._validate_base_model("_test_security_model.gguf")
            self.assertEqual(result, "_test_security_model.gguf")
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    # adapter checks -----------------------------------------------------------

    def test_adapter_none_is_allowed(self):
        """No adapter is valid."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertIsNone(self._validate_adapter(None))

    def test_adapter_path_separator_rejected(self):
        """Adapter names with path separators are rejected."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with self.assertRaises(ValueError):
            self._validate_adapter("../../.ssh")

    def test_adapter_slash_prefix_rejected(self):
        """Adapter names starting with a slash (absolute path) are rejected."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with self.assertRaises(ValueError):
            self._validate_adapter("/absolute/adapter")

    def test_adapter_empty_string_rejected(self):
        """An empty adapter name is rejected."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with self.assertRaises(ValueError):
            self._validate_adapter("   ")

    def test_adapter_valid_requires_dir_to_exist(self):
        """A valid adapter name without a matching folder raises FileNotFoundError."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with self.assertRaises(FileNotFoundError):
            self._validate_adapter("nonexistent_adapter_xyz")

    def test_adapter_valid_when_dir_exists(self):
        """A valid adapter name with an existing folder is accepted."""
        # Step 1: Validate input parameters, operational preconditions, and state
        adapter_dir = os.path.join("data", "adapters", "_test_security_adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        try:
            # Execute guarded operations and establish recovery boundaries
            result = self._validate_adapter("_test_security_adapter")
            self.assertEqual(result, "_test_security_adapter")
        finally:
            import shutil
            if os.path.isdir(adapter_dir):
                shutil.rmtree(adapter_dir)


# ── _set_active_model adapter restriction ─────────────────────────────────────

class TestSetActiveModelAdapter(unittest.TestCase):

    """Tests that _set_active_model validates model and adapter names before changing the active model."""
    def setUp(self):
        """Create a server object with mocked heavy dependencies."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.server = _make_server()

    def test_adapter_with_path_separator_raises(self):
        """_set_active_model must reject adapter names containing path components."""
        # We need an installed model to pass the model check first.
        model_dir = os.path.join("data", "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "_test_active_model.gguf")
        try:
            # Execute guarded operations and establish recovery boundaries
            Path(model_path).touch()
            with self.assertRaises(ValueError):
                self.server._set_active_model("_test_active_model.gguf", adapter="../../../etc")
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_missing_model_raises_file_not_found(self):
        """Selecting a model that is not installed raises FileNotFoundError."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with self.assertRaises(FileNotFoundError):
            self.server._set_active_model("definitely_missing.gguf")

    def test_model_with_path_separator_raises_value_error(self):
        """A model name with a path separator raises ValueError."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with self.assertRaises(ValueError):
            self.server._set_active_model("../../etc/shadow.gguf")


# ── auto_train topic / adapter_name sanitization ─────────────────────────────

class TestAutoTrainInputSanitization(unittest.TestCase):
    """
    The regex guards live inside the async handler; we test the patterns directly.
    """

    TOPIC_RE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9 _\-]*$')
    ADAPTER_RE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_\-]*$')

    # topic -------------------------------------------------------------------

    def test_valid_topic(self):
        """Letters, digits, spaces, dashes and underscores are accepted in a topic."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertIsNotNone(self.TOPIC_RE.match("Python basics"))
        self.assertIsNotNone(self.TOPIC_RE.match("data-structures"))
        self.assertIsNotNone(self.TOPIC_RE.match("intro_to_llms"))

    def test_topic_with_semicolon_rejected(self):
        """';' (command chaining) is rejected."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertIsNone(self.TOPIC_RE.match("evil; rm -rf /"))

    def test_topic_with_backtick_rejected(self):
        """Backticks (command substitution) are rejected."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertIsNone(self.TOPIC_RE.match("topic`malicious`"))

    def test_topic_with_slash_rejected(self):
        """Slashes (path traversal) are rejected."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertIsNone(self.TOPIC_RE.match("../../etc/passwd"))

    def test_topic_with_dollar_rejected(self):
        """'$' (variable expansion) is rejected."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertIsNone(self.TOPIC_RE.match("$HOME/secret"))

    def test_topic_starting_with_special_char_rejected(self):
        """A topic must start with an alphanumeric character, so leading whitespace is rejected."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertIsNone(self.TOPIC_RE.match(" leading space"))

    # adapter_name ------------------------------------------------------------

    def test_valid_adapter_name(self):
        """Letters, digits, underscore and dash are accepted for adapter names."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertIsNotNone(self.ADAPTER_RE.match("my_adapter"))
        self.assertIsNotNone(self.ADAPTER_RE.match("adapter-v2"))
        self.assertIsNotNone(self.ADAPTER_RE.match("LoRA1"))

    def test_adapter_name_with_space_rejected(self):
        """Spaces are rejected in adapter names."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertIsNone(self.ADAPTER_RE.match("has space"))

    def test_adapter_name_with_slash_rejected(self):
        """Slashes (traversal) are rejected in adapter names."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertIsNone(self.ADAPTER_RE.match("../../secrets"))

    def test_adapter_name_with_dot_rejected(self):
        """Dots are rejected, which also blocks '..' segments."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertIsNone(self.ADAPTER_RE.match("../evil"))

    def test_adapter_name_empty_rejected(self):
        """An empty adapter name is rejected."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertIsNone(self.ADAPTER_RE.match(""))


# ── HTML XSS escaping helpers ─────────────────────────────────────────────────

class TestXssEscaping(unittest.TestCase):
    """Verify that the Python HTML-escape helpers neutralize injection payloads."""

    def setUp(self):
        """Load the escape helpers under test."""
        # Step 1: Validate input parameters, operational preconditions, and state
        from app.ui.workspaces.workbench.chat_view import _escape, _escape_pre
        self._escape = _escape
        self._escape_pre = _escape_pre

    def test_escape_lt_gt(self):
        """< and > become entities, so <script> is never emitted raw."""
        # Step 1: Validate input parameters, operational preconditions, and state
        result = self._escape("<script>alert(1)</script>")
        self.assertNotIn("<script>", result)
        self.assertIn("&lt;script&gt;", result)

    def test_escape_ampersand(self):
        """& becomes &amp;."""
        # Step 1: Validate input parameters, operational preconditions, and state
        result = self._escape("foo & bar")
        self.assertIn("&amp;", result)
        self.assertNotIn(" & ", result)

    def test_escape_converts_newlines_to_br(self):
        """The normal escape turns newlines into <br> for HTML display."""
        # Step 1: Validate input parameters, operational preconditions, and state
        result = self._escape("line1\nline2")
        self.assertIn("<br>", result)
        self.assertNotIn("\n", result)

    def test_escape_pre_keeps_newlines(self):
        """The <pre> variant keeps raw newlines instead of <br>."""
        # Step 1: Validate input parameters, operational preconditions, and state
        result = self._escape_pre("line1\nline2")
        self.assertIn("\n", result)
        self.assertNotIn("<br>", result)

    def test_escape_pre_still_escapes_html(self):
        """The <pre> variant still escapes markup."""
        # Step 1: Validate input parameters, operational preconditions, and state
        result = self._escape_pre("<b>bold</b>")
        self.assertNotIn("<b>", result)
        self.assertIn("&lt;b&gt;", result)

    def test_escape_event_handler_injection(self):
        """An <img onerror=...> payload is neutralised because the tag itself is escaped."""
        # Step 1: Validate input parameters, operational preconditions, and state
        result = self._escape('<img src=x onerror="alert(1)">')
        self.assertNotIn("<img", result)
        self.assertIn("&lt;img", result)

    def test_escape_script_in_code_block(self):
        """Even inside code fences, < and > are converted to entities."""
        # Step 1: Validate input parameters, operational preconditions, and state
        payload = "```\n<script>evil()</script>\n```"
        # _escape should convert all < > to entities even inside code fences
        result = self._escape(payload)
        self.assertNotIn("<script>", result)


# ── prompt pair name sanitization ────────────────────────────────────────────

class TestPromptPairNameSanitization(unittest.TestCase):

    """Tests for _safe_prompt_pair_name, which turns user input into a safe file name for stored prompt pairs."""
    def setUp(self):
        """Create a server object with mocked heavy dependencies."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.server = _make_server()

    def test_alphanumeric_and_safe_chars_preserved(self):
        """Letters, digits, dash and underscore pass through unchanged."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertEqual(self.server._safe_prompt_pair_name("my-test_pair1"), "my-test_pair1")

    def test_special_chars_replaced(self):
        """Slashes and dots are replaced so names cannot traverse directories."""
        # Step 1: Validate input parameters, operational preconditions, and state
        result = self.server._safe_prompt_pair_name("evil/../../../etc")
        self.assertNotIn("/", result)
        self.assertNotIn(".", result)

    def test_empty_name_raises(self):
        """An empty name (nothing left to use) raises ValueError."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with self.assertRaises(ValueError):
            self.server._safe_prompt_pair_name("   ")

    def test_spaces_replaced(self):
        """Spaces are replaced."""
        # Step 1: Validate input parameters, operational preconditions, and state
        result = self.server._safe_prompt_pair_name("hello world")
        self.assertNotIn(" ", result)

    def test_null_bytes_stripped(self):
        """NUL bytes are removed (they can truncate paths in C APIs)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        result = self.server._safe_prompt_pair_name("ok\x00name")
        self.assertNotIn("\x00", result)


if __name__ == "__main__":
    unittest.main()
