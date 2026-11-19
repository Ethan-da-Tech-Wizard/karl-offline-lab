# ==============================================================================
# Program: test_keychain_atomic_write.py
# Purpose: Regression tests for keychain_manager.add_scoped_token writing the
#          token store atomically (temp file + os.replace) with mode 0600.
# ==============================================================================
# Scope: keychain_manager.add_scoped_token persists extra bearer tokens (with their RBAC scopes) to a JSON file that the WebSocket
# bridge reads. The file holds secrets, so two properties are tested: it is created owner-only (mode 0600) and it is written
# atomically (temp file + os.replace) so a crash can never leave a truncated or half-written token store.
# Run: python -m pytest tests/test_keychain_atomic_write.py -q
import pathlib
import json
import os
import stat

# The module object is imported as well as the function because the failure test replaces keychain_manager.json.dump.
from app.utils import keychain_manager
from app.utils.keychain_manager import add_scoped_token


# Scenario: add two tokens with different scopes to a fresh file.
# Expected: both are stored under data["tokens"], the file mode is exactly 0600, and no temp file is left next to it.
# Why: other local users must not be able to read the tokens, and stray .tmp files would hold secrets too.
def test_add_scoped_token_writes_private_file_and_leaves_no_temp_files(tmp_path):
    path = str(tmp_path / "tokens.json")

    # token_path points into pytest's tmp_path so the real data/bridge_token.json is never touched.
    add_scoped_token("tok-a", ["read:telemetry"], token_path=path)
    add_scoped_token("tok-b", ["admin:execute"], token_path=path)
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    assert data["tokens"]["tok-a"] == ["read:telemetry"]
    assert data["tokens"]["tok-b"] == ["admin:execute"]

    # S_IMODE strips the file-type bits so only the permission bits are compared.
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600

    # Exactly one file: proves the temp file was renamed away, not copied.
    assert os.listdir(tmp_path) == ["tokens.json"]


def test_failed_write_keeps_previous_token_file_intact(tmp_path):
    """A crash while dumping must not truncate the existing store (it used to be opened with O_TRUNC)."""
    path = str(tmp_path / "tokens.json")
    add_scoped_token("tok-a", ["read:telemetry"], token_path=path)

    # Snapshot of the good file: the test later proves a failed write leaves these exact bytes in place.
    before = pathlib.Path(path).read_text(encoding="utf-8")

    # Saved so the `finally` block can restore it even if an assertion fails; the replacement is applied to the shared json module.
    real_dump = json.dump

    # Simulates a failure in the middle of serialising (for example the disk filling up).
    def boom(*a, **k):
        raise RuntimeError("disk full")

    # Patched directly (not with monkeypatch) on the json module that keychain_manager uses.
    keychain_manager.json.dump = boom
    try:
        try:
            add_scoped_token("tok-b", ["read:telemetry"], token_path=path)
        except RuntimeError:
            pass
        else:
            # The error must reach the caller: swallowing it would hide that the token was not saved.
            raise AssertionError("expected the simulated write failure to propagate")
    finally:
        keychain_manager.json.dump = real_dump

    # Old behaviour opened the file with O_TRUNC first, which would have left it empty here; the atomic write keeps it intact.
    assert pathlib.Path(path).read_text(encoding="utf-8") == before
    assert os.listdir(tmp_path) == ["tokens.json"]
