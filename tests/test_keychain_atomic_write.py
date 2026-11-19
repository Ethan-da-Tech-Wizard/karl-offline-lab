# ==============================================================================
# Program: test_keychain_atomic_write.py
# Purpose: Regression tests for keychain_manager.add_scoped_token writing the
#          token store atomically (temp file + os.replace) with mode 0600.
# ==============================================================================
import json
import os
import stat

from app.utils import keychain_manager
from app.utils.keychain_manager import add_scoped_token


def test_add_scoped_token_writes_private_file_and_leaves_no_temp_files(tmp_path):
    path = str(tmp_path / "tokens.json")
    add_scoped_token("tok-a", ["read:telemetry"], token_path=path)
    add_scoped_token("tok-b", ["admin:execute"], token_path=path)
    data = json.loads(open(path, encoding="utf-8").read())
    assert data["tokens"]["tok-a"] == ["read:telemetry"]
    assert data["tokens"]["tok-b"] == ["admin:execute"]
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert os.listdir(tmp_path) == ["tokens.json"]


def test_failed_write_keeps_previous_token_file_intact(tmp_path):
    """A crash while dumping must not truncate the existing store (it used to be opened with O_TRUNC)."""
    path = str(tmp_path / "tokens.json")
    add_scoped_token("tok-a", ["read:telemetry"], token_path=path)
    before = open(path, encoding="utf-8").read()

    real_dump = json.dump

    def boom(*a, **k):
        raise RuntimeError("disk full")

    keychain_manager.json.dump = boom
    try:
        try:
            add_scoped_token("tok-b", ["read:telemetry"], token_path=path)
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected the simulated write failure to propagate")
    finally:
        keychain_manager.json.dump = real_dump
    assert open(path, encoding="utf-8").read() == before
    assert os.listdir(tmp_path) == ["tokens.json"]
