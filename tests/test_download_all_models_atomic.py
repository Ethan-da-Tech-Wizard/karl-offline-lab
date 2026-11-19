"""Regression: download_file writes via .part and renames only on success."""
# Scope: download_all_models.download_file must write into "<file>.part" and rename to the final name only after a complete download
# (and an optional sha256 check). That way the "file exists" test the batch downloader uses can never mistake a partial or corrupted
# download for a finished model. The `requests` and `tqdm` modules are replaced by tiny stubs, so no network or progress bar is used.
# Run: python -m pytest tests/test_download_all_models_atomic.py -q
import sys
import types

import pytest


# Fake HTTP response: reports content-length 6, yields the given chunks, and can raise partway through to simulate a dropped connection.
class _Resp:
    def __init__(self, chunks, fail=False):
        # Six bytes is the combined size of the chunks used by every test ("abc" + "def"); the downloader reads it only for the progress bar.
        self.headers = {"content-length": "6"}
        self._c, self._f = chunks, fail

    def raise_for_status(self):
        pass

    def iter_content(self, n):
        # The first chunk is delivered before any failure, so a partial .part file really exists when the error is raised.
        yield self._c[0]
        if self._f:
            # IOError is what a dropped connection looks like to requests' streaming iterator.
            raise IOError("boom")
        for c in self._c[1:]:
            yield c


# No-op replacement for tqdm's progress bar (it only needs to be a context manager with update()).
class _Bar:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def update(self, n):
        pass


@pytest.fixture
# Fixture: installs the stub modules, imports download_all_models fresh, and removes it from sys.modules afterwards so the stubs
# do not leak into other test files.
def mod(monkeypatch):
    # `get` is replaced per test; the namespace only has to satisfy `import requests` inside download_all_models.
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(get=None))
    monkeypatch.setitem(sys.modules, "tqdm", types.SimpleNamespace(tqdm=_Bar))

    # Popped before importing so each test gets a module bound to the current stubs, and popped again afterwards.
    sys.modules.pop("download_all_models", None)
    import download_all_models
    yield download_all_models
    sys.modules.pop("download_all_models", None)


# Scenario: two chunks arrive ("abc" + "def"). Expected: the final file holds all six bytes and no .part file remains.
def test_success_renames(mod, tmp_path, monkeypatch):
    # Whatever arguments the downloader passes (url, stream=True) are ignored; the fake always returns the canned response.
    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: _Resp([b"abc", b"def"]))
    dest = str(tmp_path / "m.gguf")
    mod.download_file("u", dest)
    with open(dest, "rb") as fh:
        data = fh.read()

    # Both halves matter: the content is complete and the temporary name is gone.
    assert data == b"abcdef" and not (tmp_path / "m.gguf.part").exists()


# Scenario: the connection fails after the first chunk. Expected: IOError propagates, and the directory is completely empty (neither a
# partial final file nor a stray .part file).
def test_failure_leaves_no_final_file(mod, tmp_path, monkeypatch):
    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: _Resp([b"abc", b"def"], fail=True))

    # The error must propagate so the batch loop can report the failed model.
    with pytest.raises(IOError):
        mod.download_file("u", str(tmp_path / "m.gguf"))

    # An empty directory proves the cleanup in the `except BaseException` branch ran.
    assert list(tmp_path.iterdir()) == []


# When the registry supplies a sha256 and the digest matches, the download is accepted and renamed as usual.
def test_sha256_match_accepts(mod, tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: _Resp([b"abc", b"def"]))
    dest = str(tmp_path / "m.gguf")

    # The expected digest is computed from the same bytes the fake server sends, so this checks the hashing of the streamed data.
    mod.download_file("u", dest, sha256=hashlib.sha256(b"abcdef").hexdigest())
    assert (tmp_path / "m.gguf").exists()


# A wrong digest raises ValueError and removes the .part file, so a corrupted or tampered model is never left on disk under its real name.
def test_sha256_mismatch_rejects_and_keeps_nothing(mod, tmp_path, monkeypatch):
    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: _Resp([b"abc", b"def"]))
    with pytest.raises(ValueError):
        # 64 zeros is a syntactically valid but wrong SHA-256 digest.
        mod.download_file("u", str(tmp_path / "m.gguf"), sha256="0" * 64)
    assert list(tmp_path.iterdir()) == []
