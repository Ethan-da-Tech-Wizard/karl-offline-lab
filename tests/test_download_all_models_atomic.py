"""Regression: download_file writes via .part and renames only on success."""
import sys
import types

import pytest


class _Resp:
    def __init__(self, chunks, fail=False):
        self.headers = {"content-length": "6"}
        self._c, self._f = chunks, fail

    def raise_for_status(self):
        pass

    def iter_content(self, n):
        yield self._c[0]
        if self._f:
            raise IOError("boom")
        for c in self._c[1:]:
            yield c


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
def mod(monkeypatch):
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(get=None))
    monkeypatch.setitem(sys.modules, "tqdm", types.SimpleNamespace(tqdm=_Bar))
    sys.modules.pop("download_all_models", None)
    import download_all_models
    yield download_all_models
    sys.modules.pop("download_all_models", None)


def test_success_renames(mod, tmp_path, monkeypatch):
    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: _Resp([b"abc", b"def"]))
    dest = str(tmp_path / "m.gguf")
    mod.download_file("u", dest)
    assert open(dest, "rb").read() == b"abcdef" and not (tmp_path / "m.gguf.part").exists()


def test_failure_leaves_no_final_file(mod, tmp_path, monkeypatch):
    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: _Resp([b"abc", b"def"], fail=True))
    with pytest.raises(IOError):
        mod.download_file("u", str(tmp_path / "m.gguf"))
    assert list(tmp_path.iterdir()) == []
