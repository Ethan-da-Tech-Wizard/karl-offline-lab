"""Regression: going online must not remove HF env vars the user set beforehand."""
import pytest

og = pytest.importorskip("app.engine.offline_guard")


def test_user_value_survives(monkeypatch):
    var = og._HF_OFFLINE_VARS[0]
    for v in og._HF_OFFLINE_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv(var, "1")
    monkeypatch.setattr(og, "_hf_vars_applied", False)
    online = {"v": False}
    monkeypatch.setattr(og, "online_execution_enabled", lambda: online["v"])
    og.apply_for_current_config()
    others = [v for v in og._HF_OFFLINE_VARS if v != var]
    assert all(__import__("os").environ.get(v) == "1" for v in others)
    online["v"] = True
    og.apply_for_current_config()
    import os
    assert os.environ.get(var) == "1"
    assert all(v not in os.environ for v in others)
    for v in og._HF_OFFLINE_VARS:
        os.environ.pop(v, None)
