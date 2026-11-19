"""Regression: going online must not remove HF env vars the user set beforehand."""
# Scope: app/engine/offline_guard.py forces Hugging Face libraries offline (HF_HUB_OFFLINE etc.) while Karl runs offline and
# removes those variables again when the user turns online execution on. This regression test pins down one subtle rule: a
# value the USER exported before launching Karl must survive the offline/online toggle, i.e. the guard may only undo what it
# set itself. Run: python -m pytest tests/test_offline_guard_env_restore.py -q
import pytest

# importorskip: the module imports engine code that may be unavailable in minimal environments; the test is then skipped.
og = pytest.importorskip("app.engine.offline_guard")


# Scenario: the user already set ONE of the offline variables to "1"; Karl applies offline mode, then switches online.
# Expected: while offline every variable is "1"; after going online only the guard's own variables are removed and the
# user's variable keeps its value.
# Why it matters: deleting a variable the user set deliberately would silently change behaviour of their other tools.
def test_user_value_survives(monkeypatch):
    # The first variable plays the "user-set" one; the others are set by the guard.
    var = og._HF_OFFLINE_VARS[0]

    # Start from a clean environment so the test does not depend on the developer's shell.
    for v in og._HF_OFFLINE_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv(var, "1")

    # Reset the module's own bookkeeping flag (it records that the guard has already applied the variables).
    monkeypatch.setattr(og, "_hf_vars_applied", False)

    # A mutable dict lets the lambda below read a value the test flips later.
    online = {"v": False}
    monkeypatch.setattr(og, "online_execution_enabled", lambda: online["v"])
    og.apply_for_current_config()
    others = [v for v in og._HF_OFFLINE_VARS if v != var]

    # Offline phase: the guard filled in every variable the user had not set.
    assert all(__import__("os").environ.get(v) == "1" for v in others)

    # Switch to online execution and re-apply the configuration.
    online["v"] = True
    og.apply_for_current_config()
    import os
    assert os.environ.get(var) == "1"

    # The guard removes only the variables it added itself.
    assert all(v not in os.environ for v in others)

    # Final clean-up: monkeypatch cannot undo changes the guard made directly to os.environ, so they are removed by hand.
    for v in og._HF_OFFLINE_VARS:
        os.environ.pop(v, None)
