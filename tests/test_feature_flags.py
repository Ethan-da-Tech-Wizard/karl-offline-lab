# Program: karl-offline-lab - test_feature_flags.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Tests for the feature flag store and the crash-detecting boot guard, all isolated to temporary files.
"""
Feature Flag Registry and Boot Guard tests.

All tests are filesystem-isolated: every fixture receives a tmp_path so that
nothing in the real data/ directory is read or written during the test run.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from app.engine.feature_flags import (
    FeatureFlagStore,
    check_boot_lock,
    create_boot_lock,
    release_boot_lock,
    run_boot_guard,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_safe_mode():
    """Reset the class-level safe_mode_active flag after every test."""
    FeatureFlagStore.safe_mode_active = False
    yield
    FeatureFlagStore.safe_mode_active = False


@pytest.fixture
def flags_path(tmp_path) -> str:
    """Path of a feature flag file inside the temporary folder."""
    return str(tmp_path / "feature_flags.json")


@pytest.fixture
def lock_path(tmp_path) -> str:
    """Path of a boot lock file inside the temporary folder."""
    return str(tmp_path / "boot_in_progress.lock")


@pytest.fixture
def store(flags_path) -> FeatureFlagStore:
    """A feature flag store backed by the temporary file."""
    return FeatureFlagStore(path=flags_path)


# ── Default flag state ────────────────────────────────────────────────────────

def test_default_flags_match_spec(store):
    """Scenario: fresh store. Expected: experimental flags off and the swarming loop flag on."""
    assert store.is_enabled("experimental_speculative_decoding") is False
    assert store.is_enabled("multimodal_vision_ocr") is False
    assert store.is_enabled("agentic_swarming_loops") is True


def test_unknown_flag_returns_false(store):
    """Scenario: ask about a flag that does not exist. Expected: False."""
    assert store.is_enabled("nonexistent_flag") is False


def test_defaults_are_persisted_on_first_load(flags_path):
    """Constructing a store on a fresh path must write defaults to disk."""
    assert not os.path.exists(flags_path)
    FeatureFlagStore(path=flags_path)
    assert os.path.exists(flags_path)
    with open(flags_path, "r", encoding="utf-8") as f:
        data = json.loads(f.read())
    assert "agentic_swarming_loops" in data


# ── set_flag ──────────────────────────────────────────────────────────────────

def test_set_flag_updates_in_memory(store):
    """Scenario: enable a flag. Expected: the store reports it enabled."""
    store.set_flag("experimental_speculative_decoding", True)
    assert store.is_enabled("experimental_speculative_decoding") is True


def test_set_flag_persists_to_disk(flags_path, store):
    """Scenario: enable a flag then reload from disk. Expected: still enabled."""
    store.set_flag("multimodal_vision_ocr", True)
    # Reload from disk to verify persistence
    reloaded = FeatureFlagStore(path=flags_path)
    assert reloaded.is_enabled("multimodal_vision_ocr") is True


def test_set_flag_raises_for_unknown_flag(store):
    """Scenario: set a flag that does not exist. Expected: KeyError."""
    with pytest.raises(KeyError):
        store.set_flag("not_a_real_flag", True)


def test_set_flag_persists_false(flags_path, store):
    """Scenario: turn a default-on flag off then reload. Expected: still off."""
    store.set_flag("agentic_swarming_loops", False)
    reloaded = FeatureFlagStore(path=flags_path)
    assert reloaded.is_enabled("agentic_swarming_loops") is False


# ── enter_safe_mode ───────────────────────────────────────────────────────────

def test_safe_mode_disables_experimental_flags(store):
    """Scenario: enable experimental flags then enter safe mode. Expected: they are all disabled."""
    store.set_flag("experimental_speculative_decoding", True)
    store.set_flag("multimodal_vision_ocr", True)
    store.enter_safe_mode()
    assert store.is_enabled("experimental_speculative_decoding") is False
    assert store.is_enabled("multimodal_vision_ocr") is False


def test_safe_mode_preserves_non_experimental_flags(store):
    """agentic_swarming_loops is NOT experimental — safe mode must not touch it."""
    store.enter_safe_mode()
    assert store.is_enabled("agentic_swarming_loops") is True


def test_safe_mode_sets_class_flag(store):
    """Scenario: enter safe mode. Expected: the class-level safe_mode_active flag becomes True."""
    assert FeatureFlagStore.safe_mode_active is False
    store.enter_safe_mode()
    assert FeatureFlagStore.safe_mode_active is True


def test_safe_mode_persists_experimental_flags_as_false(flags_path, store):
    """Scenario: enter safe mode then reload from disk. Expected: experimental flags stay off."""
    store.set_flag("experimental_speculative_decoding", True)
    store.enter_safe_mode()
    reloaded = FeatureFlagStore(path=flags_path)
    assert reloaded.is_enabled("experimental_speculative_decoding") is False


# ── Boot Guard — individual helpers ──────────────────────────────────────────

def test_check_boot_lock_false_when_absent(lock_path):
    """Scenario: no lock file. Expected: check reports False."""
    assert check_boot_lock(lock_path) is False


def test_create_and_check_boot_lock(lock_path):
    """Scenario: create the lock. Expected: check reports True and the file exists."""
    create_boot_lock(lock_path)
    assert check_boot_lock(lock_path) is True
    assert os.path.exists(lock_path)


def test_boot_lock_contains_timestamp(lock_path):
    """Scenario: create the lock. Expected: the file holds a time value between before and after creation."""
    before = time.time()
    create_boot_lock(lock_path)
    after = time.time()
    with open(lock_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    ts = float(raw)
    assert before <= ts <= after


def test_release_boot_lock_removes_file(lock_path):
    """Scenario: create then release the lock. Expected: the file is gone."""
    create_boot_lock(lock_path)
    release_boot_lock(lock_path)
    assert not os.path.exists(lock_path)


def test_release_boot_lock_is_idempotent(lock_path):
    """Calling release when no lock exists must not raise."""
    release_boot_lock(lock_path)  # should not raise


# ── run_boot_guard — integration ──────────────────────────────────────────────

def test_boot_guard_creates_lock_on_clean_boot(flags_path, lock_path):
    """Normal boot: lock absent → guard creates it and returns False."""
    store = FeatureFlagStore(path=flags_path)
    result = run_boot_guard(store, lock_path)
    assert result is False
    assert os.path.exists(lock_path), "Boot lock must be created on clean boot"


def test_boot_guard_returns_false_and_does_not_activate_safe_mode_on_clean_boot(
    flags_path, lock_path
):
    """Scenario: run the guard with no lock. Expected: returns False, safe mode stays off and defaults are kept."""
    store = FeatureFlagStore(path=flags_path)
    crashed = run_boot_guard(store, lock_path)
    assert crashed is False
    assert FeatureFlagStore.safe_mode_active is False
    assert store.is_enabled("experimental_speculative_decoding") is False  # default
    assert store.is_enabled("agentic_swarming_loops") is True


def test_boot_guard_detects_crash_lock_and_enters_safe_mode(flags_path, lock_path):
    """
    Regression: artificially create the boot lock (simulating a previous crash),
    then run the boot guard and assert that:
      1. It returns True (crash detected).
      2. All experimental flags are reset to False.
      3. FeatureFlagStore.safe_mode_active is True.
    """
    # Pre-condition: enable some experimental flags so we can see them get reset.
    store = FeatureFlagStore(path=flags_path)
    store.set_flag("experimental_speculative_decoding", True)
    store.set_flag("multimodal_vision_ocr", True)

    # Simulate a crashed previous run by writing the lock file.
    create_boot_lock(lock_path)
    assert check_boot_lock(lock_path), "Precondition: lock file must exist"

    # Run the startup initialization routine.
    crashed = run_boot_guard(store, lock_path)

    # ── assertions ────────────────────────────────────────────────────────────
    assert crashed is True, "Boot guard must detect the crash and return True"

    # All experimental flags must be disabled.
    assert store.is_enabled("experimental_speculative_decoding") is False, (
        "experimental_speculative_decoding must be reset to False in Safe Mode"
    )
    assert store.is_enabled("multimodal_vision_ocr") is False, (
        "multimodal_vision_ocr must be reset to False in Safe Mode"
    )

    # Non-experimental flag must be untouched.
    assert store.is_enabled("agentic_swarming_loops") is True, (
        "agentic_swarming_loops is not experimental and must not be disabled"
    )

    # Safe mode indicator must be set.
    assert FeatureFlagStore.safe_mode_active is True, (
        "FeatureFlagStore.safe_mode_active must be True after crash recovery"
    )


def test_boot_guard_crash_recovery_persists_safe_flags(flags_path, lock_path):
    """Safe-mode flag state written during crash recovery survives a reload."""
    store = FeatureFlagStore(path=flags_path)
    store.set_flag("experimental_speculative_decoding", True)
    create_boot_lock(lock_path)

    run_boot_guard(store, lock_path)

    # Reload from disk — the disabled flags must still be disabled.
    reloaded = FeatureFlagStore(path=flags_path)
    assert reloaded.is_enabled("experimental_speculative_decoding") is False


def test_boot_guard_does_not_remove_lock_on_crash_detection(flags_path, lock_path):
    """
    The guard must NOT remove the lock when a crash is detected — the lock is
    only cleared by release_boot_lock() after successful window render.
    """
    store = FeatureFlagStore(path=flags_path)
    create_boot_lock(lock_path)
    run_boot_guard(store, lock_path)
    # Lock file still present — the boot guard doesn't touch it after detection.
    assert os.path.exists(lock_path), (
        "Boot lock must remain after crash detection; "
        "only MainWindow.showEvent should remove it"
    )


# ── all_flags helper ──────────────────────────────────────────────────────────

def test_all_flags_returns_complete_dict(store):
    """Scenario: list all flags. Expected: exactly the default flag names."""
    flags = store.all_flags()
    assert set(flags.keys()) == set(FeatureFlagStore.DEFAULTS.keys())


def test_all_flags_returns_copy(store):
    """Scenario: change the returned dict. Expected: the store itself is not changed."""
    flags = store.all_flags()
    flags["agentic_swarming_loops"] = False
    # Original must not be mutated
    assert store.is_enabled("agentic_swarming_loops") is True
