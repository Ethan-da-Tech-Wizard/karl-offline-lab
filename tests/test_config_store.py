# Program: karl-offline-lab - test_config_store.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Tests for the JSON config store: safe reads and writes, active model, model registry, adapter compatibility and UI config.
"""Tests for the centralized JSON config store."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.engine import config_store


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Run every test against a throwaway data/ directory."""
    # Work inside a temporary folder so real data is never touched
    monkeypatch.chdir(tmp_path)
    # Reset registry cache between tests
    config_store._registry_cache = None
    config_store._registry_cache_mtime = None
    yield


def test_read_json_missing_returns_default():
    """Scenario: read a file that does not exist. Expected: the supplied default is returned."""
    assert config_store.read_json("data/nope.json", default={"x": 1}) == {"x": 1}


def test_read_json_corrupt_returns_default():
    """Scenario: read a file with invalid JSON. Expected: the supplied default is returned."""
    os.makedirs("data", exist_ok=True)
    with open("data/bad.json", "w", encoding="utf-8") as f:
        f.write("{not json")
    assert config_store.read_json("data/bad.json", default=[]) == []


def test_write_json_atomic_roundtrip():
    """Scenario: write JSON atomically then read it back. Expected: contents match and no temporary files remain."""
    assert config_store.write_json_atomic("data/out.json", {"a": [1, 2]})
    with open("data/out.json", "r", encoding="utf-8") as f:
        assert json.load(f) == {"a": [1, 2]}
    # No leftover temp files
    leftovers = [f for f in os.listdir("data") if f.endswith(".tmp")]
    assert leftovers == []


def test_active_model_defaults_when_missing():
    """Scenario: no active model file. Expected: the default model filename and no adapter."""
    active = config_store.get_active_model()
    assert active["filename"] == config_store.DEFAULT_MODEL_FILENAME
    assert active["adapter"] is None


def test_active_model_roundtrip():
    """Scenario: set then get the active model with an adapter. Expected: the same filename and adapter come back."""
    assert config_store.set_active_model("custom.gguf", adapter="my-adapter")
    active = config_store.get_active_model()
    assert active == {"filename": "custom.gguf", "adapter": "my-adapter"}


def test_active_model_accepts_legacy_model_key():
    """Scenario: file uses the older "model" key. Expected: it is read as the filename."""
    config_store.write_json_atomic(
        config_store.ACTIVE_MODEL_PATH, {"model": "legacy.gguf"}
    )
    assert config_store.get_active_model()["filename"] == "legacy.gguf"


def test_registry_missing_returns_empty():
    """Scenario: no registry file. Expected: an empty list."""
    assert config_store.get_model_registry() == []


def test_registry_caches_until_mtime_changes():
    """Scenario: read the registry twice, then change the file. Expected: the same cached object is reused until the modification time changes, then new data loads."""
    entries = [{"filename": "m.gguf", "n_ctx": 8192}]
    config_store.write_json_atomic(config_store.MODEL_REGISTRY_PATH, entries)
    first = config_store.get_model_registry()
    assert first == entries
    # Cached object is reused while the file is unchanged
    assert config_store.get_model_registry() is first

    updated = [{"filename": "m.gguf", "n_ctx": 16384}]
    config_store.write_json_atomic(config_store.MODEL_REGISTRY_PATH, updated)
    # Force a different modification time so the cache sees the change
    os.utime(config_store.MODEL_REGISTRY_PATH, (1, 1))
    assert config_store.get_model_registry() == updated


def test_registry_n_ctx_lookup_and_default():
    """Scenario: look up context size for a listed and an unlisted model. Expected: the registry value, or the default when unknown."""
    config_store.write_json_atomic(
        config_store.MODEL_REGISTRY_PATH, [{"filename": "m.gguf", "n_ctx": 2048}]
    )
    assert config_store.registry_n_ctx("m.gguf") == 2048
    assert config_store.registry_n_ctx("unknown.gguf") == config_store.DEFAULT_N_CTX


def test_registry_rejects_non_list_payload():
    """Scenario: registry file holds an object instead of a list. Expected: treated as an empty registry."""
    config_store.write_json_atomic(config_store.MODEL_REGISTRY_PATH, {"oops": True})
    assert config_store.get_model_registry() == []


def test_adapter_compatibility_from_config_file():
    """Scenario: adapter config names its base model. Expected: compatible with the matching base model only."""
    adapter_dir = os.path.join("data", "adapters", "tuned-1.5b")
    os.makedirs(adapter_dir, exist_ok=True)
    config_store.write_json_atomic(
        os.path.join(adapter_dir, "adapter_config.json"),
        {"base_model_name_or_path": "deepseek/DeepSeek-R1-Distill-Qwen-1.5B"},
    )
    assert config_store.is_adapter_compatible("deepseek-r1-1.5b.gguf", "tuned-1.5b")
    assert not config_store.is_adapter_compatible("deepseek-r1-llama-8b.gguf", "other")


def test_adapter_compatibility_name_fallback():
    """Scenario: no adapter config, only size hints in the names. Expected: compatibility follows the size in the name."""
    assert config_store.is_adapter_compatible("deepseek-r1-1.5b.gguf", "karl-1.5b-lora")
    assert config_store.is_adapter_compatible("deepseek-r1-llama-8b.gguf", "lora-8b")
    assert not config_store.is_adapter_compatible("deepseek-r1-1.5b.gguf", "lora-8b")


def test_ui_config_defaults_when_missing():
    """Scenario: no UI config file. Expected: the built-in defaults."""
    assert config_store.get_ui_config() == config_store.UI_CONFIG_DEFAULTS


def test_ui_config_roundtrip_ignores_unknown_keys():
    """Scenario: save a UI config containing an unknown key. Expected: known values persist and the unknown key is dropped."""
    assert config_store.save_ui_config(
        {"theme_preset": "Matrix Verdant", "glow_strength": 0.5, "junk": "x"}
    )
    config = config_store.get_ui_config()
    assert config["theme_preset"] == "Matrix Verdant"
    assert config["glow_strength"] == 0.5
    assert "junk" not in config


def test_ui_config_legacy_fallback():
    """Scenario: only the legacy theme file exists. Expected: its values are mapped onto the new UI config keys."""
    config_store.write_json_atomic(
        config_store.LEGACY_THEME_CONFIG_PATH,
        {"theme_name": "Karl Obsidian", "custom_accent": "#33FF33", "reduced_motion": True},
    )
    config = config_store.get_ui_config()
    assert config["theme_preset"] == "Karl Obsidian Core"
    assert config["custom_accent"] == "#33FF33"
    assert config["reduced_motion"] is True
