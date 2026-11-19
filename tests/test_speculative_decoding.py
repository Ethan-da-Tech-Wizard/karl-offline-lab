# ==============================================================================
# Program: test_speculative_decoding.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Speculative decoding configuration and integration guards.
#          Coordinates with adjacent subsystems within the Karl Offline Lab
#          architecture.
# ==============================================================================

"""Speculative decoding configuration and integration guards."""

from __future__ import annotations

from pathlib import Path


def test_registry_companion_draft_is_discoverable(tmp_path, monkeypatch):
    """A registry entry can name a companion draft model for a target; it is discoverable by registry_draft_model_filename and the draft config is off by default."""
    from app.engine import config_store

    monkeypatch.chdir(tmp_path)
    # The registry is cached in module globals keyed by mtime; reset them so this test's fresh file is read instead of a cached one from an earlier test.
    config_store._registry_cache = None
    config_store._registry_cache_mtime = None

    config_store.write_json_atomic(
        config_store.MODEL_REGISTRY_PATH,
        [
            {
                "filename": "target.gguf",
                "n_ctx": 4096,
                "draft_model_filename": "draft.gguf",
            }
        ],
    )
    config_store.set_active_model("target.gguf")

    draft_cfg = config_store.get_active_draft_model()

    assert config_store.registry_draft_model_filename("target.gguf") == "draft.gguf"
    # Speculative decoding is opt-in: merely naming a companion draft in the registry must not turn it on.
    assert not draft_cfg["enabled"]
    assert draft_cfg["filename"] == "draft.gguf"


def test_active_draft_model_persists_enabled_flag(tmp_path, monkeypatch):
    """set_active_draft_model persists the filename and enabled flag, and clearing it persists disabled with no filename."""
    from app.engine import config_store

    monkeypatch.chdir(tmp_path)

    assert config_store.set_active_draft_model("draft.gguf", enabled=True)
    draft_cfg1 = config_store.get_active_draft_model()
    assert draft_cfg1["enabled"]
    assert draft_cfg1["filename"] == "draft.gguf"

    # Clearing the draft must also clear the filename (None), so a stale draft file is never loaded after the user turns the feature off.
    assert config_store.set_active_draft_model(None, enabled=False)
    draft_cfg2 = config_store.get_active_draft_model()
    assert not draft_cfg2["enabled"]
    assert draft_cfg2["filename"] is None


def test_model_loader_uses_persisted_draft_setting():
    """Source-level guard: ModelLoader reads the persisted draft setting, reloads when it changes, wraps the draft in an adapter class and passes it as draft_model."""
    # Source-level guards: loading a model needs hardware, so the wiring is pinned by checking the source text. They fail loudly if a refactor removes the draft path, but do not prove it works at runtime.
    source = Path("app/engine/model_loader.py").read_text(encoding="utf-8")

    assert "config_store.get_active_draft_model()" in source
    assert "needs_reload or needs_draft_reload" in source
    assert "change speculative draft model" in source
    assert "class _LlamaDraftModelAdapter" in source
    assert "draft_model=cls._draft_instance" in source


def test_status_bar_exposes_speculative_indicator():
    """Source-level guard: the status bar has set_speculative_active and shows '[Speculative Active]'."""
    source = Path("app/ui/widgets/status_bar.py").read_text(encoding="utf-8")

    assert "def set_speculative_active" in source
    assert "[Speculative Active]" in source
