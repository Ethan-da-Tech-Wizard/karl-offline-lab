# ==============================================================================
# Program: test_kv_quantization.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Comprehensive test suite exercising kv quantization functionality.
#          Verifies state transitions, error boundaries, regression resistance,
#          and contract compliance under tests.
# ==============================================================================

from unittest.mock import MagicMock, patch


def _reset_model_loader_state(ModelLoader):
    """Blank the ModelLoader class-level singleton state before a test."""
    ModelLoader._instance = None
    ModelLoader._instance_locked = False
    ModelLoader._active_generation_count = 0
    ModelLoader._model_path = None
    ModelLoader._active_adapter = None
    ModelLoader._adapter_offloaded = False
    ModelLoader._draft_instance = None
    ModelLoader._draft_model_path = None


def _run_loader_with_patches(fake_llama):
    """Run ModelLoader.get_instance with every external dependency patched (hardware profile, config, subprocess) and Llama replaced by the supplied fake; returns after the load attempt."""
    from app.engine.model_loader import ModelLoader

    # ModelLoader keeps its state in class attributes, so a test that mutates them would leak into every later test; the originals are saved here and put back in the finally block.
    orig = {
        "instance": ModelLoader._instance,
        "locked": ModelLoader._instance_locked,
        "count": ModelLoader._active_generation_count,
        "model_path": ModelLoader._model_path,
        "adapter": ModelLoader._active_adapter,
        "offloaded": ModelLoader._adapter_offloaded,
        "draft_instance": ModelLoader._draft_instance,
        "draft_model_path": ModelLoader._draft_model_path,
    }

    try:
        _reset_model_loader_state(ModelLoader)
        # Replacing Llama (not the whole loader) keeps the real kwargs-building code under test: the assertions inspect exactly what the loader would hand to llama-cpp.
        # 8 is ggml's enum value for the Q8_0 tensor type; patching it makes the expected type_k/type_v value independent of whether llama_cpp is importable on the test machine.
        # Any subprocess the loader might spawn (probes, preflight) is stubbed to succeed, which keeps the test hermetic and fast.
        with patch("app.engine.model_loader.Llama", side_effect=fake_llama), \
             patch.object(
                 ModelLoader,
                 "_resolve_model_path",
                 return_value="data/models/deepseek-r1-1.5b.gguf",
             ), \
             patch.object(ModelLoader, "preflight_model_load", return_value=None), \
             patch.object(ModelLoader, "_bench_vram_bandwidth", return_value=None), \
             patch.object(ModelLoader, "_ggml_type_q8_0", return_value=8), \
             patch("app.engine.model_loader.get_hardware_profile", return_value={"gpu_list": []}), \
             patch("app.engine.config_store.get_ui_config", return_value={"quantized_kv_cache": True}), \
             patch("app.engine.config_store.read_json", return_value={}), \
             patch("subprocess.run", return_value=MagicMock(returncode=0)):
            try:
                ModelLoader.get_instance(model_path="data/models/deepseek-r1-1.5b.gguf")
            # The fakes raise RuntimeError to abort the load right after Llama() is called, since nothing after that point matters here.
            except RuntimeError:
                pass
    finally:
        ModelLoader._instance = orig["instance"]
        ModelLoader._instance_locked = orig["locked"]
        ModelLoader._active_generation_count = orig["count"]
        ModelLoader._model_path = orig["model_path"]
        ModelLoader._active_adapter = orig["adapter"]
        ModelLoader._adapter_offloaded = orig["offloaded"]
        ModelLoader._draft_instance = orig["draft_instance"]
        ModelLoader._draft_model_path = orig["draft_model_path"]


def test_quantized_kv_cache_passes_q8_type_overrides_to_llama():
    """With quantized_kv_cache enabled the Llama constructor receives type_k=8 and type_v=8 (q8_0 KV cache) to save VRAM."""
    captured_kwargs = []

    def fake_llama(**kwargs):
        """Record the constructor kwargs, then raise so the load stops right after the Llama() call."""
        captured_kwargs.append(kwargs)
        # Aborting here avoids needing a fully working fake model for the post-load steps (warm-up, locking).
        raise RuntimeError("stop after constructor capture")

    _run_loader_with_patches(fake_llama)

    assert captured_kwargs, "Llama() was never called"
    assert captured_kwargs[0]["type_k"] == 8
    assert captured_kwargs[0]["type_v"] == 8


def test_quantized_kv_cache_falls_back_when_type_overrides_are_unsupported():
    """If the installed llama-cpp rejects the type overrides the loader retries without type_k/type_v instead of failing the load."""
    captured_kwargs = []
    loaded = MagicMock()

    def fake_llama(**kwargs):
        """Record kwargs; raise TypeError when type_k/type_v are passed (simulating an older llama-cpp) and return the mock model otherwise."""
        captured_kwargs.append(dict(kwargs))
        if "type_k" in kwargs or "type_v" in kwargs:
            # Older llama-cpp-python builds do not know the KV-cache type parameters and fail with exactly this TypeError; the loader must treat it as 'feature unsupported' and retry.
            raise TypeError("unexpected keyword argument 'type_k'")
        return loaded

    _run_loader_with_patches(fake_llama)

    assert len(captured_kwargs) >= 2
    assert captured_kwargs[0]["type_k"] == 8
    assert captured_kwargs[0]["type_v"] == 8
    # The retry must drop BOTH overrides; keeping one would fail again for the same reason.
    assert "type_k" not in captured_kwargs[1]
    assert "type_v" not in captured_kwargs[1]
