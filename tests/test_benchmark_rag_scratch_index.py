"""Regression: eval/benchmark_rag.py must never reset the persisted data/vector_db index.

The benchmark calls rag.index.reset() before ingesting synthetic data, so the pipeline it
builds has to point at a throwaway directory instead of RAGPipeline's default index path.
"""
import importlib.util
import os

_BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "benchmark_rag.py")


def _load_benchmark():
    """Import eval/benchmark_rag.py by path (eval/ is a script folder, not a package)."""
    spec = importlib.util.spec_from_file_location("benchmark_rag_under_test", _BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scratch_pipeline_uses_temp_index_dir(monkeypatch):
    """The pipeline gets an explicit temp index_path, not the default data/vector_db."""
    mod = _load_benchmark()
    seen = {}

    class _FakePipeline:
        """Records constructor arguments so the test can inspect the chosen index path."""

        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(mod, "RAGPipeline", _FakePipeline)
    mod._scratch_pipeline(True)
    assert seen["contextual_headers"] is True
    assert "index_path" in seen, "benchmark must not rely on RAGPipeline's persisted default"
    assert os.path.isdir(seen["index_path"])
    assert os.path.abspath(seen["index_path"]) != os.path.abspath("data/vector_db")
