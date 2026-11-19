"""Regression: VRAM used is derived from hardware_scout's free/total keys."""
import ast
import pathlib


def _helper():
    src = pathlib.Path("eval/run_eval.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "gpu_used_mb")
    ns = {}
    exec(compile(ast.Module([fn], []), "run_eval", "exec"), ns)
    return ns["gpu_used_mb"]


def test_gpu_used_mb():
    assert _helper()({"id": 0, "memory_free_mb": 1000, "memory_total_mb": 8000}) == 7000
