"""Regression: VRAM used is derived from hardware_scout's free/total keys."""
# Scope: eval/run_eval.py reports GPU memory used during an evaluation. core/hardware_scout.py supplies "memory_free_mb" and
# "memory_total_mb" per GPU, so "used" is derived as total minus free. The helper gpu_used_mb is extracted with ast and exec'd
# because importing run_eval would pull in the model stack. Run: python -m pytest tests/test_run_eval_vram.py -q
import ast
import pathlib


def _helper():
    src = pathlib.Path("eval/run_eval.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "gpu_used_mb")
    ns = {}
    exec(compile(ast.Module([fn], []), "run_eval", "exec"), ns)
    return ns["gpu_used_mb"]


# 8000 MB total with 1000 MB free means 7000 MB used; the extra "id" key checks the helper ignores fields it does not need.
def test_gpu_used_mb():
    assert _helper()({"id": 0, "memory_free_mb": 1000, "memory_total_mb": 8000}) == 7000
