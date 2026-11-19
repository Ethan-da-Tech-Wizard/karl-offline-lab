"""Regression: unimplemented TS/Rust runners must not report success."""
import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location("mls", pathlib.Path("data/flywheel/multilang_sandbox.py"))
mls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mls)


def test_stubs_fail_closed():
    sb = mls.SafeMultiLangSandbox()
    for lang in ("typescript", "rust"):
        ok, log = sb.run_code(lang, "x", "y")
        assert ok is False and "NOT IMPLEMENTED" in log
