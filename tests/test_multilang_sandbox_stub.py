"""Regression: unimplemented TS/Rust runners must not report success."""
# Scope: data/flywheel/multilang_sandbox.py is the flywheel's code-execution sandbox. TypeScript and Rust runners are placeholders;
# the safety rule tested here is "fail closed": a stub must return failure with a NOT IMPLEMENTED message, never a fake success
# that would let unverified generated code into the training data. Run: python -m pytest tests/test_multilang_sandbox_stub.py -q
import importlib.util
import pathlib

# Loaded by file path because data/flywheel is a script folder, not an importable package.
spec = importlib.util.spec_from_file_location("mls", pathlib.Path("data/flywheel/multilang_sandbox.py"))
mls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mls)


# run_code(language, code, tests) returns (ok, log); for both unimplemented languages ok must be False and the log must say why.
def test_stubs_fail_closed():
    sb = mls.SafeMultiLangSandbox()
    for lang in ("typescript", "rust"):
        ok, log = sb.run_code(lang, "x", "y")
        assert ok is False and "NOT IMPLEMENTED" in log
