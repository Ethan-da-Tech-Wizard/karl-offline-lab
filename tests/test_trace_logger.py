# ==============================================================================
# Program: test_trace_logger.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Comprehensive test suite exercising trace logger functionality.
#          Verifies state transitions, error boundaries, regression resistance,
#          and contract compliance under tests.
# ==============================================================================

# Standard library only for the fixtures: tempfile/shutil give every test a disposable log
# directory, gzip/gc/sys.getrefcount support the key-zeroing checks, json parses the JSONL records.
import gc
import gzip
import os
import sys
import json
import shutil
import tempfile
# patch swaps config lookups and the archive step so tests never touch real data/ or encrypt for real.
from unittest.mock import patch

import pytest

# The repo root is put on sys.path so `import app...` works when the file is run directly with
# `python tests/test_trace_logger.py` as well as under pytest (the __main__ block at the bottom).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The module is imported as `tl` as well so rotation tests can shrink its _MAX_BYTES constant.
import app.utils.trace_logger as tl
from app.utils.trace_logger import TraceLogger

# Contract: every generation becomes one JSONL line whose keys are the Unsloth SFT/DPO source
# schema documented in AGENTS.md. Downstream curation breaks if a key is renamed or dropped, so the
# key set is compared for exact equality (not subset).
def test_schema_fields():
    """A logged generation has exactly the documented schema (model, adapter, workflow, template, hyperparams, timing breakdown with tokens/second figures, thinking, response, rag chunks, feedback), and update_feedback changes feedback to 'thumbs_up'."""
    print("Testing TraceLogger schema fields...")
    # mkdtemp + try/finally rmtree instead of the tmp_path fixture keeps the file runnable without pytest.
    temp_dir = tempfile.mkdtemp()
    try:
        # log_dir redirects output away from data/logs/traces.
        logger = TraceLogger(log_dir=temp_dir)
        # log_generation returns the path it wrote, so the test can read the file back.
        log_file = logger.log_generation(
            compiled_prompt="System rules\nUser message",
            hyperparams={"temperature": 0.7, "top_p": 0.9},
            raw_output="Thought response",
            parsed_thought="Thought",
            parsed_response="response",
            execution_time=2.5,
            rag_context=["chunk1"],
            workflow="code_review",
            template="json_extractor",
            feedback="none",
            model_name="qwen-1.5b",
            adapter_name="lora-1",
            # diagnostics is the llama-cpp timing breakdown; each key must land under record["timing"] with
            # the renamed fields (prefill_time -> prefill_seconds, etc.).
            diagnostics={
                "prompt_tokens": 10,
                "prefill_time": 0.5,
                "prefill_tps": 20.0,
                "generation_tokens": 15,
                "generation_time": 2.0,
                "generation_tps": 7.5,
                "total_time": 2.5,
                "total_tps": 10.0,
            }
        )
        
        # Read the logged record: one entry == one line, so readline() is the whole entry.
        # Read the logged record
        assert os.path.exists(log_file)
        with open(log_file, "r", encoding="utf-8") as f:
            line = f.readline().strip()
        record = json.loads(line)
        
        # Verify schema keys. Includes the thermal fields (gpu_temp_c...) that are always present, even when null.
        # Verify schema keys
        expected_keys = {
            "id", "session_id", "timestamp", "timing", "model", "adapter",
            "workflow", "template", "hyperparams", "system_prompt",
            "compiled_prompt", "thinking", "response", "raw_output",
            "rag_chunks", "feedback", "corrected_response",
            "gpu_temp_c", "throttle_reasons", "cooling_duration_sec"
        }
        assert set(record.keys()) == expected_keys, f"Missing or extra keys: {record.keys()}"
        
        # Assert values: spot-checks that each input argument is mapped to the right output field.
        # Assert values
        assert record["model"] == "qwen-1.5b"
        assert record["adapter"] == "lora-1"
        assert record["workflow"] == "code_review"
        assert record["template"] == "json_extractor"
        assert record["hyperparams"] == {"temperature": 0.7, "top_p": 0.9}
        # Timing numbers pass through unchanged (no rounding or unit conversion).
        assert record["timing"]["total_seconds"] == 2.5
        assert record["timing"]["prefill_seconds"] == 0.5
        assert record["timing"]["prefill_tps"] == 20.0
        assert record["timing"]["generation_seconds"] == 2.0
        assert record["timing"]["generation_tps"] == 7.5
        assert record["timing"]["prompt_tokens"] == 10
        assert record["timing"]["generation_tokens"] == 15
        assert record["timing"]["total_tps"] == 10.0
        assert record["thinking"] == "Thought"
        assert record["response"] == "response"
        assert record["rag_chunks"] == ["chunk1"]
        assert record["feedback"] == "none"
        
        # Test updating feedback: rewrites the last JSONL line in place; this is how thumbs-up in the
        # Workbench reaches the training curator after the trace was already written.
        # Test updating feedback
        logger.update_last_entry_feedback(feedback="thumbs_up")
        with open(log_file, "r", encoding="utf-8") as f:
            line = f.readline().strip()
        record = json.loads(line)
        assert record["feedback"] == "thumbs_up"
        
        print("TraceLogger schema fields OK.")
    finally:
        shutil.rmtree(temp_dir)

# Rotation guarantees one file never grows without bound; the old file is archived (compressed +
# encrypted) and a fresh one started. Tested with a tiny limit so a single entry trips it.
def test_file_rotation():
    """When the log file exceeds the (shrunk) size limit the next entry goes to a new rotated file and both files keep their records."""
    print("Testing TraceLogger rotation logic...")
    temp_dir = tempfile.mkdtemp()
    
    # Save original max bytes so the finally block can restore the module global; leaking 50 would
    # make every later test rotate.
    # Save original max bytes
    old_max = tl._MAX_BYTES
    # Set max bytes very small so any entry triggers rotation.
    # Set max bytes very small so any entry triggers rotation
    tl._MAX_BYTES = 50 
    
    try:
        logger = TraceLogger(log_dir=temp_dir)
        
        # _archive_log is stubbed: encryption needs the cryptography package and a hardware id, which are
        # irrelevant to the question "does the next write go to a new file?".
        with patch.object(logger, "_archive_log", return_value=None):
            # Write one entry. This file will exceed 50 bytes.
            # Write one entry. This file will exceed 50 bytes.
            file1 = logger.log_generation("prompt 1", {}, "raw 1", "thought 1", "resp 1", 1.0)
            
            # Next entry should be written to rotated file, so its path must differ.
            # Next entry should be written to rotated file
            file2 = logger.log_generation("prompt 2", {}, "raw 2", "thought 2", "resp 2", 1.0)
            
        assert file1 != file2, f"Expected rotation, but both wrote to {file1}"
        assert os.path.exists(file1)
        assert os.path.exists(file2)
        
        # Verify contents of both: rotation must not lose or mix records.
        # Verify contents of both
        with open(file1, "r") as f:
            r1 = json.loads(f.readline().strip())
        with open(file2, "r") as f:
            r2 = json.loads(f.readline().strip())
            
        assert r1["thinking"] == "thought 1"
        assert r2["thinking"] == "thought 2"
        
        print("TraceLogger rotation OK.")
    finally:
        tl._MAX_BYTES = old_max
        shutil.rmtree(temp_dir)

# Retention policy: log_retention_days bounds how long traces stay on disk (privacy + disk quota).
# One ancient and one fresh file prove pruning is age-based, not "delete everything".
def test_log_pruning():
    """Pruning deletes log files older than log_retention_days and keeps recent ones."""
    import time
    from unittest.mock import patch
    print("Testing TraceLogger pruning logic...")
    temp_dir = tempfile.mkdtemp()
    
    # Retention of 2 days; get_ui_config is patched so the user's real settings are ignored.
    with patch("app.engine.config_store.get_ui_config", return_value={"log_retention_days": 2}):
        logger = TraceLogger(log_dir=temp_dir)
        
        # The year 2020 filename is old by name; the mtime change below makes it old by timestamp too,
        # because pruning goes by mtime, not by the date in the filename.
        old_file = os.path.join(temp_dir, "trace_2020-01-01.jsonl")
        new_file = os.path.join(temp_dir, "trace_2026-06-12.jsonl")
        
        with open(old_file, "w") as f:
            f.write("{}\n")
        with open(new_file, "w") as f:
            f.write("{}\n")
            
        # Backdate the old file by 10 days (> 2-day retention). The new file keeps its fresh mtime.
        now = time.time()
        os.utime(old_file, (now - 10 * 86400, now - 10 * 86400))
        
        logger.prune_logs()
        
        assert not os.path.exists(old_file), "Expected old log to be pruned"
        assert os.path.exists(new_file), "Expected new log to be kept"
        
    shutil.rmtree(temp_dir)
    print("TraceLogger pruning OK.")

# Security property: Fernet key material must not linger in process memory after archival.
# _archive_log builds the key into a mutable bytearray and calls _zero_bytes on it; this test
# intercepts that call to inspect the very same buffer afterwards.
def test_cryptographic_memory_zeroing():
    """
    Verify that _zero_bytes() is called on key material inside _archive_log() and
    that the bytearray is fully overwritten with zeros after the call returns.
    """
    # cryptography is an optional dependency; skip rather than fail where it is not installed.
    pytest.importorskip("cryptography", reason="cryptography package not installed")

    temp_dir = tempfile.mkdtemp()
    archive_dir = tempfile.mkdtemp()
    # The list collects references (not copies) so in-place zeroing is visible after the call.
    captured: list[bytearray] = []

    # Keep the real implementation so behaviour is unchanged; the wrapper only observes.
    original_zero = TraceLogger._zero_bytes

    def intercepting_zero(ba: bytearray) -> None:
        """Wrap the zeroing helper to capture each bytearray before it is zeroed, so the test can verify afterwards that every byte became 0."""
        # Capture the bytearray reference BEFORE the real zeroing runs so we
        # can inspect the same object (in-place mutation) after the call.
        captured.append(ba)
        original_zero(ba)

    try:
        # patch.object with staticmethod because _zero_bytes is a static method on the class.
        with patch.object(TraceLogger, "_zero_bytes", staticmethod(intercepting_zero)):
            tracer = TraceLogger(log_dir=temp_dir, archive_dir=archive_dir)

            # Write a real JSONL entry so _archive_log() has a non-empty file.
            # Write a real JSONL entry so _archive_log() has a non-empty file.
            test_log = os.path.join(temp_dir, "trace_zeroing_test.jsonl")
            with open(test_log, "w", encoding="utf-8") as fh:
                fh.write('{"id":"zero-test","session_id":"s0","timestamp":"2026-06-13T00:00:00+00:00"}\n')

            # Archival runs the full gzip -> encrypt -> write path, including key derivation and zeroing.
            tracer._archive_log(test_log)

        # At least one zeroing must have happened, otherwise the sanitising step was skipped.
        assert len(captured) >= 1, (
            "_zero_bytes was never called — key material was not sanitized after archival"
        )
        # Every captured buffer must be a bytearray (immutable bytes could not be wiped), non-empty, and
        # all zero by the time the call returns.
        for slot, ba in enumerate(captured):
            assert isinstance(ba, bytearray), \
                f"Slot {slot}: expected bytearray, got {type(ba).__name__}"
            assert len(ba) > 0, f"Slot {slot}: captured bytearray is empty"
            non_zero = [i for i, v in enumerate(ba) if v != 0]
            assert len(non_zero) == 0, (
                f"Slot {slot}: bytearray not fully zeroed — "
                f"{len(non_zero)} non-zero byte(s) at indices {non_zero[:10]}"
                f"{'...' if len(non_zero) > 10 else ''}"
            )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(archive_dir, ignore_errors=True)


# Same property for the read side: decrypting an archive must hand back a mutable bytearray the
# caller can wipe, and nothing else (caches, logs) may hold another reference to the plaintext.
def test_decrypted_payload_cleanup():
    """
    Verify that decrypt_to_bytearray() returns a mutable bytearray, that
    _zero_bytes() wipes every byte to zero, and that no external references to
    the sensitive buffer linger after cleanup.
    """
    Fernet = pytest.importorskip(
        "cryptography.fernet", reason="cryptography package not installed"
    ).Fernet

    temp_dir = tempfile.mkdtemp()
    try:
        # ── 1. Build a valid .enc file with a known test key ──────────────────
        # ── 1. Build a valid .enc file with a known test key ──────────────────
        # Fernet = AES-128-CBC + HMAC, the scheme the logger uses for archives. Using a random test key
        # avoids deriving one from the machine's hardware id.
        test_key = Fernet.generate_key()
        plaintext = b'{"id":"ref-test","thinking":"sensitive reasoning"}\n'
        encrypted = Fernet(test_key).encrypt(gzip.compress(plaintext))

        enc_path = os.path.join(temp_dir, "trace_2026-06-13_cleanup.jsonl.enc")
        with open(enc_path, "wb") as fh:
            fh.write(encrypted)

        # ── 2. Decrypt to a mutable bytearray ────────────────────────────────
        # ── 2. Decrypt to a mutable bytearray ────────────────────────────────
        result = TraceLogger.decrypt_to_bytearray(enc_path, test_key)
        assert isinstance(result, bytearray), \
            "decrypt_to_bytearray must return a bytearray"
        assert len(result) > 0, "Decrypted bytearray is empty"
        assert bytes(result) == plaintext, \
            "Decrypted content does not match the original plaintext"

        # ── 3. Verify no unexpected external references before zeroing ────────
        # ── 3. Verify no unexpected external references before zeroing ────────
        # sys.getrefcount() always adds 1 for its own argument.
        # Expected: 2 = local variable 'result' + getrefcount argument.
        gc.collect()
        ref_count = sys.getrefcount(result)
        assert ref_count <= 2, (
            f"Reference leak detected before zeroing: refcount={ref_count}, "
            f"expected ≤ 2 (local var + getrefcount arg)"
        )

        # ── 4. Zero the payload in-place ──────────────────────────────────────
        TraceLogger._zero_bytes(result)

        # ── 5. Every byte must be 0x00 ────────────────────────────────────────
        non_zero = [i for i, b in enumerate(result) if b != 0]
        assert len(non_zero) == 0, (
            f"Payload not fully zeroed: {len(non_zero)} non-zero byte(s) remain "
            f"at indices {non_zero[:10]}{'...' if len(non_zero) > 10 else ''}"
        )

        # ── 6. No referrers outside the current stack frame ──────────────────
        # gc.get_referrers() returns all objects that directly reference 'result'.
        # The only legitimate referrer is the local-variables dict of this frame.
        gc.collect()
        referrers = gc.get_referrers(result)
        external = [
            r for r in referrers
            if not (isinstance(r, dict) and "result" in r)
        ]
        assert len(external) == 0, (
            f"Unexpected referrers to decrypted bytearray after zeroing: {external}"
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# Allows `python tests/test_trace_logger.py` without pytest. Note: the test defined after this
# block (rate limiting) is only collected by pytest; it is not part of this manual runner list.
if __name__ == "__main__":
    test_schema_fields()
    test_file_rotation()
    test_log_pruning()
    test_cryptographic_memory_zeroing()
    test_decrypted_payload_cleanup()
    print("All trace logger unit tests PASSED!")


# Performance regression: the retention sweep walks the whole log directory, and it used to run
# inside every log_generation call, which slowed down each token-stream completion.
def test_log_generation_rate_limits_retention_sweep():
    """log_generation used to run a full retention sweep on every call; it now sweeps at most once per interval."""
    from unittest.mock import patch
    temp_dir = tempfile.mkdtemp()
    try:
        # 30-day retention; the config is patched so no real settings file is read.
        with patch("app.engine.config_store.get_ui_config", return_value={"log_retention_days": 30}):
            logger = TraceLogger(log_dir=temp_dir, archive_dir=os.path.join(temp_dir, "archive"))
            # prune_logs is replaced by a counter so the test measures how often the sweep fires without
            # touching the file system.
            sweeps = []
            logger.prune_logs = lambda: sweeps.append(1)
            # Three quick writes in a row: the interval (RETENTION_SWEEP_INTERVAL_SEC) has not elapsed, so
            # zero sweeps are expected.
            for _ in range(3):
                logger.log_generation("p", {}, "o", "", "r", 0.1)
            assert sweeps == [], "no sweep expected within the interval"
            # Move the "last sweep" timestamp into the past by more than the interval, so the next write is due.
            logger._last_sweep -= logger.RETENTION_SWEEP_INTERVAL_SEC + 1
            logger.log_generation("p", {}, "o", "", "r", 0.1)
            assert sweeps == [1]
    finally:
        shutil.rmtree(temp_dir)
