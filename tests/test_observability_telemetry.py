# ==============================================================================
# Program: test_observability_telemetry.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Comprehensive test suite exercising observability telemetry functionality.
#          Verifies state transitions, error boundaries, regression resistance,
#          and contract compliance under tests.
# ==============================================================================

import os
import sys
import tempfile
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tests.qt_test_helper  # noqa: F401  # Ensures global QApplication runs headlessly
from app.ui.workspaces.system_config.observability_tab import ObservabilityTab

class MockState:
    """Empty stand-in for AppState; the tab only needs an object to hold."""
    pass

def test_telemetry_metrics_calculation():
    """The Observability tab averages a dummy performance-telemetry file into its metric pills: TTFT 400.0 ms, tokens/s 25.00, cache hit rate 40.0% (hits over total) and VRAM delta +15.0 MB."""
    state = MockState()
    
    # 1. Create a dummy performance telemetry file
    # A temp directory (cleaned in the finally block) holds the fake JSONL log so the real data/ telemetry is never read or modified.
    temp_dir = tempfile.mkdtemp()
    log_file = os.path.join(temp_dir, "performance_telemetry.jsonl")
    
    entries = [
        {
            "ts": "2026-06-27_120000",
            "model": "test-model.gguf",
            "prefill_tokens_count": 100,
            "prefill_duration_sec": 0.5,  # 500 ms TTFT
            "tokens_per_second": 20.0,
            # Entry 1: 20 hits on 100 prefill tokens; entry 2: 100 hits on 200. The two entries are chosen so a mean of per-entry ratios (35%) differs from the pooled ratio (40%).
            "kv_cache_hits": 20,
            "vram_usage_mb_delta": 10.0
        },
        {
            "ts": "2026-06-27_120100",
            "model": "test-model.gguf",
            "prefill_tokens_count": 200,
            "prefill_duration_sec": 0.3,  # 300 ms TTFT
            "tokens_per_second": 30.0,
            "kv_cache_hits": 100,
            "vram_usage_mb_delta": 20.0
        }
    ]
    
    with open(log_file, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
            
    import app.ui.workspaces.system_config.observability_tab as ot
    # The tab reads a module-level path constant, so the test swaps it and restores it in finally rather than passing the path in.
    original_path = ot.TELEMETRY_LOG_PATH
    ot.TELEMETRY_LOG_PATH = log_file
    
    tab = ObservabilityTab(state)
    
    try:
        tab.refresh_metrics()
        
        # Verify metric pill values
        # TTFT: (500 + 300) / 2 = 400.0 ms
        assert tab.ttft_pill.val_lbl.text() == "400.0 ms"
        # TPS: (20.0 + 30.0) / 2 = 25.00 t/s
        assert tab.tps_pill.val_lbl.text() == "25.00 t/s"
        # Cache hits: (20 + 100) / (100 + 200) = 120 / 300 = 40.0%
        # The hit rate is pooled (total hits / total prefill tokens), which weights larger prompts more; averaging ratios would show 35.0% instead.
        assert tab.cache_pill.val_lbl.text() == "40.0%"
        # VRAM Delta: (10.0 + 20.0) / 2 = +15.0 MB
        # VRAM delta is shown signed (+/-) because a negative value means memory was released.
        assert tab.vram_pill.val_lbl.text() == "+15.0 MB"
    finally:
        ot.TELEMETRY_LOG_PATH = original_path
        import shutil
        shutil.rmtree(temp_dir)
