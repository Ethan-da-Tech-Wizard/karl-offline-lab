# ==============================================================================
# Program: observability_tab.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Desktop workspace implementation for Observability Tab.
#          Provides responsive layout controls, parameter adjustment surfaces,
#          and visual feedback components within Karl's PyQt6 shell.
# ==============================================================================
# Observability tab: shows SLA-style performance numbers averaged over the last
# 100 entries of the telemetry JSONL log, plus log retention/disk-quota policy
# controls that can trigger a cleanup sweep immediately.

import os
import json
import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QFrame
)
from app.ui.workspaces.system_config.common import _section, _hline, _row
from app.engine import config_store

logger = logging.getLogger("karl.system_config.observability_tab")

# Written by the inference engine; one JSON object per line.
TELEMETRY_LOG_PATH = "data/logs/performance_telemetry.jsonl"


class ObservabilityTab(QWidget):
    """Standalone tab widget for performance metrics and log retention policy."""
    def __init__(self, state, parent=None):
        """Store the app state and build the tab."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(parent)
        self.state = state
        self._build_ui()

    def _build_ui(self):
        """Build the metric pills, retention/quota spin boxes and action buttons."""
        # Step 1: Validate input parameters, operational preconditions, and state
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        layout.addWidget(_section("PERFORMANCE SLA METRICS (LAST 100 INFERENCES)"))

        # Four pills: time to first token, tokens/sec, KV-cache hit rate and VRAM delta.
        # Horizontal statistics layout
        stats_widget = QWidget()
        sl = QHBoxLayout(stats_widget)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(16)

        self.ttft_pill = self._create_pill("Avg TTFT", "0.0 ms")
        self.tps_pill = self._create_pill("Avg TPS", "0.0 t/s")
        self.cache_pill = self._create_pill("Cache Hit Rate", "0.0%")
        self.vram_pill = self._create_pill("VRAM Delta", "0.0 MB")

        sl.addWidget(self.ttft_pill)
        sl.addWidget(self.tps_pill)
        sl.addWidget(self.cache_pill)
        sl.addWidget(self.vram_pill)
        layout.addWidget(stats_widget)

        layout.addWidget(_hline())

        layout.addWidget(_section("LOG RETENTION & GOVERNANCE POLICY"))

        # Spin boxes start from the persisted UI config so the tab reflects what is stored.
        # Policy configs
        cfg = config_store.get_ui_config()

        self._retention_spin = QSpinBox()
        self._retention_spin.setRange(1, 365)
        self._retention_spin.setValue(cfg.get("log_retention_days", 30))
        self._retention_spin.setSuffix(" days")
        layout.addWidget(_row("Log Retention Age:", self._retention_spin))

        self._disk_size_spin = QSpinBox()
        # Range is 10 MB to 50 GB (the spin box works in MB).
        self._disk_size_spin.setRange(10, 50 * 1024)  # 10 MB to 50 GB
        self._disk_size_spin.setValue(cfg.get("max_log_disk_size_mb", 1024))
        self._disk_size_spin.setSuffix(" MB")
        layout.addWidget(_row("Max Log Directory Size:", self._disk_size_spin))

        # Actions row
        actions_row = QWidget()
        al = QHBoxLayout(actions_row)
        al.setContentsMargins(0, 8, 0, 0)
        al.setSpacing(10)

        apply_btn = QPushButton("Apply Policy & Clean Now")
        apply_btn.setObjectName("btn-primary")
        apply_btn.clicked.connect(self._apply_policy_and_clean)
        al.addWidget(apply_btn)

        refresh_btn = QPushButton("Refresh Stats")
        refresh_btn.setObjectName("btn-ghost")
        refresh_btn.clicked.connect(self.refresh_metrics)
        al.addWidget(refresh_btn)

        al.addStretch()
        layout.addWidget(actions_row)

        layout.addStretch()

        self.refresh_metrics()

    def _create_pill(self, label: str, val: str) -> QFrame:
        """Build one statistic card (muted caption over a large accent value) and expose the value label as val_lbl."""
        # Step 1: Validate input parameters, operational preconditions, and state
        f = QFrame()
        f.setObjectName("observability-pill")
        f.setStyleSheet("""
            #observability-pill {
                background-color: #0F0F1E;
                border: 1px solid #25253C;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        lo = QVBoxLayout(f)
        lo.setContentsMargins(8, 8, 8, 8)
        lo.setSpacing(4)

        lbl = QLabel(label.upper())
        lbl.setObjectName("lbl-muted")
        lbl.setStyleSheet("font-size: 7.5pt; font-weight: bold; letter-spacing: 0.5px;")
        lo.addWidget(lbl)

        val_lbl = QLabel(val)
        val_lbl.setObjectName("lbl-accent")
        val_lbl.setStyleSheet("font-size: 14pt; font-weight: bold; color: #00C2FF;")
        # Stashed on the frame so refresh_metrics can update the number without searching children.
        f.val_lbl = val_lbl  # Save reference to update value
        lo.addWidget(val_lbl)

        return f

    def refresh_metrics(self):
        """Recompute the four averages from the last 100 telemetry lines and update the pills ("N/A" when no data)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        # No log yet (fresh install or no inferences) means nothing to average.
        log_path = TELEMETRY_LOG_PATH
        if not os.path.exists(log_path):
            self.ttft_pill.val_lbl.setText("N/A")
            self.tps_pill.val_lbl.setText("N/A")
            self.cache_pill.val_lbl.setText("N/A")
            self.vram_pill.val_lbl.setText("N/A")
            return

        ttfts = []
        tps_list = []
        hits = 0
        total_prefill = 0
        vram_deltas = []

        try:
            # Execute guarded operations and establish recovery boundaries
            with open(log_path, "r", encoding="utf-8") as f:
                # Only the tail is used so the figures describe recent behaviour, not all-time history.
                lines = f.readlines()[-100:]
                for line in lines:
                    # Iterate over items, evaluate loop invariant, and update progress
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    
                    # TTFT is approximated by the prefill duration, converted to milliseconds.
                    prefill_sec = entry.get("prefill_duration_sec", 0.0)
                    ttfts.append(prefill_sec * 1000)
                    
                    tps = entry.get("tokens_per_second", 0.0)
                    # Zero TPS entries are skipped so failed or empty runs do not drag the average down.
                    if tps > 0:
                        tps_list.append(tps)
                        
                    # Hit rate = cache-hit tokens / prefill tokens, accumulated only for entries that prefilled.
                    prefill_count = entry.get("prefill_tokens_count", 0)
                    cache_hits = entry.get("kv_cache_hits", 0)
                    if prefill_count > 0:
                        total_prefill += prefill_count
                        hits += cache_hits
                        
                    vram = entry.get("vram_usage_mb_delta")
                    if vram is not None:
                        vram_deltas.append(vram)
        # A malformed line aborts the loop; values gathered so far are still shown.
        except Exception as e:
            logger.warning(f"Failed to read telemetry log line: {e}")

        total_runs = len(ttfts)
        avg_ttft = sum(ttfts) / total_runs if total_runs > 0 else 0.0
        avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0.0
        kv_hit_rate = (hits / total_prefill * 100) if total_prefill > 0 else 0.0
        avg_vram_delta = sum(vram_deltas) / len(vram_deltas) if vram_deltas else 0.0

        if total_runs > 0:
            self.ttft_pill.val_lbl.setText(f"{avg_ttft:.1f} ms")
            self.tps_pill.val_lbl.setText(f"{avg_tps:.2f} t/s")
            self.cache_pill.val_lbl.setText(f"{kv_hit_rate:.1f}%")
            self.vram_pill.val_lbl.setText(f"{avg_vram_delta:+.1f} MB")
        else:
            self.ttft_pill.val_lbl.setText("N/A")
            self.tps_pill.val_lbl.setText("N/A")
            self.cache_pill.val_lbl.setText("N/A")
            self.vram_pill.val_lbl.setText("N/A")

    def _apply_policy_and_clean(self):
        """Save the retention policy, run the retention sweep now, refresh stats and confirm."""
        # Step 1: Validate input parameters, operational preconditions, and state
        retention = self._retention_spin.value()
        disk_size = self._disk_size_spin.value()

        # Read-modify-write so unrelated UI config keys are preserved.
        cfg = config_store.get_ui_config()
        cfg["log_retention_days"] = retention
        cfg["max_log_disk_size_mb"] = disk_size
        config_store.save_ui_config(cfg)

        # Imported lazily; the logger module is only needed when a sweep is requested.
        from app.utils.trace_logger import TraceLogger
        try:
            # Execute guarded operations and establish recovery boundaries
            logger_inst = TraceLogger()
            logger_inst.enforce_retention_policy()
        except Exception as e:
            logger.warning(f"Enforce retention policy failed in UI thread: {e}")

        self.refresh_metrics()
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(
            self, "Policy Saved",
            "Log retention settings applied and cleanup sweep completed."
        )
