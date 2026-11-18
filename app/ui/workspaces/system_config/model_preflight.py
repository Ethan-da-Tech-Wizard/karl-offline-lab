# Program: karl-offline-lab - model_preflight.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Mixin for the System Config workspace that checks the active GGUF model file, RAM fit and adapter compatibility and shows an HTML health report.
"""Model preflight diagnostics mixin for the System Config workspace."""
from __future__ import annotations

import logging
import os

from PyQt6.QtCore import Qt


# ----- Module setup -----
logger = logging.getLogger("karl.system_config")

# ----- Preflight mixin -----
class ModelPreflightMixin:
    """Adds model health-check behaviour to the System Config workspace widget."""
    def _run_model_preflight_checks(self):
        """Validate the active model and render a diagnostic report plus quantization info in the status labels."""
        from core.hardware_scout import get_hardware_profile
        active_name = self._get_active_model_name()

        # Nothing is loaded, so there is nothing to check; show a neutral message and stop.
        if active_name == "none":
            self._model_status.setText("<div style='color: #888;'>No active model loaded.</div>")
            self._model_status.setTextFormat(Qt.TextFormat.RichText)
            return

        # Informational lines (report) and problems (warnings) are collected separately.
        report = []
        warnings = []
        
        # 1. File existence & GGUF signature check
        # Bare names live in the models folder; absolute paths are used as given.
        model_path = os.path.join("data", "models", active_name)
        if os.path.isabs(active_name):
            model_path = active_name
            active_name = os.path.basename(active_name)
            
        if not os.path.exists(model_path):
            warnings.append(f"Model file not found at: {model_path}")
        else:
            try:
                # Read the first 4 bytes: every valid GGUF file starts with the ASCII magic GGUF.
                with open(model_path, "rb") as f:
                    header = f.read(4)
                    if header != b"GGUF":
                        warnings.append("Invalid file format: File header does not match GGUF specification.")
            except Exception as e:
                warnings.append(f"Could not verify model file header: {e}")

        # 2. RAM requirement preflight check
        # Ask the hardware scout how much system RAM is available.
        profile = get_hardware_profile()
        sys_ram = profile.get("ram_gb", 0.0)
        
        reg_item = None
        # Look for this model in the registry to learn its minimum RAM requirement.
        for item in self._registry:
            if item.get("filename") == active_name:
                reg_item = item
                break
                
        if reg_item:
            min_ram = reg_item.get("min_ram_gb", 0.0)
            if sys_ram > 0 and sys_ram < min_ram:
                warnings.append(
                    f"RAM Limit Warning: Model requires at least {min_ram} GB RAM, but system has only {sys_ram:.1f} GB."
                )
        else:
            if os.path.exists(model_path):
                try:
                    size_gb = os.path.getsize(model_path) / (1024**3)
                    # Not in the registry: estimate RAM need as 1.5x the file size (a rough heuristic).
                    est_ram = size_gb * 1.5
                    if sys_ram > 0 and sys_ram < est_ram:
                        warnings.append(
                            f"RAM Limit Warning: Estimated RAM needed is {est_ram:.1f} GB, but system has only {sys_ram:.1f} GB."
                        )
                except Exception:
                    pass

        # 3. Adapter compatibility warning
        # Find the active LoRA adapter, falling back to the saved active-model config.
        active_adapter = self.state.adapter_name or "none"
        if active_adapter == "none":
            from app.engine import config_store
            data = config_store.read_json(config_store.ACTIVE_MODEL_PATH, default=None)
            if isinstance(data, dict):
                active_adapter = data.get("adapter") or "none"

        if active_adapter and active_adapter != "none":
            model_lower = active_name.lower()
            adapter_lower = active_adapter.lower()
            
            # Guess the model family from its file name.
            m_arch = None
            if "qwen" in model_lower:
                m_arch = "qwen"
            elif "llama" in model_lower:
                m_arch = "llama"
            elif "mistral" in model_lower:
                m_arch = "mistral"
                
            # Guess the adapter family the same way; a family mismatch is flagged below.
            a_arch = None
            if "qwen" in adapter_lower:
                a_arch = "qwen"
            elif "llama" in adapter_lower:
                a_arch = "llama"
            elif "mistral" in adapter_lower:
                a_arch = "mistral"

            if m_arch and a_arch and m_arch != a_arch:
                warnings.append(
                    f"Adapter Compatibility Mismatch: Active adapter '{active_adapter}' appears to be for "
                    f"{a_arch.upper()} architecture, but active base model '{active_name}' is {m_arch.upper()}."
                )
            elif active_adapter != "none":
                report.append(f"Active Adapter: <b>{active_adapter}</b> (loaded)")

        # Green when clean, yellow when any warning exists; then build the HTML report.
        status_color = "#2DD4A0" if not warnings else "#FFD800"
        status_text = "HEALTHY" if not warnings else "WARNINGS DETECTED"
        
        html_report = [
            f"<div style='margin-top: 10px; padding: 10px; border-radius: 4px; border: 1px solid {status_color}33; background: rgba(30,30,50,0.2);'>",
            f"<div style='font-weight: bold; color: {status_color}; margin-bottom: 6px;'>MODEL DIAGNOSTIC REPORT: {status_text}</div>",
            f"<div>Active Model: <b>{active_name}</b></div>"
        ]
        
        if report:
            for r in report:
                html_report.append(f"<div>{r}</div>")
                
        if warnings:
            html_report.append("<div style='margin-top: 6px; font-weight: bold; color: #FF3366;'>Warnings:</div>")
            for w in warnings:
                html_report.append(f"<div style='color: #FFB0B0; font-size: 8.5pt;'>&bull; {w}</div>")
        else:
            html_report.append("<div style='color: #2DD4A0; font-size: 8.5pt;'>&bull; No resource conflicts or GGUF signature warnings. Ready.</div>")
            
        html_report.append("</div>")
        
        # Render the finished report as rich text.
        self._model_status.setText("".join(html_report))
        self._model_status.setTextFormat(Qt.TextFormat.RichText)

        # Show quantization tradeoff info
        from app.engine.model_loader import ModelLoader
        from core.hardware_scout import get_hardware_profile
        # Show the quantization level, its quality rating and VRAM need versus free VRAM.
        quant = ModelLoader.get_quantization()
        vram_needed = ModelLoader.vram_estimate_gb()
        if quant and hasattr(self, '_quant_info_lbl'):
            hw = get_hardware_profile()
            vram_free = hw.get("vram_gb", 0.0)
            vram_str = f"{vram_needed:.1f} GB required, {vram_free:.1f} GB free" if vram_needed else ""
            quant_quality = {"Q4_K_M": "Good", "Q5_K_M": "Better", "Q8_0": "Best", "Q6_K": "Very Good"}.get(quant, quant)
            color = "#2DD4A0" if (not vram_needed or vram_free >= vram_needed) else "#FF5C7A"
            self._quant_info_lbl.setText(
                f"<span style='color:{color};'>{quant}</span> — Quality: {quant_quality} "
                + (f"| VRAM: {vram_str}" if vram_str else "")
            )
            self._quant_info_lbl.setTextFormat(Qt.TextFormat.RichText)
            self._quant_info_lbl.setVisible(True)

