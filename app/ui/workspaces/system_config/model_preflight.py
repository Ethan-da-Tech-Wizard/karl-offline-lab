# ==============================================================================
# Program: model_preflight.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Desktop workspace implementation for Model Preflight.
#          Provides responsive layout controls, parameter adjustment surfaces,
#          and visual feedback components within Karl's PyQt6 shell.
# ==============================================================================
# Preflight diagnostics for the active model: checks the file, estimates RAM
# needs, flags adapter/base architecture mismatches, and renders a status card
# plus quantization/VRAM info on the Model tab. Advisory only; it never blocks a load.

from __future__ import annotations

import logging
import os

from PyQt6.QtCore import Qt


logger = logging.getLogger("karl.system_config")

# Advisory only: nothing here can stop a load. The report is a pure function of
# disk and hardware state plus self.state / self._registry (populated by the
# registry panel), re-rendered after every load, adapter change and filesystem
# scan so the warning card never describes a model that is no longer active.
class ModelPreflightMixin:
    """Mixin providing the model health report shown under the Model tab."""
    def _run_model_preflight_checks(self):
        """Run all preflight checks and render the HTML report into the model status label."""
        # Imported lazily to avoid pulling hardware probing in at module import time.
        from core.hardware_scout import get_hardware_profile
        # May be a bare registry filename, an absolute path, or the sentinel "none"
        # (no model configured); the three cases are handled separately below.
        active_name = self._get_active_model_name()

        if active_name == "none":
            self._model_status.setText("<div style='color: #888;'>No active model loaded.</div>")
            self._model_status.setTextFormat(Qt.TextFormat.RichText)
            return

        # Two lists on purpose: `report` holds neutral facts (shown in plain text),
        # `warnings` holds problems (shown red and flip the card from green to amber).
        report = []
        warnings = []
        
        # Check 1: the file exists and starts with the 4-byte GGUF magic number.
        # 1. File existence & GGUF signature check
        # Registry filenames are relative to data/models; absolute paths are used as-is.
        # Absolute paths are accepted for models kept outside data/models; active_name
        # is then reduced to the basename so registry lookup and display still work.
        model_path = os.path.join("data", "models", active_name)
        if os.path.isabs(active_name):
            model_path = active_name
            active_name = os.path.basename(active_name)
            
        if not os.path.exists(model_path):
            warnings.append(f"Model file not found at: {model_path}")
        else:
            try:
                with open(model_path, "rb") as f:
                    # Only 4 bytes are read: GGUF files start with the ASCII magic "GGUF", so a
                    # truncated download, a .safetensors renamed to .gguf or an HTML error page
                    # saved as a model are all caught without loading gigabytes.
                    header = f.read(4)
                    if header != b"GGUF":
                        warnings.append("Invalid file format: File header does not match GGUF specification.")
            except Exception as e:
                warnings.append(f"Could not verify model file header: {e}")

        # Check 2: RAM. Prefer the registry's min_ram_gb; for unregistered models estimate
        # 1.5x the file size. A sys_ram of 0 means detection failed, so no warning is raised.
        # 2. RAM requirement preflight check
        # sys_ram is AVAILABLE RAM in GB (psutil 'available', which counts reclaimable
        # cache) as reported by the scout, so the warning reflects what a load could
        # really get right now and can differ between two runs on the same machine.
        profile = get_hardware_profile()
        sys_ram = profile.get("ram_gb", 0.0)
        
        # Linear scan is fine: the registry has tens of entries at most.
        reg_item = None
        for item in self._registry:
            if item.get("filename") == active_name:
                reg_item = item
                break
                
        if reg_item:
            min_ram = reg_item.get("min_ram_gb", 0.0)
            # Registry min_ram_gb is the curated figure and wins over the file-size guess,
            # since it accounts for context-window memory.
            if sys_ram > 0 and sys_ram < min_ram:
                warnings.append(
                    f"RAM Limit Warning: Model requires at least {min_ram} GB RAM, but system has only {sys_ram:.1f} GB."
                )
        else:
            if os.path.exists(model_path):
                try:
                    size_gb = os.path.getsize(model_path) / (1024**3)
                    # 1.5x file size: weights are memory-mapped at roughly their file size, and the
                    # extra 50% covers the KV cache and runtime buffers for a typical context. A
                    # rule of thumb, intentionally on the conservative side for unregistered models.
                    est_ram = size_gb * 1.5
                    if sys_ram > 0 and sys_ram < est_ram:
                        warnings.append(
                            f"RAM Limit Warning: Estimated RAM needed is {est_ram:.1f} GB, but system has only {sys_ram:.1f} GB."
                        )
                except Exception:
                    pass

        # Check 3: adapter compatibility. Architecture is guessed from the file names only
        # (qwen / llama / mistral), so this is a heuristic and can miss mismatches.
        # 3. Adapter compatibility warning
        # If state has no adapter, fall back to the one persisted in active_model.json.
        # "none" (a string) is the no-adapter sentinel used across the System tab
        # combo boxes, so state.adapter_name None is normalised to it here.
        active_adapter = self.state.adapter_name or "none"
        if active_adapter == "none":
            from app.engine import config_store
            data = config_store.read_json(config_store.ACTIVE_MODEL_PATH, default=None)
            if isinstance(data, dict):
                active_adapter = data.get("adapter") or "none"

        if active_adapter and active_adapter != "none":
            # Matching is on substrings of the file names (qwen/llama/mistral), the only
            # signal available without opening the GGUF metadata. A name containing none of
            # these yields no architecture, and unknown pairs are never flagged, avoiding
            # false alarms on custom-named files.
            model_lower = active_name.lower()
            adapter_lower = active_adapter.lower()
            
            m_arch = None
            if "qwen" in model_lower:
                m_arch = "qwen"
            elif "llama" in model_lower:
                m_arch = "llama"
            elif "mistral" in model_lower:
                m_arch = "mistral"
                
            a_arch = None
            if "qwen" in adapter_lower:
                a_arch = "qwen"
            elif "llama" in adapter_lower:
                a_arch = "llama"
            elif "mistral" in adapter_lower:
                a_arch = "mistral"

            # Both sides must be recognised before a mismatch is claimed; this keeps the
            # warning high-precision at the cost of missing renamed models.
            if m_arch and a_arch and m_arch != a_arch:
                warnings.append(
                    f"Adapter Compatibility Mismatch: Active adapter '{active_adapter}' appears to be for "
                    f"{a_arch.upper()} architecture, but active base model '{active_name}' is {m_arch.upper()}."
                )
            elif active_adapter != "none":
                report.append(f"Active Adapter: <b>{active_adapter}</b> (loaded)")

        # Yellow if any warning, green otherwise; the border colour reuses it with alpha 0x33.
        # Appending "33" to a #RRGGBB colour makes #RRGGBBAA with alpha 0x33 (about
        # 20%), a faint outline that tints the card without competing with the text.
        status_color = "#2DD4A0" if not warnings else "#FFD800"
        status_text = "HEALTHY" if not warnings else "WARNINGS DETECTED"
        
        # Report is assembled as HTML strings because _model_status is a QLabel set to
        # RichText; every dynamic value comes from local file names or registry data,
        # not from remote input.
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
        
        # The same label shows load progress/errors from _start_model_load; the report
        # replaces that text after each load, so load results appear only briefly.
        self._model_status.setText("".join(html_report))
        self._model_status.setTextFormat(Qt.TextFormat.RichText)

        # Quantization tradeoff line: quality label plus required versus free VRAM.
        # Show quantization tradeoff info
        from app.engine.model_loader import ModelLoader
        from core.hardware_scout import get_hardware_profile
        # Quantization and VRAM estimate come from the loaded model instance, so this
        # section is blank until a model has actually been loaded.
        quant = ModelLoader.get_quantization()
        vram_needed = ModelLoader.vram_estimate_gb()
        if quant and hasattr(self, '_quant_info_lbl'):
            hw = get_hardware_profile()
            # vram_gb is the FREE VRAM of the best GPU, and 0.0 on CPU-only machines, in
            # which case the colour below turns red whenever an estimate exists; that is
            # the honest outcome for a GPU-offloaded load with no GPU.
            vram_free = hw.get("vram_gb", 0.0)
            vram_str = f"{vram_needed:.1f} GB required, {vram_free:.1f} GB free" if vram_needed else ""
            # Quality labels are rough rules of thumb; unknown formats are shown verbatim.
            quant_quality = {"Q4_K_M": "Good", "Q5_K_M": "Better", "Q8_0": "Best", "Q6_K": "Very Good"}.get(quant, quant)
            # Red when the estimate exceeds free VRAM.
            color = "#2DD4A0" if (not vram_needed or vram_free >= vram_needed) else "#FF5C7A"
            self._quant_info_lbl.setText(
                f"<span style='color:{color};'>{quant}</span> — Quality: {quant_quality} "
                + (f"| VRAM: {vram_str}" if vram_str else "")
            )
            self._quant_info_lbl.setTextFormat(Qt.TextFormat.RichText)
            self._quant_info_lbl.setVisible(True)

