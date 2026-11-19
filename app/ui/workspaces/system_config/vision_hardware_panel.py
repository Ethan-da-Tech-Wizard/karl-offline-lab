# ==============================================================================
# Program: vision_hardware_panel.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Desktop workspace implementation for Vision Hardware Panel.
#          Provides responsive layout controls, parameter adjustment surfaces,
#          and visual feedback components within Karl's PyQt6 shell.
# ==============================================================================
# Vision and Hardware tabs of the System workspace. Vision: choose, load and
# reset the local vision model and inspect the registry. Hardware: live CPU /
# RAM / disk meters, the scouted hardware profile, remote-engine offloading
# settings, thermal protection settings and an About blurb.

from __future__ import annotations

import html
import logging

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextBrowser, QLabel, QSpinBox, QMessageBox, QProgressBar,
    QComboBox, QCheckBox, QLineEdit,
)

from app.ui.themes import MONO
from app.vision.vision_model_loader import (
    VisionModelLoader,
    installed_vision_models,
    set_active_vision_model,
)
from .common import _section

logger = logging.getLogger("karl.system_config")

# Mixin, not a widget: it is combined into the System workspace class, which
# owns self.state (AppState) and the 2 s QTimer that drives _update_live_hardware.
# Every widget created here is stored on self under a leading underscore so the
# sibling panel mixins and the timer can reach it; nothing here emits signals of
# its own, all coupling is through those attributes and AppState fields.
class VisionHardwarePanelMixin:
    """Mixin that builds the Vision and Hardware tabs and their refresh/config handlers."""
    def _build_vision_tab(self) -> QWidget:
        """Build the Vision tab: runtime status, model picker with action buttons, and registry viewer."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Two stacked "panel" cards (objectName drives the QSS card look). The runtime
        # card is fixed-height content; the registry card gets stretch=1 below so the
        # registry list absorbs all spare vertical space when the window is resized.
        runtime_panel = QWidget()
        runtime_panel.setObjectName("panel")
        rp = QVBoxLayout(runtime_panel)
        rp.setContentsMargins(12, 12, 12, 12)
        rp.setSpacing(8)
        rp.addWidget(_section("VISION RUNTIME"))

        # OCR works with just Tesseract; describing pixels needs a vision GGUF plus its projector file.
        desc = QLabel(
            "Karl vision stays offline. OCR can run immediately when Tesseract is installed; "
            "pixel-level image description requires a local GGUF vision model plus its matching projector."
        )
        desc.setObjectName("lbl-muted")
        desc.setWordWrap(True)
        rp.addWidget(desc)

        # Filled by _refresh_vision_status from VisionModelLoader.status(); empty until
        # the first refresh at the end of this builder runs.
        self._vision_runtime_status = QLabel("")
        self._vision_runtime_status.setObjectName("lbl-muted")
        self._vision_runtime_status.setWordWrap(True)
        rp.addWidget(self._vision_runtime_status)

        model_row = QWidget()
        mr = QHBoxLayout(model_row)
        mr.setContentsMargins(0, 0, 0, 0)
        mr.setSpacing(8)
        # Item text is human-readable; item data is the registry id, which is what the
        # button handlers read back with currentData(). Stretch 1 lets the combo take
        # the row while the three buttons keep their natural width.
        self._vision_model_combo = QComboBox()
        mr.addWidget(self._vision_model_combo, 1)

        # Three-step lifecycle, deliberately separate buttons: "set active" only
        # records the choice (cheap, persisted), "load now" pays the multi-GB load, and
        # "reset" unloads to free RAM/VRAM. Splitting them lets a user pick a model
        # without committing memory, and lets the chat path load it lazily instead.
        active_btn = QPushButton("set active")
        active_btn.setObjectName("btn-secondary")
        active_btn.clicked.connect(self._set_active_vision_model_from_combo)
        mr.addWidget(active_btn)

        load_btn = QPushButton("load now")
        load_btn.setObjectName("btn-primary")
        load_btn.clicked.connect(self._load_active_vision_model)
        mr.addWidget(load_btn)

        reset_btn = QPushButton("reset")
        reset_btn.setObjectName("btn-ghost")
        reset_btn.clicked.connect(self._reset_vision_runtime)
        mr.addWidget(reset_btn)
        rp.addWidget(model_row)

        layout.addWidget(runtime_panel)

        registry_panel = QWidget()
        registry_panel.setObjectName("panel")
        vp = QVBoxLayout(registry_panel)
        vp.setContentsMargins(12, 12, 12, 12)
        vp.setSpacing(8)
        header = QWidget()
        hl = QHBoxLayout(header)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(_section("LOCAL VISION MODEL REGISTRY"))
        refresh_btn = QPushButton("refresh")
        refresh_btn.setObjectName("btn-ghost")
        refresh_btn.clicked.connect(self._refresh_vision_status)
        hl.addWidget(refresh_btn)
        vp.addWidget(header)

        self._vision_registry_view = QTextBrowser()
        # 260 px keeps roughly two registry entries visible even in the Compact Laptop
        # layout, where the surrounding panels are squeezed.
        self._vision_registry_view.setMinimumHeight(260)
        vp.addWidget(self._vision_registry_view, 1)
        layout.addWidget(registry_panel, 1)

        self._refresh_vision_status()
        return w


    def _refresh_vision_status(self):
        """Refresh the runtime status text, the model combo and the registry HTML view."""
        if not hasattr(self, "_vision_runtime_status"):
            return

        # status() is a cheap in-memory snapshot (no model load), so it is safe to call
        # from button handlers on the GUI thread. backend_available is False when the
        # llama-cpp multimodal bindings cannot be imported; backend_error says why.
        status = VisionModelLoader.status()
        backend = "available" if status["backend_available"] else "blocked"
        loaded = "loaded" if status["loaded"] else "not loaded"
        active = status.get("active_name") or "none"
        lines = [
            f"Backend: {backend}",
            f"Runtime: {loaded}",
            f"Active model: {active}",
        ]
        if status.get("backend_error"):
            lines.append(f"Backend detail: {status['backend_error']}")
        if status.get("last_error"):
            lines.append(f"Last error: {status['last_error']}")
        self._vision_runtime_status.setText("\n".join(lines))

        # Preserve the user's selection across the rebuild, with signals blocked to avoid side effects.
        # Refresh runs on a timer-less button and after every action, so it must
        # rebuild the combo from disk each time (files may have been copied in since).
        # Signals are blocked so clear()/addItem() do not fire currentIndexChanged
        # handlers while the list is momentarily empty.
        current = self._vision_model_combo.currentData()
        self._vision_model_combo.blockSignals(True)
        self._vision_model_combo.clear()
        rows = installed_vision_models()
        for row in rows:
            # A model is usable only when both the model file and its projector are installed.
            # Vision GGUFs are two files: the language model and an mmproj projector that
            # maps image embeddings into its token space. Either one missing means image
            # input cannot work, so only both present counts as ready.
            ready = row.get("model_installed") and row.get("projector_installed")
            suffix = "ready" if ready else "missing files"
            self._vision_model_combo.addItem(f"{row['name']} ({suffix})", row["id"])
        if current:
            idx = self._vision_model_combo.findData(current)
            if idx >= 0:
                self._vision_model_combo.setCurrentIndex(idx)
        self._vision_model_combo.blockSignals(False)

        # Registry entries come from disk and a JSON file, so every string is HTML-escaped.
        # Entries are rendered as inline-styled divs because QTextBrowser supports only
        # a small HTML4/CSS2 subset: no classes, no flexbox, so styling stays per-element.
        html_rows = []
        for row in rows:
            # Paths are shown even when files are absent so the user can see exactly where
            # to drop the downloaded weights.
            model_state = "present" if row.get("model_installed") else "missing"
            projector_state = "present" if row.get("projector_installed") else "missing"
            html_rows.append(
                "<div style='margin-bottom:12px; padding-bottom:10px; border-bottom:1px solid rgba(255,255,255,0.08);'>"
                f"<b style='color:#00C2FF;'>{html.escape(row['name'])}</b><br/>"
                f"<span style='color:#9090A8;'>Family:</span> {html.escape(row['family'])} &middot; "
                f"<span style='color:#9090A8;'>Context:</span> {int(row['n_ctx']):,}<br/>"
                f"<span style='color:#9090A8;'>Model:</span> {html.escape(row['model_path'])} "
                f"<b>{model_state}</b><br/>"
                f"<span style='color:#9090A8;'>Projector:</span> {html.escape(row['projector_path'])} "
                f"<b>{projector_state}</b><br/>"
                f"<span style='color:#9090A8;'>RAM:</span> &ge; {row['min_ram_gb']} GB &middot; "
                f"<span style='color:#9090A8;'>VRAM:</span> &ge; {row['min_vram_gb']} GB<br/>"
                f"<span style='color:#B8F7FF;'>{html.escape(row.get('strengths', ''))}</span><br/>"
                f"<span style='color:#9090A8;'>{html.escape(row.get('notes', ''))}</span>"
                "</div>"
            )
        self._vision_registry_view.setHtml(
            f"<div style='font-family:{MONO}; font-size:9pt; line-height:1.45;'>"
            + "".join(html_rows or ["No registry entries found."])
            + "</div>"
        )


    def _set_active_vision_model_from_combo(self):
        """Persist the selected vision model as active and reset the loaded runtime so it reloads lazily."""
        model_id = self._vision_model_combo.currentData()
        if not model_id:
            return
        # Order matters: persist the choice first, then reset the loader, so a lazy
        # reload triggered by the very next vision request already sees the new id and
        # never reloads the old model.
        set_active_vision_model(model_id)
        VisionModelLoader.reset()
        self._refresh_vision_status()
        QMessageBox.information(self, "Vision Model Selected", f"Active vision model set to {model_id}.")


    def _load_active_vision_model(self):
        """Load the selected vision model now and refresh the status."""
        model_id = self._vision_model_combo.currentData()
        # Synchronous on the GUI thread: a multi-GB GGUF load can freeze the window for
        # seconds. Failures are not raised; they are recorded in last_error, which the
        # refresh below surfaces in the status label.
        VisionModelLoader.load(model_id=model_id)
        self._refresh_vision_status()


    def _reset_vision_runtime(self):
        """Unload the vision runtime and refresh the status."""
        VisionModelLoader.reset()
        self._refresh_vision_status()

    # The meters below are plain QProgressBars driven by _update_live_hardware (a
    # workspace timer, 2 s period); the builder only sets range 0-100 and the format
    # string is rewritten on every tick.
    # ── hardware tab ──────────────────────────────────────────────────────────


    def _build_hardware_tab(self) -> QWidget:
        """Build the Hardware tab: meters, profile view, remote engine, thermal protection and About panels."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Meters: CPU, memory and disk. A QTimer in the workspace updates them every 2 seconds.
        # Hardware Panel
        hw_panel = QWidget()
        hw_panel.setObjectName("panel")
        hwp_layout = QVBoxLayout(hw_panel)
        hwp_layout.setContentsMargins(16, 16, 16, 16)
        hwp_layout.setSpacing(12)

        # Row pattern repeated three times (label + bar): kept explicit rather than
        # looped because each bar is stored under its own attribute for the timer.
        hwp_layout.addWidget(_section("HARDWARE PROFILE & METERS"))

        # CPU meter
        cpu_row = QWidget()
        cpu_l = QHBoxLayout(cpu_row)
        cpu_l.setContentsMargins(0, 0, 0, 0)
        cpu_lbl = QLabel("CPU Load:")
        # Fixed 100 px label column so the three meter bars (and the 180 px label column
        # of the config rows below) start at the same x and read as one aligned grid.
        cpu_lbl.setFixedWidth(100)
        cpu_lbl.setObjectName("lbl-muted")
        self._cpu_progress = QProgressBar()
        self._cpu_progress.setRange(0, 100)
        self._cpu_progress.setValue(0)
        # 14 px: thin HUD-style bar; the percentage text is drawn inside via setFormat.
        self._cpu_progress.setFixedHeight(14)
        cpu_l.addWidget(cpu_lbl)
        cpu_l.addWidget(self._cpu_progress)
        hwp_layout.addWidget(cpu_row)

        # RAM meter
        ram_row = QWidget()
        ram_l = QHBoxLayout(ram_row)
        ram_l.setContentsMargins(0, 0, 0, 0)
        ram_lbl = QLabel("Memory:")
        # Same 100 px label column as the CPU row above.
        ram_lbl.setFixedWidth(100)
        ram_lbl.setObjectName("lbl-muted")
        self._ram_progress = QProgressBar()
        self._ram_progress.setRange(0, 100)
        self._ram_progress.setValue(0)
        self._ram_progress.setFixedHeight(14)
        ram_l.addWidget(ram_lbl)
        ram_l.addWidget(self._ram_progress)
        hwp_layout.addWidget(ram_row)
        
        # The disk bar shows percent used (see _update_live_hardware), labelled with free space.
        # Disk Space meter
        disk_row = QWidget()
        disk_l = QHBoxLayout(disk_row)
        disk_l.setContentsMargins(0, 0, 0, 0)
        disk_lbl = QLabel("Disk Free:")
        # Same 100 px label column as the CPU/RAM rows.
        disk_lbl.setFixedWidth(100)
        disk_lbl.setObjectName("lbl-muted")
        self._disk_progress = QProgressBar()
        self._disk_progress.setRange(0, 100)
        self._disk_progress.setValue(0)
        self._disk_progress.setFixedHeight(14)
        disk_l.addWidget(disk_lbl)
        disk_l.addWidget(self._disk_progress)
        hwp_layout.addWidget(disk_row)

        self._hw_view = QTextBrowser()
        # Fixed height fits exactly the three RAM/VRAM/STORAGE rows from
        # _refresh_hardware without a scrollbar.
        self._hw_view.setFixedHeight(120)
        hwp_layout.addWidget(self._hw_view)

        refresh_btn = QPushButton("refresh profiles")
        # Scouting is separate from the live meters on purpose: meters poll cheap
        # psutil counters every tick, the profile re-probes hardware (VRAM, disks) and
        # is therefore manual.
        refresh_btn.setToolTip("Re-scout RAM, VRAM, and storage specifications")
        refresh_btn.clicked.connect(self._refresh_hardware)
        hwp_layout.addWidget(refresh_btn)

        layout.addWidget(hw_panel)

        # Remote offloading: when enabled prompts go to a remote Karl bridge over JSON-RPC/WSS
        # instead of the local llama-cpp-python model.
        # Offload settings live on this tab (not in Defaults) because they change which
        # machine runs inference, which is a hardware-level choice. Values are mirrored
        # into AppState for the engine and into config_store for the next launch.
        engine_panel = QWidget()
        engine_panel.setObjectName("panel")
        ep_layout = QVBoxLayout(engine_panel)
        ep_layout.setContentsMargins(16, 16, 16, 16)
        ep_layout.setSpacing(10)
        ep_layout.addWidget(_section("REMOTE MODEL OFFLOADING"))

        self._remote_engine_check = QCheckBox("Remote Engine Mode")
        # Both the newer remote_engine_enabled flag and the older engine_mode == "remote" count as enabled.
        self._remote_engine_check.setChecked(
            bool(getattr(self.state, "remote_engine_enabled", False))
            or getattr(self.state, "engine_mode", "local") == "remote"
        )
        # stateChanged (not toggled) also fires on programmatic setChecked, but this
        # connect happens after the initial setChecked so construction does not write
        # config. The handler persists to disk, so it must not run during build.
        self._remote_engine_check.stateChanged.connect(self._on_engine_config_changed)
        ep_layout.addWidget(self._remote_engine_check)

        url_row = QWidget()
        url_lay = QHBoxLayout(url_row)
        url_lay.setContentsMargins(0, 0, 0, 0)
        url_lbl = QLabel("Remote Server URL:")
        url_lbl.setObjectName("lbl-muted")
        # 180 px label column shared by the URL, token and threshold rows so the input
        # fields line up across the two panels.
        url_lbl.setFixedWidth(180)
        url_lay.addWidget(url_lbl)
        self._remote_url_input = QLineEdit()
        # Placeholder doubles as documentation: wss:// (TLS WebSocket) is the expected
        # scheme for the Karl bridge, and 8080 its default port.
        self._remote_url_input.setPlaceholderText("wss://localhost:8080")
        # URL fallback chain: new remote_engine_url, legacy remote_server_url, then localhost default.
        self._remote_url_input.setText(
            getattr(self.state, "remote_engine_url", "")
            or getattr(self.state, "remote_server_url", "")
            or "wss://localhost:8080"
        )
        # editingFinished (focus loss / Enter) instead of textChanged: avoids writing the
        # config file on every keystroke while a URL is half typed.
        self._remote_url_input.editingFinished.connect(self._on_engine_config_changed)
        url_lay.addWidget(self._remote_url_input, 1)
        ep_layout.addWidget(url_row)

        token_row = QWidget()
        token_lay = QHBoxLayout(token_row)
        token_lay.setContentsMargins(0, 0, 0, 0)
        token_lbl = QLabel("Auth Token:")
        token_lbl.setObjectName("lbl-muted")
        token_lbl.setFixedWidth(180)
        token_lay.addWidget(token_lbl)
        self._remote_token_input = QLineEdit()
        # Token is masked on screen; it is persisted via config_store.set_remote_engine_config.
        # Token is a credential: masked on screen, and the only place it is stored is the
        # config store, never logged here.
        self._remote_token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._remote_token_input.setPlaceholderText("bridge token or password")
        self._remote_token_input.setText(
            getattr(self.state, "remote_engine_token", "")
            or getattr(self.state, "remote_auth_token", "")
        )
        self._remote_token_input.editingFinished.connect(self._on_engine_config_changed)
        token_lay.addWidget(self._remote_token_input, 1)
        ep_layout.addWidget(token_row)

        self._remote_status_label = QLabel("")
        self._remote_status_label.setObjectName("lbl-muted")
        self._remote_status_label.setWordWrap(True)
        ep_layout.addWidget(self._remote_status_label)
        self._sync_engine_controls_enabled()

        layout.addWidget(engine_panel)

        # Thermal protection pauses/throttles work above the threshold (handlers in defaults_panel).
        # Thermal Protection Settings Panel
        # Thermal protection: when the sensed temperature exceeds the threshold below,
        # work is paused or throttled. Only the two settings are held here: the change
        # handlers in defaults_panel write them to AppState and the config store
        # (which clamps the value to 50-105 C), and the engine reads them from there.
        thermal_panel = QWidget()
        thermal_panel.setObjectName("panel")
        tp_layout = QVBoxLayout(thermal_panel)
        tp_layout.setContentsMargins(16, 16, 16, 16)
        tp_layout.setSpacing(10)

        tp_layout.addWidget(_section("THERMAL PROTECTION SETTINGS"))

        self._thermal_protection_check = QCheckBox("Enable Thermal Protection")
        # Default True: protection is opt-out because an unprotected laptop running
        # long generations is the failure this feature exists to prevent.
        self._thermal_protection_check.setChecked(getattr(self.state, "thermal_protection_enabled", True))
        self._thermal_protection_check.stateChanged.connect(self._on_thermal_protection_enabled_changed)
        tp_layout.addWidget(self._thermal_protection_check)

        threshold_row = QWidget()
        tr_lay = QHBoxLayout(threshold_row)
        tr_lay.setContentsMargins(0, 0, 0, 0)
        # Threshold is degrees Celsius; the 95 default (see getattr below) sits just
        # under typical laptop CPU throttle points (~100 C).
        tr_lbl = QLabel("Protection Threshold (°C):")
        tr_lbl.setObjectName("lbl-muted")
        tr_lbl.setFixedWidth(180)
        tr_lay.addWidget(tr_lbl)
        self._thermal_threshold_spin = QSpinBox()
        # 70-105 C: below 70 would trigger too often, above ~105 is past most CPU/GPU limits.
        self._thermal_threshold_spin.setRange(70, 105)
        # Falls back to 95 C if the saved state predates the setting.
        self._thermal_threshold_spin.setValue(getattr(self.state, "thermal_protection_threshold", 95))
        self._thermal_threshold_spin.setFixedWidth(80)
        self._thermal_threshold_spin.valueChanged.connect(self._on_thermal_threshold_changed)
        tr_lay.addWidget(self._thermal_threshold_spin)
        tr_lay.addStretch()
        tp_layout.addWidget(threshold_row)

        layout.addWidget(thermal_panel)

        # About Panel
        # Static identity blurb; kept at the bottom so the settings panels above stay
        # in the first screenful.
        about_panel = QWidget()
        about_panel.setObjectName("panel")
        ap_layout = QVBoxLayout(about_panel)
        ap_layout.setContentsMargins(12, 12, 12, 12)
        ap_layout.setSpacing(8)

        ap_layout.addWidget(_section("ABOUT KARL"))
        
        about = QLabel(
            "Karl &mdash; Privacy-first Offline LLM Introspection Environment<br/>"
            "Zero network calls &middot; In-process inference &middot; Immutable trace logs<br/>"
            "<a href='https://github.com/ethan-da-tech-wizard/karl' style='color:#00C2FF; text-decoration:none;'>https://github.com/ethan-da-tech-wizard/karl</a>"
        )
        about.setObjectName("lbl-muted")
        about.setWordWrap(True)
        about.setTextFormat(Qt.TextFormat.RichText)
        # The one deliberate external link in an otherwise offline app: opened by the
        # system browser on click only, never fetched by Karl itself.
        about.setOpenExternalLinks(True)
        ap_layout.addWidget(about)

        layout.addWidget(about_panel)
        # Stretch pushes all four panels to the top instead of letting the layout
        # spread them evenly down a tall window.
        layout.addStretch()
        return w

    # ── logic ─────────────────────────────────────────────────────────────────


    def _update_live_hardware(self):
        """Timer callback: refresh the CPU, RAM and disk meters and the circuit-breaker label."""
        # Everything is inside try so a missing psutil or a transient error only logs a warning
        # instead of killing the timer-driven slot.
        try:
            import psutil
            # cpu_percent() with no interval reports usage since the previous call, so the
            # first tick after startup reads 0.0 and later ticks are non-blocking.
            cpu = psutil.cpu_percent()
            # virtual_memory().percent already accounts for reclaimable cache on each OS,
            # so it matches what Activity Monitor / Task Manager call memory pressure.
            mem = psutil.virtual_memory()
            
            # setValue takes an int; the exact float goes into the format text so the bar
            # and the label can disagree by <1%, which is intended.
            self._cpu_progress.setValue(int(cpu))
            self._cpu_progress.setFormat(f"%p% ({cpu:.1f}%)")
            
            self._ram_progress.setValue(int(mem.percent))
            # 1_073_741_824 = 2**30: binary gigabytes (GiB), labelled "GB" for brevity.
            used_gb = mem.used / 1_073_741_824
            total_gb = mem.total / 1_073_741_824
            self._ram_progress.setFormat(f"%p% ({used_gb:.1f} GB / {total_gb:.1f} GB)")
            
            # Disk is measured on "/" only; models living on another volume are not reflected.
            # Disk space
            import shutil
            # The bar shows percent USED (fills as the disk fills, so the red zone in the
            # QSS means "nearly full") while the text shows free space, the number users
            # actually care about.
            total, used, free = shutil.disk_usage("/")
            # Guard against total == 0 is unnecessary on a mounted "/" and any
            # ZeroDivisionError would be caught and logged by the surrounding except.
            free_percent = (free / total) * 100
            self._disk_progress.setValue(int(100 - free_percent))
            free_gb = free / 1_073_741_824
            total_gb = total / 1_073_741_824
            self._disk_progress.setFormat(f"{free_gb:.1f} GB free ({free_percent:.1f}%)")

            # Colours: COOLDOWN red, HALF_OPEN amber, otherwise green (closed).
            # Update circuit breaker status label if it exists
            if hasattr(self, "_cb_status_lbl") and self._cb_status_lbl is not None:
                # Imported lazily inside the tick: the engine package pulls in heavy deps, and
                # this panel must stay importable (and the tick harmless) when it is missing;
                # any failure lands in the except below.
                from app.engine.model_loader import ModelLoader
                state = ModelLoader.get_circuit_breaker_state()
                if state == "COOLDOWN":
                    self._cb_status_lbl.setText("<span style='color:#FF5C7A; font-weight:bold;'>Circuit Breaker: COOLDOWN</span>")
                elif state == "HALF_OPEN":
                    self._cb_status_lbl.setText("<span style='color:#FFD800; font-weight:bold;'>Circuit Breaker: HALF_OPEN</span>")
                else:
                    self._cb_status_lbl.setText("<span style='color:#2DD4A0;'>Circuit Breaker: CLOSED</span>")
                self._cb_status_lbl.setTextFormat(Qt.TextFormat.RichText)
        except Exception as e:
            logger.warning(f"Error updating live hardware meters: {e}")


    def _refresh_hardware(self):
        """Re-scout RAM, VRAM and storage and render them in the profile box."""
        # Lazy import: hardware_scout probes GPUs via subprocess/driver calls, which is
        # only worth paying for when the user presses "refresh profiles".
        from core.hardware_scout import get_hardware_profile
        p = get_hardware_profile()
        
        # Values may be missing; '?' and 'N/A' are shown verbatim instead of failing.
        # The scout returns only the keys it could detect, so every read has a default.
        ram = p.get('ram_gb', '?')
        vram = p.get('vram_gb', 'N/A')
        storage = p.get('storage_gb', '?')
        
        # vram is a number on GPU machines and a string such as "N/A" otherwise; only
        # numbers can take the :.1f format.
        if isinstance(vram, (int, float)):
            vram_str = f"{vram:.1f} GB"
        else:
            vram_str = str(vram)
            
        # Inline-styled flex-looking divs: QTextBrowser ignores display:flex, so the
        # label column is faked with a fixed-width inline-block span (120 px).
        html_content = (
            f"<div style='font-family:{MONO}; color:#E4E4F0; line-height:1.6;'>"
            f"<div style='margin-bottom:8px; display:flex;'>"
            f"<span style='display:inline-block; width:120px; color:#505068; font-size:8.5pt; font-weight:bold; letter-spacing:1px;'>RAM</span>"
            f"<span style='font-size:11pt; color:#00C2FF; font-weight:bold;'>{ram} <span style='font-size:8.5pt; font-weight:normal; color:#9090A8;'>GB</span></span>"
            f"</div>"
            f"<div style='margin-bottom:8px; display:flex;'>"
            f"<span style='display:inline-block; width:120px; color:#505068; font-size:8.5pt; font-weight:bold; letter-spacing:1px;'>VRAM</span>"
            f"<span style='font-size:11pt; color:#2DD4A0; font-weight:bold;'>{vram_str}</span>"
            f"</div>"
            f"<div style='margin-bottom:8px; display:flex;'>"
            f"<span style='display:inline-block; width:120px; color:#505068; font-size:8.5pt; font-weight:bold; letter-spacing:1px;'>STORAGE</span>"
            f"<span style='font-size:11pt; color:#F0B030; font-weight:bold;'>{storage} <span style='font-size:8.5pt; font-weight:normal; color:#9090A8;'>GB free</span></span>"
            f"</div>"
            f"</div>"
        )
        self._hw_view.setHtml(html_content)


    def _on_engine_config_changed(self, *_args):
        """Copy the remote-engine controls into app state, persist them and update enabled/disabled UI."""
        # Controls may not exist yet (signals can fire during construction), hence the hasattr guards.
        # Each widget is looked up defensively: Qt can fire stateChanged while the
        # builder is still creating the later widgets.
        enabled = bool(self._remote_engine_check.isChecked()) if hasattr(self, "_remote_engine_check") else False
        # engine_mode is the older single-string switch; remote_engine_enabled is the
        # newer boolean. Both are written below so either generation of reader agrees.
        mode = "remote" if enabled else "local"
        url = self._remote_url_input.text().strip() if hasattr(self, "_remote_url_input") else "wss://localhost:8080"
        token = self._remote_token_input.text() if hasattr(self, "_remote_token_input") else ""
        # State keeps both the new and legacy field names so older readers keep working.
        self.state.engine_mode = mode
        self.state.remote_server_url = url
        self.state.remote_auth_token = token
        self.state.remote_engine_enabled = enabled
        self.state.remote_engine_url = url
        self.state.remote_engine_token = token
        # Imported here to avoid an import cycle with app.engine at module load time.
        from app.engine import config_store
        config_store.set_remote_engine_config(enabled, url, token)
        self._sync_engine_controls_enabled()


    def _sync_engine_controls_enabled(self):
        """Enable the URL/token fields only in remote mode and show matching help text."""
        if not hasattr(self, "_remote_engine_check"):
            return
        # Greying out (rather than hiding) the fields keeps the layout stable and shows
        # users that URL/token exist but only apply in remote mode.
        remote = self._remote_engine_check.isChecked()
        # Disabled widgets keep their text, so toggling remote off and on again does
        # not lose a typed URL or token.
        self._remote_url_input.setEnabled(remote)
        self._remote_token_input.setEnabled(remote)
        if remote:
            self._remote_status_label.setText(
                "Remote mode forwards prompts and generation parameters over JSON-RPC. "
                "Private LAN WSS endpoints may use self-signed certificates."
            )
        else:
            self._remote_status_label.setText("Local mode loads GGUF models with llama-cpp-python on this machine.")
