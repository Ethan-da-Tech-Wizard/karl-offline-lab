# ==============================================================================
# Program: workspace.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: System — hardware info, model management, generation defaults, about.
#          Coordinates with adjacent subsystems within the Karl Offline Lab
#          architecture.
# ==============================================================================

# The workspace itself is assembled from per-tab mixins (one file per tab) so
# that each concern lives in its own module while sharing one `self` for state.
"""
System — hardware info, model management, generation defaults, about.
"""

from __future__ import annotations

import logging
import os

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtWidgets import QLabel, QWidget, QVBoxLayout, QTabWidget

from app.engine.quantizer_thread import QuantizerThread

from .appearance_panel import AppearancePanelMixin
from .appearance_runtime import AppearanceRuntimeMixin
from .defaults_panel import DefaultsPanelMixin
from .mcp_panel import McpPanelMixin
from .model_panel import ModelPanelMixin
from .model_preflight import ModelPreflightMixin
from .quantization_panel import QuantizationPanelMixin
from .registry_panel import RegistryPanelMixin
from .vision_hardware_panel import VisionHardwarePanelMixin
from .adapter_sandbox import AdapterSandboxMixin


logger = logging.getLogger("karl.system_config")


# Mixin order matters for Python's MRO: panel mixins come first so their
# methods win over QWidget defaults; QWidget must stay last in the base list.
class SystemConfigWorkspace(
    RegistryPanelMixin,
    QuantizationPanelMixin,
    ModelPanelMixin,
    ModelPreflightMixin,
    DefaultsPanelMixin,
    VisionHardwarePanelMixin,
    AppearancePanelMixin,
    AppearanceRuntimeMixin,
    McpPanelMixin,
    AdapterSandboxMixin,
    QWidget,
):
    """Tabbed system workspace for model, registry, theme, telemetry, and hardware controls."""

    adapter_changed = pyqtSignal(str)
    appearance_changed = pyqtSignal()

    # State is the shared app state object (signals such as set_workbench_hyperparams).
    def __init__(self, state, parent=None):
        """Create system config tabs and start live hardware monitoring."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(parent)
        self.state = state
        # Background-thread handles; kept so we can avoid starting duplicate jobs
        # and so callers can cancel or wait on them.
        self._download_thread = None
        self._quantizer_thread: QuantizerThread | None = None
        self._model_load_thread = None
        self._active_threads = set()
        # Model/adapter directory scans are cached; showEvent refreshes them lazily.
        self._active_custom_accent = None
        self._cached_models_list = None
        self._cached_adapters_list = None
        self._load_registry()
        self.setObjectName("workspace-root")
        self._build_ui()
        # Hardware readout refreshes on a 2s timer; the first call fills the UI at once
        # instead of waiting for the first tick.
        self._refresh_hardware()
        
        # Start diagnostic hardware monitoring timer
        self._hardware_timer = QTimer(self)
        self._hardware_timer.timeout.connect(self._update_live_hardware)
        self._hardware_timer.start(2000)
        self._update_live_hardware()

    def showEvent(self, event):
        """Refresh scans, speculative-decoding state and hardware info whenever the tab becomes visible."""
        # Step 1: Validate input parameters, operational preconditions, and state
        # force=False lets the scans reuse their cache when nothing changed on disk.
        super().showEvent(event)
        self._scan_models(force=False)
        self._scan_adapters(force=False)
        if hasattr(self, "_scan_sandbox_adapters"):
            # The sandbox tab is optional (mixin may be absent), hence the hasattr guard.
            self._scan_sandbox_adapters()

        # Imported lazily to avoid import cycles and startup cost at module load.
        from app.engine import config_store as _cs
        draft_cfg = _cs.get_active_draft_model()
        if hasattr(self, "_speculative_toggle"):
            self._speculative_toggle.setChecked(bool(draft_cfg.get("enabled")))
        # Reflect the persisted draft-model (speculative decoding) config in the UI.
        if draft_cfg["filename"]:
            draft_path = os.path.join("data", "models", draft_cfg["filename"])
            self._draft_model_input.setText(draft_path)
            if hasattr(self, "_draft_model_combo"):
                idx = self._draft_model_combo.findText(draft_cfg["filename"])
                if idx >= 0:
                    self._draft_model_combo.setCurrentIndex(idx)
            # Only claim "active" if the loader really has speculative decoding enabled.
            from app.engine.model_loader import ModelLoader
            if ModelLoader.is_speculative():
                self._draft_status.setText(
                    f"<span style='color:#2DD4A0;'>Speculative decoding active — draft: "
                    f"{draft_cfg['filename']}</span>"
                )
                self._draft_status.setTextFormat(Qt.TextFormat.RichText)

        self._refresh_hardware()
        self._run_model_preflight_checks()

    def _build_ui(self):
        """Build the System title and tab set from mixin-provided panels."""
        # Step 1: Validate input parameters, operational preconditions, and state
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        title = QLabel("System")
        title.setObjectName("lbl-accent")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; padding-bottom: 4px;")
        root.addWidget(title)

        desc = QLabel(
            "Configure model and environment settings. Load or download GGUF models, "
            "manage default generation hyperparameters, active identity prompts, and check hardware resource specs."
        )
        desc.setObjectName("lbl-muted")
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 8.5pt; margin-bottom: 6px; padding-left: 2px;")
        root.addWidget(desc)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_model_tab(), "Model")
        self._tabs.addTab(self._build_registry_tab(), "Registry")
        self._tabs.addTab(self._build_params_tab(), "Defaults")
        self._tabs.addTab(self._build_identity_tab(), "Identity")
        self._tabs.addTab(self._build_vision_tab(), "Vision")
        self._tabs.addTab(self._build_mcp_tab(), "MCP")
        self._theme_tab = self._build_theme_tab()
        self._tabs.addTab(self._theme_tab, "Theme")
        
        # Observability is imported lazily because it pulls in telemetry modules.
        # Inject Observability panel
        from .observability_tab import ObservabilityTab
        self._observability_tab = ObservabilityTab(self.state, self)
        self._tabs.addTab(self._observability_tab, "Observability")

        self._tabs.addTab(self._build_hardware_tab(), "Hardware")
        self._tabs.addTab(self._build_adapter_sandbox_tab(), "Adapter Sandbox")
        root.addWidget(self._tabs, 1)

    def show_theme_tab(self):
        """Switch to the Theme tab, resyncing its controls from current app state first."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if hasattr(self, "_tabs") and hasattr(self, "_theme_tab"):
            self._sync_appearance_controls_from_state()
            self._tabs.setCurrentWidget(self._theme_tab)

    def show_defaults_tab(self):
        """Switch to the Defaults tab (index 2 in the tab order built in _build_ui)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if hasattr(self, "_tabs"):
            self._tabs.setCurrentIndex(2)

    # ── model tab ─────────────────────────────────────────────────────────────

    def _apply_defaults(self):
        """Publish the chosen generation defaults to the Workbench and confirm to the user."""
        # Step 1: Validate input parameters, operational preconditions, and state
        from PyQt6.QtWidgets import QMessageBox
        temp = self._temp_spin.value()
        top_p = self._topp_spin.value()
        max_tokens = self._maxtok_spin.value()
        
        self.state.set_workbench_hyperparams.emit({
            "temperature": temp,
            "top_p": top_p,
            "max_tokens": max_tokens
        })
        QMessageBox.information(
            self, "Defaults Applied",
            f"Generation defaults applied to Workbench:\n"
            f"• Temperature: {temp}\n"
            f"• Top-P: {top_p}\n"
            f"• Max Tokens: {max_tokens}"
        )

    # ── theme tab ─────────────────────────────────────────────────────────────
