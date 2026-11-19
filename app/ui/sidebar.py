# ==============================================================================
# Program: sidebar.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Collapsible navigation sidebar for Karl desktop application.
#          Provides single-click navigation between Workbench, Swarm, AI Lab,
#          Knowledge Base, and Settings.
# ==============================================================================

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QSizePolicy
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont

# QFont's family argument is a single family name, not a CSS-style comma
# list — passing "JetBrains Mono, Consolas, monospace" as one string matches
# no installed font and silently falls back to the platform default (often
# wider than any of the intended monospace fonts), which overflowed these
# fixed-width buttons and clipped their labels. Use setFamilies() for a real
# ordered fallback list instead.
_MONO_FAMILIES = ["JetBrains Mono", "Fira Code", "Cascadia Code", "Consolas", "Courier New"]


def _mono_font(point_size: float) -> QFont:
    """Build a monospace QFont using the ordered fallback list above.

    QFont needs setFamilies() for fallbacks; see the note on _MONO_FAMILIES.
    """
    # Step 1: Validate input parameters, operational preconditions, and state
    font = QFont()
    font.setFamilies(_MONO_FAMILIES)
    font.setPointSizeF(point_size)
    return font

# Navigation table: (glyph, short label, stack index). The index must match
# the order workspaces are added to the QStackedWidget in MainWindow._build_ui,
# and the Ctrl+N shortcuts there. Glyphs are plain Unicode so no icon assets
# are needed.
_ITEMS = [
    ("◈", "Workbench",  0),
    ("⊕", "Prompt Lab", 1),
    ("⊞", "Knowledge",  2),
    ("◇", "Vision",     3),
    ("⬡", "Training",   4),
    ("◎", "Eval",       5),
    ("☄", "Swarm",      6),
    ("≡", "System",     7),
    ("▤", "Codex",      8),
    ("⟳", "Flywheel",    9),
]


# One square button per workspace. Its "active" dynamic property is what the
# QSS uses to draw the highlighted state.
class _SidebarButton(QPushButton):
    """Fixed-size accessible workspace navigation button."""

    def __init__(self, icon: str, label: str, index: int, parent=None):
        """Create a sidebar button for one workspace index."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(parent)
        self.index = index
        self._icon = icon
        self._label = label
        self.setObjectName("sidebar-btn")
        # Fixed size keeps the column width stable regardless of label length.
        self.setFixedSize(62, 62)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setText(f"{icon}\n{label}")
        self.setFont(_mono_font(6.5))
        # Accessibility text for screen readers (the visible label is tiny).
        self.setAccessibleName(f"Workspace Navigator: {label}")
        self.setAccessibleDescription(f"Switch active view to the {label} workspace")

    def set_compact(self, compact: bool):
        """Switch between label and icon-only presentation for narrow windows."""
        # Step 1: Validate input parameters, operational preconditions, and state
        # Compact mode: icon only, slightly smaller button, larger glyph.
        if compact:
            self.setText(self._icon)
            self.setFixedSize(48, 50)
            self.setFont(_mono_font(12))
        # Normal mode: icon above label in a small monospace font.
        else:
            self.setText(f"{self._icon}\n{self._label}")
            self.setFixedSize(62, 62)
            self.setFont(_mono_font(6.5))

    def set_active(self, active: bool):
        """Set the active dynamic QSS property and force a style refresh."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.setProperty("active", "true" if active else "false")
        # Qt does not re-evaluate property selectors on change; unpolish+polish
        # forces the stylesheet to be re-applied to this widget.
        self.style().unpolish(self)
        self.style().polish(self)


# The navigator column. It is a pure view: it only emits workspace_changed and
# lets the main window flip the stack.
class Sidebar(QWidget):
    """Vertical workspace navigator that emits selected stack indexes."""

    # Emitted with the target stack index whenever a workspace is selected.
    workspace_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        """Build the ten-workspace navigator and select Workbench."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(62)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(2)

        logo = QLabel("K", self)
        logo.setObjectName("sidebar-logo")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setFixedHeight(44)
        layout.addWidget(logo)

        # Hover text describing each workspace, keyed by stack index.
        tooltips = {
            0: "Workbench (Main Chat Space)",
            1: "Prompt Lab (A/B testing & tokenization)",
            2: "Knowledge Base (RAG context & source ingestion)",
            3: "Vision Workbench (Saved images, OCR, and screenshot reasoning)",
            4: "Training Studio (Flywheel, LoRA/QLoRA training, and Mini-GPT sandbox)",
            5: "Eval Suite (Automated model benchmarking)",
            6: "Swarm Studio (dependency layers, task states, verification, and tracebacks)",
            7: "System Config (Hardware readout & model loader settings)",
            8: "Codex (Prompt taxonomy & information guide)",
            9: "Flywheel Studio (Telemetry & fine-tuning loop metrics)",
        }
        self._buttons: list[_SidebarButton] = []
        self._compact = False
        # Build one button per entry. `i=idx` is bound as a default argument so each
        # lambda keeps its own index instead of the loop's last value.
        for icon, label, idx in _ITEMS:
            # Iterate over items, evaluate loop invariant, and update progress
            btn = _SidebarButton(icon, label, idx, self)
            btn.setToolTip(tooltips.get(idx, ""))
            btn.clicked.connect(lambda _checked, i=idx: self._select(i))
            layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignHCenter)
            self._buttons.append(btn)

        # Push the buttons to the top of the column.
        layout.addStretch()
        # Start on the Workbench; this also emits workspace_changed(0).
        self._select(0)

    def _select(self, index: int):
        """Highlight the button for `index` and notify listeners."""
        # Step 1: Validate input parameters, operational preconditions, and state
        for btn in self._buttons:
            # Iterate over items, evaluate loop invariant, and update progress
            btn.set_active(btn.index == index)
        self.workspace_changed.emit(index)

    def select(self, index: int):
        """Programmatically select a workspace index."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._select(index)

    def set_compact(self, compact: bool):
        """Use icon-only navigation when the app is too narrow for labels."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if self._compact == compact:
            return
        self._compact = compact
        # Narrow rail in compact mode so the content area keeps its space.
        self.setFixedWidth(48 if compact else 62)
        for btn in self._buttons:
            # Iterate over items, evaluate loop invariant, and update progress
            btn.set_compact(compact)
