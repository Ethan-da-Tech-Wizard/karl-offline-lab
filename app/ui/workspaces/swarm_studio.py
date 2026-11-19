# ==============================================================================
# Program: swarm_studio.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Desktop workspace implementation for Swarm Studio.
#          Provides responsive layout controls, parameter adjustment surfaces,
#          and visual feedback components within Karl's PyQt6 shell.
# ==============================================================================
# Layout of the file:
#   _TpsChart               small custom-painted sparkline (tokens per second).
#   SwarmStudioWorkspace    the whole tab: a "Live Run" view (left: objective and
#                           settings, centre: task graph/table/log, right: code
#                           stream, debugger, steering, cherry-pick, preview), an
#                           "Agent Profiles" editor tab, and a "Replay" tab.
#
# Data flow: the workspace never runs the swarm itself. _launch() builds a
# SwarmOrchestratorThread (a QThread in app/engine/swarm_orchestrator.py),
# connects every one of its Qt signals to an _on_* slot below, and starts it.
# The thread emits signals; these slots only update widgets. Slots run on the GUI
# thread (Qt queues cross-thread signals), so no locking is needed here.
#
# State kept on the workspace (all reset by _reset_run_ui between runs):
#   _task_rows       filepath -> row index in the task table
#   _task_nodes      filepath -> rectangle item in the dependency graph
#   _file_contents   filepath -> latest content the swarm wrote (for previews)
#   _tracebacks      key (filepath or "Layer N dry-run") -> traceback text
#   _active_filepaths  files currently in progress (drives the steering target)
#
# Style notes: colours are hard-coded hex strings (dark theme) rather than theme
# tokens; object names such as "section-header" and "btn-primary" are QSS hooks
# defined in app/ui/themes.py.


from __future__ import annotations

# Standard library only: os for the default workspace path, time for the run timer.
import os
import time

# Qt value types and painting primitives, used by the chart and the graph view.
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QTextCursor
# Widgets are imported by name (not a star import) so unused ones are easy to spot.
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Project imports: the background orchestrator thread, the sibling profile editor
# tab (embedded below), and the loader that reads saved agent personas.
from app.engine.swarm_orchestrator import SwarmOrchestratorThread
from app.ui.workspaces.agent_profile_studio import AgentProfileStudioWorkspace
from app.utils.swarm_agent_profiles import load_agent_profiles


class _TpsChart(QWidget):
    """Minimal sparkline canvas for live tokens-per-second telemetry."""

    def __init__(self, parent=None):
        """Create an empty chart with a 64px minimum height and no samples."""
        super().__init__(parent)
        # Painting is custom, so the widget needs an explicit minimum height or the layout may collapse it.
        self.setMinimumHeight(64)
        # Rolling window of recent tokens/second readings (capped at 60 in add_sample).
        self._samples: list[float] = []
        self._cumulative_tokens = 0
        self._latest_tps = 0.0

    def add_sample(self, tokens_per_second: float, cumulative_tokens: int) -> None:
        """Record one telemetry reading and schedule a repaint.

        tokens_per_second is plotted; cumulative_tokens is only shown in the caption.
        Negative speeds are clamped to 0 for both the plot and the caption.
        """
        # Clamp to >= 0 so a glitchy reading cannot draw below the baseline.
        self._samples.append(max(0.0, tokens_per_second))
        # Keep only the newest 60 samples so the line is a sliding window, not an ever-growing history.
        self._samples = self._samples[-60:]
        # A negative speed is meaningless (timer glitch); clamp the caption too so it never shows "-12.0 tok/s".
        self._latest_tps = max(0.0, tokens_per_second)
        self._cumulative_tokens = cumulative_tokens
        # update() queues a paintEvent; never call paintEvent directly.
        self.update()

    def reset(self) -> None:
        """Forget all samples and counters (called at the start of each run)."""
        self._samples = []
        self._cumulative_tokens = 0
        self._latest_tps = 0.0
        self.update()

    def paintEvent(self, event):
        """Draw the background, border, sparkline and the caption text.

        Layout: 4px padding on the sides; the bottom 22px is reserved for the caption.
        """
        painter = QPainter(self)
        # Antialiasing keeps the diagonal sparkline segments smooth.
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        # Dark background and a 1px border; the -1 keeps the border inside the widget rectangle.
        painter.fillRect(rect, QColor("#0A0A14"))
        painter.setPen(QPen(QColor("#252535")))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))

        # A line needs at least two points; with fewer, only the caption is drawn.
        if len(self._samples) >= 2:
            # Scale to the peak sample; 'or 1.0' avoids dividing by zero when every sample is 0.
            max_v = max(self._samples) or 1.0
            # Drawable width/height after padding; height excludes the caption strip.
            w = rect.width() - 8
            h = rect.height() - 22
            n = len(self._samples)
            # Horizontal spacing so the samples span the full width (max() guards n == 1, though n >= 2 here).
            step = w / max(1, n - 1)
            painter.setPen(QPen(QColor("#7FDBFF"), 1.5))
            # Qt's y axis points down, so invert: a larger value maps to a smaller y.
            points = [
                QPointF(4 + i * step, 4 + h - (v / max_v) * h)
                for i, v in enumerate(self._samples)
            ]
            # Join consecutive points with line segments to form the sparkline.
            for a, b in zip(points, points[1:]):
                painter.drawLine(a, b)

        # Caption: latest tokens/second and the running token total, bottom-left.
        painter.setPen(QPen(QColor("#C0C0D0")))
        painter.setFont(QFont("Monospace", 8))
        painter.drawText(
            rect.adjusted(4, 0, -4, -4),
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
            f"{self._latest_tps:.1f} tok/s · {self._cumulative_tokens} tokens total",
        )
        # end() must be called to release the painter before the widget finishes painting.
        painter.end()


class SwarmStudioWorkspace(QWidget):
    """High-signal dashboard for Karl's multi-agent code swarm."""

    def __init__(self, state, parent=None):
        """Build the workspace UI and load saved data.

        state is the shared AppState (provides memory persistence and the swarm_*
        fields used to remember the last objective/workspace).
        """
        super().__init__(parent)
        # Shared application state; the only way workspaces talk to the rest of the app.
        self.state = state
        # Hook for the QSS workspace styling.
        self.setObjectName("workspace-root")
        # The running orchestrator thread, or None when idle.
        self._thread: SwarmOrchestratorThread | None = None
        # File header currently being streamed (None until the first coder token of a run).
        self._stream_current_file = None
        # Wall-clock start time of the run; drives the elapsed-time label.
        self._started_at = 0.0
        # Run bookkeeping, cleared by _reset_run_ui (see the file header for what each holds).
        self._task_rows: dict[str, int] = {}
        self._file_contents: dict[str, str] = {}
        self._tracebacks: dict[str, str] = {}
        # Graph scene/items are rebuilt whenever new dependency layers arrive.
        self._history = []
        self._task_nodes: dict[str, QGraphicsRectItem] = {}
        self._graph_scene: QGraphicsScene | None = None
        # Files being coded right now; used to keep the steering target pointing at live work.
        self._active_filepaths: list[str] = []
        # Cache of agent persona definitions, refreshed when the profile editor saves.
        self._agent_profiles: dict[str, dict] = {}

        # UI clock, started in _launch with a 250ms interval; it only updates the runtime label and never touches the thread.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        # Order matters: widgets must exist before the profile combos can be filled and history loaded.
        self._build_ui()
        self._refresh_agent_profile_selectors()
        self._load_history()

    def _build_ui(self):
        """Assemble the top-level layout: title row plus a tab widget.

        Tabs: "Live Run" (three-column splitter), "Agent Profiles" and "Replay".
        """
        # Outer vertical layout: title row on top, tabs filling the rest.
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # Title row: product name on the left, run state ('idle' / 'running Ns') pushed to the right by a stretch.
        title_row = QWidget()
        tl = QHBoxLayout(title_row)
        tl.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Swarm Studio")
        title.setObjectName("lbl-accent")
        # Inline style overrides the QSS size for the title only; the object name still picks up the accent colour.
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        self._runtime_lbl = QLabel("idle")
        self._runtime_lbl.setObjectName("lbl-muted")
        tl.addWidget(title)
        tl.addStretch()
        tl.addWidget(self._runtime_lbl)
        root.addWidget(title_row)

        # Three resizable columns separated by 1px handles; stretch factors 2:4:3 favour the centre.
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        # Add order = left, centre, right; stretch indexes below follow this order.
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 3)

        # Wrap the splitter in a tab page with a small top gap so it sits below the tab bar.
        live_tab = QWidget()
        live_layout = QVBoxLayout(live_tab)
        live_layout.setContentsMargins(0, 8, 0, 0)
        live_layout.setSpacing(0)
        # The splitter gets stretch factor 1 so it takes all vertical space.
        live_layout.addWidget(splitter, 1)

        # Tab order matters: _on_main_tab_changed matches on the label 'Replay'.
        self._main_tabs = QTabWidget()
        self._main_tabs.addTab(live_tab, "Live Run")
        # The profile editor shares the same AppState.
        self._profile_studio = AgentProfileStudioWorkspace(self.state)
        # When the profile editor saves, re-read profiles so the Architect/Coder/Tester selectors stay current.
        self._profile_studio.profile_saved.connect(lambda _name: self._refresh_agent_profile_selectors())
        self._main_tabs.addTab(self._profile_studio, "Agent Profiles")
        # Replay is built here but its run list is loaded lazily (see _on_main_tab_changed).
        self._main_tabs.addTab(self._build_replay_panel(), "Replay")
        # Detect tab switches so the Replay list can be loaded on first view.
        self._main_tabs.currentChanged.connect(self._on_main_tab_changed)
        root.addWidget(self._main_tabs, 1)

    def _build_left_panel(self) -> QWidget:
        """Build the left column: objective, workspace, runtime and intelligence controls.

        Returns a QScrollArea wrapping the panel (see the comment at the end).
        """
        # Right margin 8px separates this column from the centre column.
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(8)

        # Section headers use the shared 'section-header' object name for consistent QSS styling.
        header = QLabel("Objective")
        header.setObjectName("section-header")
        layout.addWidget(header)

        # Free-text objective; the stretch factor 2 lets it grow more than the smaller widgets.
        self._objective_input = QTextEdit()
        self._objective_input.setPlaceholderText("Describe the codebase change the swarm should plan, edit, and verify.")
        self._objective_input.setMinimumHeight(120)
        layout.addWidget(self._objective_input, 2)

        # Default the workspace to the last one used, falling back to the process working directory.
        self._workspace_input = QLineEdit()
        self._workspace_input.setPlaceholderText(os.getcwd())
        # state.swarm_last_workspace may be empty on first run, hence the cwd fallback.
        self._workspace_input.setText(self.state.swarm_last_workspace or os.getcwd())
        layout.addWidget(QLabel("Workspace Path"))
        layout.addWidget(self._workspace_input)

        # The same default is used for the placeholder and the initial text, and again in _load_history_item.
        self._test_command_input = QLineEdit()
        self._test_command_input.setPlaceholderText("venv/bin/python -m pytest tests/ -q")
        self._test_command_input.setText("venv/bin/python -m pytest tests/ -q")
        layout.addWidget(QLabel("Verification Command"))
        layout.addWidget(self._test_command_input)

        # Basic runtime knobs: sampling temperature (as a percent, divided by 100 in _launch) and max tokens per generation.
        group = QGroupBox("Swarm Runtime")
        gl = QGridLayout(group)
        # Spin boxes keep inputs in range so _launch needs no extra validation of them.
        self._temp_spin = QSpinBox()
        self._temp_spin.setRange(0, 100)
        self._temp_spin.setValue(20)
        self._temp_spin.setSuffix("%")
        # Token limit moves in steps of 256 between 256 and 8192.
        self._tokens_spin = QSpinBox()
        self._tokens_spin.setRange(256, 8192)
        self._tokens_spin.setSingleStep(256)
        self._tokens_spin.setValue(2048)
        # Grid rows: label in column 0, control in column 1.
        gl.addWidget(QLabel("Coder Temp"), 0, 0)
        gl.addWidget(self._temp_spin, 0, 1)
        gl.addWidget(QLabel("Max Tokens"), 1, 0)
        gl.addWidget(self._tokens_spin, 1, 1)
        layout.addWidget(group)

        # Swarm 2.0 options. Each toggle below maps to one boolean in the hyperparams dict built by _launch.
        intel_group = QGroupBox("Swarm Intelligence")
        ig = QGridLayout(intel_group)
        self._candidates_spin = QSpinBox()
        self._candidates_spin.setRange(1, 5)
        self._candidates_spin.setValue(1)
        # Range 1-5; 1 means a single candidate, so the Judge is not involved.
        self._candidates_spin.setToolTip(
            "Multiverse execution: generate N independent candidate solutions per\n"
            "task (different temperatures) and let the Judge pick the strongest one."
        )
        # Candidate count control sits in row 0 of the intelligence grid.
        ig.addWidget(QLabel("Candidates / Task"), 0, 0)
        ig.addWidget(self._candidates_spin, 0, 1)

        # Checkable QPushButtons styled as toggles are used instead of QCheckBox for a more compact, themed look.
        self._memory_check = QPushButton("Cross-Run Memory")
        self._memory_check.setCheckable(True)
        self._memory_check.setChecked(True)
        self._memory_check.setToolTip("Recall past failure patterns on this codebase before coding.")
        self._specialists_check = QPushButton("Specialist Audit")
        self._specialists_check.setCheckable(True)
        self._specialists_check.setChecked(True)
        self._specialists_check.setToolTip("Auto-run security/performance auditors on tasks that touch sensitive code.")
        self._critic_check = QPushButton("Critic Review")
        self._critic_check.setCheckable(True)
        self._critic_check.setChecked(True)
        self._critic_check.setToolTip("Red-team pass for silent failure modes and code smells.")
        self._adaptive_check = QPushButton("Adaptive Concurrency")
        self._adaptive_check.setCheckable(True)
        self._adaptive_check.setChecked(True)
        self._adaptive_check.setToolTip("Size the parallel worker pool from live GPU/CPU/VRAM headroom.")
        # Give all four toggles the shared 'btn-toggle' QSS look.
        for btn in (self._memory_check, self._specialists_check, self._critic_check, self._adaptive_check):
            btn.setObjectName("btn-toggle")
        # Toggles are placed two per row in rows 1 and 2.
        ig.addWidget(self._memory_check, 1, 0)
        ig.addWidget(self._specialists_check, 1, 1)
        ig.addWidget(self._critic_check, 2, 0)
        ig.addWidget(self._adaptive_check, 2, 1)
        # Persona selectors; contents are filled by _refresh_agent_profile_selectors.
        self._architect_profile_combo = QComboBox()
        self._coder_profile_combo = QComboBox()
        self._tester_profile_combo = QComboBox()
        # Rows 3-5: one persona selector per role.
        ig.addWidget(QLabel("Architect"), 3, 0)
        ig.addWidget(self._architect_profile_combo, 3, 1)
        ig.addWidget(QLabel("Coder"), 4, 0)
        ig.addWidget(self._coder_profile_combo, 4, 1)
        ig.addWidget(QLabel("Tester"), 5, 0)
        ig.addWidget(self._tester_profile_combo, 5, 1)
        layout.addWidget(intel_group)

        # Launch / Stop row. Stop stays disabled until a run is active.
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        # Primary action; connected to _launch.
        self._launch_btn = QPushButton("Launch Swarm")
        self._launch_btn.setObjectName("btn-primary")
        self._launch_btn.clicked.connect(self._launch)
        # 'btn-danger' styles Stop as destructive; it is wired to the cooperative _stop.
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setObjectName("btn-danger")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop)
        rl.addWidget(self._launch_btn)
        rl.addWidget(self._stop_btn)
        layout.addWidget(row)

        # Clicking a recent objective restores its stored settings (see _load_history_item).
        layout.addWidget(QLabel("Recent Objectives"))
        # Stretch 1 lets the history list absorb leftover height.
        self._recent_list = QListWidget()
        self._recent_list.itemClicked.connect(self._load_history_item)
        layout.addWidget(self._recent_list, 1)

        # This panel stacks the objective box, two line-edit fields, two
        # QGroupBoxes of runtime/intelligence controls, the launch row, and
        # a history list -- with no room to spare, all of it got squeezed
        # below readable height at the app's 760x560 minimum size (observed:
        # the objective placeholder text and the verification-command field
        # both visibly clipped). Scrolling beats every widget losing height.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(panel)
        return scroll

    def _build_center_panel(self) -> QWidget:
        """Build the centre column: progress summary, dependency graph, layers tree, task table and status log."""
        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(8, 0, 8, 0)
        # Margins are 8px left/right so the centre column is padded away from its neighbours.
        outer.setSpacing(8)

        # Summary strip: layer count, task count, verification state, and an overall progress bar.
        self._summary_frame = QFrame()
        self._summary_frame.setObjectName("panel")
        sl = QGridLayout(self._summary_frame)
        # Initial placeholders; replaced as layer/task events arrive.
        self._layer_lbl = QLabel("Layers: 0")
        self._task_lbl = QLabel("Tasks: 0")
        self._verify_lbl = QLabel("Verification: idle")
        # Muted style for the three summary labels.
        for label in (self._layer_lbl, self._task_lbl, self._verify_lbl):
            label.setObjectName("lbl-muted")
        # Grid row 0: three status labels; row 1 (below) spans all three columns with the bar.
        sl.addWidget(self._layer_lbl, 0, 0)
        sl.addWidget(self._task_lbl, 0, 1)
        sl.addWidget(self._verify_lbl, 0, 2)
        # Progress is a plain 0-100 percentage; the text is hidden because the labels above already describe state.
        self._progress = QProgressBar()
        self._progress.setTextVisible(False)
        self._progress.setRange(0, 100)
        # Span 1 row x 3 columns.
        sl.addWidget(self._progress, 1, 0, 1, 3)
        outer.addWidget(self._summary_frame)

        # Vertical splitter: graph on top, details (tree, table, log) below, 2:3.
        vsplit = QSplitter(Qt.Orientation.Vertical)
        vsplit.setHandleWidth(1)

        # Container so the graph header and view share one layout inside the splitter.
        graph_container = QWidget()
        gc_layout = QVBoxLayout(graph_container)
        gc_layout.setContentsMargins(0, 0, 0, 0)
        gc_layout.setSpacing(4)

        # Heading above the graph view.
        graph_lbl = QLabel("Task Dependency Graph")
        graph_lbl.setObjectName("section-header")
        gc_layout.addWidget(graph_lbl)

        # ScrollHandDrag lets the user pan a graph that is larger than the view.
        self._graph_scene = QGraphicsScene(self)
        self._graph_view = QGraphicsView(self._graph_scene)
        self._graph_view.setMinimumHeight(180)
        self._graph_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._graph_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        # Local stylesheet gives the graph a dark background and a rounded border.
        self._graph_view.setStyleSheet(
            "background: #0D0D16; border: 1px solid #252535; border-radius: 4px;"
        )
        gc_layout.addWidget(self._graph_view, 1)
        vsplit.addWidget(graph_container)

        # Lower pane: dependency tree, task table and log stacked vertically.
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(8)

        # Tree of dependency layers; each top-level item is one layer, children are its files.
        self._layers_tree = QTreeWidget()
        self._layers_tree.setHeaderLabels(["Dependency Layers"])
        # Minimum heights stop the tree being squeezed to nothing.
        self._layers_tree.setMinimumHeight(80)
        bl.addWidget(self._layers_tree, 1)

        # Four read-only columns; rows are keyed by filepath in _task_rows.
        self._task_table = QTableWidget(0, 4)
        # Header labels double as column indices: 0 File, 1 Status, 2 Layer, 3 Detail (used by _ensure_task_row).
        self._task_table.setHorizontalHeaderLabels(["File", "Status", "Layer", "Detail"])
        # Display-only table: the user cannot edit cells.
        self._task_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        # Clicking selects the whole row so selectedRows() works in _sync_selection_preview.
        self._task_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        # Selecting a row loads that file's content/traceback into the right-hand previews.
        self._task_table.itemSelectionChanged.connect(self._sync_selection_preview)
        # Let the Detail column take the remaining width.
        self._task_table.horizontalHeader().setStretchLastSection(True)
        bl.addWidget(self._task_table, 2)

        # Read-only log fed by status messages from the swarm.
        self._status_log = QTextEdit()
        self._status_log.setReadOnly(True)
        # Monospace keeps log columns aligned.
        self._status_log.setFont(QFont("Monospace", 9))
        self._status_log.setPlaceholderText("Swarm status stream.")
        bl.addWidget(self._status_log, 2)

        # Stretch 2:3 matches the graph/bottom ratio set below.
        vsplit.addWidget(bottom)
        vsplit.setStretchFactor(0, 2)
        vsplit.setStretchFactor(1, 3)
        outer.addWidget(vsplit, 1)
        return panel

    def _build_right_panel(self) -> QWidget:
        """Build the right column: code stream, telemetry, debugger, steering, cognition log, cherry-pick, preview and traceback."""
        # Right column; left margin 8px, no right margin so it sits against the window edge.
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(8)

        # Heading for the live token stream.
        stream_header = QLabel("Live Code Stream")
        stream_header.setObjectName("section-header")
        layout.addWidget(stream_header)

        # Live token stream; capped at 180px so the other widgets keep room.
        self._stream_view = QPlainTextEdit()
        self._stream_view.setReadOnly(True)
        # Monospace 8pt keeps lots of code visible in a small box.
        self._stream_view.setFont(QFont("Monospace", 8))
        self._stream_view.setPlaceholderText("Tokens appear here as each coder agent writes...")
        self._stream_view.setMaximumHeight(180)
        # Colours match a terminal look: dark background, green text.
        self._stream_view.setStyleSheet(
            "background: #080810; color: #A0D8B0; border: 1px solid #202040; border-radius: 4px; padding: 4px;"
        )
        layout.addWidget(self._stream_view)

        # Heading for the telemetry chart.
        chart_header = QLabel("Live Telemetry")
        chart_header.setObjectName("section-header")
        layout.addWidget(chart_header)
        # Custom sparkline for tokens/second.
        self._tps_chart = _TpsChart()
        layout.addWidget(self._tps_chart)

        # ── Interactive debugger: pause/step/resume mid-run ──────────────────
        # Group box holds the pause/step/resume controls.
        debugger_group = QGroupBox("Interactive Debugger")
        dg = QVBoxLayout(debugger_group)
        dg.setSpacing(6)

        # Pause on Step is read once at launch (hyperparams['pause_on_step']); the Pause button below can engage pausing mid-run.
        self._pause_on_step_check = QPushButton("Pause on Step")
        self._pause_on_step_check.setCheckable(True)
        # The 'btn-toggle' look makes a checkable button read as an on/off switch.
        self._pause_on_step_check.setObjectName("btn-toggle")
        self._pause_on_step_check.setToolTip(
            "Pause before each coder task starts so you can inspect or override it before it runs."
        )
        dg.addWidget(self._pause_on_step_check)

        # Pause, Step and Resume sit on one row with tight spacing.
        debug_btn_row = QWidget()
        dbr = QHBoxLayout(debug_btn_row)
        dbr.setContentsMargins(0, 0, 0, 0)
        dbr.setSpacing(4)
        self._debug_pause_btn = QPushButton("Pause")
        self._debug_pause_btn.setToolTip("Engage step-pausing now; takes effect at the next task boundary.")
        # Pause only takes effect at the next task boundary (the tooltip says so).
        self._debug_pause_btn.clicked.connect(self._on_debug_pause)
        self._debug_step_btn = QPushButton("Step")
        # Step/Resume start disabled; _on_swarm_paused enables them when the swarm is actually paused.
        self._debug_step_btn.setEnabled(False)
        self._debug_step_btn.clicked.connect(self._on_debug_step)
        self._debug_resume_btn = QPushButton("Resume")
        self._debug_resume_btn.setEnabled(False)
        self._debug_resume_btn.clicked.connect(self._on_debug_resume)
        dbr.addWidget(self._debug_pause_btn)
        dbr.addWidget(self._debug_step_btn)
        dbr.addWidget(self._debug_resume_btn)
        dg.addWidget(debug_btn_row)

        # Describes which task the swarm is paused before; word wrap keeps long instructions visible.
        self._debug_active_step_lbl = QLabel("No task currently paused.")
        self._debug_active_step_lbl.setObjectName("lbl-muted")
        self._debug_active_step_lbl.setWordWrap(True)
        dg.addWidget(self._debug_active_step_lbl)

        layout.addWidget(debugger_group)

        # ── Live steering: inject a human correction into an in-flight task ──
        # Doubles as the "editable text box" for the interactive debugger —
        # when paused, target the shown filepath and send an instruction.
        # Target path, message, and button used to share one row -- three
        # widgets fighting for a panel that's only ~1/3 of the app's 760px
        # minimum width left the message field showing a couple of
        # characters of its own placeholder. Message gets its own full-width
        # row since it's the one that actually needs the room.
        # Row 1 of steering: target filepath box (stretch 1) and the Steer button.
        steer_target_row = QWidget()
        str_l = QHBoxLayout(steer_target_row)
        str_l.setContentsMargins(0, 0, 0, 0)
        str_l.setSpacing(4)
        # Steering: the target filepath box is auto-filled from the active task; the message box sends on Enter or via the Steer button.
        self._steering_target = QLineEdit()
        self._steering_target.setPlaceholderText("target filepath (e.g. app/foo.py)")
        # Both the button and Enter in the message box call _on_send_guidance.
        self._steering_btn = QPushButton("Steer")
        self._steering_btn.clicked.connect(self._on_send_guidance)
        str_l.addWidget(self._steering_target, 1)
        str_l.addWidget(self._steering_btn)
        layout.addWidget(steer_target_row)

        self._steering_input = QLineEdit()
        self._steering_input.setPlaceholderText("Live steering: correct the agent mid-task, no restart needed...")
        self._steering_input.returnPressed.connect(self._on_send_guidance)
        layout.addWidget(self._steering_input)

        # Heading for the cognition/intelligence log (same text as the left group box title).
        cog_header = QLabel("Swarm Intelligence")
        cog_header.setObjectName("section-header")
        layout.addWidget(cog_header)
        # Read-only log of multiverse/memory/specialist/concurrency/steering events (see _log_intel).
        self._cognition_log = QPlainTextEdit()
        self._cognition_log.setReadOnly(True)
        self._cognition_log.setFont(QFont("Monospace", 8))
        self._cognition_log.setPlaceholderText(
            "Multiverse candidates, memory recalls, specialist reviews, and concurrency\n"
            "decisions stream here in real time."
        )
        # Purple-on-dark styling distinguishes this log from the green code stream.
        self._cognition_log.setMaximumHeight(140)
        self._cognition_log.setStyleSheet(
            "background: #0A0812; color: #C9A8FF; border: 1px solid #26203A; border-radius: 4px; padding: 4px;"
        )
        layout.addWidget(self._cognition_log)

        # ── Cherry-pick panel (hidden until the orchestrator proposes edits) ──
        # Group title uses an em dash.
        self._cherry_pick_group = QGroupBox("Proposed Edits — Cherry Pick")
        cg = QVBoxLayout(self._cherry_pick_group)
        cg.setSpacing(6)

        # Checkable list of proposed file edits; item data holds the new file content.
        self._cherry_pick_list = QListWidget()
        self._cherry_pick_list.setMinimumHeight(100)
        self._cherry_pick_list.itemClicked.connect(self._on_cherry_pick_item_clicked)
        cg.addWidget(self._cherry_pick_list)

        # Button row for All / None / Commit.
        btn_row = QWidget()
        br = QHBoxLayout(btn_row)
        br.setContentsMargins(0, 0, 0, 0)
        br.setSpacing(4)
        # Fixed-width All/None helpers keep the row compact.
        _all_btn = QPushButton("All")
        _all_btn.setFixedWidth(42)
        _all_btn.clicked.connect(self._cherry_select_all)
        _none_btn = QPushButton("None")
        _none_btn.setFixedWidth(42)
        _none_btn.clicked.connect(self._cherry_select_none)
        # Commit stays disabled until edits are proposed.
        self._commit_btn = QPushButton("Commit Checked Edits")
        self._commit_btn.setObjectName("btn-primary")
        self._commit_btn.setEnabled(False)
        self._commit_btn.clicked.connect(self._on_commit_edits)
        # Layout: All, None, then a stretch so Commit sits at the right edge.
        br.addWidget(_all_btn)
        br.addWidget(_none_btn)
        br.addStretch()
        br.addWidget(self._commit_btn)
        cg.addWidget(btn_row)

        # Hidden until the orchestrator proposes edits (_on_edits_proposed shows it).
        self._cherry_pick_group.setVisible(False)
        layout.addWidget(self._cherry_pick_group)

        # Preview shows the selected file's content or a cherry-pick item's proposed content.
        layout.addWidget(QLabel("File Preview"))
        # Read-only preview; content comes from selected tasks or cherry-pick items.
        self._preview = QTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setFont(QFont("Monospace", 9))
        layout.addWidget(self._preview, 2)

        layout.addWidget(QLabel("Verification Traceback"))
        # setOpenLinks(False) stops the browser from navigating when a link in the traceback is clicked.
        self._traceback_browser = QTextBrowser()
        self._traceback_browser.setFont(QFont("Monospace", 9))
        self._traceback_browser.setOpenLinks(False)
        layout.addWidget(self._traceback_browser, 2)

        # Result banner under the traceback; text and colour are set by _reset_run_ui and _on_finished.
        self._result_banner = QLabel("No run active.")
        self._result_banner.setWordWrap(True)
        self._result_banner.setObjectName("lbl-muted")
        layout.addWidget(self._result_banner)
        return panel

    def _build_replay_panel(self) -> QWidget:
        """Build the Replay tab: a past-run picker, a step list, and diff/test output viewers."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Created lazily in _refresh_replay_runs (imported there so startup stays light).
        self._replay_manager = None
        # Steps of the currently selected past run.
        self._replay_steps: list[dict] = []

        # Header row: run selector and Refresh button.
        header_row = QWidget()
        hl = QHBoxLayout(header_row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(8)
        hl.addWidget(QLabel("Past Run:"))
        # Combo item data holds the run id; item text is a readable label.
        self._replay_run_combo = QComboBox()
        self._replay_run_combo.setMinimumWidth(280)
        self._replay_run_combo.currentIndexChanged.connect(self._on_replay_run_selected)
        hl.addWidget(self._replay_run_combo, 1)
        # Refresh re-lists runs without needing to leave the tab.
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_replay_runs)
        hl.addWidget(refresh_btn)
        layout.addWidget(header_row)

        # Shows the selected run's objective (or a prompt before one is chosen).
        self._replay_objective_lbl = QLabel("Select a past run to inspect its step-by-step timeline.")
        self._replay_objective_lbl.setObjectName("lbl-muted")
        self._replay_objective_lbl.setWordWrap(True)
        layout.addWidget(self._replay_objective_lbl)

        # Left: step list. Right: diff and test output for the selected step, 1:2.
        replay_split = QSplitter(Qt.Orientation.Horizontal)
        replay_split.setHandleWidth(1)

        # Clicking a step shows its diff and test output.
        self._replay_steps_list = QListWidget()
        self._replay_steps_list.itemClicked.connect(self._on_replay_step_selected)
        replay_split.addWidget(self._replay_steps_list)

        # Right pane: diff above, test output below, each stretch 1.
        detail_widget = QWidget()
        dw_layout = QVBoxLayout(detail_widget)
        dw_layout.setContentsMargins(0, 0, 0, 0)
        dw_layout.setSpacing(6)
        dw_layout.addWidget(QLabel("Diff"))
        # Monospace read-only text for diffs.
        self._replay_diff_view = QPlainTextEdit()
        self._replay_diff_view.setReadOnly(True)
        self._replay_diff_view.setFont(QFont("Monospace", 8))
        dw_layout.addWidget(self._replay_diff_view, 1)
        dw_layout.addWidget(QLabel("Test Output"))
        self._replay_test_view = QPlainTextEdit()
        self._replay_test_view.setReadOnly(True)
        self._replay_test_view.setFont(QFont("Monospace", 8))
        dw_layout.addWidget(self._replay_test_view, 1)
        replay_split.addWidget(detail_widget)
        replay_split.setStretchFactor(0, 1)
        replay_split.setStretchFactor(1, 2)

        layout.addWidget(replay_split, 1)
        return panel

    def _on_main_tab_changed(self, index: int) -> None:
        """Lazy-load past runs the first time the Replay tab is opened and the run list is still empty."""
        # Only load when the Replay tab is shown and the combo is empty, so it is loaded once, not on every tab switch.
        if self._main_tabs.tabText(index) == "Replay" and self._replay_run_combo.count() == 0:
            self._refresh_replay_runs()

    def _refresh_replay_runs(self) -> None:
        """Reload the list of past swarm runs into the run combo box.

        Selects the first run automatically; if none exist, shows a message and clears the views.
        """
        # Imported lazily so this module can load without the replay utilities being touched until needed.
        from app.utils.swarm_replay import SwarmReplayManager
        # Create the manager on first use and reuse it afterwards.
        if self._replay_manager is None:
            self._replay_manager = SwarmReplayManager()

        # Returns a list of run dicts with run_id, timestamp, objective and success.
        runs = self._replay_manager.list_past_runs()
        # Block signals while repopulating, otherwise every addItem would fire currentIndexChanged and load a run.
        self._replay_run_combo.blockSignals(True)
        self._replay_run_combo.clear()
        for run in runs:
            # Label: timestamp (first 19 chars), objective (first 50 chars, falling back to the run id) and status.
            status = "ok" if run.get("success") else "failed"
            objective = (run.get("objective") or run["run_id"])[:50]
            label = f"{run.get('timestamp', '')[:19]} — {objective} ({status})"
            # The run id travels as item data so it can be retrieved with itemData().
            self._replay_run_combo.addItem(label, run["run_id"])
        self._replay_run_combo.blockSignals(False)

        # Signals were blocked above, so the first run must be selected manually.
        if runs:
            self._on_replay_run_selected(0)
        else:
            # Empty state: explain that no runs exist and clear the right-hand views.
            self._replay_objective_lbl.setText("No past swarm runs found yet.")
            self._replay_steps_list.clear()
            self._replay_diff_view.clear()
            self._replay_test_view.clear()

    def _on_replay_run_selected(self, index: int) -> None:
        """Load the steps of the chosen past run into the step list.

        index is the combo box index; negative or missing data is ignored.
        """
        if index < 0 or self._replay_manager is None:
            return
        # Combo item data = run id; empty/None means nothing selected.
        run_id = self._replay_run_combo.itemData(index)
        if not run_id:
            return
        # get_run_details returns the objective and the list of step dicts.
        details = self._replay_manager.get_run_details(run_id)
        self._replay_objective_lbl.setText(details.get("objective") or "(no objective recorded)")
        # Keep the raw step dicts; each list item also carries its own step dict as UserRole data.
        self._replay_steps = details.get("steps", [])
        self._replay_steps_list.clear()
        for step in self._replay_steps:
            # Label format: [index] type.
            label = f"[{step['step_index']}] {step['type']}"
            # Optional filepath and drift marker are appended to the label only when present.
            if step.get("filepath"):
                label += f" — {step['filepath']}"
            if step.get("is_drift"):
                label += "  ⚠ drift"
            item = QListWidgetItem(label)
            # UserRole data carries the full step dict for _on_replay_step_selected.
            item.setData(Qt.ItemDataRole.UserRole, step)
            self._replay_steps_list.addItem(item)
        # Clear stale diff/test output from the previous selection.
        self._replay_diff_view.clear()
        self._replay_test_view.clear()

    def _on_replay_step_selected(self, item: QListWidgetItem) -> None:
        """Show the diff and test output stored on the clicked replay step, with placeholders when absent."""
        # The item may carry no data, so default to an empty dict.
        step = item.data(Qt.ItemDataRole.UserRole) or {}
        self._replay_diff_view.setPlainText(step.get("diff") or "(no diff for this step)")
        self._replay_test_view.setPlainText(step.get("test_output") or "(no test output for this step)")

    def _load_history(self):
        """Reload recent objectives from persistent memory and rebuild the Recent Objectives list."""
        # Memory is the source of truth for history; this is a read, not a write.
        self._history = self.state.memory.load_swarm_history()
        self._recent_list.clear()
        # One list row per history entry.
        for item in self._history:
            objective = item.get("objective", "")
            # Truncate long objectives to 70 chars (67 + '...'); the full text goes in the tooltip.
            row = QListWidgetItem(objective if len(objective) < 70 else objective[:67] + "...")
            row.setToolTip(objective)
            # Store the whole history dict so a click can restore every setting.
            row.setData(Qt.ItemDataRole.UserRole, item)
            self._recent_list.addItem(row)

    def _save_history_item(self, objective: str, workspace_path: str, test_command: str):
        """Record a launched objective at the top of the recent list and persist it.

        The most recent 10 distinct objectives are kept (an existing entry with the same
        objective is replaced, not duplicated).
        """
        # Settings stored with the objective so a later click can recreate the run's setup.
        item = {
            "objective": objective,
            "workspace_path": workspace_path,
            "test_command": test_command,
            "temperature": self._temp_spin.value(),
            "max_tokens": self._tokens_spin.value(),
        }
        # De-duplicate by objective text, then put the new entry first.
        self._history = [old for old in self._history if old.get("objective") != objective]
        # insert(0) puts the newest entry at the top.
        self._history.insert(0, item)
        # Cap history at 10 entries.
        self._history = self._history[:10]
        # Persist, then rebuild the visible list from the new list.
        self.state.memory.save_swarm_history(self._history)
        self._load_history()

    def _load_history_item(self, item: QListWidgetItem):
        """Restore a history entry's objective, workspace, test command and (if stored) temperature/max tokens."""
        data = item.data(Qt.ItemDataRole.UserRole) or {}
        # Fill the editable fields from the stored entry; missing values become empty strings or defaults.
        self._objective_input.setPlainText(data.get("objective", ""))
        self._workspace_input.setText(data.get("workspace_path", ""))
        self._test_command_input.setText(data.get("test_command", "venv/bin/python -m pytest tests/ -q"))
        # Older entries may lack these keys, so only apply them when present.
        if "temperature" in data:
            self._temp_spin.setValue(int(data["temperature"]))
        if "max_tokens" in data:
            self._tokens_spin.setValue(int(data["max_tokens"]))

    def _refresh_agent_profile_selectors(self):
        """Refill the Architect/Coder/Tester combo boxes from the saved agent profiles.

        The previous selection is kept when it still exists; otherwise the role's
        default profile id is used, then the first entry. Safe to call before the combos
        exist (getattr fallback) and from the profile editor's saved signal.
        """
        self._agent_profiles = load_agent_profiles()
        # The dict key doubles as the default profile id for that role.
        combos = {
            "architect": getattr(self, "_architect_profile_combo", None),
            "coder": getattr(self, "_coder_profile_combo", None),
            "tester": getattr(self, "_tester_profile_combo", None),
        }
        for default_id, combo in combos.items():
            # Skip roles whose combo has not been created yet.
            if combo is None:
                continue
            # Remember the current choice so a refresh does not reset the user's selection.
            previous = combo.currentData() or default_id
            # Block signals so clearing/refilling does not emit change events.
            combo.blockSignals(True)
            combo.clear()
            # Sorted by profile id for a stable order; display text is icon + name.
            for profile_id, profile in sorted(self._agent_profiles.items()):
                combo.addItem(f"{profile.get('icon', '')} {profile.get('name', profile_id)}".strip(), profile_id)
            # Fall back: previous selection, then role default, then index 0.
            index = combo.findData(previous)
            if index < 0:
                index = combo.findData(default_id)
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.blockSignals(False)

    def _launch(self):
        """Validate inputs, build hyperparameters, start the orchestrator thread.

        Objective and workspace path are mandatory; the verification command may be empty.
        """
        # Read user input and trim whitespace so blank-only input counts as empty.
        objective = self._objective_input.toPlainText().strip()
        workspace_path = self._workspace_input.text().strip()
        test_command = self._test_command_input.text().strip()
        # Refuse to launch without the two required fields and tell the user why.
        if not objective or not workspace_path:
            QMessageBox.warning(self, "Swarm Studio", "Objective and workspace path are required.")
            return

        # Clear the previous run's widgets before recording the new run.
        self._reset_run_ui()
        # History is saved before the thread starts, so the objective is kept even if the run fails right away.
        self._save_history_item(objective, workspace_path, test_command)
        # Mark the app as running the swarm and remember the inputs so they survive a restart.
        self.state.swarm_running = True
        # Keeps the objective and workspace for later sessions (see how the workspace field is initialised in _build_left_panel).
        self.state.swarm_last_objective = objective
        self.state.swarm_last_workspace = workspace_path

        # Settings handed to the orchestrator. Temperature is converted from percent to 0-1.
        hyperparams = {
            "temperature": self._temp_spin.value() / 100.0,
            # max_tokens is passed through unchanged.
            "max_tokens": self._tokens_spin.value(),
            "candidates_per_task": self._candidates_spin.value(),
            # Feature toggles come straight from the checkable buttons.
            "enable_memory": self._memory_check.isChecked(),
            "enable_specialists": self._specialists_check.isChecked(),
            "enable_critic": self._critic_check.isChecked(),
            "adaptive_concurrency": self._adaptive_check.isChecked(),
            "pause_on_step": self._pause_on_step_check.isChecked(),
            # Selected profile ids per role; fall back to the role name when a combo has no data.
            "agent_profiles": {
                "architect": self._architect_profile_combo.currentData() or "architect",
                "coder": self._coder_profile_combo.currentData() or "coder",
                "tester": self._tester_profile_combo.currentData() or "tester",
            },
        }
        # Wire every orchestrator signal to its slot. All slots run on the GUI thread.
        self._thread = SwarmOrchestratorThread(workspace_path, objective, test_command, hyperparams)
        # Progress, plan and layer events.
        self._thread.status_update.connect(self._on_status)
        self._thread.task_plan_created.connect(self._on_plan)
        self._thread.dependency_layers_built.connect(self._on_layers)
        self._thread.layer_started.connect(self._on_layer_started)
        self._thread.layer_finished.connect(self._on_layer_finished)
        # Per-task and verification events.
        self._thread.task_status_changed.connect(self._on_task_status)
        self._thread.verification_started.connect(self._on_verification_started)
        self._thread.proposal_verification_finished.connect(self._on_proposal_verification_finished)
        self._thread.traceback_captured.connect(self._on_traceback)
        self._thread.verification_failed.connect(self._on_verification_failed)
        # Edits, results and completion.
        self._thread.edits_proposed.connect(self._on_edits_proposed)
        self._thread.file_edited.connect(self._on_file_edited)
        self._thread.test_result.connect(self._on_test_result)
        self._thread.finished_swarm.connect(self._on_finished)
        # Token streaming and multiverse (candidate) events.
        self._thread.coder_token.connect(self._on_coder_token)
        self._thread.candidates_generated.connect(self._on_candidates_generated)
        self._thread.candidate_scored.connect(self._on_candidate_scored)
        self._thread.winner_selected.connect(self._on_winner_selected)
        # Memory, specialist and concurrency events.
        self._thread.memory_recalled.connect(self._on_memory_recalled)
        self._thread.specialist_review.connect(self._on_specialist_review)
        self._thread.concurrency_adjusted.connect(self._on_concurrency_adjusted)
        # Steering, debugger and telemetry events.
        self._thread.guidance_injected.connect(self._on_guidance_injected)
        self._thread.swarm_paused.connect(self._on_swarm_paused)
        self._thread.telemetry_update.connect(self._on_telemetry_update)
        # Free the QThread object after it finishes.
        self._thread.finished.connect(self._thread.deleteLater)

        # Flip button states: launching is blocked while running, stop is enabled.
        self._launch_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._started_at = time.time()
        # Refresh the runtime label every 250ms.
        self._timer.start(250)
        self._thread.start()

    def _reset_run_ui(self):
        """Clear every widget and cache that belongs to a previous run so a new run starts clean."""
        # Dictionaries first, then widgets, then labels.
        self._task_rows.clear()
        self._file_contents.clear()
        self._tracebacks.clear()
        self._layers_tree.clear()
        # The scene may be None only if construction failed; guard keeps this safe.
        if self._graph_scene is not None:
            self._graph_scene.clear()
        self._task_nodes.clear()
        self._task_table.setRowCount(0)
        self._status_log.clear()
        self._preview.clear()
        self._traceback_browser.clear()
        self._cherry_pick_list.clear()
        self._cherry_pick_group.setVisible(False)
        self._commit_btn.setEnabled(False)
        self._stream_view.clear()
        # Reset per run; also initialised in __init__ so it always exists.
        self._stream_current_file = None
        self._cognition_log.clear()
        self._active_filepaths = []
        self._progress.setValue(0)
        self._layer_lbl.setText("Layers: 0")
        self._task_lbl.setText("Tasks: 0")
        self._verify_lbl.setText("Verification: idle")
        self._result_banner.setText("Run active.")
        # Empty stylesheet removes the success/failure colouring from the last run.
        self._result_banner.setStyleSheet("")
        self._tps_chart.reset()
        # Step/Resume are re-enabled only when the swarm reports it is paused.
        self._debug_step_btn.setEnabled(False)
        self._debug_resume_btn.setEnabled(False)
        self._debug_active_step_lbl.setText("No task currently paused.")

    def _stop(self):
        """Ask the running thread to stop (cooperative; it stops at its next checkpoint) and log it."""
        # Only act when a thread exists and is still running; otherwise do nothing.
        if self._thread and self._thread.isRunning():
            self._thread.request_stop()
            self._status_log.append("[Swarm] Stop requested.")

    def _tick(self):
        """Timer slot: update the elapsed-time label while a run is in progress."""
        # _started_at is 0.0 until the first launch, so the label is only updated after a run has started.
        if self._started_at:
            elapsed = time.time() - self._started_at
            # One decimal place keeps the label from jittering.
            self._runtime_lbl.setText(f"running {elapsed:.1f}s")

    def _on_status(self, message: str):
        """Append a status message from the swarm to the status log."""
        self._status_log.append(message)

    def _on_plan(self, plan: dict):
        """Show the planned tasks: update the task count and add a 'planned' row per file."""
        # tasks may be missing, so default to an empty list.
        tasks = plan.get("tasks", [])
        self._task_lbl.setText(f"Tasks: {len(tasks)}")
        for task in tasks:
            # A plan task starts as 'planned' with no layer yet; instructions become the detail text.
            self._ensure_task_row(task.get("filepath", ""), "planned", "", task.get("instructions", ""))

    def _on_layers(self, layers: list):
        """Show the dependency layers in the tree and graph, and update the counters.

        layers is a list of layers, each a list of task dicts with 'filepath' and 'instructions'.
        """
        self._layers_tree.clear()
        # Rebuilding from scratch keeps the tree in sync with the latest plan.
        total_tasks = sum(len(layer) for layer in layers)
        self._layer_lbl.setText(f"Layers: {len(layers)}")
        self._task_lbl.setText(f"Tasks: {total_tasks}")
        for idx, layer in enumerate(layers, start=1):
            # Top-level item per layer; the layer number is stored for _layer_for_file.
            # Parent item = one layer.
            parent = QTreeWidgetItem(self._layers_tree)
            parent.setText(0, f"Layer {idx}: {len(layer)} task(s)")
            parent.setData(0, Qt.ItemDataRole.UserRole, idx)
            for task in layer:
                child = QTreeWidgetItem(parent)
                child.setText(0, task.get("filepath", "unknown"))
                # Tooltip shows the task's instructions.
                child.setToolTip(0, task.get("instructions", ""))
            parent.setExpanded(True)
        # Draw the node graph last so it matches the tree.
        self._render_dependency_graph(layers)

    def _on_layer_started(self, index: int, total: int, tasks: list):
        """A layer began coding: update the verification label and mark its tasks in_progress."""
        # Index is 1-based as sent by the thread.
        self._verify_lbl.setText(f"Layer {index}/{total}: coding")
        for task in tasks:
            # Reuse _on_task_status so the table, graph and progress all update the same way.
            self._on_task_status(task.get("filepath", ""), "in_progress", f"Layer {index} started")

    def _on_layer_finished(self, index: int, success: bool, summary: str):
        """A layer finished: show verified/failed and append the thread's summary to the log."""
        # The label summarises success or failure only; the summary text goes to the log.
        self._verify_lbl.setText(f"Layer {index}: {'verified' if success else 'failed'}")
        self._status_log.append(f"[Layer] {summary}")

    def _on_task_status(self, filepath: str, status: str, detail: str):
        """Update a task's table row, overall progress, graph colour and steering target.

        Progress counts rows whose status is 'completed' or 'failed' ('skipped' does not
        count). While a file is in progress it becomes a candidate steering target.
        """
        # Row layer text is looked up from the tree, which is built by _on_layers earlier.
        self._ensure_task_row(filepath, status, self._layer_for_file(filepath), detail)
        completed = 0
        # max(1, ...) avoids division by zero when no rows exist yet.
        total = max(1, len(self._task_rows))
        for row in self._task_rows.values():
            item = self._task_table.item(row, 1)
            # Column 1 is Status; item() can be None for rows without items.
            if item and item.text() in {"completed", "failed"}:
                completed += 1
        # Progress shown as an integer percentage.
        self._progress.setValue(int((completed / total) * 100))
        # Recolour the graph node to match.
        self._update_graph_node_status(filepath, status)
        # Track active files; only auto-fill the steering box when it is empty so the user's choice is never overwritten.
        if status == "in_progress":
            if filepath not in self._active_filepaths:
                self._active_filepaths.append(filepath)
            if not self._steering_target.text().strip():
                self._steering_target.setText(filepath)
        # When a file ends, drop it from the active list; if it was the steering target, move to another active file (or clear).
        elif filepath in self._active_filepaths and status in {"completed", "failed", "skipped"}:
            self._active_filepaths.remove(filepath)
            if self._steering_target.text().strip() == filepath:
                self._steering_target.setText(self._active_filepaths[0] if self._active_filepaths else "")

    # ── Swarm 2.0 telemetry ──────────────────────────────────────────────────

    def _log_intel(self, line: str):
        """Append one line to the Swarm Intelligence log."""
        # Plain text is appended as a new line.
        self._cognition_log.appendPlainText(line)

    def _on_candidates_generated(self, filepath: str, count: int):
        """Log that N candidate solutions are being generated for a file (multiverse mode)."""
        self._log_intel(f"[Multiverse] {filepath}: generating {count} candidate solutions...")

    def _on_candidate_scored(self, filepath: str, index: int, score: dict):
        """Log one candidate's score breakdown; the tick/cross reflects the syntax check."""
        # Syntax OK -> tick, otherwise cross.
        ok = "✓" if score.get("syntax_ok") else "✗"
        self._log_intel(
            f"  candidate {index} [{ok}] score={score.get('total_score')} "
            f"lint={score.get('lint_violations')} diff={score.get('diff_size')} "
            f"align={score.get('signature_alignment')}"
        )

    def _on_winner_selected(self, filepath: str, index: int, reason: str):
        """Log which candidate won and why."""
        self._log_intel(f"[Multiverse] {filepath}: winner is candidate {index} ({reason})")

    def _on_memory_recalled(self, filepath: str, text: str):
        """Log past failure patterns recalled from cross-run memory."""
        self._log_intel(f"[Memory] {filepath}: recalled past pattern(s):\n{text}")

    def _on_specialist_review(self, filepath: str, specialist: str, result: dict):
        """Log a specialist reviewer's verdict, risk score and concerns (or 'clean')."""
        # Verdict defaults to '?' if the specialist omitted it.
        verdict = result.get("verdict", "?")
        concerns = result.get("concerns", [])
        if concerns:
            # With concerns, join them on one line after the risk score.
            self._log_intel(
                f"[{specialist.title()}] {filepath}: verdict={verdict} risk={result.get('risk_score')} — "
                + "; ".join(concerns)
            )
        else:
            self._log_intel(f"[{specialist.title()}] {filepath}: clean (verdict={verdict})")

    def _on_concurrency_adjusted(self, workers: int, reason: str):
        """Log the adaptive worker-pool size and reason."""
        self._log_intel(f"[Concurrency] {workers} parallel workers — {reason}")

    def _on_guidance_injected(self, filepath: str, message: str):
        """Echo a steering message that was injected into a task."""
        self._log_intel(f"[Steering] → {filepath}: \"{message}\"")

    def _on_send_guidance(self):
        """Send the typed correction to the target file's agent; ignored if no run is active or either field is empty."""
        # Needs an active run; otherwise there is nobody to receive the guidance.
        if not self._thread or not self._thread.isRunning():
            return
        target = self._steering_target.text().strip()
        message = self._steering_input.text().strip()
        # Both a target and a message are required.
        if not target or not message:
            return
        # inject_guidance is thread-safe on the orchestrator side; the field is cleared on success so the same text is not sent twice.
        self._thread.inject_guidance(target, message)
        self._steering_input.clear()

    # ── Interactive debugger ─────────────────────────────────────────────────

    def _on_debug_pause(self):
        """Request a pause (takes effect at the next task boundary)."""
        if self._thread and self._thread.isRunning():
            # pause() only requests a pause; the swarm acknowledges it by emitting swarm_paused.
            self._thread.pause()
            self._status_log.append("[Debugger] Pause requested — takes effect at the next task boundary.")

    def _on_debug_resume(self):
        """Resume a paused swarm and reset the debugger controls."""
        if self._thread and self._thread.isRunning():
            # If the thread is gone, still reset the buttons.
            self._thread.resume()
        self._debug_step_btn.setEnabled(False)
        self._debug_resume_btn.setEnabled(False)
        self._debug_active_step_lbl.setText("No task currently paused.")
        self._status_log.append("[Debugger] Resumed.")

    def _on_debug_step(self):
        """Run exactly one task forward from a paused state, then pause again."""
        if self._thread and self._thread.isRunning():
            # step() lets exactly one task run, then the swarm pauses again.
            self._thread.step()
        self._debug_step_btn.setEnabled(False)
        self._debug_resume_btn.setEnabled(False)
        self._status_log.append("[Debugger] Stepping one task forward.")

    def _on_swarm_paused(self, task_info: dict):
        """Swarm paused before a task: show what is about to run and enable Step/Resume."""
        # task_info comes from the thread: 'filepath' and 'instructions'.
        filepath = task_info.get("filepath", "")
        instructions = task_info.get("instructions", "")
        self._debug_active_step_lbl.setText(f"Paused before: {filepath}\n{instructions}")
        self._debug_step_btn.setEnabled(True)
        self._debug_resume_btn.setEnabled(True)
        # Fill the steering target only when empty.
        if not self._steering_target.text().strip():
            self._steering_target.setText(filepath)
        self._status_log.append(f"[Debugger] Paused before {filepath}.")

    def _on_telemetry_update(self, payload: dict):
        """Feed a telemetry payload (tokens_per_second, cumulative_tokens) into the chart."""
        # Missing keys default to 0.
        self._tps_chart.add_sample(
            payload.get("tokens_per_second", 0.0), payload.get("cumulative_tokens", 0)
        )

    def _on_verification_started(self, layer_index: int, command: str):
        """Show that verification started for a layer, with the command used."""
        # Shows which layer is being verified.
        self._verify_lbl.setText(f"Layer {layer_index}: verifying")
        self._status_log.append(f"[Tester] Layer {layer_index}: {command}")

    def _on_proposal_verification_finished(self, layer_index: int, passed: bool, trace: str):
        """A pre-approval dry-run of a layer finished.

        The trace is always stored, but it is only shown immediately when the dry-run failed.
        """
        # Status text used in both the label and the log line.
        status = "passed" if passed else "failed"
        self._verify_lbl.setText(f"Layer {layer_index}: dry-run {status}")
        self._status_log.append(f"[Tester] Layer {layer_index} dry-run {status} before approval.")
        if trace:
            # Key under which the dry-run trace is stored; also used by _layer_for_file lookups.
            key = f"Layer {layer_index} dry-run"
            self._tracebacks[key] = trace
            # Passing dry-runs are stored but not displayed, to avoid hiding an existing failure.
            if not passed:
                self._traceback_browser.setHtml(self._format_traceback_html(trace, key))

    def _on_traceback(self, key: str, trace: str):
        """Store a traceback under the given key and show it."""
        # Overwrite any earlier traceback with the same key.
        self._tracebacks[key] = trace
        self._traceback_browser.setHtml(self._format_traceback_html(trace, key))

    def _on_file_edited(self, filepath: str, content: str):
        """A file was written: cache its content, mark the row 'written' and show it in the preview."""
        # Record content so selecting the row later can show it.
        self._file_contents[filepath] = content
        self._ensure_task_row(filepath, "written", self._layer_for_file(filepath), "File content written")
        self._preview.setPlainText(content)

    def _on_test_result(self, passed: bool, trace: str):
        """Final test result: a green message on pass, or a formatted traceback on failure."""
        if passed:
            # Green 'passed' message in the same monospaced container style.
            self._traceback_browser.setHtml(
                '<div style="font-family:Monospace,monospace;padding:8px;color:#4CAF50;">Verification passed.</div>'
            )
        else:
            self._traceback_browser.setHtml(self._format_traceback_html(trace))

    def _on_finished(self, success: bool, summary: str):
        """Run finished: stop the timer, restore buttons and show the result banner.

        On success progress jumps to 100 and the banner is green; on failure it keeps the
        current progress and is red. Drops the thread reference (deleteLater frees it).
        """
        # Tell the rest of the app the swarm is no longer running.
        self.state.swarm_running = False
        # Stop the runtime timer; the label is reset next.
        self._timer.stop()
        self._runtime_lbl.setText("idle")
        self._launch_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        # On failure keep the current value so the user sees how far it got.
        self._progress.setValue(100 if success else self._progress.value())
        self._result_banner.setText(summary)
        # Green for success, red for failure.
        if success:
            self._result_banner.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self._result_banner.setStyleSheet("color: #FF5C7A; font-weight: bold;")
        # Release the reference; the QThread object is destroyed via deleteLater.
        self._thread = None

    def _on_coder_token(self, filepath: str, token: str):
        """Stream one token of coder output into the live code view.

        A header is written whenever the file changes, so interleaved output from
        parallel coders may repeat headers.
        """
        # Set header on first token for this file
        # _stream_current_file is created in _reset_run_ui; getattr keeps this safe if a token arrives first.
        current_header = getattr(self, '_stream_current_file', None)
        if current_header != filepath:
            self._stream_current_file = filepath
            # A header line precedes the first token of each file.
            self._stream_view.appendPlainText(f"\n── {filepath} ──")
        # Move the cursor to the end before inserting so tokens append rather than overwrite wherever the cursor was.
        self._stream_view.moveCursor(QTextCursor.MoveOperation.End)
        # insertPlainText inserts at the cursor, which is now the end.
        self._stream_view.insertPlainText(token)

    def _ensure_task_row(self, filepath: str, status: str, layer: str, detail: str):
        """Create or update the table row for a file.

        Empty filepaths are ignored. New files get a row (recorded in _task_rows);
        existing files get their status, layer and detail updated in place.
        """
        # Empty filepaths are ignored.
        if not filepath:
            return
        # A new file gets a row appended at the bottom; its index is stored for later updates.
        if filepath not in self._task_rows:
            row = self._task_table.rowCount()
            self._task_table.insertRow(row)
            self._task_rows[filepath] = row
            self._task_table.setItem(row, 0, QTableWidgetItem(filepath))
            self._task_table.setItem(row, 1, QTableWidgetItem(status))
            self._task_table.setItem(row, 2, QTableWidgetItem(layer))
            self._task_table.setItem(row, 3, QTableWidgetItem(detail))
        else:
            # Known file: update the cells in place.
            row = self._task_rows[filepath]
            self._task_table.item(row, 1).setText(status)
            self._task_table.item(row, 2).setText(layer)
            self._task_table.item(row, 3).setText(detail)
        # Resize on every update so the columns fit the latest text.
        self._task_table.resizeColumnsToContents()

    def _render_dependency_graph(self, layers: list[list[dict]]) -> None:
        """Build a node-link graph from dependency layers into _graph_scene."""
        # Without a scene there is nothing to draw on.
        if self._graph_scene is None:
            return
        # Rebuild from scratch every time layers arrive.
        self._graph_scene.clear()
        self._task_nodes.clear()

        # Geometry: each node is 160x36; 60px between layers, 14px between nodes in a layer.
        node_w, node_h = 160, 36
        h_gap, v_gap = 60, 14
        layer_step = node_w + h_gap

        # Fill and border colours per status. The same palette is repeated in _update_graph_node_status, so keep them in sync.
        status_colors = {
            "planned": QColor("#1E1E3C"),
            "in_progress": QColor("#7A4800"),
            "written": QColor("#005A4A"),
            "completed": QColor("#174430"),
            "failed": QColor("#5A0020"),
        }
        border_colors = {
            "planned": QColor("#35356E"),
            "in_progress": QColor("#FFB400"),
            "written": QColor("#00C2A0"),
            "completed": QColor("#2DD4A0"),
            "failed": QColor("#FF3366"),
        }
        text_color = QColor("#C8C8E0")
        # Remember each node's centre per layer so edges can be drawn afterwards.
        layer_centers: list[list[tuple[str, float, float]]] = []

        # One column per layer.
        for layer_idx, layer_tasks in enumerate(layers):
            # Layers are laid out in columns left to right; each column is vertically centred around y = 0.
            x = layer_idx * layer_step
            centers = []
            # Total column height; subtract the final gap.
            total_h = len(layer_tasks) * (node_h + v_gap) - v_gap
            y_start = -total_h / 2

            for task_idx, task in enumerate(layer_tasks):
                filepath = task.get("filepath", "")
                # Skip tasks without a filepath; they cannot be keyed in _task_nodes.
                if not filepath:
                    continue
                # Show only the file's base name, truncated to 20 chars; the full path is in the tooltip.
                label = filepath.split("/")[-1] if "/" in filepath else filepath
                # Vertical offset by task index.
                y = y_start + task_idx * (node_h + v_gap)

                rect = QGraphicsRectItem(QRectF(x, y, node_w, node_h))
                # All nodes start as 'planned'; _update_graph_node_status recolours them as events arrive.
                rect.setBrush(QBrush(status_colors["planned"]))
                rect.setPen(QPen(border_colors["planned"], 1.5))
                rect.setToolTip(filepath)
                self._graph_scene.addItem(rect)
                self._task_nodes[filepath] = rect

                # Text items sit on top of the rectangles (z=1) and are centred inside them.
                text = self._graph_scene.addText(
                    label if len(label) <= 20 else label[:18] + "...",
                    QFont("Monospace", 7),
                )
                text.setDefaultTextColor(text_color)
                # Use the text's own size to centre it inside its rectangle.
                text_rect = text.boundingRect()
                text.setPos(
                    x + (node_w - text_rect.width()) / 2,
                    y + (node_h - text_rect.height()) / 2,
                )
                text.setZValue(1)

                centers.append((filepath, x + node_w / 2, y + node_h / 2))

            layer_centers.append(centers)

        # Dashed connectors between consecutive layers. Every node in layer i-1 is joined
        # to every node in layer i; this shows layer ordering, not exact per-file dependencies.
        edge_pen = QPen(QColor("#2A2A50"), 1, Qt.PenStyle.DashLine)
        for i in range(1, len(layer_centers)):
            # Right edge of the previous layer's nodes and left edge of the next layer's.
            src_x_right = (i - 1) * layer_step + node_w
            dst_x_left = i * layer_step
            # Underscore-prefixed names mark the unused tuple parts (filepath and x).
            for _a, _cx_a, cy_a in layer_centers[i - 1]:
                for _b, _cx_b, cy_b in layer_centers[i]:
                    line = QGraphicsLineItem(src_x_right, cy_a, dst_x_left, cy_b)
                    line.setPen(edge_pen)
                    # Negative z keeps lines behind the nodes.
                    line.setZValue(-1)
                    self._graph_scene.addItem(line)

        # Zoom the view to fit the whole graph, but only when there is something to show.
        if self._graph_scene.items():
            self._graph_view.fitInView(
                self._graph_scene.itemsBoundingRect(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )

    def _update_graph_node_status(self, filepath: str, status: str) -> None:
        """Recolour a graph node for the given status; unknown statuses (for example 'skipped') fall back to the 'planned' colours."""
        # No node means the file is not in the graph (for example, planned before layers arrived).
        node = self._task_nodes.get(filepath)
        if node is None:
            return
        # Same palette as in _render_dependency_graph.
        status_fill = {
            "planned": QColor("#1E1E3C"),
            "in_progress": QColor("#7A4800"),
            "written": QColor("#005A4A"),
            "completed": QColor("#174430"),
            "failed": QColor("#5A0020"),
        }
        status_border = {
            "planned": QColor("#35356E"),
            "in_progress": QColor("#FFB400"),
            "written": QColor("#00C2A0"),
            "completed": QColor("#2DD4A0"),
            "failed": QColor("#FF3366"),
        }
        # Brush is the fill, pen is the border.
        node.setBrush(QBrush(status_fill.get(status, status_fill["planned"])))
        node.setPen(QPen(status_border.get(status, status_border["planned"]), 1.5))

    def _layer_for_file(self, filepath: str) -> str:
        """Return the layer number (as text) containing the file, or '' if it is not in any layer."""
        # Walk top-level layers, then their children, comparing filepaths.
        for idx in range(self._layers_tree.topLevelItemCount()):
            parent = self._layers_tree.topLevelItem(idx)
            # The layer number stored in UserRole by _on_layers.
            layer_num = parent.data(0, Qt.ItemDataRole.UserRole)
            for child_idx in range(parent.childCount()):
                if parent.child(child_idx).text(0) == filepath:
                    return str(layer_num)
        return ""

    def _sync_selection_preview(self):
        """Show the selected task row's file content and traceback in the right-hand previews."""
        # selectionModel() can be None, so guard before asking for rows.
        rows = self._task_table.selectionModel().selectedRows() if self._task_table.selectionModel() else []
        if not rows:
            return
        # Use the first selected row.
        row = rows[0].row()
        # Column 0 is File.
        filepath_item = self._task_table.item(row, 0)
        if not filepath_item:
            return
        filepath = filepath_item.text()
        # Use the cached content, or tell the user nothing has been captured yet.
        self._preview.setPlainText(self._file_contents.get(filepath, "No file content captured yet."))
        # Prefer a file-specific traceback, then its layer's traceback.
        trace = (
            self._tracebacks.get(filepath)
            or self._tracebacks.get(f"Layer {self._layer_for_file(filepath)}", "")
        )
        if trace:
            # Show with the filepath as the label.
            self._traceback_browser.setHtml(self._format_traceback_html(trace, filepath))
        else:
            # No traceback for this file -> clear the browser.
            self._traceback_browser.clear()

    # ── Cherry-pick helpers ──────────────────────────────────────────────────

    def _on_edits_proposed(self, edits: list):
        """Show the proposed edits in the cherry-pick list, all checked by default.

        Each edit dict needs 'filepath' and 'content'. The user then reviews and presses Commit.
        """
        self._cherry_pick_list.clear()
        # Each edit is a dict with 'filepath' and 'content'.
        for edit in edits:
            item = QListWidgetItem(edit["filepath"])
            # Make the row checkable and keep the proposed content as item data for previewing.
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            # Content is stored per item so previews do not need a lookup.
            item.setData(Qt.ItemDataRole.UserRole, edit["content"])
            self._cherry_pick_list.addItem(item)
        # Commit is enabled and the panel is shown.
        self._commit_btn.setEnabled(True)
        self._cherry_pick_group.setVisible(True)
        self._status_log.append("[Swarm] Edits proposed — review and commit to proceed.")

    def _on_cherry_pick_item_clicked(self, item: QListWidgetItem):
        """Preview the proposed content of the clicked cherry-pick item."""
        # Empty or missing content is ignored.
        content = item.data(Qt.ItemDataRole.UserRole)
        if content:
            self._preview.setPlainText(content)

    def _cherry_select_all(self):
        """Check every proposed edit."""
        # Loop over every row index.
        for i in range(self._cherry_pick_list.count()):
            self._cherry_pick_list.item(i).setCheckState(Qt.CheckState.Checked)

    def _cherry_select_none(self):
        """Uncheck every proposed edit."""
        for i in range(self._cherry_pick_list.count()):
            self._cherry_pick_list.item(i).setCheckState(Qt.CheckState.Unchecked)

    def _on_commit_edits(self):
        """Send the checked edits to the orchestrator and hide the cherry-pick panel.

        Only the checked file paths are sent; the unchecked count is reported as skipped.
        """
        # Nothing to commit to when there is no active thread.
        if not self._thread:
            return
        # Collect the filepaths of rows that are checked.
        selected = [
            self._cherry_pick_list.item(i).text()
            for i in range(self._cherry_pick_list.count())
            if self._cherry_pick_list.item(i).checkState() == Qt.CheckState.Checked
        ]
        # Count of unchecked edits.
        skipped = self._cherry_pick_list.count() - len(selected)
        # The orchestrator applies only the selected edits.
        self._thread.commit_selected_edits(selected)
        # Commit is disabled and the panel hidden so the same edits cannot be committed twice.
        self._commit_btn.setEnabled(False)
        self._cherry_pick_group.setVisible(False)
        # Build the log message; mention skipped edits only when any were skipped.
        msg = f"[Swarm] Committed {len(selected)} edit(s)"
        if skipped:
            msg += f", skipped {skipped}"
        self._status_log.append(msg + ".")

    # ── Verification traceback viewer ────────────────────────────────────────

    def _on_verification_failed(self, context: str, traceback_text: str):
        """Show a verification failure's traceback, labelled with its context."""
        self._traceback_browser.setHtml(self._format_traceback_html(traceback_text, context))

    def _format_traceback_html(self, trace: str, context: str = "") -> str:
        """Return the traceback as colour-highlighted HTML for the QTextBrowser.

        File lines are cyan, exception lines red/bold, caret/tilde marker lines orange.
        All text is HTML-escaped first so traceback content cannot inject markup.
        """
        # Imported locally; only needed in this method.
        import html as _html
        # Line prefixes treated as exception headings (including the '# SYNTAX' marker the swarm uses for syntax errors).
        _exc_starts = (
            "Traceback", "Error", "Exception", "SyntaxError", "TypeError",
            "ValueError", "KeyError", "IndexError", "AttributeError",
            "NameError", "ImportError", "AssertionError", "JSONDecodeError",
            "# SYNTAX",
        )
        # Open a monospaced dark container div; closed after the loop.
        parts = [
            '<div style="font-family:Monospace,monospace;font-size:9pt;'
            'background:#0A0A14;padding:8px;color:#C0C0D0;">'
        ]
        if context:
            parts.append(f'<b style="color:#FFB400;">[{_html.escape(context)}]</b><br/>')
        # Classify each line; checks are ordered, the first match wins.
        for line in trace.splitlines():
            # Preserve indentation in the output; strip only for matching.
            stripped = line.strip()
            escaped = _html.escape(line)
            # Frame lines look like: File "x.py", line N.
            if stripped.startswith("File ") and ("line " in stripped or '", line' in stripped):
                parts.append(f'<span style="color:#7FDBFF;">{escaped}</span><br/>')
            # Exception headings start with one of the known prefixes.
            elif any(stripped.startswith(p) for p in _exc_starts):
                parts.append(f'<b style="color:#FF5C7A;">{escaped}</b><br/>')
            # Caret / tilde lines point at the faulty code.
            elif stripped.startswith("^") or "~~~" in stripped:
                parts.append(f'<span style="color:#FF9900;">{escaped}</span><br/>')
            else:
                # Everything else is plain text.
                parts.append(f'{escaped}<br/>')
        parts.append("</div>")
        return "".join(parts)

    def closeEvent(self, event):
        """On close, ask a running swarm to stop and wait for it so the thread is not destroyed while running."""
        if self._thread and self._thread.isRunning():
            # Asking for a stop first lets the thread finish its current step.
            self._thread.request_stop()
            # Bounded wait so a hung model call cannot freeze window close forever.
            if not self._thread.wait(3000):
                # Last resort: destroying a still-running QThread aborts the process, so force it down.
                self._thread.terminate()
                self._thread.wait(1000)
        # Always run the parent close handler.
        super().closeEvent(event)
