# ==============================================================================
# Program: command_palette.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Reusable desktop widget component implementing command palette.
#          Enforces styling standards, event handling, and visual telemetry
#          across Karl offline workspaces.
# ==============================================================================

from PyQt6.QtWidgets import QDialog, QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QKeyEvent

# Modal Ctrl+K / Ctrl+P launcher: a search box over a list of commands that
# each map to a MainWindow method. It deliberately uses a frameless dialog so
# it looks like a floating overlay.
class CommandPalette(QDialog):
    def __init__(self, main_window, parent=None):
        """Build the palette UI and the static command table.

        `main_window` supplies the actions (switch_workspace, start_new_workbench_session, ...).
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(parent or main_window)
        self.main_window = main_window
        # Frameless + translucent: the palette draws its own border via stylesheet.
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setModal(True)
        self.setFixedSize(QSize(500, 320))

        # Build UI
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)

        self.input_edit = QLineEdit(self)
        self.input_edit.setObjectName("command-input")
        self.input_edit.setPlaceholderText("Search commands or switch workspaces... (Esc to close)")
        # Colours are hard-coded (not themed) so the palette looks the same in every
        # theme and stays readable even if the app stylesheet fails to load.
        self.input_edit.setStyleSheet(
            "background-color: #0D0D1B; border: 1px solid #00E5FF; border-radius: 4px; "
            "color: #F0F5FF; font-family: 'JetBrains Mono', monospace; font-size: 10pt; "
            "padding: 10px; margin: 4px;"
        )
        self.input_edit.textChanged.connect(self._filter_commands)
        layout.addWidget(self.input_edit)

        self.list_widget = QListWidget(self)
        self.list_widget.setStyleSheet(
            "QListWidget { background-color: #0D0D1B; border: 1px solid #1F1F3D; border-top: none; "
            "border-radius: 0 0 4px 4px; color: #A0AEC0; font-family: 'JetBrains Mono', monospace; "
            "font-size: 9.5pt; padding: 4px; }"
            "QListWidget::item { padding: 8px; border-radius: 3px; }"
            "QListWidget::item:selected { background-color: #00C2FF; color: #020205; font-weight: bold; }"
            "QListWidget::item:hover { background-color: #14142D; color: #F0F5FF; }"
        )
        self.list_widget.itemDoubleClicked.connect(self._execute_selected)
        layout.addWidget(self.list_widget)

        # (title, callable) pairs. Titles are what the user searches, so keep them
        # descriptive. Workspace indexes 0-9 mirror the sidebar order; lambdas are safe
        # here because each literal index is fixed, not a loop variable.
        self._commands = [
            ("Switch to Workbench (Chat Space)", lambda: self.main_window.switch_workspace(0)),
            ("Switch to Prompt Lab (Playground)", lambda: self.main_window.switch_workspace(1)),
            ("Switch to Knowledge Base (RAG)", lambda: self.main_window.switch_workspace(2)),
            ("Switch to Vision Workbench", lambda: self.main_window.switch_workspace(3)),
            ("Switch to Training Studio", lambda: self.main_window.switch_workspace(4)),
            ("Switch to Eval Suite (Benchmarks)", lambda: self.main_window.switch_workspace(5)),
            ("Switch to Swarm Studio", lambda: self.main_window.switch_workspace(6)),
            ("Switch to System Config (Settings)", lambda: self.main_window.switch_workspace(7)),
            ("Switch to Codex (Documentation)", lambda: self.main_window.switch_workspace(8)),
            ("Switch to Flywheel Studio", lambda: self.main_window.switch_workspace(9)),
            ("Workbench: Start New Session", self.main_window.start_new_workbench_session),
            ("Workbench: Save Session", self.main_window.save_workbench_session),
            ("Workbench: Toggle RAG Pipeline", self.main_window.toggle_workbench_rag),
            ("Workbench: Toggle Agentic Loop", self.main_window.toggle_workbench_agentic_loop),
            ("Knowledge Base: Ingest Document", self.main_window.open_knowledge_ingest),
            ("Knowledge Base: Rebuild Search Index", self.main_window.rebuild_knowledge_index),
            ("Eval Suite: Run Benchmarks", self.main_window.run_eval_suite),
            ("System Config: Open Defaults Tab", self.main_window.open_system_defaults),
        ]

        self._populate_list()
        self.input_edit.setFocus()

    def _populate_list(self):
        """Fill the list with every command and select the first row."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.list_widget.clear()
        for title, action in self._commands:
            # Iterate over items, evaluate loop invariant, and update progress
            item = QListWidgetItem(title)
            # Store the callable on the item so execution does not depend on the title.
            item.setData(Qt.ItemDataRole.UserRole, action)
            self.list_widget.addItem(item)
        self.list_widget.setCurrentRow(0)

    def _filter_commands(self, text):
        """Show only commands whose title contains the typed text (case-insensitive)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        query = text.strip().lower()
        self.list_widget.clear()
        for title, action in self._commands:
            # Iterate over items, evaluate loop invariant, and update progress
            if not query or query in title.lower():
                item = QListWidgetItem(title)
                item.setData(Qt.ItemDataRole.UserRole, action)
                self.list_widget.addItem(item)
        self.list_widget.setCurrentRow(0)

    def keyPressEvent(self, event: QKeyEvent):
        """Route Esc/Up/Down/Enter to the list while focus stays in the search box.

        Up/Down are handled manually because the QLineEdit would otherwise consume them.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
        elif event.key() == Qt.Key.Key_Up:
            row = self.list_widget.currentRow()
            if row > 0:
                self.list_widget.setCurrentRow(row - 1)
            event.accept()
        elif event.key() == Qt.Key.Key_Down:
            row = self.list_widget.currentRow()
            if row < self.list_widget.count() - 1:
                self.list_widget.setCurrentRow(row + 1)
            event.accept()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._execute_selected()
            event.accept()
        else:
            super().keyPressEvent(event)

    def _execute_selected(self):
        """Close the palette and run the highlighted command (if any).

        The dialog is accepted first so the action runs with the palette gone, since
        actions such as switching workspaces change focus.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        curr_item = self.list_widget.currentItem()
        if not curr_item:
            return
        
        title = curr_item.text()
        action = curr_item.data(Qt.ItemDataRole.UserRole)
        
        # Defensive: recover the action from the table if the item carries none.
        # Fallback to search list of commands if not set via filtering UserRole
        if not action:
            for t, act in self._commands:
                # Iterate over items, evaluate loop invariant, and update progress
                if t == title:
                    action = act
                    break
        
        self.accept()
        if action:
            action()

    def showEvent(self, event):
        """Place the palette near the top centre of the parent window when shown."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().showEvent(event)
        # Position command palette at top center of main window
        if self.parentWidget():
            parent_geom = self.parentWidget().geometry()
            x = parent_geom.x() + (parent_geom.width() - self.width()) // 2
            y = parent_geom.y() + 80
            self.move(x, y)
