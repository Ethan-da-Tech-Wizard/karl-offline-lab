# Program: karl-offline-lab - dataset_tab.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Qt tab for browsing curated training examples with a preview pane, deleting examples and merging another curated dataset file in on a background thread.
"""Dataset tab for the Training Studio."""
import html

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListView,
    QTextBrowser, QPushButton, QMessageBox, QLabel,
    QFileDialog, QProgressDialog,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QAbstractListModel, QModelIndex


class DatasetListModel(QAbstractListModel):
    """Simple list model that shows pre-formatted text rows for the dataset list view."""
    def __init__(self, dataset_list=None):
        """Store the initial list of display rows."""
        super().__init__()
        # Rows are plain strings built by DatasetTab.refresh.
        self._dataset = dataset_list or []

    def rowCount(self, parent=QModelIndex()):
        """Return the number of rows in the model."""
        return len(self._dataset)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        """Return the display text for a row; other roles give nothing."""
        if not index.isValid():
            return None
        row = index.row()
        if role == Qt.ItemDataRole.DisplayRole:
            item = self._dataset[row]
            return str(item)
        return None

    def get_item(self, row):
        """Return the row at the given index, or None when it is out of range."""
        if 0 <= row < len(self._dataset):
            return self._dataset[row]
        return None

    def update_data(self, new_dataset):
        # Bracket the data swap so views redraw correctly.
        """Replace all rows and notify attached views with a model reset."""
        self.beginResetModel()
        self._dataset = new_dataset
        self.endResetModel()


def _section(text: str) -> QLabel:
    """Return a styled section-header label."""
    lbl = QLabel(text)
    lbl.setObjectName("section-header")
    return lbl


# ── Background merge thread ────────────────────────────────────────────────────

class _MergeThread(QThread):
    """Runs DatasetMerger.merge_files() off the UI thread."""
    # Note: this signal name shadows QThread's built-in finished signal.
    finished = pyqtSignal(dict)   # stats dict on success
    error = pyqtSignal(str)       # error message on failure

    def __init__(self, primary: str, incoming: str, parent=None):
        """Remember the local (primary) and incoming dataset file paths."""
        super().__init__(parent)
        self._primary = primary
        self._incoming = incoming

    def run(self):
        """Merge the files and emit stats on success or the error message on failure."""
        try:
            from app.utils.dataset_merger import DatasetMerger
            stats = DatasetMerger.merge_files(self._primary, self._incoming)
            self.finished.emit(stats)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Dataset Tab ────────────────────────────────────────────────────────────────

class DatasetTab(QWidget):
    """Tab with the curated example list, a preview pane and import/delete actions."""
    dataset_changed = pyqtSignal()

    def __init__(self, state, parent=None):
        """Set up state, merge thread and dialog handles and build the UI."""
        super().__init__(parent)
        self.state = state
        self._merge_thread: _MergeThread | None = None
        self._progress_dialog: QProgressDialog | None = None
        self._build_ui()

    def _build_ui(self):
        """Lay out the example list with actions on the left and the preview on the right."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # ── Left Column: list + actions ────────────────────────────────────
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(6)
        ll.addWidget(_section("EXAMPLES"))

        self._example_list = QListView()
        self._list_model = DatasetListModel()
        self._example_list.setModel(self._list_model)
        # Selecting a row shows its preview.
        self._example_list.selectionModel().currentRowChanged.connect(self._on_row_changed)
        self._example_list.setToolTip("Double-click or select a curated example to preview")
        ll.addWidget(self._example_list, 1)

        # Import button — visually distinct from the destructive delete action
        import_btn = QPushButton("⬆  Import Team Traces…")
        import_btn.setObjectName("btn-secondary")
        import_btn.setToolTip(
            "Merge another developer's curated.jsonl into your local dataset.\n"
            "Duplicates are dropped; conflicts are resolved by timestamp."
        )
        import_btn.clicked.connect(self._import_team_traces)
        ll.addWidget(import_btn)

        del_btn = QPushButton("delete selected")
        del_btn.setObjectName("btn-danger")
        del_btn.setToolTip("Remove the selected curation example from training database")
        del_btn.clicked.connect(self._delete_selected)
        ll.addWidget(del_btn)

        layout.addWidget(left, 1)

        # ── Right Column: detail preview ───────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)
        rl.addWidget(_section("PREVIEW"))

        self._detail_view = QTextBrowser()
        self._detail_view.setPlaceholderText("Select an example to preview.")
        rl.addWidget(self._detail_view, 1)

        layout.addWidget(right, 2)

    def _on_row_changed(self, current, previous):
        """Show the preview for the newly selected row."""
        row = current.row()
        self._on_example_selected(row)

    # ── Refresh ────────────────────────────────────────────────────────────────

    def refresh(self):
        """Reload all curated examples into the list, labelled by source with a short user-prompt preview."""
        self._detail_view.clear()
        formatted_list = []
        for ex in self.state.curator.get_all_examples():
            source = ex.get("source", "unknown")
            if source == "thumbs_up":
                # Label each example by how it was curated.
                tag = "✓ positive"
            elif source == "corrected":
                tag = "✎ corrected"
            elif source == "thumbs_down":
                tag = "✗ negative"
            else:
                tag = f"● {source}"

            messages = ex.get("messages", [])
            user_text = ""
            # Use the first user message as the row preview, cut to 60 characters.
            for m in messages:
                if m.get("role") == "user":
                    user_text = m.get("content", "")
                    break
            preview = user_text[:60].replace("\n", " ")
            formatted_list.append(f"[{tag:<10}]  {preview}")
        self._list_model.update_data(formatted_list)

    # ── Preview ────────────────────────────────────────────────────────────────

    def _on_example_selected(self, row: int):
        """Render the selected example's messages as colour-coded, HTML-escaped blocks."""
        if row < 0:
            return
        examples = self.state.curator.get_all_examples()
        # Ignore stale selections after the dataset changed.
        if row >= len(examples):
            return
        ex = examples[row]

        messages = ex.get("messages", [])
        timestamp = ex.get("timestamp", "")
        source = ex.get("source", "unknown")

        html_parts = [
            f"<div style='font-size:9pt;color:#9090A8;margin-bottom:12px;"
            f"border-bottom:1px solid #252535;padding-bottom:6px;'>"
            f"Source: <b style='color:#00C2FF;'>{source}</b> &middot; Created: {timestamp}"
            f"</div>"
        ]

        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")

            # Pick colours per role (system, assistant, user).
            color = "#00C2FF" if role == "SYSTEM" else ("#2DD4A0" if role == "ASSISTANT" else "#E4E4F0")
            bg = "#14141F" if role == "SYSTEM" else ("#0D0D16" if role == "ASSISTANT" else "#1C1C2A")
            border = "#252535" if role == "SYSTEM" else ("#1A1A25" if role == "ASSISTANT" else "#383850")

            html_parts.append(
                f"<div style='margin-bottom:10px;'>"
                f"<div style='font-size:7.5pt;font-weight:bold;color:#505068;"
                f"margin-bottom:3px;letter-spacing:1px;'>{role}</div>"
                f"<div style='background:{bg};border:1px solid {border};border-radius:4px;"
                f"padding:8px 12px;color:{color};font-size:9.5pt;white-space:pre-wrap;'>"
                # Escape the content so message text cannot inject markup.
                f"{html.escape(content)}</div>"
                f"</div>"
            )

        self._detail_view.setHtml("".join(html_parts))

    # ── Delete ─────────────────────────────────────────────────────────────────

    def _delete_selected(self):
        """Delete the selected example after confirmation and notify listeners."""
        row = self._example_list.currentIndex().row()
        if row < 0:
            return
        reply = QMessageBox.question(
            self, "Delete example", "Delete this example?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            # The row index matches the curator's example order.
            self.state.curator.delete_example(row)
            self.refresh()
            self.dataset_changed.emit()

    # ── Import ─────────────────────────────────────────────────────────────────

    def _import_team_traces(self):
        """Open file dialog → run merge in background → show summary."""
        if self._merge_thread and self._merge_thread.isRunning():
            QMessageBox.information(
                self, "Import in Progress",
                "A merge is already running. Please wait for it to finish.",
            )
            return

        incoming_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Team curated.jsonl",
            "",
            "JSONL Files (*.jsonl);;All Files (*)",
        )
        if not incoming_path:
            return

        # Resolve the primary (local) dataset path.
        # The local curated dataset is the merge target.
        from app.utils.training_curator import CURATED_PATH
        primary_path = CURATED_PATH

        # Pre-flight: warn if primary doesn't exist yet (merge will create it).
        import os
        if not os.path.exists(primary_path):
            reply = QMessageBox.question(
                self,
                "Local Dataset Missing",
                f"Your local dataset does not exist yet:\n{primary_path}\n\n"
                "The import will create it. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Show indeterminate progress dialog while the thread runs.
        # A busy dialog with no cancel button keeps the user from interacting during the merge.
        self._progress_dialog = QProgressDialog(
            "Merging team traces…", None, 0, 0, self
        )
        self._progress_dialog.setWindowTitle("Import Team Traces")
        self._progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress_dialog.setMinimumDuration(0)
        self._progress_dialog.setCancelButton(None)
        self._progress_dialog.show()

        # Launch background thread.
        self._merge_thread = _MergeThread(primary_path, incoming_path, parent=self)
        self._merge_thread.finished.connect(self._on_merge_finished)
        self._merge_thread.error.connect(self._on_merge_error)
        self._merge_thread.start()

    def _on_merge_finished(self, stats: dict):
        """Handle successful merge completion."""
        if self._progress_dialog:
            self._progress_dialog.close()
            self._progress_dialog = None

        added = stats.get("added", 0)
        duplicates = stats.get("duplicates_skipped", 0)
        errors = stats.get("errors", 0)

        summary = (
            f"<b>Merge complete.</b><br><br>"
            f"<table style='font-size:10pt;'>"
            f"<tr><td>Added&nbsp;&nbsp;</td>"
            f"<td><b style='color:#2DD4A0;'>{added}</b> new examples</td></tr>"
            f"<tr><td>Duplicates&nbsp;&nbsp;</td>"
            f"<td><b style='color:#9090A8;'>{duplicates}</b> skipped</td></tr>"
            f"<tr><td>Errors&nbsp;&nbsp;</td>"
            f"<td><b style='color:#FF5C7A;'>{errors}</b> filtered</td></tr>"
            f"</table>"
        )

        msg = QMessageBox(self)
        msg.setWindowTitle("Import Complete")
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(summary)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()

        # Reload the dataset browser so the user sees the imported examples.
        self.refresh()
        self.dataset_changed.emit()

    def _on_merge_error(self, error_text: str):
        """Handle merge failure."""
        if self._progress_dialog:
            self._progress_dialog.close()
            self._progress_dialog = None

        QMessageBox.critical(
            self, "Import Failed",
            f"The merge encountered an error:\n\n{error_text}",
        )
