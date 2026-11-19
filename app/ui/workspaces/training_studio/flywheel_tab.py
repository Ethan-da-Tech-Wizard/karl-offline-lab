# ==============================================================================
# Program: flywheel_tab.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Desktop workspace implementation for Flywheel Tab.
#          Provides responsive layout controls, parameter adjustment surfaces,
#          and visual feedback components within Karl's PyQt6 shell.
# ==============================================================================
# Flywheel tab: a dashboard for the self-improvement loop (interactions ->
# feedback -> training data -> eval score -> export). Statistics are gathered in
# a background thread; the tab also offers quick SFT/DPO export buttons.

# Where it fits: TrainingStudioWorkspace (training_studio/__init__.py) adds one FlywheelTab as a page and calls load_stats()
# when the tab becomes visible. Statistics come from _FlywheelStatsThread (threads.py), which reads traces, saved sessions,
# data/training/curated.jsonl and the last eval report off the GUI thread. Exports go through state.curator
# (TrainingCurator.export_unsloth / export_dpo). The tab holds no data of its own: every number shown is recomputed on Refresh.
# Widget object names such as "panel", "section-header", "lbl-muted", "lbl-accent" and "btn-primary" are QSS hooks defined in
# app/ui/themes.py. Tests: tests/test_training_curator.py covers the exporters this tab calls.
import os
import sys
import json

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFrame, QTextBrowser, QMessageBox, QFileDialog, QScrollArea
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

# Imported from the sibling module that owns every QThread of the Training Studio, so this file contains no thread code.
from app.ui.workspaces.training_studio.threads import _FlywheelStatsThread

def _section(text: str) -> QLabel:
    """Return a QLabel styled as a section heading."""
    lbl = QLabel(text)

    # The object name selects the small uppercase heading style in the QSS.
    lbl.setObjectName("section-header")
    return lbl


def _hline() -> QFrame:
    """Return a horizontal rule used inside the cards."""
    f = QFrame()

    # A QFrame configured as a thin separator line; the theme colours it through the QFrame rules.
    f.setFrameShape(QFrame.Shape.HLine)
    return f


class FlywheelTab(QWidget):
    """Tab widget showing flywheel statistic cards and an export preview."""
    def __init__(self, state, parent=None):
        """Store state and build the tab; statistics load later through load_stats()."""
        super().__init__(parent)
        self.state = state
        # Strong references to running stats threads, released when each one finishes.
        # Qt only keeps a running QThread alive while Python holds a reference; the set prevents garbage collection mid-run.
        self._active_threads = set()
        self._build_ui()

    def _build_ui(self):
        """Build the header, the pipeline caption and the row of five cards (interactions, feedback, data, eval, export)."""
        layout = QVBoxLayout(self)

        # 12 px outer padding is the workspace standard (AGENTS.md "Design System").
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        header_row = QWidget()
        hl = QHBoxLayout(header_row)
        hl.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel("Self-Improvement Flywheel")
        lbl.setObjectName("lbl-accent")
        hl.addWidget(lbl)
        hl.addStretch()
        self._flywheel_refresh_btn = QPushButton("Refresh")
        self._flywheel_refresh_btn.setObjectName("btn-ghost")

        # The button is disabled while a stats thread runs (see load_stats) so refreshes cannot pile up.
        self._flywheel_refresh_btn.clicked.connect(self.load_stats)
        hl.addWidget(self._flywheel_refresh_btn)
        layout.addWidget(header_row)

        # Static caption describing the loop the four stat cards represent, left to right.
        pipeline_lbl = QLabel(
            "Interactions  →  Feedback  →  Training Data  →  Eval Score  →  Export"
        )
        pipeline_lbl.setObjectName("lbl-muted")
        pipeline_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(pipeline_lbl)

        # Cards row. The four stat cards share one helper; the export card is built by hand.
        # Cards row
        cards = QWidget()
        cl = QHBoxLayout(cards)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(10)

        # Field ids double as dictionary keys of the stats dict emitted by _FlywheelStatsThread; the label text is derived from them.
        self._fw_interactions = self._flywheel_card(
            "Interactions", ["traces_total", "sessions_saved", "last_session"]
        )
        self._fw_feedback = self._flywheel_card(
            "Feedback", ["thumbs_up", "thumbs_down", "corrections"]
        )
        self._fw_data = self._flywheel_card(
            "Training Data", ["sft_examples", "dpo_pairs", "last_export"]
        )
        # Export buttons are inserted just above the card's trailing stretch so they sit under the stat rows.
        # Add SFT and DPO export buttons directly inside the Training Data card
        btn_lay_data = QHBoxLayout()
        btn_lay_data.setSpacing(6)

        # SFT = supervised fine-tuning pairs (messages format); DPO = preference pairs (prompt/chosen/rejected).
        card_export_sft = QPushButton("Export SFT")
        card_export_sft.setObjectName("btn-primary")
        card_export_sft.clicked.connect(self._flywheel_export_sft)
        card_export_dpo = QPushButton("Export DPO")
        card_export_dpo.setObjectName("btn-secondary")
        card_export_dpo.clicked.connect(self._flywheel_export_dpo)
        btn_lay_data.addWidget(card_export_sft)
        btn_lay_data.addWidget(card_export_dpo)
        self._fw_data.layout().insertLayout(self._fw_data.layout().count() - 1, btn_lay_data)

        # The Eval card links to the Eval workspace instead of running an evaluation itself.
        self._fw_eval = self._flywheel_card(
            "Eval Score", ["eval_score", "eval_dataset", "eval_date"]
        )
        # Add Run Eval button inside the Eval Score card
        btn_lay_eval = QHBoxLayout()
        card_run_eval = QPushButton("Run Eval")
        card_run_eval.setObjectName("btn-primary")
        card_run_eval.clicked.connect(self._flywheel_goto_eval)
        btn_lay_eval.addWidget(card_run_eval)
        self._fw_eval.layout().insertLayout(self._fw_eval.layout().count() - 1, btn_lay_eval)

        # Export & Preview Card
        # Built by hand (not with _flywheel_card) because it holds a button and a preview instead of label/value rows.
        self._fw_export_card = QFrame()
        self._fw_export_card.setObjectName("panel")
        ec_lay = QVBoxLayout(self._fw_export_card)
        ec_lay.setContentsMargins(10, 10, 10, 10)
        ec_lay.setSpacing(6)

        export_title = QLabel("EXPORT & PREVIEW")
        export_title.setObjectName("section-header")
        ec_lay.addWidget(export_title)
        ec_lay.addWidget(_hline())

        open_folder_btn = QPushButton("Open training folder")
        open_folder_btn.setObjectName("btn-primary")
        open_folder_btn.clicked.connect(self._open_training_folder)
        ec_lay.addWidget(open_folder_btn)

        preview_title = QLabel("Last Export Preview:")
        preview_title.setObjectName("lbl-muted")

        # Inline QSS is used for these two preview widgets because they need a fixed dark editor look independent of the active theme.
        preview_title.setStyleSheet("font-size: 8pt; font-weight: bold;")
        ec_lay.addWidget(preview_title)

        # Read-only viewer (QTextBrowser) for the first lines of the most recent SFT export.
        self._sft_preview = QTextBrowser()

        # Monospace so JSON lines align; 8.5 pt keeps long lines compact.
        preview_font = QFont("Monospace")
        preview_font.setPointSizeF(8.5)
        self._sft_preview.setFont(preview_font)
        self._sft_preview.setStyleSheet("background-color: #0E0F15; border: 1px solid #1A1A24;")
        self._sft_preview.setPlaceholderText("No recent SFT exports found.")
        ec_lay.addWidget(self._sft_preview, 1)

        # Each card needs real width for its labels/buttons to stay readable
        # (e.g. "Export SFT"/"Export DPO" buttons) -- 5 equal-stretch columns
        # squeezed into the app's 760px minimum width clipped them to
        # unreadable fragments. Give each a floor and let the row scroll
        # horizontally at narrow widths instead of shrinking below it.
        for card in (self._fw_interactions, self._fw_feedback, self._fw_data, self._fw_eval, self._fw_export_card):
            # 240 px is the narrowest width at which the card labels and the two export buttons stay readable.
            card.setMinimumWidth(240)
            cl.addWidget(card, 1)

        # Horizontal scrolling instead of shrinking: at the 760 px minimum window width five cards cannot fit side by side.
        cards_scroll = QScrollArea()
        cards_scroll.setWidgetResizable(True)
        cards_scroll.setFrameShape(QFrame.Shape.NoFrame)

        # Cards are as tall as their content, so no vertical bar is wanted; only the horizontal one appears when needed.
        cards_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        cards_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        cards_scroll.setWidget(cards)
        layout.addWidget(cards_scroll)

        # One-line status ("Loading stats..." / "Stats loaded.") shown below the cards.
        self._fw_status_lbl = QLabel("")
        self._fw_status_lbl.setObjectName("lbl-muted")
        layout.addWidget(self._fw_status_lbl)

        layout.addStretch()

    def _flywheel_card(self, title: str, field_ids: list) -> QWidget:
        """Build one stat card whose rows are label/value pairs, registering each value label in self._fw_fields by id."""
        card = QFrame()
        card.setObjectName("panel")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(10, 10, 10, 10)
        cl.setSpacing(6)

        title_lbl = QLabel(title.upper())
        title_lbl.setObjectName("section-header")
        cl.addWidget(title_lbl)
        cl.addWidget(_hline())

        # _fw_fields is shared across cards (created on first use) so stats can be applied by key later.
        # The first card call creates the dict; later calls reuse it, so all value labels live in one lookup table.
        self._fw_fields = getattr(self, "_fw_fields", {})
        for fid in field_ids:
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(4)

            # "traces_total" becomes "Traces Total:", so the id doubles as the label text with no separate translation table.
            name_lbl = QLabel(fid.replace("_", " ").title() + ":")
            name_lbl.setObjectName("lbl-muted")
            name_lbl.setFixedWidth(100)

            # An em dash means "not loaded yet" until the first stats arrive.
            val_lbl = QLabel("—")
            # A few headline values get the accent style.
            val_lbl.setObjectName("lbl-accent" if fid in ("traces_total", "thumbs_up", "sft_examples", "eval_score") else "")
            rl.addWidget(name_lbl)
            rl.addWidget(val_lbl, 1)
            cl.addWidget(row)
            self._fw_fields[fid] = val_lbl

        cl.addStretch()
        return card

    def load_stats(self):
        """Start a background thread that computes the stats and re-enable Refresh when it finishes."""
        self._fw_status_lbl.setText("Loading stats...")
        self._flywheel_refresh_btn.setEnabled(False)

        # New thread per refresh: simpler than reusing one object, and it cleans itself up through deleteLater below.
        t = _FlywheelStatsThread()

        # Signal delivery is queued to the GUI thread, so the slot may touch widgets safely.
        t.stats_ready.connect(self._apply_flywheel_stats)
        t.finished.connect(lambda: self._flywheel_refresh_btn.setEnabled(True))

        # Frees the QThread C++ object after it finishes; the Python reference is dropped by the discard below.
        t.finished.connect(t.deleteLater)
        self._active_threads.add(t)
        t.finished.connect(lambda: self._active_threads.discard(t))
        t.start()

    def _apply_flywheel_stats(self, stats: dict):
        """Write each statistic into its matching label and show the latest SFT export preview."""
        # Unknown keys in the stats dict are ignored, so the thread can add fields without breaking the UI.
        fields = getattr(self, "_fw_fields", {})
        for fid, val in stats.items():
            if fid in fields:
                fields[fid].setText(str(val))

        # Apply preview content
        if "last_sft_content" in stats:
            self._sft_preview.setPlainText(stats["last_sft_content"])

        self._fw_status_lbl.setText("Stats loaded.")

    def _flywheel_export_sft(self):
        """Export SFT data, then refresh the statistics so counts and preview update."""
        # Export first, refresh second: the refreshed counts and preview then include the file just written.
        self._export("sft")
        self.load_stats()

    def _flywheel_export_dpo(self):
        """Export DPO data, then refresh the statistics."""
        self._export("dpo")
        self.load_stats()

    def _open_training_folder(self):
        """Open data/training in the system file browser (creating it if needed)."""
        # Local import: only needed for this rarely used button.
        import subprocess
        try:
            # Relative to the working directory, matching where TrainingCurator writes curated.jsonl.
            path = os.path.abspath("data/training")
            os.makedirs(path, exist_ok=True)
            # Each OS has a different "open in file manager" entry point.
            if sys.platform == "win32":
                os.startfile(path)  # type: ignore[attr-defined]  # Windows-only API
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to open folder: {e}")

    def _flywheel_goto_eval(self):
        """Switch the main window to the Eval workspace (sidebar index 5)."""
        try:
            # The window() lookup keeps this tab decoupled from MainWindow; silently do nothing if the sidebar is absent.
            # window() returns the top-level MainWindow; hasattr guards against embedding the tab somewhere else (for example in tests).
            main_win = self.window()
            if hasattr(main_win, "_sidebar"):
                # 5 is the Eval Suite's index in the sidebar stack (see the architecture list in AGENTS.md); update it if the order changes.
                main_win._sidebar.select(5)
        except Exception:
            pass

    def _export(self, mode: str):
        """Validate curated.jsonl, ask where to save, and export in the given mode ("sft" or "dpo")."""
        # Same preflight as ExportTab._export: non-empty file and each line has "messages" or "prompt".
        # Preflight avoids passing an empty or malformed file to the exporters, which would fail with a less helpful message.
        curated_file = "data/training/curated.jsonl"
        if not os.path.exists(curated_file) or os.path.getsize(curated_file) == 0:
            QMessageBox.warning(self, "Validation Failed", "The dataset is empty. Curate some examples first.")
            return
            
        try:
            with open(curated_file, "r") as f:
                for line_idx, line in enumerate(f):
                    if not line.strip():
                        continue

                    # Each line must be a JSON object holding either chat-format "messages" (SFT) or "prompt" (DPO) keys.
                    obj = json.loads(line)
                    if "messages" not in obj and "prompt" not in obj:
                        raise ValueError(f"Line {line_idx+1}: Missing both 'messages' and 'prompt' keys.")
        except Exception as e:
            QMessageBox.critical(self, "Dataset Validation Error", f"curated.jsonl format validation failed:\n{e}")
            return

        # Default name unsloth_<mode>.jsonl hints at the format the exporters produce (Unsloth-compatible JSONL).
        path, _ = QFileDialog.getSaveFileName(
            self, "Save export", f"unsloth_{mode}.jsonl", "JSONL (*.jsonl)"
        )
        if not path:
            return
        try:
            if mode == "sft":
                out_path = self.state.curator.export_unsloth(path)

                # One line per example, so counting lines gives the example count shown to the user.
                count = sum(1 for line in open(out_path, "r", encoding="utf-8"))
                QMessageBox.information(self, "Export Successful", f"Saved {count} SFT examples to:\n{out_path}")
            else:
                out_path = self.state.curator.export_dpo(path)
                count = sum(1 for line in open(out_path, "r", encoding="utf-8"))
                QMessageBox.information(self, "Export Successful", f"Saved {count} DPO pairs to:\n{out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export: {e}")
