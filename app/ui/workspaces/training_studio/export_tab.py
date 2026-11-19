# ==============================================================================
# Program: export_tab.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Desktop workspace implementation for Export Tab.
#          Provides responsive layout controls, parameter adjustment surfaces,
#          and visual feedback components within Karl's PyQt6 shell.
# ==============================================================================
# Export tab: writes the curated dataset (data/training/curated.jsonl) out as
# Unsloth-compatible SFT or DPO JSONL, after a quick structural validation.

import os
import json

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox
)
from app.ui.widgets.glow_panel import GlowPanel

# Local copy of the section-header helper rather than importing from
# system_config.common: the Training Studio package stays independent of the
# System workspace, and the helper is three lines.
def _section(text: str) -> QLabel:
    """Return a QLabel styled as a section heading."""
    lbl = QLabel(text)
    lbl.setObjectName("section-header")
    return lbl


# Self-contained tab. The only shared object is `state`, whose .curator
# (TrainingCurator) does the real conversion; this widget only validates the
# source file, picks the destination and reports the outcome, so there are no
# signals to other tabs.
class ExportTab(QWidget):
    """Tab widget with one card per export format (SFT and DPO) and a status line."""
    def __init__(self, state, parent=None):
        """Store state and build the tab."""
        super().__init__(parent)
        self.state = state
        self._build_ui()

    def _build_ui(self):
        """Build the two export cards and the status label."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)

        # Two side-by-side cards of equal stretch: SFT and DPO are alternative outputs of
        # the same dataset, so they are shown together and neither is the "default".
        cards = QWidget()
        cards_layout = QHBoxLayout(cards)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(16)

        # SFT card: instruction-following chat traces.
        # SFT Panel
        # GlowPanel gives the card the themed glow border; it takes state so it follows
        # the glow / reduced-motion settings from the Theme tab.
        sft_box = GlowPanel(self.state)
        sft_l = QVBoxLayout(sft_box)
        sft_l.setContentsMargins(16, 16, 16, 16)
        sft_l.setSpacing(12)
        sft_l.addWidget(_section("UNSLOTH / SFT FORMAT"))
        sft_desc = QLabel(
            "Exports curated examples in Unsloth-compatible JSONL format. "
            "Includes instruction-following message traces ideal for Supervised Fine-Tuning (SFT)."
        )
        sft_desc.setObjectName("lbl-muted")
        sft_desc.setWordWrap(True)
        sft_l.addWidget(sft_desc)
        # Stretch pushes the button to the card bottom so the SFT and DPO buttons sit on
        # the same baseline even though the description texts differ in length.
        sft_l.addStretch()
        sft_btn = QPushButton("export SFT  →  unsloth_sft.jsonl")
        sft_btn.setObjectName("btn-primary")
        sft_btn.setToolTip("Export curated dataset in Unsloth SFT chat format")
        # Lambda binds the mode string; the two buttons share one _export slot so the
        # validation and dialog logic exist once.
        sft_btn.clicked.connect(lambda: self._export("sft"))
        sft_l.addWidget(sft_btn)
        cards_layout.addWidget(sft_box, 1)

        # DPO card: needs both a thumbs-up (chosen) and a thumbs-down (rejected) sample to form pairs.
        # DPO Panel
        dpo_box = GlowPanel(self.state)
        dpo_l = QVBoxLayout(dpo_box)
        dpo_l.setContentsMargins(16, 16, 16, 16)
        dpo_l.setSpacing(12)
        dpo_l.addWidget(_section("DPO FORMAT"))
        dpo_desc = QLabel(
            "Exports paired chosen (thumbs-up) vs rejected (thumbs-down) examples. "
            "Requires at least one positive and one negative sample to construct comparison pairs."
        )
        dpo_desc.setObjectName("lbl-muted")
        dpo_desc.setWordWrap(True)
        dpo_l.addWidget(dpo_desc)
        dpo_l.addStretch()
        # No btn-primary on DPO: only the SFT export is highlighted as the common
        # path, since DPO needs both ratings to exist before it yields anything.
        dpo_btn = QPushButton("export DPO  →  unsloth_dpo.jsonl")
        dpo_btn.setToolTip("Export paired chosen/rejected examples in DPO format")
        dpo_btn.clicked.connect(lambda: self._export("dpo"))
        dpo_l.addWidget(dpo_btn)
        cards_layout.addWidget(dpo_box, 1)

        # Stretch 1: the cards absorb vertical space; the status line stays one row high.
        layout.addWidget(cards, 1)

        self._export_status = QLabel("")
        # Single status line for both exporters; it holds the last result (saved N
        # examples or an error) until the next export overwrites it.
        self._export_status.setObjectName("lbl-mid")
        self._export_status.setWordWrap(True)
        layout.addWidget(self._export_status)

    def _export(self, mode: str):
        """Validate the curated dataset, ask for a destination file, and export in the given mode ("sft" or "dpo")."""
        # Preflight: refuse empty datasets, then confirm every non-blank line is JSON carrying
        # either "messages" or "prompt", so a corrupt file is caught before the exporter runs.
        # Validation preflight
        # The validation reads the same file the curator appends to, so it checks what
        # the export will consume. Each accepted line must carry "messages" (SFT chat
        # records) or "prompt" (DPO-style records); anything else means the file was
        # hand-edited or written by another tool.
        curated_file = "data/training/curated.jsonl"
        if not os.path.exists(curated_file) or os.path.getsize(curated_file) == 0:
            QMessageBox.warning(self, "Validation Failed", "The dataset is empty. Curate some examples first.")
            return
            
        try:
            # Streamed line by line: curated sets can be large and only structural keys are
            # checked, so there is no need to hold the parsed objects.
            with open(curated_file, "r") as f:
                for line_idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    # A JSONDecodeError here is caught by the surrounding except and shown with the
                    # bad content; line_idx + 1 in the ValueError below is a 1-based line number to
                    # match what a text editor shows.
                    obj = json.loads(line)
                    if "messages" not in obj and "prompt" not in obj:
                        raise ValueError(f"Line {line_idx+1}: Missing both 'messages' and 'prompt' keys.")
        except Exception as e:
            QMessageBox.critical(self, "Dataset Validation Error", f"curated.jsonl format validation failed:\n{e}")
            return

        # The curator does the actual conversion; the line count is re-read from the output file to report what was written.
        # Default file name encodes the mode so SFT and DPO exports saved to the same
        # folder cannot overwrite each other. Empty path means the user cancelled.
        path, _ = QFileDialog.getSaveFileName(
            self, "Save export", f"unsloth_{mode}.jsonl", "JSONL (*.jsonl)"
        )
        if not path:
            return
        try:
            if mode == "sft":
                # The curator filters degraded examples and balances classes while exporting, so
                # the count reported below can be lower than the lines in curated.jsonl.
                out_path = self.state.curator.export_unsloth(path)
                count = sum(1 for line in open(out_path, "r", encoding="utf-8"))
                self._export_status.setText(f"saved {count} SFT examples: {out_path}")
            else:
                # DPO groups answers by (system prompt, user message) and pairs a chosen with a
                # rejected one, so fewer pairs than ratings is normal.
                out_path = self.state.curator.export_dpo(path)
                count = sum(1 for line in open(out_path, "r", encoding="utf-8"))
                self._export_status.setText(f"saved {count} DPO pairs: {out_path}")
        except Exception as e:
            # Errors go to the status line instead of a modal dialog because a failed export
            # loses nothing and the user can retry immediately.
            self._export_status.setText(f"error: {e}")
