# ==============================================================================
# Program: common.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Desktop workspace implementation for Common.
#          Provides responsive layout controls, parameter adjustment surfaces,
#          and visual feedback components within Karl's PyQt6 shell.
# ==============================================================================

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QFrame
# Shared layout helpers used by every System tab panel. They keep section
# headers, separators and label/value rows visually identical across tabs, so
# the QSS rules keyed on the object names below apply uniformly.


def _section(text: str) -> QLabel:
    """Return a QLabel styled as a section heading via the "section-header" QSS id."""
    lbl = QLabel(text)
    lbl.setObjectName("section-header")
    return lbl


def _hline() -> QFrame:
    """Return a horizontal rule (QFrame HLine) used to separate groups in a panel."""
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    return f


def _row(label_text: str, widget: QWidget) -> QWidget:
    """Build a "label: widget" row with a fixed-width muted label column."""
    # Plain container widget; the caller adds the returned row to its own layout.
    w = QWidget()
    lo = QHBoxLayout(w)
    lo.setContentsMargins(0, 2, 0, 2)
    # Fixed 130px label column so inputs line up vertically across rows.
    lo.setSpacing(12)
    lbl = QLabel(label_text)
    lbl.setFixedWidth(130)
    lbl.setObjectName("lbl-muted")
    lo.addWidget(lbl)
    lo.addWidget(widget)
    lo.addStretch()
    return w
