# Program: karl-offline-lab - common.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Small shared widget helpers (section header, horizontal rule, labelled row) used by the system config panels.
"""Small shared widget helpers (section header, horizontal rule, labelled row) used by the system config panels."""

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QFrame


def _section(text: str) -> QLabel:
    """Return a QLabel styled as a section header."""
    lbl = QLabel(text)
    lbl.setObjectName("section-header")
    return lbl


def _hline() -> QFrame:
    """Return a horizontal line frame used as a divider."""
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    return f


def _row(label_text: str, widget: QWidget) -> QWidget:
    """Build a row with a fixed-width muted label on the left and the given widget on the right."""
    w = QWidget()
    lo = QHBoxLayout(w)
    lo.setContentsMargins(0, 2, 0, 2)
    lo.setSpacing(12)
    lbl = QLabel(label_text)
    lbl.setFixedWidth(130)
    lbl.setObjectName("lbl-muted")
    lo.addWidget(lbl)
    lo.addWidget(widget)
    lo.addStretch()
    return w
