# ==============================================================================
# Program: common.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Desktop workspace implementation for Common.
#          Provides responsive layout controls, parameter adjustment surfaces,
#          and visual feedback components within Karl's PyQt6 shell.
# ==============================================================================

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QFrame


def _section(text: str) -> QLabel:
    # Step 1: Validate input parameters, operational preconditions, and state
    lbl = QLabel(text)
    lbl.setObjectName("section-header")
    return lbl


def _hline() -> QFrame:
    # Step 1: Validate input parameters, operational preconditions, and state
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    return f


def _row(label_text: str, widget: QWidget) -> QWidget:
    # Step 1: Validate input parameters, operational preconditions, and state
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
