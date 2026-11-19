# ==============================================================================
# Program: shortcuts_overlay.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Floating, semi-transparent keyboard shortcuts reference overlay.
#          Coordinates with adjacent subsystems within the Karl Offline Lab
#          architecture.
# ==============================================================================

"""Floating, semi-transparent keyboard shortcuts reference overlay."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton,
)
from PyQt6.QtCore import Qt, QEvent


# Display-only list; the real bindings live in the Workbench/MainWindow.
# Keys are joined with " + " so _build can render each part as its own badge
# (note "Ctrl + Shift+H" therefore renders as two badges: "Ctrl" and "Shift+H").
_SHORTCUTS = [
    # (key display, description)
    ("Ctrl + Enter",  "Send message"),
    ("Esc",           "Cancel generation"),
    ("Ctrl + R",      "Reload active model"),
    ("Ctrl + S",      "Save current session"),
    ("Ctrl + N",      "New session"),
    ("Ctrl + V",      "Paste image from clipboard"),
    ("Ctrl + Shift+H","Toggle all HUD panels"),
    ("?",             "Toggle this shortcuts panel"),
]


def _kbd(text: str, accent: str, border: str, bg_raised: str, text_hi: str) -> QLabel:
    """Render a key name as a styled kbd badge."""
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setStyleSheet(f"""
        background-color: {bg_raised};
        border: 1px solid {border};
        border-radius: 3px;
        color: {accent};
        font-family: 'JetBrains Mono', 'Consolas', monospace;
        font-size: 8pt;
        font-weight: bold;
        padding: 2px 7px;
    """)
    return lbl


# Overlay is a child widget (not a window), so it is clipped to and moves with
# its parent. It does not call installEventFilter itself; whoever creates it
# must install it on the parent for resize repositioning to work.
class ShortcutsOverlay(QWidget):
    """Semi-transparent floating panel listing Workbench keyboard shortcuts.

    Parent must be the widget over which the overlay should be positioned.
    Call show_overlay() / hide() to toggle. The overlay repositions itself
    whenever its parent is resized.
    """

    def __init__(self, parent: QWidget):
        """Create the hidden overlay as a child of `parent`, build its contents and position it."""
        super().__init__(parent)
        self.setObjectName("shortcuts-overlay")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.hide()

        self._build(parent)
        self._reposition()

    # ── build ─────────────────────────────────────────────────────────────────

    def _build(self, parent: QWidget):
        """Construct the header, divider and one row per entry in _SHORTCUTS.

        Theme colours are read from the parent's AppState when available; the
        hard-coded fallbacks keep the overlay usable if theming fails.
        """
        from app.ui.themes import get_theme_colors
        try:
            # Any failure (no state, unknown theme) just means fallback colours are used.
            colors = get_theme_colors(getattr(parent, "state", None))
        except Exception:
            colors = {}

        # Fallback colours match the default dark palette.
        bg_raised = colors.get("bg_raised", "#1C1C2A")
        border    = colors.get("border",    "#252535")
        border_hi = colors.get("border_hi", "#383850")
        accent    = colors.get("accent",    "#00C2FF")
        text_hi   = colors.get("text_hi",   "#E4E4F0")
        text_lo   = colors.get("text_lo",   "#505068")

        self.setStyleSheet(f"""
            ShortcutsOverlay {{
                background-color: rgba(13, 13, 22, 220);
                border: 1px solid {border_hi};
                border-radius: 8px;
            }}
        """)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(12)

        # Title on the left, a round close button on the right.
        # ── header ────────────────────────────────────────────────────────────
        header_row = QHBoxLayout()
        header_row.setSpacing(8)

        title = QLabel("Keyboard Shortcuts")
        title.setStyleSheet(
            f"color: {accent}; font-size: 10pt; font-weight: bold; "
            f"letter-spacing: 1px; font-family: 'JetBrains Mono', monospace; "
            f"background: transparent;"
        )
        header_row.addWidget(title)
        header_row.addStretch()

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(22, 22)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: 1px solid {border};
                border-radius: 11px;
                color: {text_lo};
                font-size: 9pt;
            }}
            QPushButton:hover {{
                color: {text_hi};
                border-color: {accent};
            }}
        """)
        close_btn.clicked.connect(self.hide)
        header_row.addWidget(close_btn)
        outer.addLayout(header_row)

        # 1px horizontal rule under the header.
        # ── divider ───────────────────────────────────────────────────────────
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(f"color: {border}; background: {border};")
        divider.setFixedHeight(1)
        outer.addWidget(divider)

        # One row per shortcut: key badges (with '+' separators) on the left,
        # description right-aligned.
        # ── shortcut rows ─────────────────────────────────────────────────────
        for key_text, description in _SHORTCUTS:
            row = QHBoxLayout()
            row.setSpacing(12)
            row.setContentsMargins(0, 0, 0, 0)

            kbd_parts = key_text.split(" + ")
            for i, part in enumerate(kbd_parts):
                row.addWidget(_kbd(part.strip(), accent, border, bg_raised, text_hi))
                if i < len(kbd_parts) - 1:
                    plus = QLabel("+")
                    plus.setStyleSheet(
                        f"color: {text_lo}; font-size: 8pt; background: transparent;"
                    )
                    row.addWidget(plus)

            row.addStretch()

            desc_lbl = QLabel(description)
            desc_lbl.setStyleSheet(
                f"color: {text_hi}; font-size: 8.5pt; background: transparent;"
            )
            row.addWidget(desc_lbl)

            outer.addLayout(row)

    # ── positioning ───────────────────────────────────────────────────────────

    def _reposition(self):
        """Anchor the overlay to the parent's top-right corner.

        Width is fixed at 340px; height follows the content but never below 300px.
        """
        parent = self.parent()
        if parent is None:
            return
        w = 340
        h = self.sizeHint().height() or 320
        pw = parent.width()
        # y=156 is a hard-coded offset that clears the Workbench header rows; it will
        # need updating if those rows change height.
        # Anchor: top-right corner, below the Workbench header rows
        # (session bar, model bar, agent bar, HUD toolbar).
        x = max(0, pw - w - 12)
        y = 156
        self.setGeometry(x, y, w, max(h, 300))

    def show_overlay(self):
        """Reposition, raise above sibling widgets and show."""
        self._reposition()
        self.raise_()
        self.show()

    def toggle(self):
        """Hide the overlay if visible, otherwise show it."""
        if self.isVisible():
            self.hide()
        else:
            self.show_overlay()

    # ── respond to parent resize ──────────────────────────────────────────────

    def eventFilter(self, obj, event):
        """Re-anchor the overlay whenever the filtered (parent) widget is resized."""
        if event.type() == QEvent.Type.Resize:
            self._reposition()
        return super().eventFilter(obj, event)
