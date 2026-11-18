# Program: karl-offline-lab - shortcuts_overlay.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Implements ShortcutsOverlay, a floating panel that lists the Workbench keyboard shortcuts and repositions itself when its parent resizes.

"""Floating, semi-transparent keyboard shortcuts reference overlay."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton,
)
from PyQt6.QtCore import Qt, QEvent


# ---------------------------------------------------------------------------
# Shortcut table
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _kbd(text: str, accent: str, border: str, bg_raised: str, text_hi: str) -> QLabel:
    """Render a key name as a styled kbd badge."""
    # Centred label with a rounded border so it resembles a keyboard key
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


# ---------------------------------------------------------------------------
# Overlay widget
# ---------------------------------------------------------------------------
class ShortcutsOverlay(QWidget):
    """Semi-transparent floating panel listing Workbench keyboard shortcuts.

    Parent must be the widget over which the overlay should be positioned.
    Call show_overlay() / hide() to toggle. The overlay repositions itself
    whenever its parent is resized.
    """

    def __init__(self, parent: QWidget):
        """Create the hidden frameless overlay, build its contents and position it over the parent."""
        super().__init__(parent)
        self.setObjectName("shortcuts-overlay")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.hide()

        self._build(parent)
        self._reposition()

    # ── build ─────────────────────────────────────────────────────────────────

    def _build(self, parent: QWidget):
        """Fill the overlay with a title, close button, divider and one row per shortcut, using theme colours."""
        # Imported here to avoid a circular import at module load
        from app.ui.themes import get_theme_colors
        try:
            colors = get_theme_colors(getattr(parent, "state", None))
        except Exception:
            colors = {}

        # Theme colours with fallbacks if the theme lookup fails
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

        # ── divider ───────────────────────────────────────────────────────────
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(f"color: {border}; background: {border};")
        divider.setFixedHeight(1)
        outer.addWidget(divider)

        # ── shortcut rows ─────────────────────────────────────────────────────
        for key_text, description in _SHORTCUTS:
            row = QHBoxLayout()
            row.setSpacing(12)
            row.setContentsMargins(0, 0, 0, 0)

            # Keys joined by " + " become separate badges with a "+" between them
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
        """Place the overlay in the parent's top-right corner below the Workbench header rows."""
        parent = self.parent()
        if parent is None:
            return
        w = 340
        h = self.sizeHint().height() or 320
        pw = parent.width()
        # Anchor: top-right corner, below the Workbench header rows
        # (session bar, model bar, agent bar, HUD toolbar).
        x = max(0, pw - w - 12)
        y = 156
        self.setGeometry(x, y, w, max(h, 300))

    def show_overlay(self):
        """Reposition, bring to front and show the overlay."""
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
        """Reposition whenever the watched parent is resized; other events pass through."""
        if event.type() == QEvent.Type.Resize:
            self._reposition()
        return super().eventFilter(obj, event)
