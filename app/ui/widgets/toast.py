# Program: karl-offline-lab - toast.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Implements ToastOverlay, a short-lived floating notification that fades in at the top of its parent and fades out.

"""Toast notification overlay widget."""
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QGraphicsOpacityEffect
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QTimer, QPoint


# ---------------------------------------------------------------------------
# Toast overlay
# ---------------------------------------------------------------------------
class ToastOverlay(QWidget):
    """A non-intrusive floating toast overlay for success notifications."""
    
    def __init__(self, parent: QWidget, message: str, duration_ms: int = 3000):
        """Build the label, fade animation and auto-dismiss timer for the given message and duration."""
        # Frameless tooltip-style window that does not steal keyboard focus
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.ToolTip)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        
        layout = QVBoxLayout(self)
        self._lbl = QLabel(message)
        self._lbl.setObjectName("toast-label")
        # Inline stylesheet gives the toast its dark card look
        self._lbl.setStyleSheet("""
            QLabel#toast-label {
                background-color: #0D0D1B;
                color: #00C2FF;
                border: 1px solid #00C2FF;
                border-radius: 4px;
                padding: 8px 16px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 9pt;
            }
        """)
        layout.addWidget(self._lbl)
        
        # Opacity effect drives the fade; it starts transparent via the animation
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity_effect)
        
        self._fade_anim = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_anim.setDuration(500)
        self._fade_anim.setStartValue(0.0)
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        # Single-shot timer starts the fade-out once the display time has passed
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._fade_out)
        
        self._duration = duration_ms

    def show_toast(self):
        """Show the toast near the top centre of its parent, fade it in and schedule dismissal."""
        self.show()
        # Position at top center of parent
        if self.parentWidget():
            p_rect = self.parentWidget().rect()
            p_global = self.parentWidget().mapToGlobal(QPoint(0, 0))
            self.move(
                p_global.x() + (p_rect.width() - self.width()) // 2,
                p_global.y() + 40
            )
            
        self._fade_anim.setDirection(QPropertyAnimation.Direction.Forward)
        self._fade_anim.start()
        self._timer.start(self._duration)

    def _fade_out(self):
        """Fade the toast out and close it when the animation finishes."""
        self._fade_anim.setDirection(QPropertyAnimation.Direction.Backward)
        self._fade_anim.finished.connect(self.close)
        self._fade_anim.start()
