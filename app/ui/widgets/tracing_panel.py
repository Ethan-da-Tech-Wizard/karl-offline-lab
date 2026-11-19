# ==============================================================================
# Program: tracing_panel.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Reusable desktop widget component implementing tracing panel.
#          Enforces styling standards, event handling, and visual telemetry
#          across Karl offline workspaces.
# ==============================================================================

from PyQt6.QtWidgets import QFrame
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPainter, QPainterPath, QPen, QColor
from app.ui.themes import get_theme_colors

# Animation state: _offset drives the travelling dash, _pulse_val/_pulse_direction
# drive the breathing alpha used by the "pulse" motion style.
class TracingPanel(QFrame):
    """
    A customizable QFrame that draws an animated neon segment tracing along
    its border when set_active(True) is called. Adapts to theme-specific
    motion styles (normal, slow, pulse, static) and user speed configurations.
    """
    def __init__(self, state, parent=None):
        """Create an idle panel; the 25 ms timer is only started while active."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(parent)
        self.state = state
        self._active = False
        self._offset = 0
        self._pulse_val = 0.6
        self._pulse_direction = 1
        self._accent_color = None
        
        # Timer is created stopped; _update_timer_state decides when to run it.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        
        self.setObjectName("panel")
        self.setStyleSheet("border-radius: 4px;")

    def set_active(self, active: bool):
        """Turn the tracing animation on or off."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._active = active
        self._update_timer_state()
        self.update()

    def set_accent_color(self, hex_color: str):
        """Override the trace colour (hex string); None-equivalent falls back to the theme accent."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._accent_color = hex_color
        self.update()

    def _update_timer_state(self):
        """Start the animation timer only when it can be seen.

        No timer when inactive, when reduced motion is on, or when the theme's
        motion style is "static" -- saves CPU and respects accessibility settings.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        is_reduced = getattr(self.state, "reduced_motion", False) if self.state else False
        colors = get_theme_colors(self.state) if self.state else {}
        motion_style = colors.get("motion_style", "normal")
        
        if self._active and not is_reduced and motion_style != "static":
            if not self._timer.isActive():
                self._timer.start(25)
        else:
            self._timer.stop()

    def _on_tick(self):
        """Advance one animation frame (called every 25 ms)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        intensity = getattr(self.state, "animation_intensity", 1.0) if self.state else 1.0
        colors = get_theme_colors(self.state) if self.state else {}
        motion_style = colors.get("motion_style", "normal")
        
        # Pulse: bounce alpha between 0.2 and 1.0 (never fully invisible).
        if motion_style == "pulse":
            step = 0.03 * intensity
            self._pulse_val += step * self._pulse_direction
            if self._pulse_val >= 1.0:
                self._pulse_val = 1.0
                self._pulse_direction = -1
            elif self._pulse_val <= 0.2:
                self._pulse_val = 0.2
                self._pulse_direction = 1
        # Normal/slow: slide the dash pattern; wrap before the offset grows unbounded.
        else:
            step = 2.0 * intensity
            if motion_style == "slow":
                step = 0.8 * intensity
            self._offset -= int(step)
            if self._offset <= -1000:
                self._offset = 0
        self.update()

    def update_style(self):
        """Re-evaluate the timer and repaint after theme/appearance changes."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._update_timer_state()
        self.update()

    def paintEvent(self, event):
        """Draw the border outline: static/pulse as a solid line, otherwise a moving dash."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().paintEvent(event)
        if not self._active:
            return
            
        is_reduced = getattr(self.state, "reduced_motion", False) if self.state else False
        colors = get_theme_colors(self.state) if self.state else {}
        motion_style = colors.get("motion_style", "normal")
        accent = self._accent_color or colors.get("accent", "#00C2FF")
        color = QColor(accent)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        path = QPainterPath()
        # Inset by 1px so the 1.5px pen is not clipped at the widget edge.
        path.addRoundedRect(1.0, 1.0, self.width() - 2.0, self.height() - 2.0, 4.0, 4.0)
        
        if is_reduced or motion_style == "static":
            pen = QPen(color, 1.5, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawPath(path)
        elif motion_style == "pulse":
            pulse_color = QColor(color)
            pulse_color.setAlphaF(self._pulse_val)
            pen = QPen(pulse_color, 1.5, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawPath(path)
        else:
            pen = QPen(color, 1.5)
            # 60px lit segment followed by a 180px gap gives the 'tracer' look.
            pen.setDashPattern([60, 180])
            pen.setDashOffset(self._offset)
            painter.setPen(pen)
            painter.drawPath(path)
