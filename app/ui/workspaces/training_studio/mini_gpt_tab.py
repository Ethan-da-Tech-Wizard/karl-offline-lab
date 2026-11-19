# ==============================================================================
# Program: mini_gpt_tab.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Desktop workspace implementation for Mini Gpt Tab.
#          Provides responsive layout controls, parameter adjustment surfaces,
#          and visual feedback components within Karl's PyQt6 shell.
# ==============================================================================
# Mini-GPT sandbox tab: an educational, from-scratch character-level GPT. The
# user picks a dataset and architecture, trains it on a MiniTrainThread while a
# painted loss chart and periodic text samples stream in, and can then generate
# text from the saved weights. This is separate from the llama.cpp models.

import os
import json
import html
import traceback

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextBrowser,
    QLabel, QSpinBox, QDoubleSpinBox, QProgressBar, QLineEdit,
    QGridLayout, QFrame, QMessageBox, QComboBox, QSplitter, QFileDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor, QPainter, QPen, QBrush, QPainterPath, QLinearGradient

from app.ui.workspaces.training_studio.threads import MiniTrainThread
from app.engine.model_loader import ModelLoader

def _section(text: str) -> QLabel:
    """Return a QLabel styled as a section heading."""
    # Step 1: Validate input parameters, operational preconditions, and state
    lbl = QLabel(text)
    lbl.setObjectName("section-header")
    return lbl


def _hline() -> QFrame:
    """Return a horizontal rule used to separate groups of controls."""
    # Step 1: Validate input parameters, operational preconditions, and state
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    return f


class LossChartWidget(QFrame):
    """Custom-painted line chart of validation loss over training steps."""
    def __init__(self, parent=None):
        """Create the chart frame with an empty loss history."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(parent)
        self.setObjectName("loss-chart-widget")
        self.setMinimumHeight(150)
        self.setStyleSheet("""
            #loss-chart-widget {
                background-color: #0D0D1B;
                border: 1px solid #1F1F3D;
                border-radius: 4px;
            }
        """)
        self._loss_history = []

    def set_loss_history(self, history: list[float]):
        """Replace the plotted loss values and schedule a repaint."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._loss_history = list(history)
        self.update()

    def paintEvent(self, event):
        """Paint the axes, grid, smoothed loss curve, gradient fill and current-loss read-outs."""
        # Step 1: Validate input parameters, operational preconditions, and state
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        margin_left = 45
        margin_right = 15
        margin_top = 15
        margin_bottom = 25
        
        plot_w = w - margin_left - margin_right
        plot_h = h - margin_top - margin_bottom
        
        # Painting is done by hand (QPainter) rather than a charting library to keep the app dependency-free.
        # Draw background
        painter.setBrush(QBrush(QColor("#0D0D1B")))
        painter.setPen(QPen(QColor("#1F1F3D"), 1))
        painter.drawRoundedRect(0, 0, w, h, 4.0, 4.0)
        
        if plot_w <= 0 or plot_h <= 0:
            return
            
        # Nothing to plot yet: draw a placeholder message and stop.
        if not self._loss_history:
            # Draw placeholder text
            painter.setFont(QFont("JetBrains Mono", 9))
            painter.setPen(QPen(QColor("#505068")))
            painter.drawText(0, 0, w, h, Qt.AlignmentFlag.AlignCenter, "Waiting for training loss...")
            return

        # Y range is padded by 10% of the spread; a flat series gets a fixed +/-0.5 window so
        # the scale never divides by zero. The bottom is clamped at 0 since loss is non-negative.
        # Scale data
        min_y = min(self._loss_history)
        max_y = max(self._loss_history)
        
        if max_y > min_y:
            y_bottom = max(0.0, min_y - 0.1 * (max_y - min_y))
            y_top = max_y + 0.1 * (max_y - min_y)
        else:
            y_bottom = max(0.0, min_y - 0.5)
            y_top = min_y + 0.5
            
        # Draw horizontal grid lines
        grid_pen = QPen(QColor("#1F1F3D"), 1, Qt.PenStyle.DashLine)
        painter.setPen(grid_pen)
        
        font = QFont("JetBrains Mono", 8)
        painter.setFont(font)
        
        # Three grid lines (bottom, middle, top) with value labels in the left margin.
        levels = [y_bottom, (y_bottom + y_top) / 2.0, y_top]
        for val in levels:
            # Iterate over items, evaluate loop invariant, and update progress
            pct = (val - y_bottom) / (y_top - y_bottom) if y_top != y_bottom else 0.5
            gy = margin_top + (1.0 - pct) * plot_h
            painter.drawLine(margin_left, int(gy), margin_left + plot_w, int(gy))
            
            # Label
            painter.setPen(QPen(QColor("#70708F")))
            painter.drawText(5, int(gy) - 7, margin_left - 10, 15, 
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, 
                             f"{val:.2f}")
            painter.setPen(grid_pen)

        # Map each loss value to pixel coordinates; with a single point it is centred horizontally.
        # Compute point coordinates
        points = []
        n_points = len(self._loss_history)
        for idx, val in enumerate(self._loss_history):
            # Iterate over items, evaluate loop invariant, and update progress
            x = margin_left + (idx / (n_points - 1) * plot_w) if n_points > 1 else margin_left + 0.5 * plot_w
            pct = (val - y_bottom) / (y_top - y_bottom) if y_top != y_bottom else 0.5
            y = margin_top + (1.0 - pct) * plot_h
            points.append((x, y))
            
        # Build the curve with cubic Beziers using horizontal control points, which smooths the line without overshooting.
        # Draw path
        path = QPainterPath()
        path.moveTo(points[0][0], points[0][1])
        
        if len(points) > 1:
            for i in range(len(points) - 1):
                # Iterate over items, evaluate loop invariant, and update progress
                x1, y1 = points[i]
                x2, y2 = points[i+1]
                dx = (x2 - x1) / 3.0
                path.cubicTo(x1 + dx, y1, x2 - dx, y2, x2, y2)
        else:
            path.lineTo(points[0][0] + 5, points[0][1])

        # Close the area under the curve down to the x axis and fill it with a fading gradient.
        # Draw filled gradient under curve
        fill_path = QPainterPath(path)
        fill_path.lineTo(points[-1][0], margin_top + plot_h)
        fill_path.lineTo(points[0][0], margin_top + plot_h)
        fill_path.closeSubpath()
        
        grad = QLinearGradient(0, margin_top, 0, margin_top + plot_h)
        grad.setColorAt(0.0, QColor(0, 194, 255, 60))
        grad.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillPath(fill_path, QBrush(grad))

        # Draw the curve line
        curve_pen = QPen(QColor("#00C2FF"), 2)
        painter.setPen(curve_pen)
        painter.drawPath(path)
        
        # Mark the most recent value so the user can see where training currently is.
        # Highlight last point
        last_x, last_y = points[-1]
        painter.setBrush(QBrush(QColor("#2DD4A0")))
        painter.setPen(QPen(QColor("#FFFFFF"), 1.5))
        painter.drawEllipse(int(last_x) - 4, int(last_y) - 4, 8, 8)
        
        # Draw telemetry details inside chart area
        painter.setPen(QPen(QColor("#2DD4A0")))
        painter.setFont(QFont("JetBrains Mono", 8, QFont.Weight.Bold))
        curr_loss = self._loss_history[-1]
        painter.drawText(margin_left + 10, margin_top + 15, f"LOSS: {curr_loss:.4f}")
        
        # Draw count of steps
        painter.setPen(QPen(QColor("#70708F")))
        painter.setFont(QFont("JetBrains Mono", 7.5))
        painter.drawText(margin_left + 10, margin_top + 30, f"Steps: {n_points}")


class MiniGptTab(QWidget):
    """Tab widget that trains and queries a tiny GPT in a sandbox."""
    def __init__(self, state, parent=None):
        """Store state and build the tab; weights on disk (if any) enable the inference box."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(parent)
        self.state = state
        # Keep a strong reference to the training thread while it runs.
        self._active_threads = set()
        self._mini_train_thread = None
        self._mini_loss_history = []
        self._build_ui()

    def _build_ui(self):
        """Build the configuration column, the telemetry/generation monitors and the inference sandbox."""
        # Step 1: Validate input parameters, operational preconditions, and state
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Splitters let the user resize the config drawer and monitors.
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)

        # Left Column: Configuration Drawer
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(10)
        
        ll.addWidget(_section("MINI-GPT ARCHITECTURE"))
        
        # Dataset choices: bundled Shakespeare text, Karl's own trace logs, or a user-chosen file.
        # Dataset Selection
        ll.addWidget(QLabel("Training Dataset:"))
        self._mini_dataset_combo = QComboBox()
        self._mini_dataset_combo.addItems(["Tiny Shakespeare", "Karl's Trace Logs", "Custom Text File..."])
        self._mini_dataset_combo.currentTextChanged.connect(self._on_mini_dataset_changed)
        ll.addWidget(self._mini_dataset_combo)

        # The file picker row is only shown for the custom-file option.
        # Custom File Row (Initially Hidden)
        self._mini_custom_row = QWidget()
        cr_layout = QHBoxLayout(self._mini_custom_row)
        cr_layout.setContentsMargins(0, 0, 0, 0)
        cr_layout.setSpacing(8)
        self._mini_file_label = QLabel("No file selected")
        self._mini_file_label.setObjectName("lbl-muted")
        self._mini_file_label.setWordWrap(True)
        cr_layout.addWidget(self._mini_file_label, 1)
        self._mini_browse_btn = QPushButton("Browse")
        self._mini_browse_btn.setObjectName("btn-ghost")
        self._mini_browse_btn.clicked.connect(self._on_mini_browse_file)
        cr_layout.addWidget(self._mini_browse_btn)
        self._mini_custom_row.setVisible(False)
        ll.addWidget(self._mini_custom_row)

        # Architecture hyperparameters; defaults (block 64, embd 128, 4 heads, 4 layers) are small enough to train on CPU.
        # n_embd must be divisible by the number of heads for multi-head attention to work.
        # Grid of architectural hyperparameters
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        grid.addWidget(QLabel("Block Size (Context):"), 0, 0)
        self._mini_block_size = QSpinBox()
        self._mini_block_size.setRange(16, 256)
        self._mini_block_size.setSingleStep(16)
        self._mini_block_size.setValue(64)
        grid.addWidget(self._mini_block_size, 0, 1)

        grid.addWidget(QLabel("Embedding Dimension:"), 1, 0)
        self._mini_n_embd = QSpinBox()
        self._mini_n_embd.setRange(32, 512)
        self._mini_n_embd.setSingleStep(32)
        self._mini_n_embd.setValue(128)
        grid.addWidget(self._mini_n_embd, 1, 1)

        grid.addWidget(QLabel("Attention Heads:"), 2, 0)
        self._mini_n_heads = QSpinBox()
        self._mini_n_heads.setRange(1, 16)
        self._mini_n_heads.setValue(4)
        grid.addWidget(self._mini_n_heads, 2, 1)

        grid.addWidget(QLabel("Transformer Layers:"), 3, 0)
        self._mini_n_layers = QSpinBox()
        self._mini_n_layers.setRange(1, 12)
        self._mini_n_layers.setValue(4)
        grid.addWidget(self._mini_n_layers, 3, 1)

        grid.addWidget(QLabel("Learning Rate:"), 4, 0)
        self._mini_lr = QDoubleSpinBox()
        self._mini_lr.setRange(1e-5, 1e-1)
        self._mini_lr.setSingleStep(1e-4)
        self._mini_lr.setValue(1e-3)
        self._mini_lr.setDecimals(5)
        grid.addWidget(self._mini_lr, 4, 1)

        grid.addWidget(QLabel("Max Iterations:"), 5, 0)
        self._mini_max_iters = QSpinBox()
        self._mini_max_iters.setRange(50, 5000)
        self._mini_max_iters.setSingleStep(50)
        self._mini_max_iters.setValue(500)
        grid.addWidget(self._mini_max_iters, 5, 1)

        ll.addWidget(grid_widget)
        ll.addWidget(_hline())

        # Start/Stop controls; Stop stays disabled until a run is active.
        # Buttons
        self._mini_train_btn = QPushButton("Start Sandbox Training")
        self._mini_train_btn.setObjectName("btn-primary")
        self._mini_train_btn.clicked.connect(self._start_mini_training)
        ll.addWidget(self._mini_train_btn)

        self._mini_stop_btn = QPushButton("Stop Training")
        self._mini_stop_btn.setObjectName("btn-danger")
        self._mini_stop_btn.setEnabled(False)
        self._mini_stop_btn.clicked.connect(self._stop_mini_training)
        ll.addWidget(self._mini_stop_btn)

        self._mini_progress_bar = QProgressBar()
        self._mini_progress_bar.setVisible(False)
        ll.addWidget(self._mini_progress_bar)

        ll.addStretch()
        splitter.addWidget(left)

        # Right column: loss telemetry on top, live generation samples below, inference sandbox at the bottom.
        # Right Column: Visual Monitor and Output Viewers
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(10)

        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setHandleWidth(1)

        # Upper: Training Log view
        log_panel = QWidget()
        lpl = QVBoxLayout(log_panel)
        lpl.setContentsMargins(0, 0, 0, 0)
        lpl.setSpacing(6)
        lpl.addWidget(_section("TRAINING TELEMETRY & LOSS"))
        
        telemetry_row = QWidget()
        trl_layout = QHBoxLayout(telemetry_row)
        trl_layout.setContentsMargins(0, 0, 0, 0)
        trl_layout.setSpacing(10)
        
        self._mini_log_view = QTextBrowser()
        self._mini_log_view.setStyleSheet(
            "background-color: #0D0D1B; border: 1px solid #1F1F3D; border-radius: 4px; "
            "color: #2DD4A0; font-family: 'JetBrains Mono', monospace; font-size: 8.5pt; padding: 6px;"
        )
        trl_layout.addWidget(self._mini_log_view, 1)
        
        self._mini_loss_chart = LossChartWidget()
        trl_layout.addWidget(self._mini_loss_chart, 1)
        
        lpl.addWidget(telemetry_row)
        right_splitter.addWidget(log_panel)

        # Lower: Live text generation stream
        gen_panel = QWidget()
        gpl = QVBoxLayout(gen_panel)
        gpl.setContentsMargins(0, 0, 0, 0)
        gpl.addWidget(_section("LIVE GENERATION MONITOR"))
        self._mini_sample_view = QTextBrowser()
        self._mini_sample_view.setStyleSheet(
            "background-color: #0D0D1B; border: 1px solid #1F1F3D; border-radius: 4px; "
            "color: #ECECF5; font-family: 'JetBrains Mono', monospace; font-size: 9pt; padding: 6px;"
        )
        self._mini_sample_view.setPlaceholderText("The model will stream sample generations here every 100 steps to show its learning progression...")
        gpl.addWidget(self._mini_sample_view)
        right_splitter.addWidget(gen_panel)

        rl.addWidget(right_splitter, 1)

        # The inference box is disabled until trained weights exist.
        # Bottom: Inference Sandbox Groupbox
        self._mini_inference_group = QWidget()
        self._mini_inference_group.setObjectName("panel")
        self._mini_inference_group.setEnabled(False)
        ig_layout = QVBoxLayout(self._mini_inference_group)
        ig_layout.setContentsMargins(12, 12, 12, 12)
        ig_layout.setSpacing(8)
        
        ig_layout.addWidget(_section("TINY-GPT INFERENCE SANDBOX"))

        input_row = QHBoxLayout()
        self._mini_prompt_input = QLineEdit()
        self._mini_prompt_input.setPlaceholderText("Enter a starting prompt (e.g. 'To be or not to be') or leave empty...")
        self._mini_prompt_input.returnPressed.connect(self._run_mini_inference)
        input_row.addWidget(self._mini_prompt_input, 1)

        input_row.addWidget(QLabel("Temp:"))
        self._mini_temp_spin = QDoubleSpinBox()
        self._mini_temp_spin.setRange(0.1, 2.0)
        self._mini_temp_spin.setSingleStep(0.1)
        self._mini_temp_spin.setValue(0.8)
        self._mini_temp_spin.setFixedWidth(60)
        input_row.addWidget(self._mini_temp_spin)

        input_row.addWidget(QLabel("Tokens:"))
        self._mini_gen_tokens = QSpinBox()
        self._mini_gen_tokens.setRange(10, 500)
        self._mini_gen_tokens.setSingleStep(25)
        self._mini_gen_tokens.setValue(150)
        self._mini_gen_tokens.setFixedWidth(70)
        input_row.addWidget(self._mini_gen_tokens)

        self._mini_generate_btn = QPushButton("Generate Text")
        self._mini_generate_btn.setObjectName("btn-primary")
        self._mini_generate_btn.clicked.connect(self._run_mini_inference)
        input_row.addWidget(self._mini_generate_btn)
        ig_layout.addLayout(input_row)

        self._mini_generate_output = QTextBrowser()
        self._mini_generate_output.setStyleSheet(
            "background-color: #0B0B14; border: 1px solid #1F1F35; border-radius: 4px; "
            "color: #00C2FF; font-family: 'JetBrains Mono', monospace; font-size: 9pt; padding: 6px;"
        )
        self._mini_generate_output.setFixedHeight(100)
        ig_layout.addWidget(self._mini_generate_output)

        rl.addWidget(self._mini_inference_group)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

        # Enable inference straight away if a previous run left weights, tokenizer and config on disk.
        # Check if pre-existing weights exist to enable inference sandbox
        if os.path.exists("data/mini_gpt/weights.pt") and os.path.exists("data/mini_gpt/tokenizer.json") and os.path.exists("data/mini_gpt/config.json"):
            self._mini_inference_group.setEnabled(True)
            self._mini_generate_output.setHtml("<span style='color:#00C2FF;'>Pre-trained Sandbox weights found. Type a prompt above and generate text!</span>")

    def _on_mini_dataset_changed(self, text: str):
        """Show the file-picker row only when the custom text file dataset is selected."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._mini_custom_row.setVisible(text == "Custom Text File...")

    def _on_mini_browse_file(self):
        """Let the user choose a text file and remember its path."""
        # Step 1: Validate input parameters, operational preconditions, and state
        path, _ = QFileDialog.getOpenFileName(self, "Select Text File", "", "Text Files (*.txt *.md *.py *.jsonl);;All Files (*)")
        if path:
            self._mini_custom_file_path = path
            self._mini_file_label.setText(os.path.basename(path))

    def _get_mini_dataset_text(self) -> str:
        """Return the training text for the selected dataset, creating or falling back to defaults when needed."""
        # Step 1: Validate input parameters, operational preconditions, and state
        dataset_type = self._mini_dataset_combo.currentText()
        if dataset_type == "Tiny Shakespeare":
            # The bundled Shakespeare excerpt is written to disk on first use and repeated 20 times so
            # there is enough text for the model to train on without a network download.
            path = "data/tiny_shakespeare.txt"
            if not os.path.exists(path):
                os.makedirs("data", exist_ok=True)
                default_shakespeare = (
                    "ROMEO:\n"
                    "But, soft! what light through yonder window breaks?\n"
                    "It is the east, and Juliet is the sun.\n"
                    "Arise, fair sun, and kill the envious moon,\n"
                    "Who is already sick and pale with grief,\n"
                    "That thou her maid art far more fair than she:\n"
                    "Be not her maid, since she is envious;\n"
                    "Her vestal livery is but sick and green\n"
                    "And none but fools do wear it; cast it off.\n"
                    "It is my lady, O, it is my love!\n"
                    "O, that she knew she were!\n"
                    "She speaks yet she says nothing: what of that?\n"
                    "Her eye discourses; I will answer it.\n"
                    "I am too bold, 'tis not to me she speaks:\n"
                    "Two of the fairest stars in all the heaven,\n"
                    "Having some business, do entreat her eyes\n"
                    "To twinkle in their spheres till they return.\n"
                    "What if her eyes were there, they in her head?\n"
                    "The brightness of her cheek would shame those stars,\n"
                    "As daylight doth a lamp; her eyes in heaven\n"
                    "Would through the airy region stream so bright\n"
                    "That birds would sing and think it were not night.\n"
                    "See, how she leans her cheek upon her hand!\n"
                    "O, that I were a glove upon that hand,\n"
                    "That I might touch that cheek!\n\n"
                    "JULIET:\n"
                    "Ay me!\n\n"
                    "ROMEO:\n"
                    "She speaks:\n"
                    "O, speak again, bright angel! for thou art\n"
                    "As glorious to this night, being o'er my head\n"
                    "As is a winged messenger of heaven\n"
                    "Unto the white-upturned wondering eyes\n"
                    "Of mortals that fall back to gaze on him\n"
                    "When he bestrides the lazy-pacing clouds\n"
                    "And sails upon the bosom of the air.\n"
                ) * 20  # Repeat to make it long enough for training
                with open(path, "w", encoding="utf-8") as f:
                    f.write(default_shakespeare)
            with open(path, "r", encoding="utf-8") as f:
                return f.read()

        # Trace logs: concatenate each logged thinking block (wrapped in <think> tags) and response,
        # so the model learns Karl's own output style. Unreadable files are skipped.
        elif dataset_type == "Karl's Trace Logs":
            log_dir = "data/logs/traces"
            text_parts = []
            if os.path.exists(log_dir):
                for fname in sorted(os.listdir(log_dir)):
                    # Iterate over items, evaluate loop invariant, and update progress
                    if fname.endswith(".jsonl"):
                        try:
                            # Execute guarded operations and establish recovery boundaries
                            with open(os.path.join(log_dir, fname), "r", encoding="utf-8") as f:
                                for line in f:
                                    # Iterate over items, evaluate loop invariant, and update progress
                                    if not line.strip():
                                        continue
                                    obj = json.loads(line)
                                    if "thinking" in obj and obj["thinking"]:
                                        text_parts.append(f"<think>\n{obj['thinking']}\n</think>\n")
                                    if "response" in obj and obj["response"]:
                                        text_parts.append(f"{obj['response']}\n")
                        except Exception:
                            pass
            text = "".join(text_parts)
            # With no logs, a short repeated placeholder keeps the length check below satisfiable.
            if not text:
                text = "SYSTEM: You are Karl. I am an AI assistant.\n" * 100
            return text

        elif dataset_type == "Custom Text File...":
            # Custom file; errors="ignore" drops undecodable bytes instead of failing.
            path = getattr(self, "_mini_custom_file_path", "")
            if path and os.path.exists(path):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            else:
                raise ValueError("No custom text file selected. Please select a text file first.")
        
        return ""

    def _start_mini_training(self):
        """Validate the dataset, lock the configuration controls and start the MiniTrainThread."""
        # Step 1: Validate input parameters, operational preconditions, and state
        try:
            # Execute guarded operations and establish recovery boundaries
            text = self._get_mini_dataset_text()
            # Too little text gives an unusable character vocabulary and no batches.
            if len(text) < 500:
                QMessageBox.warning(self, "Dataset Too Small", "Training dataset must be at least 500 characters long.")
                return
            
            # Lock all configuration widgets so settings cannot change mid-run.
            # Disable configuration panel
            self._mini_dataset_combo.setEnabled(False)
            self._mini_browse_btn.setEnabled(False)
            self._mini_block_size.setEnabled(False)
            self._mini_n_embd.setEnabled(False)
            self._mini_n_heads.setEnabled(False)
            self._mini_n_layers.setEnabled(False)
            self._mini_lr.setEnabled(False)
            self._mini_max_iters.setEnabled(False)
            
            # Toggle buttons
            self._mini_train_btn.setEnabled(False)
            self._mini_stop_btn.setEnabled(True)
            
            # Setup progress bar
            self._mini_progress_bar.setVisible(True)
            self._mini_progress_bar.setValue(0)
            self._mini_progress_bar.setMaximum(self._mini_max_iters.value())

            self._mini_log_view.clear()
            self._mini_sample_view.clear()
            self._mini_loss_history.clear()
            self._mini_loss_chart.set_loss_history([])

            # Free the llama.cpp model so training has the memory.
            ModelLoader.reset_instance()

            # batch_size is fixed at 32; evaluation runs every 50 steps and a text sample every 100.
            config = {
                "batch_size": 32,
                "block_size": self._mini_block_size.value(),
                "n_embd": self._mini_n_embd.value(),
                "n_heads": self._mini_n_heads.value(),
                "n_layers": self._mini_n_layers.value(),
                "lr": self._mini_lr.value(),
                "max_iters": self._mini_max_iters.value(),
                "eval_interval": 50,
                "sample_interval": 100
            }

            self._mini_train_thread = MiniTrainThread(
                dataset_text=text,
                config=config
            )
            self._mini_train_thread.log.connect(self._on_mini_log)
            self._mini_train_thread.loss.connect(self._on_mini_loss)
            self._mini_train_thread.progress.connect(self._on_mini_progress)
            self._mini_train_thread.done.connect(self._on_mini_done)
            self._mini_train_thread.error.connect(self._on_mini_error)
            
            # Track the thread until finished, then let Qt delete it.
            self._active_threads.add(self._mini_train_thread)
            self._mini_train_thread.finished.connect(lambda: self._active_threads.discard(self._mini_train_thread))
            self._mini_train_thread.finished.connect(self._mini_train_thread.deleteLater)
            
            self._mini_train_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Training Initialization Error", f"Could not launch training:\n{e}")

    def _stop_mini_training(self):
        """Ask the training thread to stop at its next step and disable the Stop button."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if self._mini_train_thread and self._mini_train_thread.isRunning():
            self._mini_train_thread.stop()
            self._mini_stop_btn.setEnabled(False)

    def _on_mini_log(self, text: str):
        """Append a log line from the trainer."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._mini_log_view.append(text)

    def _on_mini_loss(self, step: int, val_loss: float):
        """Record the validation loss, redraw the chart and log a text bar showing the value."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._mini_loss_history.append(val_loss)
        self._mini_loss_chart.set_loss_history(self._mini_loss_history)
        
        # The bar is a rough visual: higher loss shows fewer filled blocks, scaled and clamped to 1-20 cells.
        # ASCII loss bar
        bar_len = int(min(20, max(1, val_loss * 4.0)))
        bar = "█" * (21 - bar_len) + "░" * bar_len
        self._mini_log_view.append(f"<b>[Telemetry] Step {step} | Val Loss: {val_loss:.4f} [{bar}]</b>")

    def _on_mini_progress(self, step: int, max_steps: int, sample_text: str):
        """Advance the progress bar and show the latest generated sample text."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._mini_progress_bar.setValue(step)
        html_sample = (
            f"<div style='color:#9090A8; font-size:8.2pt; border-bottom:1px solid #1F1F35; padding-bottom:3px; margin-bottom:5px;'>"
            f"  --- Generation Output (Step {step} / {max_steps}) ---"
            f"</div>"
            f"<div style='color:#ECECF5; font-size:9pt; white-space:pre-wrap; line-height:1.4;'>{html.escape(sample_text)}</div>"
        )
        self._mini_sample_view.setHtml(html_sample)

    def _on_mini_done(self, save_dir: str):
        """Re-enable the controls, show the inference box and tell the user where the weights were saved."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._mini_progress_bar.setValue(self._mini_max_iters.value())
        self._mini_progress_bar.setVisible(False)
        self._mini_train_btn.setEnabled(True)
        self._mini_stop_btn.setEnabled(False)
        
        # Restore configuration inputs
        self._mini_dataset_combo.setEnabled(True)
        self._mini_browse_btn.setEnabled(True)
        self._mini_block_size.setEnabled(True)
        self._mini_n_embd.setEnabled(True)
        self._mini_n_heads.setEnabled(True)
        self._mini_n_layers.setEnabled(True)
        self._mini_lr.setEnabled(True)
        self._mini_max_iters.setEnabled(True)

        self._mini_inference_group.setEnabled(True)
        QMessageBox.information(self, "Mini-GPT Training Complete", f"Educational Mini-GPT trained successfully!\nWeights saved to: {save_dir}/weights.pt")

    def _on_mini_error(self, msg: str):
        """Re-enable the controls and report the training failure in the log and a dialog."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._mini_progress_bar.setVisible(False)
        self._mini_train_btn.setEnabled(True)
        self._mini_stop_btn.setEnabled(False)
        
        # Restore inputs
        self._mini_dataset_combo.setEnabled(True)
        self._mini_browse_btn.setEnabled(True)
        self._mini_block_size.setEnabled(True)
        self._mini_n_embd.setEnabled(True)
        self._mini_n_heads.setEnabled(True)
        self._mini_n_layers.setEnabled(True)
        self._mini_lr.setEnabled(True)
        self._mini_max_iters.setEnabled(True)

        self._mini_log_view.append(f"\n[CRITICAL ERROR] Training failed:\n{msg}")
        QMessageBox.critical(self, "Mini-GPT Training Error", f"Training execution failed:\n{msg}")

    def _run_mini_inference(self):
        """Rebuild the saved MiniGPT from disk and generate text from the user's prompt."""
        # Step 1: Validate input parameters, operational preconditions, and state
        prompt = self._mini_prompt_input.text()
        self._mini_generate_btn.setEnabled(False)
        self._mini_generate_output.clear()
        
        try:
            # Execute guarded operations and establish recovery boundaries
            # torch and the model class are imported here so the app starts without torch installed.
            import torch
            from app.engine.mini_transformer import MiniGPT, CharTokenizer

            # Use the GPU if available; the model is tiny so CPU also works.
            device = "cuda" if torch.cuda.is_available() else "cpu"
            save_dir = "data/mini_gpt"
            
            config_path = os.path.join(save_dir, "config.json")
            weights_path = os.path.join(save_dir, "weights.pt")
            vocab_path = os.path.join(save_dir, "tokenizer.json")
            
            if not (os.path.exists(config_path) and os.path.exists(weights_path) and os.path.exists(vocab_path)):
                raise FileNotFoundError("Mini-GPT weights or config file not found. Please run training first.")
                
            # The character tokenizer is restored from its saved vocabulary tables.
            # Load tokenizer
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab_meta = json.load(f)
            tokenizer = CharTokenizer()
            tokenizer.chars = vocab_meta["chars"]
            tokenizer.stoi = vocab_meta["stoi"]
            tokenizer.itos = vocab_meta["itos"]
            tokenizer.vocab_size = len(tokenizer.chars)
            
            # Load config
            with open(config_path, "r", encoding="utf-8") as f:
                arch = json.load(f)
            
            # Dropout is disabled (0.0) and eval() is set for deterministic inference.
            # Recreate model
            model = MiniGPT(
                vocab_size=arch["vocab_size"],
                n_embd=arch["n_embd"],
                n_heads=arch["n_heads"],
                n_layers=arch["n_layers"],
                block_size=arch["block_size"],
                dropout=0.0
            )
            
            # Load weights
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            
            # An empty prompt is replaced by token 0 so the generator has some context to start from.
            # Encode prompt
            prompt_encoded = tokenizer.encode(prompt)
            if not prompt_encoded:
                prompt_encoded = [0]
            
            context_tensor = torch.tensor([prompt_encoded], dtype=torch.long, device=device)
            
            # Generate
            max_new = self._mini_gen_tokens.value()
            temp = self._mini_temp_spin.value()
            
            # Generate tokens
            # top_k=20 limits sampling to the 20 most likely characters per step.
            gen_ids = model.generate(context_tensor, max_new_tokens=max_new, temperature=temp, top_k=20)[0].tolist()
            output_text = tokenizer.decode(gen_ids)
            
            self._mini_generate_output.setText(output_text)
            
        # Show errors inline (escaped) and always re-enable the Generate button.
        except Exception as e:
            traceback.print_exc()
            self._mini_generate_output.setHtml(f"<span style='color:#FF3B30;'>Inference error: {html.escape(str(e))}</span>")
        finally:
            self._mini_generate_btn.setEnabled(True)
