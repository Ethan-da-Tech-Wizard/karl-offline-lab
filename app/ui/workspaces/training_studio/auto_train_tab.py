# ==============================================================================
# Program: auto_train_tab.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Desktop workspace implementation for Auto Train Tab.
#          Provides responsive layout controls, parameter adjustment surfaces,
#          and visual feedback components within Karl's PyQt6 shell.
# ==============================================================================
# Auto-Train tab: a one-click front end for the flywheel pipeline. It gathers a
# topic, adapter name and LoRA hyperparameters, starts an AutoTrainThread, and
# streams that thread's log lines into a read-only view.

import re

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextBrowser,
    QLabel, QSpinBox, QDoubleSpinBox, QProgressBar, QCheckBox,
    QLineEdit, QGridLayout, QFrame, QMessageBox
)

from app.ui.widgets.glow_panel import GlowPanel
from app.ui.workspaces.training_studio.threads import AutoTrainThread
from app.engine.model_loader import ModelLoader

# Per-tab copy of the section-header helper (also in export_tab and others): keeps
# each Training Studio tab importable on its own, at the cost of three duplicated
# lines.
def _section(text: str) -> QLabel:
    """Return a QLabel styled as a section heading (same helper as in the other tabs)."""
    lbl = QLabel(text)
    lbl.setObjectName("section-header")
    return lbl


def _hline() -> QFrame:
    """Return a horizontal rule used to separate groups of controls."""
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    return f


# Thin front end: it validates inputs, frees the GPU and launches AutoTrainThread
# (threads.py), which runs the pipeline as an external process and emits log(str),
# done(adapter_name) and error(str). Everything after "start" arrives through those
# three signals, so this class holds no pipeline logic of its own.
class AutoTrainTab(QWidget):
    """Tab widget that configures and launches the automatic synthesize-solve-verify-train loop."""
    def __init__(self, state, parent=None):
        """Store state and build the tab; no thread runs until the user starts one."""
        super().__init__(parent)
        self.state = state
        # _active_threads keeps a strong reference to running QThreads so they are not garbage collected.
        self._active_threads = set()
        # Only the most recent thread; _active_threads is the authoritative set used for
        # shutdown. A second run cannot start while the first is active because the
        # start button is disabled for the duration.
        self._auto_thread = None
        self._build_ui()

    def _build_ui(self):
        """Build the configuration column (left) and the guide plus log column (right)."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Left column: configuration form in a GlowPanel.
        # GlowPanel (not a plain "panel") makes the configuration card pick up the
        # glow/motion settings from the Theme tab; stretch 1:1 with the right column
        # gives the log as much width as the form.
        left_col = GlowPanel(self.state)
        left_layout = QVBoxLayout(left_col)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(8)

        left_layout.addWidget(_section("FLYWHEEL AUTO-TRAIN CONFIG"))

        # Form layout-like structure
        left_layout.addWidget(QLabel("Target Behavior / Topic:"))
        # Free text handed to the synthesizer as the behaviour to generate problems
        # for; no validation beyond "non-empty" because any topic is legitimate.
        self._auto_topic_input = QLineEdit()
        self._auto_topic_input.setPlaceholderText("e.g., modular arithmetic, binary search")
        self._auto_topic_input.setToolTip("Enter the specific capability you want Karl to learn")
        left_layout.addWidget(self._auto_topic_input)

        left_layout.addWidget(QLabel("Adapter Save Name:"))
        # Becomes both a folder and a GGUF file name under data/adapters (see
        # _on_auto_done), which is why _begin_auto_training sanitises it.
        self._auto_adapter_input = QLineEdit()
        self._auto_adapter_input.setPlaceholderText("e.g., math_specialist")
        self._auto_adapter_input.setToolTip("Save folder name under data/adapters/")
        left_layout.addWidget(self._auto_adapter_input)

        # Two label/value pairs per row (columns 0-1 and 2-3) so six numeric settings fit
        # in three rows without making the left column tall.
        grid = QGridLayout()
        grid.setSpacing(8)
        grid.setContentsMargins(0, 4, 0, 4)

        # Ranges: examples 2-200, epochs 1-20, LoRA rank 1-256, alpha 1-512, learning rate 1e-6..1e-2.
        # Defaults (15 examples, 3 epochs, rank 16, alpha 32, lr 2e-4) are a conventional small-LoRA starting point.
        # Generate Count
        self._auto_count_spin = QSpinBox()
        # Minimum 2 (a single example cannot form a dataset worth training on);
        # maximum 200 bounds the synthesis/verification time, which scales with count
        # (each problem is generated, solved and run in the sandbox).
        self._auto_count_spin.setRange(2, 200)
        self._auto_count_spin.setValue(15)
        self._auto_count_spin.setToolTip("Number of training problems to generate and verify")
        grid.addWidget(QLabel("Examples Count:"), 0, 0)
        grid.addWidget(self._auto_count_spin, 0, 1)

        # Epochs
        self._auto_epochs_spin = QSpinBox()
        # Small datasets overfit quickly, so more than 20 passes is not offered.
        self._auto_epochs_spin.setRange(1, 20)
        self._auto_epochs_spin.setValue(3)
        grid.addWidget(QLabel("Epochs:"), 0, 2)
        grid.addWidget(self._auto_epochs_spin, 0, 3)

        # Rank
        self._auto_rank_spin = QSpinBox()
        # LoRA rank r: adapter capacity (and file size). 16 is a common small-model default.
        self._auto_rank_spin.setRange(1, 256)
        self._auto_rank_spin.setValue(16)
        grid.addWidget(QLabel("Rank:"), 1, 0)
        grid.addWidget(self._auto_rank_spin, 1, 1)

        # Alpha
        self._auto_alpha_spin = QSpinBox()
        # LoRA alpha scales the adapter output by alpha/rank. The default 32 with rank
        # 16 gives the usual factor of 2.
        self._auto_alpha_spin.setRange(1, 512)
        self._auto_alpha_spin.setValue(32)
        grid.addWidget(QLabel("Alpha:"), 1, 2)
        grid.addWidget(self._auto_alpha_spin, 1, 3)

        # LR
        self._auto_lr_spin = QDoubleSpinBox()
        # Six decimals so 2e-4 (0.000200) is displayed exactly; the 1e-5 step moves it by
        # roughly one order of fine granularity without jumping past useful values.
        self._auto_lr_spin.setDecimals(6)
        self._auto_lr_spin.setRange(1e-6, 1e-2)
        self._auto_lr_spin.setSingleStep(1e-5)
        self._auto_lr_spin.setValue(2e-4)
        grid.addWidget(QLabel("Learning Rate:"), 2, 0)
        grid.addWidget(self._auto_lr_spin, 2, 1)

        # QLoRA (4-bit) is on by default so training fits on small GPUs.
        # QLoRA Checkbox
        self._auto_qlora_check = QCheckBox("4-bit QLoRA")
        self._auto_qlora_check.setChecked(True)
        grid.addWidget(self._auto_qlora_check, 2, 2, 1, 2)

        left_layout.addLayout(grid)
        left_layout.addWidget(_hline())

        # Auto Train Button
        self._auto_train_btn = QPushButton("▶ start auto-training flywheel")
        self._auto_train_btn.setObjectName("btn-primary")
        # Taller than normal buttons: this is the one primary action on the page.
        self._auto_train_btn.setFixedHeight(36)
        self._auto_train_btn.clicked.connect(self._begin_auto_training)
        left_layout.addWidget(self._auto_train_btn)

        # Progress bar
        # Hidden until a run starts; configured as indeterminate in _begin_auto_training.
        self._auto_progress = QProgressBar()
        self._auto_progress.setVisible(False)
        left_layout.addWidget(self._auto_progress)

        left_layout.addStretch()
        layout.addWidget(left_col, 1)

        # Right column: explanation of the pipeline and the live log.
        # Right Column: Guide & Output Logs
        right_col = QWidget()
        right_layout = QVBoxLayout(right_col)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # Guide panel explaining the flywheel
        # Static explainer of the five flywheel stages so the log lines the user will see
        # (synthesizer, solver, sandbox, reflection, training) are recognisable.
        guide_box = GlowPanel(self.state)
        gl = QVBoxLayout(guide_box)
        gl.setContentsMargins(12, 10, 12, 10)
        
        guide_text = (
            "🚀 <b>One-Click Auto-Train Flywheel:</b><br>"
            "This tab runs Karl's self-improvement flywheel completely autonomously:<br>"
            "1. <b>Synthesizer:</b> Generates synthetic problems for your custom behavior.<br>"
            "2. <b>Solver:</b> Attempts to solve the generated problems.<br>"
            "3. <b>Curator & Sandbox:</b> Verifies solutions in a secure execution sandbox.<br>"
            "4. <b>Self-Reflection:</b> Mistakes trigger an LLM-based debugging correction loop.<br>"
            "5. <b>Training:</b> Custom datasets are created, and SFT LoRA training begins."
        )
        guide_lbl = QLabel(guide_text)
        guide_lbl.setWordWrap(True)
        guide_lbl.setStyleSheet("font-size: 8.5pt; color: #00C2FF; line-height: 1.4;")
        gl.addWidget(guide_lbl)
        right_layout.addWidget(guide_box)

        # Output Logs
        log_box = QWidget()
        log_lay = QVBoxLayout(log_box)
        log_lay.setContentsMargins(0, 0, 0, 0)
        log_lay.setSpacing(4)
        log_lay.addWidget(_section("FLYWHEEL AUTO-TRAIN LOGS"))
        # Object name "reasoning-view" opts this log into the shared QSS rule for read-only
        # log/reasoning text; append() adds a paragraph and keeps the view scrolled to the
        # newest line while the user has not scrolled away.
        self._auto_log = QTextBrowser()
        self._auto_log.setObjectName("reasoning-view")
        self._auto_log.setPlaceholderText("Logs will stream here when auto-training begins...")
        self._auto_log.setStyleSheet("font-family: monospace; font-size: 9pt;")
        log_lay.addWidget(self._auto_log)
        right_layout.addWidget(log_box, 1)

        layout.addWidget(right_col, 1)

    def _begin_auto_training(self):
        """Validate the inputs, free VRAM, then start the AutoTrainThread with the chosen configuration."""
        # Validation failures are written to the log view instead of a dialog: they are
        # recoverable, and the log is already where the user is looking.
        topic = self._auto_topic_input.text().strip()
        adapter_name = self._auto_adapter_input.text().strip()
        if not topic:
            self._auto_log.append("Please specify a target behavior/topic.")
            return
        if not adapter_name:
            self._auto_log.append("Please specify an adapter save name.")
            return

        # Adapter name becomes a folder under data/adapters, so strip anything but letters, digits, _ and -.
        # Sanitize adapter name for file path
        adapter_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', adapter_name)
        # Echoed back so the user sees the name that will actually be used on disk.
        self._auto_adapter_input.setText(adapter_name)

        # Unload the inference model first; training needs the GPU/RAM that the loaded model occupies.
        # Release VRAM first
        # Unconditional, even if a generation is running: unlike the System tab loads,
        # this path does not check is_instance_locked, so starting training during active
        # inference drops the model under it.
        ModelLoader.reset_instance()

        # Disabled until done or error so only one flywheel run exists at a time (it
        # owns the GPU and the adapter folder).
        self._auto_train_btn.setEnabled(False)
        self._auto_progress.setVisible(True)
        # A (0, 0) range makes the bar an indeterminate busy indicator; the pipeline reports no percentage.
        self._auto_progress.setRange(0, 0)  # Indeterminate progress bar
        self._auto_log.clear()
        self._auto_log.append("Starting automated flywheel training pipeline...")

        # Keys match what AutoTrainThread expects in its config dict.
        config = {
            "count": self._auto_count_spin.value(),
            "epochs": self._auto_epochs_spin.value(),
            "rank": self._auto_rank_spin.value(),
            "alpha": self._auto_alpha_spin.value(),
            "lr": self._auto_lr_spin.value(),
            "use_qlora": self._auto_qlora_check.isChecked()
        }

        # Keep the thread referenced until finished, then let Qt delete it.
        # Lambda default argument (t=...) binds this thread object; connected finished
        # slots run on the GUI thread after run() returns, regardless of success.
        self._auto_thread = AutoTrainThread(topic, adapter_name, config)
        self._active_threads.add(self._auto_thread)
        self._auto_thread.finished.connect(
            lambda t=self._auto_thread: self._active_threads.discard(t)
        )
        self._auto_thread.finished.connect(self._auto_thread.deleteLater)
        # done and error are mutually exclusive outcomes and both re-enable the button;
        # the thread's finished signal only does memory cleanup.
        self._auto_thread.log.connect(self._on_auto_log)
        self._auto_thread.done.connect(self._on_auto_done)
        self._auto_thread.error.connect(self._on_auto_error)
        self._auto_thread.start()

    def _on_auto_log(self, text: str):
        """Append a log line from the worker to the log view."""
        # append() adds a new paragraph per call, matching the one-line-per-callback
        # emission of the worker.
        self._auto_log.append(text)

    def _on_auto_done(self, adapter_name: str):
        """Re-enable the controls, announce the adapter location and clear the inputs."""
        self._auto_train_btn.setEnabled(True)
        self._auto_progress.setVisible(False)
        QMessageBox.information(
            self, "Auto-Training Complete",
            f"Auto-training completed successfully!\nAdapter GGUF ready: data/adapters/{adapter_name}/{adapter_name}.gguf"
        )
        # Both inputs are cleared on success so the next run starts from a clean form; on
        # error they are kept so the user can fix the problem and retry the same request.
        self._auto_topic_input.clear()
        self._auto_adapter_input.clear()

    def _on_auto_error(self, msg: str):
        """Re-enable the controls and show the error both in the log and in a dialog."""
        self._auto_train_btn.setEnabled(True)
        self._auto_progress.setVisible(False)
        # Logged as well as shown in a dialog so the failure stays visible in the scrollback
        # after the modal is dismissed.
        self._auto_log.append(f"\n[ERROR] Auto-training failed: {msg}")
        QMessageBox.critical(self, "Auto-Training Error", f"Auto-training failed:\n{msg}")
