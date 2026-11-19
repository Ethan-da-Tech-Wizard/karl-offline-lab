# ==============================================================================
# Program: ai_lab.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: AI Lab Workspace — Interactive pipeline visualizer and agent composer.
#          Coordinates with adjacent subsystems within the Karl Offline Lab
#          architecture.
# ==============================================================================

"""
AI Lab Workspace — Interactive pipeline visualizer and agent composer.
"""

from __future__ import annotations

import os
import json
import re
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QTabWidget,
    QPushButton, QTextBrowser, QLineEdit, QLabel, QFrame,
    QComboBox, QSpinBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QMessageBox, QTextEdit, QStackedWidget,
    QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QFont, QPainterPath

from app.state import AppState
from app.ui.workspaces.knowledge_base import VectorProjectionWidget
from app.utils.custom_embeddings import TfidfEmbedder
from app.ui.themes import get_theme_colors, MONO


def _hline() -> QFrame:
    """Create a horizontal divider (QSS object name 'line')."""
    # Step 1: Validate input parameters, operational preconditions, and state
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setObjectName("line")
    return f


def _section(text: str) -> QLabel:
    """Create a bold, letter-spaced section header label."""
    # Step 1: Validate input parameters, operational preconditions, and state
    lbl = QLabel(text)
    lbl.setObjectName("section-header")
    lbl.setStyleSheet("font-weight: bold; letter-spacing: 1.5px;")
    return lbl


def _label(text: str, obj: str = "") -> QLabel:
    """Create a QLabel, optionally tagging it with a QSS object name (e.g. 'lbl-muted')."""
    # Step 1: Validate input parameters, operational preconditions, and state
    lbl = QLabel(text)
    if obj:
        lbl.setObjectName(obj)
    return lbl


def _row(label_text: str, widget: QWidget) -> QWidget:
    """Wrap `widget` in a row with a fixed-width muted caption on the left (form-style layout)."""
    # Step 1: Validate input parameters, operational preconditions, and state
    w = QWidget()
    lo = QHBoxLayout(w)
    lo.setContentsMargins(0, 2, 0, 2)
    lo.setSpacing(12)
    lbl = QLabel(label_text)
    lbl.setFixedWidth(150)
    lbl.setObjectName("lbl-muted")
    lo.addWidget(lbl)
    lo.addWidget(widget)
    lo.addStretch()
    return w


# Custom-painted vertical flowchart of the five RAG retrieval steps. It draws
# itself with QPainter so it needs no image assets and follows the theme.
class ExecutionTraceFlowchart(QWidget):
    def __init__(self, parent=None):
        """Create the flowchart with its default (sparse/TF-IDF) step texts."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(parent)
        self.setMinimumHeight(440)
        self.state = None
        self._is_sparse = True
        
        # (title, description) per box. Index 1 is rewritten by set_mode() to name the
        # active vectorizer.
        self.steps = [
            ("1. USER QUERY INPUT", "Captures raw query string from the user input."),
            ("2. QUERY VECTORIZATION (TF-IDF)", "Converts query into a normalized vector."),
            ("3. COSINE SIMILARITY SCAN", "Computes dot product of query vector against document vectors."),
            ("4. SCORE RANKING & FILTERING", "Sorts documents by similarity and filters by top-k / threshold."),
            ("5. CONTEXT SYNTHESIS", "Constructs prompt context with retrieved reference text.")
        ]

    def set_mode(self, is_sparse: bool):
        """Switch the vectorization step's label between TF-IDF and SentenceTransformer and repaint."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._is_sparse = is_sparse
        mode_str = "TF-IDF" if is_sparse else "SentenceTransformer"
        self.steps[1] = (f"2. QUERY VECTORIZATION ({mode_str})", f"Converts query into a normalized vector using {mode_str}.")
        self.update()

    def paintEvent(self, event):
        """Draw the card background, one box per step, and arrows between boxes.

        Layout is computed from the widget width each paint (boxes centred, fixed
        420x56 size with a 20px gap), so the chart stays centred on resize.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # Hard-coded dark defaults are used until `state` is assigned by the owner.
        # Get theme colors
        p = {
            "bg_deep": "#020205",
            "bg_surface": "#0D0D1B",
            "border": "#1F1F3D",
            "accent": "#00E5FF",
            "text_hi": "#F0F5FF",
            "text_mid": "#A0AEC0"
        }
        if self.state:
            p = get_theme_colors(self.state)
            
        # Draw background card
        painter.setBrush(QBrush(QColor(p["bg_deep"])))
        painter.setPen(QPen(QColor(p["border"]), 1))
        painter.drawRoundedRect(0, 0, w, h, 8.0, 8.0)
        
        box_w = 420
        box_h = 56
        gap = 20
        start_y = 20
        
        for i, (title, desc) in enumerate(self.steps):
            # Iterate over items, evaluate loop invariant, and update progress
            box_x = (w - box_w) // 2
            box_y = start_y + i * (box_h + gap)
            
            # Draw box
            painter.setBrush(QBrush(QColor(p["bg_surface"])))
            painter.setPen(QPen(QColor(p["accent"]), 1.2))
            painter.drawRoundedRect(box_x, box_y, box_w, box_h, 6.0, 6.0)
            
            # Title
            painter.setPen(QPen(QColor(p["accent"])))
            painter.setFont(QFont("JetBrains Mono", 8, QFont.Weight.Bold))
            painter.drawText(box_x + 16, box_y + 22, title)
            
            # Description
            painter.setPen(QPen(QColor(p["text_mid"])))
            painter.setFont(QFont("JetBrains Mono", 7.5))
            painter.drawText(box_x + 16, box_y + 40, desc)
            
            # Vertical line plus a small filled triangle pointing down at the next box.
            # Draw arrow to next box
            if i < len(self.steps) - 1:
                arrow_x = w // 2
                arrow_y1 = box_y + box_h
                arrow_y2 = arrow_y1 + gap
                
                painter.setPen(QPen(QColor(p["accent"]), 1.2))
                painter.drawLine(arrow_x, arrow_y1, arrow_x, arrow_y2)
                
                # Draw arrowhead
                arrowhead_size = 5
                arrowhead = QPainterPath()
                arrowhead.moveTo(arrow_x, arrow_y2)
                arrowhead.lineTo(arrow_x - arrowhead_size, arrow_y2 - arrowhead_size)
                arrowhead.lineTo(arrow_x + arrowhead_size, arrow_y2 - arrowhead_size)
                arrowhead.closeSubpath()
                painter.setBrush(QBrush(QColor(p["accent"])))
                painter.drawPath(arrowhead)


# Two tabs: a Pipeline Visualizer that shows, step by step, how documents are
# vectorized and ranked against a query (sparse TF-IDF or dense neural
# embeddings), and an Agent Composer that saves custom agent definitions to
# data/custom_agents.json.
class AILabWorkspace(QWidget):
    # Emitted with the agent id after a custom agent is saved to disk.
    agent_published = pyqtSignal(str)

    def __init__(self, state: AppState, parent=None):
        """Build both tabs and subscribe to AppState changes (active model/adapter labels)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(parent)
        self.setObjectName("workspace-root")
        self.state = state
        # The sandbox TF-IDF model is local to this tab: it is fitted on the user's
        # sandbox sentences only, never on the real knowledge base.
        self._tfidf = TfidfEmbedder()
        self._cached_embeddings = []
        self._cached_sentences = []

        # Main Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Tab Widget
        self._tabs = QTabWidget()
        self._tabs.setObjectName("tabs")
        
        self._build_pipeline_tab()
        self._build_composer_tab()

        self._tabs.addTab(self._pipeline_tab_w, "Pipeline Visualizer")
        self._tabs.addTab(self._composer_tab_w, "Agent Composer")

        layout.addWidget(self._tabs)

        # Keeps the 'Active Model/Adapter' labels in sync with the rest of the app.
        # Listen to state changes for active model / adapter labels
        self.state.state_changed.connect(self._on_state_changed)

    # ── TAB A: Pipeline Visualizer ──────────────────────────────────────────

    def _build_pipeline_tab(self):
        """Build tab A: configuration drawer (left) and visual inspector tabs (right)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._pipeline_tab_w = QWidget()
        main_lay = QHBoxLayout(self._pipeline_tab_w)
        main_lay.setContentsMargins(0, 8, 0, 0)
        main_lay.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("splitter")

        # Left Panel (Configurations Drawer)
        left_panel = QWidget()
        left_panel.setObjectName("panel")
        left_lay = QVBoxLayout(left_panel)
        left_lay.setContentsMargins(12, 12, 12, 12)
        left_lay.setSpacing(10)

        left_lay.addWidget(_section("PIPELINE CONFIGURATIONS"))

        # Custom sentences text area
        left_lay.addWidget(_label("Sandbox Documents (One per line):", "lbl-muted"))
        self._sentences_input = QTextEdit()
        self._sentences_input.setPlaceholderText("Enter sandbox sentences...")
        self._sentences_input.setMinimumHeight(150)
        
        # Two similar pairs (fox/dog, doctor/physician) make the similarity ranking
        # visibly meaningful for a first run.
        # Populate with default sentences
        default_docs = [
            "The quick brown fox jumps over the lazy dog",
            "Dogs and foxes are animals that run fast",
            "A doctor heals patients in a hospital clinic",
            "The medical physician treated patients yesterday"
        ]
        self._sentences_input.setPlainText("\n".join(default_docs))
        left_lay.addWidget(self._sentences_input)

        # Index 0 = sparse TF-IDF, index 1 = dense SentenceTransformer; the rest of
        # the code tests `currentIndex() == 0` for sparse mode.
        # Vectorizer combo
        left_lay.addWidget(_label("Vectorizer Mode:", "lbl-muted"))
        self._vectorizer_combo = QComboBox()
        self._vectorizer_combo.addItems(["Sparse (TF-IDF)", "Dense (Neural - SentenceTransformer)"])
        self._vectorizer_combo.currentIndexChanged.connect(self._on_vectorizer_changed)
        left_lay.addWidget(self._vectorizer_combo)

        left_lay.addWidget(_hline())

        # Informational only: shows which model/adapter the rest of the app uses.
        # Read-only model info labels
        active_model = self.state.model_name or "unknown"
        active_adapter = self.state.adapter_name or "None"
        self._active_model_lbl = _label(f"Active Model: {active_model}", "lbl-muted")
        self._active_adapter_lbl = _label(f"Adapter State: {active_adapter}", "lbl-muted")
        
        left_lay.addWidget(self._active_model_lbl)
        left_lay.addWidget(self._active_adapter_lbl)

        left_lay.addStretch()

        # Run Button
        self._run_pipeline_btn = QPushButton("Run Pipeline")
        self._run_pipeline_btn.setObjectName("btn-primary")
        self._run_pipeline_btn.clicked.connect(self._run_pipeline)
        left_lay.addWidget(self._run_pipeline_btn)

        splitter.addWidget(left_panel)

        # Right Panel (Visual Inspector)
        right_panel = QWidget()
        right_lay = QVBoxLayout(right_panel)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(8)

        self._inspector_tabs = QTabWidget()
        self._inspector_tabs.setObjectName("inspector-tabs")

        # Sub-tab 1: Vector Math
        self._build_vector_math_subtab()
        self._inspector_tabs.addTab(self._vector_math_tab, "Vector Math")

        # Sub-tab 2: RAG Playground
        self._build_rag_playground_subtab()
        self._inspector_tabs.addTab(self._rag_playground_tab, "RAG Playground")

        # Sub-tab 3: 2D Projection
        self._build_projection_subtab()
        self._inspector_tabs.addTab(self._projection_tab, "2D Projection")

        # Sub-tab 4: Execution Trace
        self._build_execution_trace_subtab()
        self._inspector_tabs.addTab(self._execution_trace_tab, "Execution Trace")

        right_lay.addWidget(self._inspector_tabs)
        splitter.addWidget(right_panel)

        # Initial split roughly 30/70.
        # Set ratio 30% left, 70% right
        splitter.setSizes([300, 700])
        main_lay.addWidget(splitter)

    # ── TAB A - SUBTABS ──────────────────────────────────────────────────────

    def _build_vector_math_subtab(self):
        """Build the 'Vector Math' sub-tab: a stack with a sparse page and a dense page."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._vector_math_tab = QWidget()
        layout = QVBoxLayout(self._vector_math_tab)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(10)

        self._math_stack = QStackedWidget()

        # Sparse page: vocabulary table (DF/IDF) above the document-term matrix.
        # Page 0: Sparse (TF-IDF) View
        sparse_w = QWidget()
        sparse_lay = QVBoxLayout(sparse_w)
        sparse_lay.setContentsMargins(0, 0, 0, 0)
        sparse_lay.setSpacing(8)

        # Vocabulary Table
        sparse_lay.addWidget(_section("VOCABULARY (DF & IDF)"))
        self._vocab_table = QTableWidget()
        self._vocab_table.setColumnCount(3)
        self._vocab_table.setHorizontalHeaderLabels(["Word/Token", "DF (Doc Freq)", "IDF Score"])
        self._vocab_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._vocab_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        sparse_lay.addWidget(self._vocab_table, 1)

        # Matrix Table
        sparse_lay.addWidget(_section("DOCUMENT-TERM MATRIX"))
        self._matrix_table = QTableWidget()
        self._matrix_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        sparse_lay.addWidget(self._matrix_table, 1)

        # Dense page: embedding dimension and a preview of the first vector.
        # Page 1: Dense (Neural) View
        dense_w = QWidget()
        dense_lay = QVBoxLayout(dense_w)
        dense_lay.setContentsMargins(0, 0, 0, 0)
        dense_lay.setSpacing(8)

        dense_lay.addWidget(_section("NEURAL VECTOR SPECIFICATIONS"))
        self._neural_dim_lbl = _label("Dimension: N/A", "lbl-muted")
        dense_lay.addWidget(self._neural_dim_lbl)

        dense_lay.addWidget(_section("EMBEDDING PREVIEW (FIRST 20 VALUES OF SENTENCE 1)"))
        self._neural_table = QTableWidget()
        self._neural_table.setColumnCount(2)
        self._neural_table.setHorizontalHeaderLabels(["Index", "Float Value"])
        self._neural_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._neural_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        dense_lay.addWidget(self._neural_table, 1)

        self._math_stack.addWidget(sparse_w)
        self._math_stack.addWidget(dense_w)
        layout.addWidget(self._math_stack)

    def _build_rag_playground_subtab(self):
        """Build the 'RAG Playground' sub-tab: query box plus a step-by-step cosine math report."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._rag_playground_tab = QWidget()
        layout = QVBoxLayout(self._rag_playground_tab)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(10)

        # Query Input row
        row_w = QWidget()
        row_lay = QHBoxLayout(row_w)
        row_lay.setContentsMargins(0, 0, 0, 0)
        row_lay.setSpacing(8)

        self._query_input = QLineEdit()
        self._query_input.setPlaceholderText("Enter test query sentence to scan similarity...")
        self._query_input.returnPressed.connect(self._run_query)
        row_lay.addWidget(self._query_input, 1)

        self._run_query_btn = QPushButton("Run Query")
        self._run_query_btn.clicked.connect(self._run_query)
        row_lay.addWidget(self._run_query_btn)

        layout.addWidget(row_w)

        # Math formulas display
        layout.addWidget(_section("COSINE SIMILARITY MATH CALCULATIONS"))
        self._rag_results_browser = QTextBrowser()
        self._rag_results_browser.setHtml("<span style='color:#6A7B95;'>Run pipeline first, then enter a query above.</span>")
        layout.addWidget(self._rag_results_browser, 1)

    def _build_projection_subtab(self):
        """Build the '2D Projection' sub-tab hosting the shared VectorProjectionWidget."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._projection_tab = QWidget()
        layout = QVBoxLayout(self._projection_tab)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(_section("2D VECTOR PROJECTION"))
        self._projection_widget = VectorProjectionWidget()
        layout.addWidget(self._projection_widget, 1)

    def _build_execution_trace_subtab(self):
        """Build the 'Execution Trace' sub-tab hosting the flowchart."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._execution_trace_tab = QWidget()
        layout = QVBoxLayout(self._execution_trace_tab)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(_section("RAG COMPILATION EXECUTION FLOWCHART"))
        self._flowchart_widget = ExecutionTraceFlowchart()
        self._flowchart_widget.state = self.state
        layout.addWidget(self._flowchart_widget, 1)

    # ── TAB B: Agent Composer ────────────────────────────────────────────────

    def _build_composer_tab(self):
        """Build tab B: a scrollable form describing a custom agent and a Publish button."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._composer_tab_w = QWidget()
        
        # Scroll area keeps the form usable in small windows.
        # We wrap in a scroll area to guarantee readability in small layouts
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        content_w = QWidget()
        content_lay = QVBoxLayout(content_w)
        content_lay.setContentsMargins(12, 12, 12, 12)
        content_lay.setSpacing(12)

        content_lay.addWidget(_section("CUSTOM AGENT COMPOSITION FORM"))

        # The name is the unique id (alphanumeric/underscore); it is validated on publish.
        # Agent Name
        self._agent_name_input = QLineEdit()
        self._agent_name_input.setPlaceholderText("e.g. doc_karl (alphanumeric and underscores)")
        content_lay.addWidget(_row("Agent Name (ID):", self._agent_name_input))

        # Display Label
        self._agent_label_input = QLineEdit()
        self._agent_label_input.setPlaceholderText("e.g. Dr. Karl")
        content_lay.addWidget(_row("Display Label:", self._agent_label_input))

        # Description
        self._agent_desc_input = QLineEdit()
        self._agent_desc_input.setPlaceholderText("Brief description of agent's specialty...")
        content_lay.addWidget(_row("Description:", self._agent_desc_input))

        # System Prompt
        self._agent_system_input = QTextEdit()
        self._agent_system_input.setPlaceholderText("Enter the identity prompt that governs this agent's behavior...")
        self._agent_system_input.setMinimumHeight(100)
        content_lay.addWidget(_row("System Prompt:", self._agent_system_input))

        # Base Model
        self._base_model_combo = QComboBox()
        content_lay.addWidget(_row("Base Model:", self._base_model_combo))

        # LoRA Adapter
        self._lora_adapter_combo = QComboBox()
        content_lay.addWidget(_row("LoRA Adapter:", self._lora_adapter_combo))

        # RAG Context
        self._rag_enabled_check = QCheckBox("Inject Knowledge Base context")
        self._rag_enabled_check.setChecked(True)
        content_lay.addWidget(_row("RAG Integration:", self._rag_enabled_check))

        # 1-20 chunks; default 3.
        # RAG Top-K
        self._rag_top_k_spin = QSpinBox()
        self._rag_top_k_spin.setRange(1, 20)
        self._rag_top_k_spin.setValue(3)
        self._rag_top_k_spin.setFixedWidth(80)
        content_lay.addWidget(_row("RAG Top-K Chunks:", self._rag_top_k_spin))

        content_lay.addWidget(_hline())

        # Publish Button row
        btn_lay = QHBoxLayout()
        self._publish_btn = QPushButton("Publish Custom Agent")
        self._publish_btn.setObjectName("btn-primary")
        self._publish_btn.clicked.connect(self._publish_agent)
        btn_lay.addWidget(self._publish_btn)
        btn_lay.addStretch()

        content_lay.addLayout(btn_lay)
        content_lay.addStretch()

        scroll.setWidget(content_w)
        
        main_lay = QVBoxLayout(self._composer_tab_w)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.addWidget(scroll)

    # ── Pipeline / Visualizer Logic ──────────────────────────────────────────

    def _on_state_changed(self, name: str, value: object):
        """Slot: refresh the active model/adapter labels when AppState changes."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if name == "model_name":
            self._active_model_lbl.setText(f"Active Model: {value}")
        elif name == "adapter_name":
            self._active_adapter_lbl.setText(f"Adapter State: {value or 'None'}")

    def _on_vectorizer_changed(self):
        """Slot: switch the math page and flowchart to the chosen mode and recompute."""
        # Step 1: Validate input parameters, operational preconditions, and state
        is_sparse = (self._vectorizer_combo.currentIndex() == 0)
        self._math_stack.setCurrentIndex(0 if is_sparse else 1)
        self._flowchart_widget.set_mode(is_sparse)
        self._run_pipeline()

    def _run_pipeline(self):
        """Vectorize the sandbox sentences with the selected mode and fill the math tables.

        Sparse mode fits TF-IDF locally. Dense mode encodes each sentence with the
        app's SentenceTransformer and L2-normalises it, so dot product equals cosine
        similarity. Errors are shown in the dimension label, not raised.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        text = self._sentences_input.toPlainText().strip()
        if not text:
            return

        sentences = [s.strip() for s in text.split("\n") if s.strip()]
        if not sentences:
            return

        self._cached_sentences = sentences
        is_sparse = (self._vectorizer_combo.currentIndex() == 0)

        if is_sparse:
            # fit() learns vocabulary/IDF; transform() yields one vector per sentence.
            # Fit TF-IDF embedder
            self._tfidf.fit(sentences)
            self._cached_embeddings = [self._tfidf.transform(s) for s in sentences]
            self._populate_sparse_math_tables(sentences)
        else:
            # The encoder may take a while to load on first use; show a hint first.
            # Neural Mode
            self._neural_dim_lbl.setText(f"Dimension: [{self.state.rag.dimension}]")
            
            # Run neural encoding (SentenceTransformer)
            # Show a loading hint if encoder takes time to load
            self._neural_dim_lbl.setText("Loading neural encoder...")
            self.update()
            
            try:
                # Retrieve dense embeddings
                embeddings = []
                for s in sentences:
                    # Iterate over items, evaluate loop invariant, and update progress
                    emb = self.state.rag.encoder.encode([s])[0]
                    # L2 normalized
                    norm = np.linalg.norm(emb)
                    if norm > 0.0:
                        emb = emb / norm
                    embeddings.append(emb)
                
                self._cached_embeddings = embeddings
                self._neural_dim_lbl.setText(f"Dimension: [{self.state.rag.dimension}]")
                self._populate_dense_math_table()
            except Exception as e:
                self._neural_dim_lbl.setText(f"Error encoding: {e}")
                self._cached_embeddings = []

        # Refresh the similarity report for any query already typed.
        # Run query if input is not empty to update calculations
        self._run_query()

    def _populate_sparse_math_tables(self, sentences: list[str]):
        """Fill the vocabulary (word, DF, IDF) and document-term matrix tables."""
        # Populate Vocab Table
        vocab = self._tfidf.vocabulary
        self._vocab_table.setRowCount(len(vocab))
        
        # Document frequency = number of sentences containing the token (set() so
        # repeats within one sentence count once).
        # Calculate DF for each vocab word
        df_counts = {}
        for s in sentences:
            # Iterate over items, evaluate loop invariant, and update progress
            tokens = set(self._tfidf.tokenize(s))
            for w in tokens:
                # Iterate over items, evaluate loop invariant, and update progress
                df_counts[w] = df_counts.get(w, 0) + 1

        for word, idx in sorted(vocab.items(), key=lambda x: x[1]):
            # Iterate over items, evaluate loop invariant, and update progress
            df = df_counts.get(word, 0)
            idf = self._tfidf.idf[idx]
            
            self._vocab_table.setItem(idx, 0, QTableWidgetItem(word))
            self._vocab_table.setItem(idx, 1, QTableWidgetItem(str(df)))
            self._vocab_table.setItem(idx, 2, QTableWidgetItem(f"{idf:.4f}"))

        # One row per document, one column per vocabulary word, TF-IDF weight in each cell.
        # Populate Matrix Table
        self._matrix_table.setRowCount(len(sentences))
        self._matrix_table.setColumnCount(len(vocab) + 1)
        
        headers = ["Sentence/Doc"] + [w for w, _ in sorted(vocab.items(), key=lambda x: x[1])]
        self._matrix_table.setHorizontalHeaderLabels(headers)
        self._matrix_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        for s_idx, s in enumerate(sentences):
            # Iterate over items, evaluate loop invariant, and update progress
            self._matrix_table.setItem(s_idx, 0, QTableWidgetItem(f"Doc {s_idx + 1}"))
            vec = self._cached_embeddings[s_idx]
            for w_idx in range(len(vocab)):
                # Iterate over items, evaluate loop invariant, and update progress
                val = vec[w_idx]
                self._matrix_table.setItem(s_idx, w_idx + 1, QTableWidgetItem(f"{val:.4f}"))

    def _populate_dense_math_table(self):
        """Show the first 20 components of the first sentence's dense embedding."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if not self._cached_embeddings:
            self._neural_table.setRowCount(0)
            return

        first_emb = self._cached_embeddings[0]
        # Show first 20 floats
        show_count = min(20, len(first_emb))
        self._neural_table.setRowCount(show_count)
        
        for i in range(show_count):
            # Iterate over items, evaluate loop invariant, and update progress
            val = first_emb[i]
            self._neural_table.setItem(i, 0, QTableWidgetItem(str(i)))
            self._neural_table.setItem(i, 1, QTableWidgetItem(f"{val:.6f}"))

    def _run_query(self):
        """Rank sandbox documents against the query and render the math trace.

        For each document it shows the dot product, both norms and the resulting
        cosine similarity, and updates the 2D projection (TF-IDF: the two most
        common terms as axes; dense: a PCA projection).
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        query = self._query_input.text().strip()
        if not query or not self._cached_sentences:
            return

        is_sparse = (self._vectorizer_combo.currentIndex() == 0)
        p = get_theme_colors(self.state)

        if is_sparse:
            q_vec = self._tfidf.transform(query)
            if len(q_vec) == 0:
                self._rag_results_browser.setHtml("<span style='color:#FF3366;'>Query has no overlapping vocabulary words with sandbox documents.</span>")
                return
        else:
            try:
                # Execute guarded operations and establish recovery boundaries
                q_vec = self.state.rag.encoder.encode([query])[0]
                norm = np.linalg.norm(q_vec)
                if norm > 0.0:
                    q_vec = q_vec / norm
            except Exception as e:
                self._rag_results_browser.setHtml(f"<span style='color:#FF3366;'>Error encoding query: {e}</span>")
                return

        # HTML report is built piece by piece, then set once at the end.
        # Calculate similarity and construct step-by-step formula trace
        html_output = [
            f"<div style='font-family:{MONO}; font-size:9pt; line-height:1.5; color:{p['text_mid']};'>"
        ]

        q_norm = np.linalg.norm(q_vec)
        
        # Projection lists
        doc_points_to_set = []
        
        # Axes are the two terms with highest document frequency (ties alphabetical),
        # so the plot has a stable, meaningful basis.
        # In TF-IDF, get top 2 terms for projection axes
        if is_sparse:
            vocab = self._tfidf.vocabulary
            # Sort terms by document frequency (DF)
            df_counts = {}
            for s in self._cached_sentences:
                # Iterate over items, evaluate loop invariant, and update progress
                tokens = set(self._tfidf.tokenize(s))
                for w in tokens:
                    # Iterate over items, evaluate loop invariant, and update progress
                    df_counts[w] = df_counts.get(w, 0) + 1
            
            sorted_terms = sorted(vocab.keys(), key=lambda w: (-df_counts[w], w))
            
            term_x, term_y = "X Axis", "Y Axis"
            idx_x, idx_y = -1, -1
            if len(sorted_terms) >= 1:
                term_x = sorted_terms[0]
                idx_x = vocab[term_x]
            if len(sorted_terms) >= 2:
                term_y = sorted_terms[1]
                idx_y = vocab[term_y]
                
            self._projection_widget.set_axes(term_x, term_y)
            
            q_x = float(q_vec[idx_x]) if idx_x >= 0 and len(q_vec) > idx_x else 0.0
            q_y = float(q_vec[idx_y]) if idx_y >= 0 and len(q_vec) > idx_y else 0.0
            self._projection_widget.set_query(q_x, q_y)
        else:
            # Dense vectors have no natural 2D axes, so project with PCA instead.
            # PCA projection for Dense
            self._projection_widget.set_axes("PCA Component 1", "PCA Component 2")
            try:
                # Execute guarded operations and establish recovery boundaries
                pca_docs, q_proj = self._project_pca_docs_and_query(self._cached_embeddings, q_vec)
                self._projection_widget.set_query(q_proj[0], q_proj[1])
            except Exception:
                # Centre the points so the plot still renders.
                # Fallback if SVD fails
                pca_docs = [(0.5, 0.5) for _ in self._cached_embeddings]
                self._projection_widget.set_query(0.5, 0.5)

        for idx, doc in enumerate(self._cached_sentences):
            # Iterate over items, evaluate loop invariant, and update progress
            d_vec = self._cached_embeddings[idx]
            dot_product = float(np.dot(q_vec, d_vec))
            d_norm = float(np.linalg.norm(d_vec))
            
            # Guard against division by zero for empty/zero vectors.
            if q_norm * d_norm > 0:
                cos_sim = dot_product / (q_norm * d_norm)
            else:
                cos_sim = 0.0

            # Same projection coordinates the plot widget expects: x, y, similarity, text.
            # Store point for 2D Projection widget
            if is_sparse:
                d_x = float(d_vec[idx_x]) if idx_x >= 0 and len(d_vec) > idx_x else 0.0
                d_y = float(d_vec[idx_y]) if idx_y >= 0 and len(d_vec) > idx_y else 0.0
            else:
                d_x, d_y = pca_docs[idx]

            doc_points_to_set.append({
                "id": idx + 1,
                "x": d_x,
                "y": d_y,
                "similarity": cos_sim,
                "text": doc
            })

            # One card per document showing the formula with the real numbers filled in.
            # Render calculation trace
            html_output.append(
                f"<div style='margin-bottom: 12px; padding: 10px; border-radius: 4px; border: 1px solid {p['border']}; background: {p['bg_surface']};'>"
                f"<div style='font-weight: bold; color: {p['accent']};'>Doc {idx + 1}: \"{doc}\"</div>"
                f"<div style='margin-top: 6px; font-family: {MONO}; font-size: 8.5pt; color: {p['text_hi']};'>"
                f"&bull; Dot Product (Q &middot; D) = <b>{dot_product:.6f}</b><br/>"
                f"&bull; Norm(Q) = {q_norm:.6f} &middot; Norm(D) = {d_norm:.6f}<br/>"
                f"&bull; Cosine Similarity = {dot_product:.6f} / ({q_norm:.6f} &times; {d_norm:.6f}) = <b style='color:{p['green']};'>{cos_sim:.6f}</b>"
                f"</div>"
                f"</div>"
            )

        html_output.append("</div>")
        self._rag_results_browser.setHtml("".join(html_output))
        
        self._projection_widget.set_documents(doc_points_to_set)

    def _project_pca_docs_and_query(self, doc_embeddings: list[np.ndarray], query_embedding: np.ndarray) -> tuple[list[tuple[float, float]], tuple[float, float]]:
        """Project documents and query to 2D using PCA (via SVD) and scale to [0.15, 0.85].

        Returns (document points, query point). The 0.15-0.85 margin keeps points
        away from the plot edges.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        all_embeds = np.vstack(doc_embeddings + [query_embedding])
        
        # PCA requires mean-centred data.
        # Center components
        mean = np.mean(all_embeds, axis=0)
        centered = all_embeds - mean
        
        # Rows of Vt are principal axes; projecting on the first two gives 2D coordinates.
        # SVD projection
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        projected = np.dot(centered, Vt[:2].T)
        
        # Constant axes (range 0) are given range 1 to avoid division by zero.
        # Normalize to [0.15, 0.85] bounds
        min_val = np.min(projected, axis=0)
        max_val = np.max(projected, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0
        
        normalized = 0.15 + 0.70 * (projected - min_val) / range_val
        
        doc_points = [(float(p[0]), float(p[1])) for p in normalized[:-1]]
        query_point = (float(normalized[-1][0]), float(normalized[-1][1]))
        return doc_points, query_point

    # ── Composer / Form Logic ───────────────────────────────────────────────

    def showEvent(self, event):
        """Refresh the model/adapter dropdowns and (re)run the visualizer whenever the tab is shown."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().showEvent(event)
        self._refresh_combos()
        # Automatically run visualizer on display if it hasn't run yet
        self._run_pipeline()

    def _refresh_combos(self):
        """Repopulate base model (*.gguf) and LoRA adapter (directory) dropdowns from disk."""
        # Scan data/models/ for .gguf files
        self._base_model_combo.clear()
        models_dir = "data/models"
        if os.path.exists(models_dir):
            try:
                # Execute guarded operations and establish recovery boundaries
                for f in sorted(os.listdir(models_dir)):
                    # Iterate over items, evaluate loop invariant, and update progress
                    if f.endswith(".gguf"):
                        self._base_model_combo.addItem(f)
            except Exception:
                pass
        
        # Fallback entry so the form stays usable when no model file is installed.
        if self._base_model_combo.count() == 0:
            self._base_model_combo.addItem("deepseek-r1-1.5b.gguf")

        # Scan data/adapters/ for LoRA adapter subdirectories
        self._lora_adapter_combo.clear()
        self._lora_adapter_combo.addItem("none")
        adapters_dir = "data/adapters"
        if os.path.exists(adapters_dir):
            try:
                # Execute guarded operations and establish recovery boundaries
                for d in sorted(os.listdir(adapters_dir)):
                    # Iterate over items, evaluate loop invariant, and update progress
                    d_path = os.path.join(adapters_dir, d)
                    if os.path.isdir(d_path):
                        self._lora_adapter_combo.addItem(d)
            except Exception:
                pass

    def _publish_agent(self):
        """Validate the form and save the agent into data/custom_agents.json.

        The JSON file maps agent id -> profile. An unreadable existing file is
        replaced (with a warning) rather than blocking the user.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        name = self._agent_name_input.text().strip()
        label = self._agent_label_input.text().strip()
        desc = self._agent_desc_input.text().strip()
        sys_prompt = self._agent_system_input.toPlainText().strip()
        base_model = self._base_model_combo.currentText()
        adapter = self._lora_adapter_combo.currentText()
        rag_enabled = self._rag_enabled_check.isChecked()
        rag_top_k = self._rag_top_k_spin.value()

        # Name must be alphanumeric/underscore; label and system prompt are required.
        # 1. Validation
        if not name:
            QMessageBox.warning(self, "Validation Error", "Agent Name (ID) is required.")
            return
        if not re.match(r"^[a-zA-Z0-9_]+$", name):
            QMessageBox.warning(self, "Validation Error", "Agent Name must contain only alphanumeric characters and underscores.")
            return
        if not label:
            QMessageBox.warning(self, "Validation Error", "Display Label is required.")
            return
        if not sys_prompt:
            QMessageBox.warning(self, "Validation Error", "System Prompt is required.")
            return

        # The UI uses the string 'none'; the saved profile uses null.
        # Prepare adapter value
        adapter_val = None if adapter == "none" else adapter

        # `prompt` and `system_prompt` hold the same text, presumably for consumers that
        # use either key.
        profile = {
            "name": name,
            "label": label,
            "description": desc,
            "prompt": sys_prompt,
            "system_prompt": sys_prompt,
            "base_model": base_model,
            "adapter": adapter_val,
            "rag_enabled": rag_enabled,
            "rag_top_k": rag_top_k
        }

        # Read-modify-write the whole file (no locking): fine for a single desktop user.
        # 2. File I/O for data/custom_agents.json
        agents_file = "data/custom_agents.json"
        os.makedirs("data", exist_ok=True)
        
        agents_data = {}
        if os.path.exists(agents_file):
            try:
                # Execute guarded operations and establish recovery boundaries
                with open(agents_file, "r", encoding="utf-8") as f:
                    agents_data = json.load(f)
            except Exception as e:
                QMessageBox.warning(self, "JSON Read Warning", f"Could not read custom_agents.json: {e}. Rewriting file.")

        agents_data[name] = profile

        try:
            # Execute guarded operations and establish recovery boundaries
            with open(agents_file, "w", encoding="utf-8") as f:
                json.dump(agents_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            QMessageBox.critical(self, "File Error", f"Failed to write to custom_agents.json: {e}")
            return

        # Emit the signal after the file is safely written.
        # 3. Success notification & custom signal emission
        QMessageBox.information(self, "Agent Published", f"Custom agent '{label}' has been successfully compiled and published.")
        self.agent_published.emit(name)

        # Clear the form for the next agent.
        # 4. Form resetting
        self._agent_name_input.clear()
        self._agent_label_input.clear()
        self._agent_desc_input.clear()
        self._agent_system_input.clear()
        self._rag_top_k_spin.setValue(3)
        self._rag_enabled_check.setChecked(True)
        self._base_model_combo.setCurrentIndex(0)
        self._lora_adapter_combo.setCurrentIndex(0)
