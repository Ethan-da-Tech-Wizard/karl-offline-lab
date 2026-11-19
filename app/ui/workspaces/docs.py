# ==============================================================================
# Program: docs.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Desktop workspace implementation for Docs.
#          Provides responsive layout controls, parameter adjustment surfaces,
#          and visual feedback components within Karl's PyQt6 shell.
# ==============================================================================

import logging
import html
import os
import re
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QTextBrowser,
    QLabel, QFrame, QLineEdit, QPushButton, QMessageBox, QSplitter
)
from PyQt6.QtCore import Qt

from app.ui.themes import MONO, get_theme_colors
from app.ui.workspaces.docs_data import DEFAULT_LIBRARY

# Logger name keeps the legacy 'codex' label: this workspace is shown to users as Codex.
logger = logging.getLogger("karl.codex")


def _section(text: str) -> QLabel:
    """Create a small section-header label (styled by the 'section-header' QSS id)."""
    # Step 1: Validate input parameters, operational preconditions, and state
    lbl = QLabel(text)
    lbl.setObjectName("section-header")
    return lbl

def _hline() -> QFrame:
    """Create a horizontal divider line."""
    # Step 1: Validate input parameters, operational preconditions, and state
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    return f

# The Codex: an in-app documentation reader. Guides are seeded from
# DEFAULT_LIBRARY into data/codex_library as HTML files (so users can add or edit
# files on disk), cached in memory, indexed into a RAG store for search, and
# shown in a QTextBrowser with a clickable heading outline.
class DocsWorkspace(QWidget):
    def __init__(self, state, parent=None):
        """Seed/scan the on-disk library, then build the UI. `state` is the shared AppState."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(parent)
        self.state = state
        self.setObjectName("workspace-root")
        self._init_library()
        self._build_ui()

    def _init_library(self):
        """Ensure the library directory is populated and fill the in-memory cache.

        Steps: (1) decide whether the shipped guides need (re)writing using a
        .version marker file, (2) write them, (3) scan the directory into
        self._cache, (4) ingest into the codex RAG index if that index is empty.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        self._library_dir = "data/codex_library"
        os.makedirs(self._library_dir, exist_ok=True)
        
        version_filepath = os.path.join(self._library_dir, ".version")
        # Bump this string whenever DEFAULT_LIBRARY changes so existing installs
        # receive the updated guides (default files are overwritten on upgrade).
        current_version = "7.0"
        
        # Upgrade unless a readable .version file already matches. An unreadable
        # version file counts as 'needs upgrade'.
        # Determine if we need to seed or upgrade
        needs_upgrade = True
        if os.path.exists(version_filepath):
            try:
                # Execute guarded operations and establish recovery boundaries
                with open(version_filepath, "r", encoding="utf-8") as vf:
                    installed_version = vf.read().strip()
                if installed_version == current_version:
                    needs_upgrade = False
            except Exception:
                pass
                
        # An empty library (e.g. user deleted files) is always re-seeded.
        # If files don't exist, we definitely seed/upgrade
        existing_files = [f for f in os.listdir(self._library_dir) if f.endswith((".html", ".md"))]
        if not existing_files:
            needs_upgrade = True
            
        if needs_upgrade:
            # safe_name strips characters unsafe for filenames from the topic title.
            # Seed or overwrite default files in DEFAULT_LIBRARY
            for topic, content in DEFAULT_LIBRARY.items():
                # Iterate over items, evaluate loop invariant, and update progress
                safe_name = "".join(c for c in topic if c.isalnum() or c in (" ", "_", "-")).strip()
                filepath = os.path.join(self._library_dir, f"{safe_name}.html")
                try:
                    # Execute guarded operations and establish recovery boundaries
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(content)
                except Exception as e:
                    logger.warning(f"Error seeding topic {topic}: {e}")
                    
            # Written last so a failed seed is retried on the next launch.
            # Write new version file
            try:
                # Execute guarded operations and establish recovery boundaries
                with open(version_filepath, "w", encoding="utf-8") as vf:
                    vf.write(current_version)
            except Exception as e:
                logger.warning(f"Error writing version: {e}")
            
        # Cache every .html/.md file, including user-added ones; the key is the
        # file name without extension and is what the topic list displays.
        # Re-scan to build cache
        self._cache = {}
        for f in sorted(os.listdir(self._library_dir)):
            # Iterate over items, evaluate loop invariant, and update progress
            if f.endswith((".html", ".md")):
                filepath = os.path.join(self._library_dir, f)
                topic_name = os.path.splitext(f)[0]
                try:
                    # Execute guarded operations and establish recovery boundaries
                    with open(filepath, "r", encoding="utf-8") as file:
                        content = file.read()
                    self._cache[topic_name] = {
                        "filepath": filepath,
                        "content": content
                    }
                except Exception as e:
                    logger.warning(f"Error reading {f}: {e}")

        # Only when the index is empty, to avoid re-embedding on every launch.
        # A failure stops the loop (break) so one bad file does not spam warnings.
        # Auto-ingest Codex files into codex_rag index if it is empty.
        if hasattr(self.state, "codex_rag") and self.state.codex_rag.total_chunks == 0:
            for topic_name, item in self._cache.items():
                # Iterate over items, evaluate loop invariant, and update progress
                try:
                    # Execute guarded operations and establish recovery boundaries
                    self.state.codex_rag.ingest_file(item["filepath"])
                except Exception as e:
                    logger.warning(f"Could not ingest {topic_name} into codex RAG index: {e}")
                    break

    def _build_ui(self):
        """Create the splitter layout: topic list + outline on the left, reader on the right."""
        # Step 1: Validate input parameters, operational preconditions, and state
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # 1px handle keeps the split subtle; stretch factors below give the
        # reader 4x the width of the list.
        # Main horizontal splitter
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.setHandleWidth(1)

        # Contains the search box, topic list and the heading outline.
        # Left list panel: guide chapters
        left_panel = QWidget()
        left_panel.setObjectName("panel")
        lp_layout = QVBoxLayout(left_panel)
        lp_layout.setContentsMargins(12, 12, 12, 12)
        lp_layout.setSpacing(8)

        lp_layout.addWidget(_section("CODEX"))

        # Filters the topic list on every keystroke.
        # Live Search Input
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search Codex...")
        self._search_input.setToolTip("Search for keywords in Codex titles or guides")
        self._search_input.textChanged.connect(self._filter_topics)
        lp_layout.addWidget(self._search_input)

        self._topics_list = QListWidget()
        self._topics_list.setToolTip("Select a topic to view its in-app documentation")
        self._topics_list.currentTextChanged.connect(self._on_topic_selected)
        
        # Items are added in cache (alphabetical file) order.
        # Populate topics list from scanned cache keys
        for k in self._cache.keys():
            # Iterate over items, evaluate loop invariant, and update progress
            self._topics_list.addItem(k)

        lp_layout.addWidget(self._topics_list, 1)

        # Outline of the open guide's headings; clicking jumps to the anchor.
        # TOC Panel (Internal Hierarchy)
        lp_layout.addWidget(_hline())
        lp_layout.addWidget(_section("HIERARCHY"))
        self._toc_list = QListWidget()
        self._toc_list.setFixedHeight(150)
        self._toc_list.currentTextChanged.connect(self._jump_to_anchor)
        lp_layout.addWidget(self._toc_list)

        self._main_splitter.addWidget(left_panel)

        # Reader: header row (title + 'Send to Workbench') above the browser.
        # Right browser panel: content reader
        right_panel = QWidget()
        rp_layout = QVBoxLayout(right_panel)
        rp_layout.setContentsMargins(0, 0, 0, 0)
        rp_layout.setSpacing(8)

        # Header for Document Reader Panel
        right_header = QWidget()
        rh_layout = QHBoxLayout(right_header)
        rh_layout.setContentsMargins(0, 0, 0, 0)
        
        self._active_topic_lbl = QLabel("SELECT A TOPIC")
        self._active_topic_lbl.setObjectName("section-header")
        rh_layout.addWidget(self._active_topic_lbl, 1)
        
        self._send_to_wb_btn = QPushButton("Send to Workbench")
        self._send_to_wb_btn.setObjectName("btn-secondary")
        self._send_to_wb_btn.setFixedHeight(24)
        self._send_to_wb_btn.setStyleSheet("font-size: 8.5pt; padding: 2px 10px;")
        self._send_to_wb_btn.clicked.connect(self._send_to_workbench)
        rh_layout.addWidget(self._send_to_wb_btn)
        
        rp_layout.addWidget(right_header)

        self._browser = QTextBrowser()
        self._browser.setToolTip("Read-only documentation viewer")
        # Let guide links open in the system browser instead of navigating inside.
        self._browser.setOpenExternalLinks(True)
        rp_layout.addWidget(self._browser, 1)
        self._main_splitter.addWidget(right_panel)

        self._main_splitter.setStretchFactor(0, 1)
        self._main_splitter.setStretchFactor(1, 4)
        root.addWidget(self._main_splitter)

        # Selecting a row triggers _on_topic_selected, which renders it.
        # Select first topic by default
        if self._cache:
            self._topics_list.setCurrentRow(0)

    def _filter_topics(self, text: str):
        """Narrow the topic list to guides that match the search text.

        Uses the codex RAG index when it has content (results are ordered by
        relevance, but note the list itself keeps its order and non-matches are
        hidden). Falls back to plain substring matching on title/content when the
        index is unavailable or returns nothing.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        query = text.strip()
        if not query:
            # Show all topics and restore default order
            for idx in range(self._topics_list.count()):
                # Iterate over items, evaluate loop invariant, and update progress
                item = self._topics_list.item(idx)
                item.setHidden(False)
            return

        # Top 10 chunks, de-duplicated into an ordered list of topic names.
        # Use codex_rag to find matching chunks
        matched_topics = []
        if hasattr(self.state, "codex_rag") and self.state.codex_rag.total_chunks > 0:
            results = self.state.codex_rag.retrieve_with_metadata(query, top_k=10)
            for r in results:
                # Iterate over items, evaluate loop invariant, and update progress
                sf = r.get("source_file", "")
                if sf:
                    topic = os.path.splitext(sf)[0]
                    if topic not in matched_topics:
                        matched_topics.append(topic)
        
        # Simple case-insensitive substring match on title or body text.
        # Fallback to basic keyword matching if no RAG matches
        if not matched_topics:
            query_lower = query.lower()
            for idx in range(self._topics_list.count()):
                # Iterate over items, evaluate loop invariant, and update progress
                item = self._topics_list.item(idx)
                topic_name = item.text()
                item_data = self._cache.get(topic_name, {})
                content = item_data.get("content", "").lower()
                match = (query_lower in topic_name.lower()) or (query_lower in content)
                item.setHidden(not match)
            return

        # Disconnect the selection signal while toggling visibility so hiding the
        # currently selected row does not trigger intermediate renders; it is
        # reconnected before the deliberate selection below.
        # Sort topics by relevance
        self._topics_list.currentTextChanged.disconnect(self._on_topic_selected)
        items = []
        for idx in range(self._topics_list.count()):
            # Iterate over items, evaluate loop invariant, and update progress
            items.append(self._topics_list.item(idx))
            
        for item in items:
            # Iterate over items, evaluate loop invariant, and update progress
            item.setHidden(True)
            
        first_visible = None
        for topic in matched_topics:
            # Iterate over items, evaluate loop invariant, and update progress
            for item in items:
                # Iterate over items, evaluate loop invariant, and update progress
                if item.text() == topic:
                    item.setHidden(False)
                    if first_visible is None:
                        first_visible = item
                    break
                    
        self._topics_list.currentTextChanged.connect(self._on_topic_selected)
        
        if first_visible:
            self._topics_list.setCurrentItem(first_visible)
            self._on_topic_selected(first_visible.text())
        else:
            self._browser.setHtml(
                f"<div style='font-family:{MONO}; font-size:10pt; color:#9090A8; line-height:1.5; padding:8px;'>"
                f"No results found for query: <i>{html.escape(text)}</i>"
                f"</div>"
            )

    def _on_topic_selected(self, text: str):
        """Render the selected guide: update the title, rebuild the outline and show the HTML.

        The file is re-read from disk (not the cache) so external edits show up.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        if not text:
            self._browser.clear()
            self._toc_list.clear()
            self._active_topic_lbl.setText("SELECT A TOPIC")
            return
            
        self._active_topic_lbl.setText(text.upper())
        item_data = self._cache.get(text)
        if not item_data:
            self._browser.setHtml(
                f"<div style='font-family:{MONO}; font-size:10pt; color:#9090A8; line-height:1.5; padding:8px;'>"
                f"No guide found for: <i>{html.escape(text)}</i>"
                f"</div>"
            )
            self._toc_list.clear()
            return
            
        filepath = item_data["filepath"]
        try:
            # Execute guarded operations and establish recovery boundaries
            with open(filepath, "r", encoding="utf-8") as f:
                doc_content = f.read()
        except Exception as e:
            doc_content = f"Error reading guide from disk: {e}"

        # Outline first, then anchors, both derived from the same h2/h4 headings.
        # Generate TOC from headers
        self._generate_toc(doc_content)
            
        # Add anchors to headers for jumping
        modified_content = self._add_anchors(doc_content)

        # Wrap in a div so the guide inherits the theme's text colour and mono font.
        # Get theme colors for styling
        colors = get_theme_colors(self.state)
        
        styled_html = (
            f"<div style='font-family:{MONO}; font-size:10pt; color:{colors['text_hi']}; line-height:1.6; padding:15px;'>"
            f"{modified_content}"
            f"</div>"
        )
        self._browser.setHtml(styled_html)

    def _generate_toc(self, content: str):
        """Fill the outline list from the guide's <h2>/<h4> headings (h4 entries are indented)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._toc_list.clear()
        # Find all h2 and h4 headers
        headers = re.findall(r"<(h[24])[^>]*>(.*?)</\1>", content, re.IGNORECASE)
        for h_tag, h_text in headers:
            # Clean HTML tags from header text
            clean_text = re.sub(r"<[^>]+>", "", h_text).strip()
            indent = "  " if h_tag.lower() == "h4" else ""
            self._toc_list.addItem(f"{indent}{clean_text}")

    def _add_anchors(self, content: str) -> str:
        """Return the HTML with an id added to every <h2>/<h4> so the outline can scroll to it."""
        # Replace headers with anchored versions
        def replacer(match):
            # Step 1: Validate input parameters, operational preconditions, and state
            h_tag = match.group(1)
            h_attrs = match.group(2)
            h_text = match.group(3)
            clean_text = re.sub(r"<[^>]+>", "", h_text).strip()
            # Anchor ids are the heading text lowercased with spaces as underscores; this
            # must match the computation in _jump_to_anchor.
            anchor_id = clean_text.replace(" ", "_").lower()
            return f'<{h_tag} id="{anchor_id}" {h_attrs}>{h_text}</{h_tag}>'

        return re.sub(r"<(h[24])([^>]*)>(.*?)</\1>", replacer, content, flags=re.IGNORECASE)

    def _jump_to_anchor(self, text: str):
        """Scroll the reader to the heading named by the clicked outline entry."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if not text:
            return
        anchor_id = text.strip().replace(" ", "_").lower()
        self._browser.scrollToAnchor(anchor_id)
    def _send_to_workbench(self):
        """Strip the guide down to plain text and append it to the Workbench input.

        Goes through an AppState signal so the Docs view needs no direct reference
        to the Workbench widget.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        current_item = self._topics_list.currentItem()
        if not current_item:
            return
        topic = current_item.text()
        item_data = self._cache.get(topic)
        if not item_data:
            return
            
        filepath = item_data["filepath"]
        try:
            # Execute guarded operations and establish recovery boundaries
            with open(filepath, "r", encoding="utf-8") as f:
                doc_content = f.read()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not read topic file: {e}")
            return
            
        # Remove tags and blank lines so the pasted text is compact prompt context.
        clean_text = re.sub(r"<[^>]+>", "", doc_content)
        clean_text = "\n".join(line.strip() for line in clean_text.splitlines() if line.strip())
        
        self.state.append_to_workbench_input.emit(f"\n[Codex: {topic}]\n{clean_text}\n")
        QMessageBox.information(
            self, "Sent to Workbench",
            f"Topic '{topic}' content appended to chat input context."
        )
