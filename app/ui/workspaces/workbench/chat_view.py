# ==============================================================================
# Program: chat_view.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Scrollable conversation display with streaming support.
#          Coordinates with adjacent subsystems within the Karl Offline Lab
#          architecture.
# ==============================================================================

"""Scrollable conversation display with streaming support."""

# Chat transcript widget for the Workbench. It stores messages as tuples and
# re-renders them to HTML on demand, except while a reply is streaming, when
# tokens are appended directly to the open assistant block (see AGENTS.md
# "Key Gotchas" #10: do not call append() during an active stream).
from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QTextBrowser, QTextEdit
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QTextCursor


def _escape(text: str) -> str:
    """Escape text for HTML and turn newlines into <br> (used for user and assistant bodies)."""
    # Step 1: Validate input parameters, operational preconditions, and state
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
    )


def _escape_pre(text: str) -> str:
    """Escape text for HTML but keep newlines as they are (for pre-formatted blocks)."""
    # Step 1: Validate input parameters, operational preconditions, and state
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )


# Branch links use the custom "branch:<node_id>" URL scheme; setOpenLinks(False) stops Qt from
# navigating, so the owning workspace handles the anchorClicked signal instead.
class ChatView(QTextBrowser):
    """Scrollable conversation display with streaming support."""

    def __init__(self, parent=None):
        """Configure a read-only, wrapping, vertically scrolling view with default theme colours."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(parent)
        self.setOpenLinks(False)
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.document().setDocumentMargin(0)
        # Each entry is (role, text, node_id, attachments); older 3-tuples without attachments are still accepted by _render_all.
        self._messages: list[tuple[str, str, str]] = []   # (role, text, node_id)
        self._streaming_buf = ""
        self._streaming = False
        self._streaming_node_id = ""
        # Fallback palette used until the workspace supplies the active theme through set_theme().
        # Default fallback theme colors
        self.theme_colors = {
            "text_lo": "#505068",
            "accent": "#00C2FF",
            "bg_raised": "#1C1C2A",
            "border_hi": "#383850",
            "text_hi": "#E4E4F0",
            "bg_surface": "#14141F",
            "border": "#252535",
            "yellow": "#F0B030",
        }

    def set_theme(self, theme_colors: dict):
        """Store the theme colour dict and redraw the whole conversation with it."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.theme_colors = theme_colors
        self._render_all()

    # public API ──────────────────────────────────────────────────────────────

    def push_user(self, text: str, node_id: str, attachments: list[dict] | None = None):
        """Add a user message (finishing any open stream first) and re-render."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._finalize_stream()
        self._messages.append(("user", text, node_id, attachments or []))
        self._render_all()

    def _get_karl_hdr(self, node_id: str) -> str:
        """Return the HTML that opens an assistant bubble; the caller closes the two divs afterwards."""
        # Step 1: Validate input parameters, operational preconditions, and state
        text_lo = self.theme_colors.get("text_lo", "#505068")
        accent = self.theme_colors.get("accent", "#00C2FF")
        bg_surface = self.theme_colors.get("bg_surface", "#14141F")
        border = self.theme_colors.get("border", "#252535")
        text_hi = self.theme_colors.get("text_hi", "#E4E4F0")
        return (
            f'<div style="margin:16px 80px 4px 0px;">'
            f'<div style="color:{text_lo};font-size:7.5pt;font-weight:bold;margin-bottom:4px;letter-spacing:1.5px;">'
            f'KARL &nbsp;|&nbsp; <a href="branch:{node_id}" style="color:{accent};text-decoration:none;font-weight:bold;">↳ branch</a></div>'
            f'<div style="background:{bg_surface};border:1px solid {border};border-radius:6px;'
            f'padding:12px 16px;color:{text_hi};font-size:10pt;'
            f'line-height:1.4;white-space:pre-wrap;min-height:1em;">'
        )

    def _get_user_html(self, text: str, node_id: str, attachments: list[dict] | None = None) -> str:
        """Return the HTML for a right-aligned user bubble with its branch link and attachments."""
        # Step 1: Validate input parameters, operational preconditions, and state
        text_lo = self.theme_colors.get("text_lo", "#505068")
        accent = self.theme_colors.get("accent", "#00C2FF")
        bg_raised = self.theme_colors.get("bg_raised", "#1C1C2A")
        border_hi = self.theme_colors.get("border_hi", "#383850")
        text_hi = self.theme_colors.get("text_hi", "#E4E4F0")
        safe_text = _escape(text)
        # Qt's rich-text engine doesn't size `display:inline-block` to its
        # content (it always rendered full-width, turning this into a flat
        # highlighted bar instead of a bubble). A single-cell <table
        # align="right"> is what Qt actually shrink-wraps to content, and it
        # still wraps naturally within the viewport for long messages.
        # Qt's rich-text engine doesn't size `display:inline-block` to its
        # content (it always rendered full-width, turning this into a flat
        # highlighted bar instead of a bubble). A single-cell <table
        # align="right"> is what Qt actually shrink-wraps to content, and it
        # still wraps naturally within the viewport for long messages.
        return (
            f'<div style="margin:16px 0px 2px 80px; text-align:right;">'
            f'<span style="color:{text_lo};font-size:7.5pt;font-weight:bold;letter-spacing:1.5px;">'
            f'YOU &nbsp;|&nbsp; <a href="branch:{node_id}" style="color:{accent};text-decoration:none;font-weight:bold;">↳ branch</a></span>'
            f'</div>'
            f'<table align="right" cellspacing="0" cellpadding="0" style="margin:0 0 4px 0;"><tr>'
            f'<td style="background:{bg_raised};border:1px solid {border_hi};border-radius:6px;'
            f'padding:12px 16px;color:{text_hi};font-size:10pt;'
            f'line-height:1.4;white-space:pre-wrap;text-align:left;">{safe_text}'
            f'{self._attachments_html(attachments or [])}</td>'
            f'</tr></table>'
        )

    def _attachments_html(self, attachments: list[dict]) -> str:
        """Return HTML for image attachments (label and thumbnail); other attachment types are skipped."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if not attachments:
            return ""
        accent = self.theme_colors.get("accent", "#00C2FF")
        text_lo = self.theme_colors.get("text_lo", "#505068")
        border = self.theme_colors.get("border", "#252535")
        parts = []
        for attachment in attachments:
            # Iterate over items, evaluate loop invariant, and update progress
            if attachment.get("type") != "image":
                continue
            # Prefer the thumbnail, then the original path; paths become file:// URIs so QTextBrowser can load them.
            path = attachment.get("thumbnail_path") or attachment.get("path") or ""
            image_id = attachment.get("id", "")
            label = attachment.get("label") or image_id[:8] or "image"
            uri = ""
            try:
                # Execute guarded operations and establish recovery boundaries
                if path:
                    uri = QUrl.fromLocalFile(str(Path(path).resolve())).toString()
            except Exception:
                uri = ""
            image_html = f'<img src="{uri}" style="max-width:260px;max-height:180px;border-radius:4px;margin-top:6px;">' if uri else ""
            parts.append(
                f'<div style="margin-top:10px;padding:8px;border:1px solid {border};border-radius:5px;">'
                f'<div style="color:{accent};font-size:8pt;font-weight:bold;">IMAGE ATTACHMENT</div>'
                f'<div style="color:{text_lo};font-size:8pt;">{_escape(label)}</div>'
                f'{image_html}'
                f'</div>'
            )
        return "".join(parts)


    def begin_stream(self, node_id: str = ""):
        """Open an assistant bubble and start receiving streamed tokens."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._streaming = True
        self._streaming_buf = ""
        self._streaming_node_id = node_id
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.setTextCursor(cursor)
        # Only the header HTML is inserted; tokens are then added as plain text into the same paragraph.
        self.insertHtml(self._get_karl_hdr(node_id))

    def append_token(self, token: str):
        """Append one streamed token to the open bubble and keep the view scrolled to the end."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if not self._streaming:
            return
        # The text is also buffered so the final message can be stored once the stream ends.
        self._streaming_buf += token
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(token)
        self.setTextCursor(cursor)
        self._scroll_to_bottom()

    def finalize_stream(self, node_id: str = ""):
        """Close the streamed bubble, store the full reply in the message list and re-render."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if not self._streaming:
            return
        final_node_id = node_id or self._streaming_node_id
        self._messages.append(("assistant", self._streaming_buf, final_node_id, []))
        self._streaming_buf = ""
        self._streaming = False
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        # Closing tags for the two divs opened by _get_karl_hdr; the re-render below replaces this anyway.
        self.insertHtml('</div></div>')
        self.setTextCursor(cursor)
        self._render_all()

    def clear_display(self):
        """Remove all messages and clear the display."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._messages.clear()
        self._streaming_buf = ""
        self._streaming = False
        self.clear()

    def replace_last_assistant(self, text: str):
        """Swap the text of the last assistant message (for example after an edit or retry) and re-render."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if self._messages and self._messages[-1][0] == "assistant":
            self._messages[-1] = ("assistant", text, self._messages[-1][2], self._messages[-1][3] if len(self._messages[-1]) > 3 else [])
            self._render_all()

    def append_system_note(self, text: str):
        """Insert a small centred muted note such as a status message; not stored in the message list."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._finalize_stream()
        safe = _escape(text)
        text_lo = self.theme_colors.get("text_lo", "#505068")
        html = (
            f'<div style="margin:6px 0;color:{text_lo};font-size:8pt;'
            f'text-align:center;">{safe}</div>'
        )
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(html)
        self._scroll_to_bottom()

    def append_diagnostics(self, model: str, n_ctx: int, diag: dict):
        """Insert a formatted card with prompt/generation/total token counts and speeds for the last reply."""
        # Step 1: Validate input parameters, operational preconditions, and state
        # Shown after the reply, so it is not part of _messages and disappears on the next full re-render.
        self._finalize_stream()
        bg_raised = self.theme_colors.get("bg_raised", "#1C1C2A")
        border = self.theme_colors.get("border", "#252535")
        text_hi = self.theme_colors.get("text_hi", "#E4E4F0")
        text_lo = self.theme_colors.get("text_lo", "#505068")
        accent = self.theme_colors.get("accent", "#00C2FF")
        yellow = self.theme_colors.get("yellow", "#F0B030")

        html_str = (
            f'<div style="margin: 8px 80px 8px 0px; padding: 10px 14px; '
            f'background: {bg_raised}; border: 1px solid {border}; border-radius: 4px; '
            f'font-family: \'JetBrains Mono\', monospace; font-size: 8.5pt;">'
            f'<div style="color: {accent}; font-weight: bold; margin-bottom: 6px; letter-spacing: 1.5px;">📊 GENERATION DIAGNOSTICS</div>'
            f'<div style="color: {text_hi}; margin-bottom: 4px;"><b>Model:</b> {_escape(str(model))} (n_ctx={n_ctx})</div>'
            f'<div style="color: {text_hi}; margin-bottom: 4px;">'
            f'<b>Prompt:</b> {diag.get("prompt_tokens", 0)} tokens '
            f'<span style="color: {text_lo};">(prefill: {diag.get("prefill_time", 0):.2f}s @ {diag.get("prefill_tps", 0):.1f} t/s)</span>'
            f'</div>'
            f'<div style="color: {text_hi}; margin-bottom: 4px;">'
            f'<b>Generation:</b> {diag.get("generation_tokens", 0)} tokens '
            f'<span style="color: {text_lo};">(generated: {diag.get("generation_time", 0):.2f}s @ {diag.get("generation_tps", 0):.1f} t/s)</span>'
            f'</div>'
            f'<div style="color: {yellow}; font-weight: bold; margin-top: 6px; letter-spacing: 0.5px;">'
            f'Total Time: {diag.get("total_time", 0):.2f}s @ {diag.get("total_tps", 0):.1f} t/s'
            f'</div>'
            f'</div>'
        )
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(html_str)
        self._scroll_to_bottom()

    def append_rag_sources(self, results: list[dict]):
        """Insert a card listing which RAG chunks (file, chunk id, distance) were injected into the prompt."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._finalize_stream()
        if not results:
            return
        bg_raised = self.theme_colors.get("bg_raised", "#1C1C2A")
        border_hi = self.theme_colors.get("border_hi", "#383850")
        accent = self.theme_colors.get("accent", "#00C2FF")
        text_hi = self.theme_colors.get("text_hi", "#E4E4F0")
        yellow = self.theme_colors.get("yellow", "#F0B030")
        
        lines = []
        lines.append(
            f'<div style="margin:8px 60px 8px 10px; padding:10px 12px; '
            f'background:{bg_raised}; border:1px solid {border_hi}; border-radius:4px; '
            f'font-family: \'JetBrains Mono\', monospace; font-size:8.5pt;">'
        )
        lines.append(f'<div style="color:{accent}; font-weight:bold; margin-bottom:6px;">🔍 Injected RAG Context:</div>')
        for r in results:
            # Iterate over items, evaluate loop invariant, and update progress
            lines.append(
                f'<div style="color:{text_hi}; margin-bottom:4px;">'
                f'• <b>{_escape(r["source_file"])}</b> (Chunk {r["chunk_id"]}, distance: '
                f'<span style="color:{yellow};">{r["distance"]:.4f}</span>)'
                f'</div>'
            )
        lines.append('</div>')
        
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml("".join(lines))
        self._scroll_to_bottom()

    # internals ───────────────────────────────────────────────────────────────

    def _finalize_stream(self):
        """Finish an open stream if there is one (no-op otherwise)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if self._streaming:
            self.finalize_stream()

    def _render_all(self):
        """Rebuild the entire conversation HTML from the stored messages."""
        # Step 1: Validate input parameters, operational preconditions, and state
        parts = []
        for raw in self._messages:
            # Iterate over items, evaluate loop invariant, and update progress
            if len(raw) == 3:
                role, text, node_id = raw
                attachments = []
            else:
                role, text, node_id, attachments = raw
            if role == "user":
                parts.append(self._get_user_html(text, node_id, attachments))
            else:
                parts.append(self._get_karl_hdr(node_id) + _escape(text) + '</div></div>')
        self.setHtml(
            '<html><body style="background:transparent;margin:8px 10px 28px 10px;'
            'white-space:normal;overflow-x:hidden;">'
            + "".join(parts)
            + "</body></html>"
        )
        self.moveCursor(QTextCursor.MoveOperation.End)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        """Scroll the view to the end."""
        # Step 1: Validate input parameters, operational preconditions, and state
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())
