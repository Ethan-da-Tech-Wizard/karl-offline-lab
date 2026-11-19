# ==============================================================================
# Program: vision_workbench.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Desktop workspace implementation for Vision Workbench.
#          Provides responsive layout controls, parameter adjustment surfaces,
#          and visual feedback components within Karl's PyQt6 shell.
# ==============================================================================
# Vision Workbench workspace: a library of saved images (screenshots pasted into
# the Workbench or imported from disk) with a preview, editable OCR text and
# caption, tags/kind metadata, and analysis via OCR and an optional local
# multimodal model. Image records live in state.image_store.

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.engine.image_analysis_thread import ImageAnalysisThread
from app.vision.vision_model_loader import VisionModelLoader, installed_vision_models


# Browser/annotator for the image store (state.image_store, an ImageStore from
# app.vision.image_store). The widget never mutates records itself: every edit
# goes through the store (update_metadata, save_ocr_correction, ...), which returns
# the updated record that is then re-rendered. Cross-workspace messaging uses two
# state signals: change_workspace_requested(int) and attach_image_to_workbench(str).
class VisionWorkbench(QWidget):
    """Workspace for browsing, analysing and annotating saved images."""
    def __init__(self, state, parent=None):
        """Store state, build the UI and load the saved image list."""
        super().__init__(parent)
        self.state = state
        self.setObjectName("workspace-root")
        # _records caches the last list; _selected_id survives refreshes so the selection does not jump.
        # _records is only a cache of the last list_recent() result, used to repopulate the
        # list; details always re-read the single record from the store so they never show
        # stale OCR after a background analysis finished.
        self._records = []
        self._selected_id = ""
        # Strong references to running analysis threads until they finish.
        self._active_threads = set()
        # True when pixel analysis cannot run (no multimodal backend or no model); set by _update_vision_status.
        # Starts False (optimistic) and is corrected by _update_vision_status at the end
        # of _build_ui, which runs before the window is first shown.
        self._vision_analysis_blocked = False
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        """Build the toolbar and a three-pane splitter: image library, preview/actions, and details/analysis editors."""
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # Toolbar: refresh, import a file, or re-run analysis on the selected image.
        # Toolbar duplicates "run analysis" with the button inside the analysis panel on
        # purpose: the toolbar one is the quick re-run for the selected image, the panel
        # one sits next to the mode and prompt controls it depends on. Both call _run_analysis.
        title_row = QWidget()
        tl = QHBoxLayout(title_row)
        tl.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Vision Workbench")
        title.setObjectName("lbl-accent")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; padding-bottom: 4px;")
        tl.addWidget(title)
        tl.addStretch()

        # Refresh exists because other workspaces (the Workbench paste, a background
        # analysis) add images to the store while this tab is hidden; nothing pushes
        # those changes here until the user refreshes or acts on a record.
        self._refresh_btn = QPushButton("refresh")
        self._refresh_btn.setObjectName("btn-ghost")
        self._refresh_btn.clicked.connect(self.refresh)
        tl.addWidget(self._refresh_btn)

        # Primary-styled because importing is how a first image enters this workspace
        # on a fresh install; the extension filter in the dialog mirrors what QImage
        # can decode.
        self._import_btn = QPushButton("import image")
        self._import_btn.setObjectName("btn-primary")
        self._import_btn.clicked.connect(self._import_image)
        tl.addWidget(self._import_btn)

        self._rerun_analysis_btn = QPushButton("run analysis")
        self._rerun_analysis_btn.setObjectName("btn-ghost")
        self._rerun_analysis_btn.clicked.connect(self._run_analysis)
        tl.addWidget(self._rerun_analysis_btn)
        root.addWidget(title_row)

        # Left pane: list of saved images.
        # Three panes in a horizontal splitter so the user can trade library width for
        # preview width; 1 px handle keeps it visually quiet.
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(8)
        self._library = QListWidget()
        # currentItemChanged fires for keyboard navigation as well as clicks, so the details
        # follow arrow-key browsing. refresh() blocks it while repopulating (see there).
        self._library.currentItemChanged.connect(self._on_selected)
        # The list has stretch 1 so it fills the whole left pane height.
        ll.addWidget(self._library, 1)
        splitter.addWidget(left)

        # Centre pane: scaled preview with "open file" and "send to workbench" actions.
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(8)
        self._preview = QLabel("No image selected.")
        self._preview.setObjectName("panel")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # 360x300 keeps screenshots legible; the preview is re-scaled from the original
        # file on each selection, so there is no resize handler and a window resize does
        # not rescale until the next selection.
        self._preview.setMinimumSize(360, 300)
        # False because stretching would distort the aspect ratio; scaling is done
        # manually with KeepAspectRatio in _render_record.
        self._preview.setScaledContents(False)
        # Preview gets stretch 1 and the action row stays one button high below it.
        cl.addWidget(self._preview, 1)
        action_row = QWidget()
        al = QHBoxLayout(action_row)
        al.setContentsMargins(0, 0, 0, 0)
        self._open_btn = QPushButton("open file")
        self._open_btn.clicked.connect(self._open_file)
        # Attach, not copy: only the record id travels through the signal, and the
        # Workbench composer loads the image from the shared store.
        self._send_btn = QPushButton("send to workbench")
        self._send_btn.setObjectName("btn-primary")
        self._send_btn.clicked.connect(self._send_to_workbench)
        # Open file hands the original to the OS default viewer for full-resolution
        # inspection, which the scaled preview cannot give.
        al.addWidget(self._open_btn)
        al.addWidget(self._send_btn)
        cl.addWidget(action_row)
        splitter.addWidget(center)

        # Right pane: metadata, analysis controls, kind/tags editors and the OCR and caption correction editors.
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)
        self._meta = QLabel("Select an image to inspect metadata.")
        self._meta.setObjectName("lbl-muted")
        # Word-wrap because the metadata includes long absolute file paths.
        self._meta.setWordWrap(True)
        rl.addWidget(self._meta)

        analysis_panel = QWidget()
        # Everything that controls an analysis run (warning, status, mode, prompt, run
        # button, registry summary) is grouped in one card, separate from the editors
        # that show the results.
        analysis_panel.setObjectName("panel")
        apl = QVBoxLayout(analysis_panel)
        apl.setContentsMargins(10, 10, 10, 10)
        apl.setSpacing(8)

        # Red warning box shown when vision analysis cannot run (see _update_vision_status).
        # Hidden until needed. It is a separate label from the neutral status line below
        # so the warning can carry its own red box styling.
        self._vision_warning = QLabel("")
        self._vision_warning.setObjectName("lbl-danger")
        self._vision_warning.setWordWrap(True)
        self._vision_warning.setVisible(False)
        self._vision_warning.setStyleSheet(
            "QLabel#lbl-danger {"
            "border: 1px solid rgba(255, 92, 122, 0.65);"
            "background: rgba(120, 20, 40, 0.28);"
            "color: #ff9aae;"
            "padding: 8px;"
            "border-radius: 6px;"
            "}"
        )
        apl.addWidget(self._vision_warning)

        # Neutral multi-line status (backend / loaded / active model, plus any error
        # strings the loader recorded) as opposed to the red warning above.
        self._vision_status = QLabel("")
        self._vision_status.setObjectName("lbl-muted")
        self._vision_status.setWordWrap(True)
        apl.addWidget(self._vision_status)

        # Modes: OCR plus vision model, OCR only (works with just Tesseract), or vision model only.
        # Item data (not text) is what _run_analysis reads: "ocr_vision", "ocr_only",
        # "vision_only" are the mode names ImageAnalysisThread understands.
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("OCR + Vision", "ocr_vision")
        self._mode_combo.addItem("OCR Only", "ocr_only")
        self._mode_combo.addItem("Vision Only", "vision_only")
        apl.addWidget(self._mode_combo)

        self._analysis_prompt = QTextEdit()
        # 72 px is about three lines: this is a question box, not a document editor,
        # and the OCR/caption editors below need the vertical space.
        self._analysis_prompt.setMaximumHeight(72)
        self._analysis_prompt.setPlaceholderText(
            "Optional image question for local vision model. Leave blank for Karl's screenshot/debugging prompt."
        )
        # An empty prompt is meaningful: it selects the built-in screenshot/debugging
        # prompt in ImageAnalysisThread, so the placeholder says so.
        apl.addWidget(self._analysis_prompt)

        analysis_row = QWidget()
        arl = QHBoxLayout(analysis_row)
        arl.setContentsMargins(0, 0, 0, 0)
        self._analysis_btn = QPushButton("analyze selected image")
        self._analysis_btn.setObjectName("btn-primary")
        self._analysis_btn.clicked.connect(self._run_analysis)
        arl.addWidget(self._analysis_btn)
        apl.addWidget(analysis_row)

        # Lists which vision models are fully installed (see _update_vision_status), so
        # a user who sees the "blocked" warning can tell what is missing.
        self._registry_label = QLabel("")
        self._registry_label.setObjectName("lbl-muted")
        self._registry_label.setWordWrap(True)
        apl.addWidget(self._registry_label)
        rl.addWidget(analysis_panel)

        self._kind_combo = QComboBox()
        # The kind values are stored verbatim on the record; the title-cased text is only
        # the display form. "unknown" is first so an unclassified image defaults to it.
        for kind in ("unknown", "code_screenshot", "error", "document", "diagram", "ui_mockup", "photo"):
            self._kind_combo.addItem(kind.replace("_", " ").title(), kind)
        rl.addWidget(self._kind_combo)

        # Comma-separated free text; _save_metadata splits and trims it, dropping empties.
        self._tags = QLineEdit()
        self._tags.setPlaceholderText("tags: code, error, vscode")
        rl.addWidget(self._tags)

        meta_row = QWidget()
        ml = QHBoxLayout(meta_row)
        ml.setContentsMargins(0, 0, 0, 0)
        # Metadata, OCR and caption each have their own save button because they are
        # separate fields on the record; saving one must not overwrite edits the user is
        # still making in another.
        self._save_meta_btn = QPushButton("save metadata")
        self._save_meta_btn.clicked.connect(self._save_metadata)
        ml.addWidget(self._save_meta_btn)
        rl.addWidget(meta_row)

        self._ocr = QTextEdit()
        # Editable on purpose: OCR is imperfect. A saved correction is written both to
        # record.corrections["ocr_text"] and over ocr.text (save_ocr_correction), so
        # readers of the OCR field see the fixed text; the engine name and confidence
        # are kept, but the raw engine text itself is not preserved.
        self._ocr.setReadOnly(False)
        self._ocr.setPlaceholderText("OCR text will appear here after analysis. Edit and save corrections here.")
        rl.addWidget(self._ocr, 1)

        ocr_row = QWidget()
        ol = QHBoxLayout(ocr_row)
        ol.setContentsMargins(0, 0, 0, 0)
        self._save_ocr_btn = QPushButton("save OCR correction")
        self._save_ocr_btn.setObjectName("btn-primary")
        self._save_ocr_btn.clicked.connect(self._save_ocr_correction)
        ol.addWidget(self._save_ocr_btn)
        rl.addWidget(ocr_row)

        # The caption editor works the same way as the OCR one but is largely
        # forward-looking: captions appear only when a vision model produced one.
        self._caption = QTextEdit()
        self._caption.setPlaceholderText("Future visual summary/caption correction area.")
        rl.addWidget(self._caption, 1)

        caption_row = QWidget()
        crl = QHBoxLayout(caption_row)
        crl.setContentsMargins(0, 0, 0, 0)
        self._save_caption_btn = QPushButton("save caption correction")
        self._save_caption_btn.clicked.connect(self._save_caption_correction)
        crl.addWidget(self._save_caption_btn)
        rl.addWidget(caption_row)

        # The right panel stacks a lot of controls (metadata, warning/status
        # labels, mode combo, prompt box, kind combo, tags, OCR editor,
        # caption editor, and their action buttons) with no stretch to
        # spare -- at the app's 760x560 minimum size there isn't enough
        # vertical room for all of it, so several got squeezed down far
        # enough to visibly clip their own (word-wrapped) text. Scrolling
        # beats every widget silently losing height it needs.
        # Without this the right pane's many fixed-minimum widgets are squeezed below
        # their needed height at the minimum window size.
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setWidget(right)
        splitter.addWidget(right_scroll)

        # Stretch factors give the preview the most room, then the details pane, then the library.
        # Factors are relative shares of extra width: library 1, preview 3, details 2.
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        root.addWidget(splitter, 1)
        self._update_vision_status()

    def refresh(self):
        """Reload up to 500 recent images, preserve the current selection if it still exists, otherwise select the first row."""
        # 500 caps the load time and list length; images beyond it stay in the store but
        # are not listed. Within the returned set rows are ordered newest first (by
        # created_at), so row 0 is the latest. Caveat: the store applies the limit while
        # scanning files in reverse uuid-name order, so with more than 500 images the
        # set is not guaranteed to be the 500 newest.
        self._records = self.state.image_store.list_recent(limit=500)
        current = self._selected_id
        # Block signals while repopulating so selection handlers do not fire for every row.
        # Selection is restored by id, not row index, because new images shift every row.
        self._library.blockSignals(True)
        self._library.clear()
        selected_item = None
        for record in self._records:
            # created_at is datetime.now().isoformat() (local time, no zone, microseconds);
            # [:19] trims it to "YYYY-MM-DDTHH:MM:SS" for a compact, sortable label.
            label = f"{record.created_at[:19]}  {record.width}x{record.height}  {record.kind}"
            item = QListWidgetItem(label)
            # The record id rides in UserRole so selection handlers can look the record up
            # without parsing the label.
            item.setData(Qt.ItemDataRole.UserRole, record.id)
            item.setToolTip(record.original_path)
            self._library.addItem(item)
            if record.id == current:
                selected_item = item
        self._library.blockSignals(False)
        # Selection fallback order: previously selected image, else the newest (row 0),
        # else empty state. setCurrentItem fires _on_selected (signals are unblocked now).
        if selected_item:
            self._library.setCurrentItem(selected_item)
        elif self._library.count():
            self._library.setCurrentRow(0)
        else:
            self._clear_details()

    def _on_selected(self, current, _previous):
        """Show the clicked image, or clear the panels when nothing is selected."""
        if not current:
            self._clear_details()
            return
        # currentItemChanged also fires with current=None when the list is cleared; that
        # case was handled above, so the id here is always a string.
        image_id = current.data(Qt.ItemDataRole.UserRole)
        self._selected_id = image_id
        self._render_record(self.state.image_store.get(image_id))

    def _render_record(self, record):
        """Fill the preview, metadata text, kind, tags, OCR and caption editors from an image record."""
        # The pixmap is scaled to the current preview size, keeping the aspect ratio.
        # Loads the original file, not a thumbnail: the store keeps one copy and scaling
        # happens here. A moved or deleted original gives a null pixmap, handled below
        # with a text fallback rather than an exception.
        pix = QPixmap(record.original_path)
        if not pix.isNull():
            scaled = pix.scaled(
                self._preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._preview.setPixmap(scaled)
        else:
            self._preview.setText("Could not load image preview.")

        self._meta.setText(
            "\n".join([
                f"ID: {record.id}",
                f"Source: {record.source}",
                f"Kind: {record.kind}",
                f"Size: {record.width}x{record.height}",
                # Confidence is the OCR engine's own score for the whole text; the schema
                # default is 0.0, so 0.00 also means "no score reported".
                f"OCR: {record.ocr.engine} · confidence {record.ocr.confidence:.2f}",
                f"Vision: {record.vision.engine} · {record.vision.model or 'no model'}",
                f"Detected code: {record.vision.detected_code}",
                f"Detected error: {record.vision.detected_error}",
                f"Path: {record.original_path}",
            ])
        )
        # findData returns -1 for a kind outside the fixed list (for example one written
        # by an older version); falling back to index 0 ("unknown") shows a valid value,
        # but saving then rewrites the kind.
        idx = self._kind_combo.findData(record.kind)
        self._kind_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._tags.setText(", ".join(record.tags))
        self._ocr.setPlainText(record.ocr.text or "(OCR pending or unavailable.)")
        self._caption.setPlainText(record.vision.caption or "")
        self._update_vision_status()

    def _clear_details(self):
        """Reset every detail widget to its empty state."""
        self._selected_id = ""
        self._preview.setText("No image selected.")
        self._preview.setPixmap(QPixmap())
        self._meta.setText("No images saved yet. Paste one into Workbench or import a file.")
        self._kind_combo.setCurrentIndex(0)
        self._tags.clear()
        self._ocr.clear()
        self._caption.clear()
        self._update_vision_status()

    def _selected_record(self):
        """Return the selected image record from the store, or None."""
        if not self._selected_id:
            return None
        return self.state.image_store.get(self._selected_id)

    def _import_image(self):
        """Import an image file into the store, select it, and start analysis right away."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import image into Karl Vision",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff);;All files (*)",
        )
        if not path:
            return
        try:
            # The store decodes the file once to reject non-images, then copies it into its
            # inbox under a new uuid (the original stays untouched) and stamps source="file";
            # pasted images arrive from the Workbench with source="clipboard".
            record = self.state.image_store.import_file(path, source="file")
        except Exception as exc:
            QMessageBox.critical(self, "Image Import Failed", str(exc))
            return
        self._selected_id = record.id
        self.refresh()
        self._run_analysis()

    def _open_file(self):
        """Open the original image file with the system default application."""
        record = self._selected_record()
        if record:
            # resolve() makes the path absolute so the OS handler does not depend on the
            # process working directory.
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(record.original_path).resolve())))

    def _send_to_workbench(self):
        """Switch to the Workbench (index 0) and attach the selected image to its composer."""
        record = self._selected_record()
        if not record:
            return
        # Index 0 is the Workbench in the main window's workspace stack; emitting the
        # switch first means the composer is already visible when the attach arrives.
        self.state.change_workspace_requested.emit(0)
        self.state.attach_image_to_workbench.emit(record.id)
        QMessageBox.information(self, "Image Sent", "Image attached to the Workbench composer.")

    def _run_analysis(self):
        """Start a background analysis of the selected image in the chosen mode."""
        record = self._selected_record()
        if not record:
            return
        # Refresh status first so the blocked flag is current before deciding whether to run.
        self._update_vision_status()
        mode = self._mode_combo.currentData() or "ocr_vision"
        # OCR-only needs no vision model, so only the other modes are blocked.
        if mode != "ocr_only" and self._vision_analysis_blocked:
            QMessageBox.warning(
                self,
                "Vision Model Required",
                self._vision_warning.text(),
            )
            return
        # An empty prompt becomes None so the thread uses its default screenshot/debugging prompt.
        prompt = self._analysis_prompt.toPlainText().strip() or None
        # Progress is appended to the metadata text instead of a separate widget; it is
        # wiped by the next _render_record, which also runs when analysis finishes.
        self._meta.setText(self._meta.text() + f"\nAnalysis: running ({mode})")
        # ImageAnalysisThread reports through signals; widgets are only touched from these slots.
        # Runs OCR and/or the local vision model off the GUI thread; results are keyed by
        # image id so a late result for a previously selected image is ignored by the
        # slots below instead of overwriting the current view.
        thread = ImageAnalysisThread(self.state.image_store, record.id, mode=mode, prompt=prompt)
        # Signal set: progress(str), ocr_done(id, OcrResult), vision_done(id, VisionResult),
        # done(id, record), error(str). ocr_done arrives before vision_done in the
        # combined mode so OCR text shows while the slower vision model still runs.
        thread.progress.connect(self._on_analysis_progress)
        thread.ocr_done.connect(self._on_ocr_done)
        thread.vision_done.connect(self._on_vision_done)
        thread.done.connect(self._on_analysis_done)
        thread.error.connect(self._on_analysis_error)
        # Strong reference until finished, otherwise Qt could destroy a running QThread
        # when this function returns.
        self._active_threads.add(thread)
        thread.finished.connect(lambda: self._active_threads.discard(thread))
        # Registered after the discard so the Python reference is gone before Qt
        # schedules deletion of the C++ object.
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def _on_analysis_progress(self, msg: str):
        """Append a progress line to the metadata text."""
        if self._selected_id:
            self._meta.setText(self._meta.text() + f"\nAnalysis: {msg}")

    def _on_ocr_done(self, image_id: str, ocr):
        """Show OCR text as soon as it arrives, if the image is still selected."""
        if image_id == self._selected_id:
            self._ocr.setPlainText(ocr.text or "(OCR unavailable or no text detected.)")

    def _on_vision_done(self, image_id: str, vision):
        """Show the vision caption when ready, if the image is still selected."""
        if image_id == self._selected_id:
            self._caption.setPlainText(vision.caption or "")
            self._update_vision_status()

    # _record is ignored deliberately: the store is the source of truth, so the final
    # state is re-read (get) rather than trusting the thread's object.
    def _on_analysis_done(self, image_id: str, _record):
        """Re-render the finished record (when still selected) and refresh the list."""
        if image_id == self._selected_id:
            self._render_record(self.state.image_store.get(image_id))
        self.refresh()

    def _on_analysis_error(self, msg: str):
        """Warn the user that analysis failed and refresh the vision status."""
        # Not tied to an image id, so it warns regardless of the current selection; the
        # status refresh picks up a backend error the failure may have recorded.
        QMessageBox.warning(self, "Image Analysis Failed", msg)
        self._update_vision_status()

    def _save_metadata(self):
        """Save the kind and comma-separated tags of the selected image."""
        record = self._selected_record()
        if not record:
            return
        # Tags are not lowercased or deduplicated here; the store owns normalisation.
        tags = [t.strip() for t in self._tags.text().split(",") if t.strip()]
        updated = self.state.image_store.update_metadata(
            record.id,
            kind=self._kind_combo.currentData() or "unknown",
            tags=tags,
        )
        self._render_record(updated)
        self.refresh()

    def _save_ocr_correction(self):
        """Save the user's edited OCR text as the correction for the selected image."""
        record = self._selected_record()
        if not record:
            return
        # Correction overwrites ocr.text (and is also recorded under corrections); the
        # engine name and confidence stay, so _render_record still shows which engine
        # produced the original result even after a human edit.
        updated = self.state.image_store.save_ocr_correction(record.id, self._ocr.toPlainText().strip())
        self._render_record(updated)
        self.refresh()

    def _save_caption_correction(self):
        """Save the user's edited caption as the correction for the selected image."""
        record = self._selected_record()
        if not record:
            return
        updated = self.state.image_store.save_caption_correction(record.id, self._caption.toPlainText().strip())
        self._render_record(updated)
        self.refresh()

    def _update_vision_status(self):
        """Refresh runtime/registry labels and decide whether to show a warning and block pixel analysis."""
        # Called from five places (build, selection render, vision-done, error, run):
        # the status is cheap and in-memory, and recomputing it each time means the
        # warning box and the blocked flag can never disagree with the loader.
        status = VisionModelLoader.status()
        backend = "ready" if status["backend_available"] else "blocked"
        active = status.get("active_name") or "none"
        loaded = "loaded" if status["loaded"] else "not loaded"
        lines = [f"Vision runtime: {backend} · {loaded} · active: {active}"]
        if status.get("backend_error"):
            lines.append(status["backend_error"])
        if status.get("last_error"):
            lines.append(status["last_error"])
        self._vision_status.setText("\n".join(lines))

        # An installed model needs both its model file and its projector file.
        # Registry rows come from the vision model registry on disk, not from loaded
        # state, so this reflects files present even before anything is loaded.
        rows = installed_vision_models()
        installed = [
            row["name"]
            for row in rows
            if row.get("model_installed") and row.get("projector_installed")
        ]
        # Registry entries whose model or projector file is absent; shown as a count
        # so the user knows more models are available once files are downloaded.
        missing_count = len(rows) - len(installed)
        if installed:
            registry = "Installed vision models: " + ", ".join(installed)
        else:
            registry = "No complete local vision model/projector pair installed."
        if missing_count:
            registry += f"\nRegistry entries waiting for local files: {missing_count}"
        self._registry_label.setText(registry)

        # Backend missing or nothing installed: block analysis. Installed but unloaded: warn only, since analysis loads it on demand.
        # Two independent outcomes: needs_warning controls the red box, blocks_analysis
        # controls whether vision modes may start. They differ in the "installed but not
        # loaded" case, which warns without blocking because analysis loads it lazily.
        needs_warning = False
        blocks_analysis = False
        warning_lines = []
        if not status["backend_available"]:
            needs_warning = True
            blocks_analysis = True
            warning_lines.append("Vision pixel analysis is blocked because llama-cpp multimodal support is unavailable.")
        if not installed:
            needs_warning = True
            blocks_analysis = True
            warning_lines.append(
                "Vision requires a local multimodal GGUF plus matching projector file. "
                "The current text model can still chat, but it cannot see image pixels."
            )
        elif not status["loaded"]:
            needs_warning = True
            warning_lines.append(
                "A vision model/projector pair is installed but not loaded yet. "
                "Run analysis to load it, or use OCR Only for text extraction."
            )

        self._vision_warning.setText("\n".join(warning_lines))
        self._vision_warning.setVisible(needs_warning)
        # Read by _run_analysis. OCR-only mode ignores it, so a machine with only
        # Tesseract installed still gets useful analysis.
        self._vision_analysis_blocked = blocks_analysis
