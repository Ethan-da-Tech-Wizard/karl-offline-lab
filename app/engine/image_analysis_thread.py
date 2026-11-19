# ==============================================================================
# Program: image_analysis_thread.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Background worker thread managing vision and multimodal inference.
#          Processes images, generates embeddings, performs OCR analysis, and
#          returns structured findings to the UI.
# ==============================================================================

"""
Qt worker thread that analyzes one stored image (OCR, then vision) off the GUI thread.

The thread reads an ImageRecord from the ImageStore, runs Tesseract OCR and/or
the vision analyzer depending on ``mode``, writes results back to the store,
and reports progress through Qt signals. All errors are turned into the
``error`` signal; nothing is raised into the Qt event loop.
"""

from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal

from app.vision.ocr_engine import TesseractOcrEngine
from app.vision.vision_analyzer import VisionAnalyzer


class ImageAnalysisThread(QThread):
    # Signals are the only safe way to hand results back to the GUI thread.
    # The ``object`` payloads carry OcrResult / VisionResult / ImageRecord instances.
    """Run OCR and/or vision analysis for a single image id in the background.

    Modes: "ocr_only", "ocr_vision" (default) and "vision_only".
    """
    progress = pyqtSignal(str)
    ocr_done = pyqtSignal(str, object)  # image_id, OcrResult
    vision_done = pyqtSignal(str, object)  # image_id, VisionResult
    done = pyqtSignal(str, object)      # image_id, ImageRecord
    error = pyqtSignal(str)

    def __init__(
        self,
        image_store,
        image_id: str,
        lang: str = "eng",
        mode: str = "ocr_vision",
        prompt: str | None = None,
        parent=None,
    ):
        """Store the image store, target image id, OCR language, analysis mode and optional prompt."""
        super().__init__(parent)
        self.image_store = image_store
        self.image_id = image_id
        self.lang = lang
        self.mode = mode
        self.prompt = prompt

    def run(self):
        """Thread body: OCR (if the mode asks for it), then vision analysis, then emit ``done``.

        Each stage persists its result immediately so partial work survives a later failure.
        """
        try:
            record = self.image_store.get(self.image_id)
            # Prefer the OCR-normalized copy; fall back to the original file.
            image_path = record.processed_path or record.original_path

            ocr = record.ocr
            # Stage 1: OCR.
            if self.mode in {"ocr_only", "ocr_vision"}:
                self.progress.emit("running OCR")
                ocr = TesseractOcrEngine().analyze(image_path, lang=self.lang)
                record.ocr = ocr
                updated = self.image_store.update_analysis(self.image_id, ocr=ocr.to_dict())
                record = updated
                self.ocr_done.emit(self.image_id, ocr)

            # Stage 2: vision/heuristic analysis (all modes reach here, but the analyzer itself
            # skips the model for "ocr_only").
            if self.mode in {"ocr_only", "ocr_vision", "vision_only"}:
                self.progress.emit("running vision analysis")
                vision, suggestion = VisionAnalyzer().analyze_record(
                    record,
                    ocr=ocr,
                    mode=self.mode,
                    prompt=self.prompt,
                )
                fields = {
                    "vision": vision.to_dict(),
                    "tags": suggestion.tags,
                }
                # Only overwrite the kind when it was never set, or when the heuristics found a
                # high-value kind (error / code screenshot); this avoids clobbering a user-chosen kind.
                if record.kind == "unknown" or suggestion.kind in {"error", "code_screenshot"}:
                    fields["kind"] = suggestion.kind
                updated = self.image_store.update_analysis(self.image_id, **fields)
                self.vision_done.emit(self.image_id, vision)
            else:
                updated = record

            self.done.emit(self.image_id, updated)
        except Exception as exc:
            self.error.emit(str(exc))
