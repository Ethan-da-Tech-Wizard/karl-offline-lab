# Program: karl-offline-lab - image_analysis_thread.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Background Qt thread that runs OCR and vision analysis on a stored image and reports progress, results and errors through signals.
"""Worker thread for analysing a stored image with OCR and a vision model without blocking the UI."""
from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal

from app.vision.ocr_engine import TesseractOcrEngine
from app.vision.vision_analyzer import VisionAnalyzer


class ImageAnalysisThread(QThread):
    """QThread that analyses one stored image and emits progress, OCR, vision, done and error signals."""
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
        """Remember the image store, the image id, the OCR language, the analysis mode and an optional vision prompt."""
        super().__init__(parent)
        self.image_store = image_store
        self.image_id = image_id
        self.lang = lang
        self.mode = mode
        self.prompt = prompt

    def run(self):
        """Run OCR and/or vision analysis according to the mode, save the results to the image store, then emit done (or error on failure)."""
        try:
            record = self.image_store.get(self.image_id)
            # Prefer the cleaned-up image when one exists, otherwise use the original.
            image_path = record.processed_path or record.original_path

            ocr = record.ocr
            # OCR runs in the ocr_only and ocr_vision modes.
            if self.mode in {"ocr_only", "ocr_vision"}:
                self.progress.emit("running OCR")
                ocr = TesseractOcrEngine().analyze(image_path, lang=self.lang)
                record.ocr = ocr
                updated = self.image_store.update_analysis(self.image_id, ocr=ocr.to_dict())
                record = updated
                self.ocr_done.emit(self.image_id, ocr)

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
                # Only overwrite the image kind when it is still unknown or the analyzer detected an error or code screenshot.
                if record.kind == "unknown" or suggestion.kind in {"error", "code_screenshot"}:
                    fields["kind"] = suggestion.kind
                updated = self.image_store.update_analysis(self.image_id, **fields)
                self.vision_done.emit(self.image_id, vision)
            else:
                updated = record

            self.done.emit(self.image_id, updated)
        except Exception as exc:
            self.error.emit(str(exc))
