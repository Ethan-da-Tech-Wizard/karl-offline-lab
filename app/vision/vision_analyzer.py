# Program: karl-offline-lab - vision_analyzer.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Classifies images from their OCR text (error, code, document, UI) and combines that with a vision model caption to analyze a stored image.
"""Heuristic image classification and vision analysis."""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.vision.schemas import ImageRecord, OcrResult, VisionResult
from app.vision.vision_model_loader import VisionModelLoader


# === Keyword and regex lists used to guess what an image shows ===
# Lower-case words that suggest the text is an error message or traceback.
ERROR_PATTERNS = (
    "traceback",
    "exception",
    "error",
    "failed",
    "failure",
    "warning",
    "segmentation fault",
    "nameerror",
    "typeerror",
    "valueerror",
    "importerror",
    "modulenotfounderror",
    "syntaxerror",
    "referenceerror",
    "stack trace",
)

# Regular expressions that suggest source code.
CODE_PATTERNS = (
    r"\bdef\s+\w+",
    r"\bclass\s+\w+",
    r"\bimport\s+\w+",
    r"\bfrom\s+\w+\s+import\b",
    r"\bfunction\s+\w+",
    r"\bconst\s+\w+",
    r"\blet\s+\w+",
    r"\bvar\s+\w+",
    r"=>",
    r"\{.*\}",
    r"\bline\s+\d+",
)

# Words that suggest a scanned or photographed document.
DOCUMENT_PATTERNS = (
    "invoice",
    "receipt",
    "page",
    "chapter",
    "paragraph",
    "document",
    "signature",
)

# Words that suggest a user-interface screenshot.
UI_PATTERNS = (
    "button",
    "menu",
    "dialog",
    "sidebar",
    "toolbar",
    "modal",
    "panel",
)


@dataclass(frozen=True)
class AnalysisSuggestion:
    """Heuristic result: suggested kind, tags, code/error flags and a layout hint."""
    kind: str
    tags: list[str]
    detected_code: bool
    detected_error: bool
    layout: str


class VisionAnalyzer:
    """Runs the vision model (or an OCR-only heuristic) for an image record."""
    def __init__(self, vision_loader=VisionModelLoader):
        """Use vision_loader (class or object with describe_image) to describe images."""
        self.vision_loader = vision_loader

    def analyze_record(
        self,
        record: ImageRecord,
        ocr: OcrResult | None = None,
        mode: str = "ocr_vision",
        prompt: str | None = None,
    ) -> tuple[VisionResult, AnalysisSuggestion]:
        """Analyze one image record.

        mode 'ocr_only' skips the vision model and builds a caption from heuristics; any other mode calls the vision loader and fills gaps (layout, flags, caption) from the heuristics.
        Returns the vision result and the classification suggestion.
        """
        effective_ocr = ocr or record.ocr
        suggestion = classify_image(record, effective_ocr.text)

        # OCR-only mode never loads a model.
        if mode == "ocr_only":
            vision = VisionResult(
                engine="ocr-heuristic",
                model=None,
                caption=self._heuristic_caption(record, effective_ocr, suggestion),
                layout=suggestion.layout,
                detected_code=suggestion.detected_code,
                detected_error=suggestion.detected_error,
            )
            return vision, suggestion

        vision = self.vision_loader.describe_image(
            record.original_path,
            prompt=prompt,
        )
        if not vision.layout:
            vision.layout = suggestion.layout
        # Keep a detection the model made, and add the heuristic's detection.
        vision.detected_code = vision.detected_code or suggestion.detected_code
        vision.detected_error = vision.detected_error or suggestion.detected_error
        if not vision.caption:
            vision.caption = self._heuristic_caption(record, effective_ocr, suggestion)
        return vision, suggestion

    @staticmethod
    def _heuristic_caption(record: ImageRecord, ocr: OcrResult, suggestion: AnalysisSuggestion) -> str:
        """Build a plain-text caption from size, kind and OCR state when no model caption is available."""
        ocr_state = "OCR text detected" if ocr.text.strip() else "No OCR text detected"
        details = [
            f"{record.width}x{record.height} image classified as {suggestion.kind}.",
            ocr_state + f" with engine {ocr.engine}.",
        ]
        if suggestion.detected_error:
            details.append("Visible text appears to contain an error or traceback.")
        elif suggestion.detected_code:
            details.append("Visible text appears to contain source code or terminal output.")
        details.append("Load a local vision model for pixel-level description beyond OCR and heuristics.")
        return " ".join(details)


def classify_image(record: ImageRecord, text: str) -> AnalysisSuggestion:
    """Guess the image kind and tags from OCR text using keyword rules.

    Rules are checked in priority order: error, code, document, UI.
    """
    lowered = text.lower()
    detected_error = any(marker in lowered for marker in ERROR_PATTERNS)
    # An error screenshot is also treated as code-like text.
    detected_code = detected_error or any(re.search(pattern, text, re.IGNORECASE | re.DOTALL) for pattern in CODE_PATTERNS)

    tags: list[str] = []
    kind = record.kind if record.kind != "unknown" else "unknown"

    if detected_error:
        kind = "error"
        tags.extend(["error", "debug"])
    elif detected_code:
        kind = "code_screenshot"
        tags.extend(["code", "screenshot"])
    elif any(marker in lowered for marker in DOCUMENT_PATTERNS):
        kind = "document"
        tags.extend(["document", "ocr"])
    elif any(marker in lowered for marker in UI_PATTERNS):
        kind = "ui_mockup"
        tags.extend(["ui", "interface"])

    if record.source == "clipboard":
        tags.append("clipboard")
    if text.strip():
        tags.append("ocr")

    layout = _layout_hint(record, text)
    return AnalysisSuggestion(
        kind=kind,
        tags=_dedupe([*record.tags, *tags]),
        detected_code=detected_code,
        detected_error=detected_error,
        layout=layout,
    )


def _layout_hint(record: ImageRecord, text: str) -> str:
    """Describe the frame shape (from aspect ratio) and how dense the OCR text is."""
    aspect = record.width / record.height if record.height else 1.0
    lines = [line for line in text.splitlines() if line.strip()]
    if aspect > 1.6:
        shape = "wide landscape"
    elif aspect < 0.75:
        shape = "tall portrait"
    else:
        shape = "balanced frame"
    if len(lines) > 12:
        return f"{shape}; dense text layout with {len(lines)} OCR lines"
    if lines:
        return f"{shape}; sparse text layout with {len(lines)} OCR lines"
    return f"{shape}; no OCR text layout signal"


def _dedupe(values: list[str]) -> list[str]:
    """Lower-case and strip values, dropping blanks and duplicates while keeping first-seen order."""
    seen = set()
    out = []
    for value in values:
        clean = value.strip().lower()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out
