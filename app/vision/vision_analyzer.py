"""
Combines OCR text, simple keyword heuristics and an optional local vision model.

classify_image() looks only at the OCR text (and record metadata) to guess
what kind of picture it is: error screenshot, code screenshot, document or UI
mock-up. VisionAnalyzer.analyze_record() then either returns a heuristic-only
result (mode "ocr_only") or asks the vision model for a caption and merges
the heuristic flags into it.
"""
# ==============================================================================
# Program: vision_analyzer.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Core utility providing vision analyzer capabilities.
#          Implements data structures and algorithms supporting offline
#          introspection and model execution.
# ==============================================================================

from __future__ import annotations

import re
from dataclasses import dataclass

from app.vision.schemas import ImageRecord, OcrResult, VisionResult
from app.vision.vision_model_loader import VisionModelLoader


# Lower-case substrings; matched against lower-cased OCR text. Deliberately broad
# (e.g. "error", "warning") so a likely traceback is never missed - false
# positives only cost an extra tag.
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

# Regexes for common code syntax across Python/JS (def, class, import, arrow
# functions, braces) plus "line N" as seen in stack traces.
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

# Words typical of scanned documents; checked only if no error/code was found.
DOCUMENT_PATTERNS = (
    "invoice",
    "receipt",
    "page",
    "chapter",
    "paragraph",
    "document",
    "signature",
)

# Words typical of application UI screenshots; lowest-priority classification.
UI_PATTERNS = (
    "button",
    "menu",
    "dialog",
    "sidebar",
    "toolbar",
    "modal",
    "panel",
)


# Frozen so a suggestion cannot be mutated after classification.
@dataclass(frozen=True)
class AnalysisSuggestion:
    """Heuristic guess about an image: kind, tags, code/error flags and a layout hint."""
    kind: str
    tags: list[str]
    detected_code: bool
    detected_error: bool
    layout: str


class VisionAnalyzer:
    """Produces a VisionResult for an image record, with or without a vision model."""
    def __init__(self, vision_loader=VisionModelLoader):
        # Step 1: Validate input parameters, operational preconditions, and state
        """``vision_loader`` defaults to the VisionModelLoader class (its methods are class-level); tests can inject a stub."""
        self.vision_loader = vision_loader

    def analyze_record(
        self,
        record: ImageRecord,
        ocr: OcrResult | None = None,
        mode: str = "ocr_vision",
        prompt: str | None = None,
    ) -> tuple[VisionResult, AnalysisSuggestion]:
        # Step 1: Validate input parameters, operational preconditions, and state
        # Prefer freshly supplied OCR over what is stored on the record.
        """Analyze one image record and return (VisionResult, AnalysisSuggestion).

        mode "ocr_only" skips the vision model and builds a caption from heuristics.
        Any other mode calls the vision loader and then fills gaps: missing layout,
        code/error flags (OR-ed with the heuristics) and an empty caption.
        """
        effective_ocr = ocr or record.ocr
        suggestion = classify_image(record, effective_ocr.text)

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

        # Heavy path: may load a local vision model; failures are reported inside the returned VisionResult by the loader.
        vision = self.vision_loader.describe_image(
            record.original_path,
            prompt=prompt,
        )
        if not vision.layout:
            vision.layout = suggestion.layout
        # OR the flags: trust either the model or the text heuristics if one saw a problem.
        vision.detected_code = vision.detected_code or suggestion.detected_code
        vision.detected_error = vision.detected_error or suggestion.detected_error
        if not vision.caption:
            vision.caption = self._heuristic_caption(record, effective_ocr, suggestion)
        return vision, suggestion

    @staticmethod
    def _heuristic_caption(record: ImageRecord, ocr: OcrResult, suggestion: AnalysisSuggestion) -> str:
        # Step 1: Validate input parameters, operational preconditions, and state
        """Build a plain-English caption from size, kind and OCR state without any model."""
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
    # Step 1: Validate input parameters, operational preconditions, and state
    """Classify an image from its OCR ``text`` and record metadata.

    Priority order: error > code > document > UI. An error also implies code
    (tracebacks are code-like). Existing tags on the record are preserved.
    """
    lowered = text.lower()
    detected_error = any(marker in lowered for marker in ERROR_PATTERNS)
    detected_code = detected_error or any(re.search(pattern, text, re.IGNORECASE | re.DOTALL) for pattern in CODE_PATTERNS)

    tags: list[str] = []
    # Keep a kind the user already set unless a heuristic below overrides it.
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
    # Step 1: Validate input parameters, operational preconditions, and state
    # Guard against height 0 to avoid ZeroDivisionError.
    """Describe image shape (wide/tall/balanced) and how much text OCR found.

    More than 12 non-empty lines counts as "dense".
    """
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
    # Step 1: Validate input parameters, operational preconditions, and state
    """Lower-case, strip and de-duplicate tags while keeping first-seen order."""
    seen = set()
    out = []
    for value in values:
        # Iterate over items, evaluate loop invariant, and update progress
        clean = value.strip().lower()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out
