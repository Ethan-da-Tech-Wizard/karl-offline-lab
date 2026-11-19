# ==============================================================================
# Program: vision_analyzer.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Core utility providing vision analyzer capabilities.
#          Implements data structures and algorithms supporting offline
#          introspection and model execution.
# ==============================================================================

"""
Combines OCR text, simple keyword heuristics and an optional local vision model.

classify_image() looks only at the OCR text (and record metadata) to guess
what kind of picture it is: error screenshot, code screenshot, document or UI
mock-up. VisionAnalyzer.analyze_record() then either returns a heuristic-only
result (mode "ocr_only") or asks the vision model for a caption and merges
the heuristic flags into it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.vision.schemas import ImageRecord, OcrResult, VisionResult
from app.vision.vision_model_loader import VisionModelLoader


# Lower-case substrings; matched against lower-cased OCR text. Deliberately broad
# (e.g. "error", "warning") so a likely traceback is never missed - false
# positives only cost an extra tag.
# Tuple order has no meaning (any() short-circuits on the first hit). Entries are
# plain substrings, so "error" also matches "terror" or "errors.log"; that is
# acceptable because the only consequence is an extra "error"/"debug" tag and
# the kind "error", which the user can change in the Vision Workbench.
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
# Applied with re.IGNORECASE | re.DOTALL to the ORIGINAL-case text. r"\{.*\}" is the
# broadest entry: with DOTALL it matches any text that has a "{" before a "}"
# anywhere, which makes prose with braces count as code. Kept because OCR often
# breaks code across lines, and a missed code screenshot is worse than an extra tag.
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
# Generic office vocabulary. "page" and "document" are common in UI text too, which
# is why this group is checked after code/error and before the UI group only.
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
# Checked last: a screenshot of an app window frequently contains a panel or a
# button label, so these words are weak evidence and only win when nothing
# stronger matched.
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
# frozen=True prevents accidental edits, but note the tags list inside is still a
# mutable list object; callers should copy before changing it. detected_* are
# booleans from the text heuristics only, not from any model.
@dataclass(frozen=True)
class AnalysisSuggestion:
    """Heuristic guess about an image: kind, tags, code/error flags and a layout hint."""
    kind: str
    tags: list[str]
    detected_code: bool
    detected_error: bool
    layout: str


# Stateless apart from the injected loader. All real work for pixel analysis is
# delegated to the loader (describe_image); this class adds the heuristic fallback
# that makes the feature useful with nothing but Tesseract installed.
class VisionAnalyzer:
    """Produces a VisionResult for an image record, with or without a vision model."""
    def __init__(self, vision_loader=VisionModelLoader):
        """``vision_loader`` defaults to the VisionModelLoader class (its methods are class-level); tests can inject a stub."""
        self.vision_loader = vision_loader

    def analyze_record(
        self,
        record: ImageRecord,
        ocr: OcrResult | None = None,
        mode: str = "ocr_vision",
        prompt: str | None = None,
    ) -> tuple[VisionResult, AnalysisSuggestion]:
        # Prefer freshly supplied OCR over what is stored on the record.
        """Analyze one image record and return (VisionResult, AnalysisSuggestion).

        mode "ocr_only" skips the vision model and builds a caption from heuristics.
        Any other mode calls the vision loader and then fills gaps: missing layout,
        code/error flags (OR-ed with the heuristics) and an empty caption.
        """
        # Caller-supplied OCR wins because it is the result of the run that is still in
        # progress; record.ocr may be the previous (or empty) state of the record.
        effective_ocr = ocr or record.ocr
        # Always computed, even for model modes: its flags are merged into the model's
        # result below and its tags/kind are returned to the caller to store.
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
            # engine "ocr-heuristic" with model=None marks a result that no model produced, so
            # the UI can show "no model" in the metadata block.
            return vision, suggestion

        # Heavy path: may load a local vision model; failures are reported inside the returned VisionResult by the loader.
        # describe_image does not raise: with no backend or model it returns a VisionResult
        # (engine "llama-cpp-unavailable") whose caption holds the error message, so no
        # try/except is needed here. Consequence: that message is non-empty, so the
        # heuristic-caption fallback below only fires when a model answered with nothing.
        vision = self.vision_loader.describe_image(
            record.original_path,
            prompt=prompt,
        )
        # The model rarely describes layout; the pixel-shape hint (wide/tall, text
        # density) is still useful metadata, so it fills the gap.
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
        """Build a plain-English caption from size, kind and OCR state without any model."""
        # Caption is a deterministic template, so ocr_only runs are reproducible and
        # testable; the closing sentence nudges the user toward loading a vision model.
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
    """Classify an image from its OCR ``text`` and record metadata.

    Priority order: error > code > document > UI. An error also implies code
    (tracebacks are code-like). Existing tags on the record are preserved.
    """
    # Two views of the text on purpose: plain substring lists run on `lowered`, while
    # the code regexes run on the original `text` with IGNORECASE (case-insensitive
    # for words, but keeps symbols untouched).
    lowered = text.lower()
    detected_error = any(marker in lowered for marker in ERROR_PATTERNS)
    # An error screenshot always counts as code: tracebacks contain code-like
    # lines (file paths, "line N", indented source), so the two flags can be set together.
    detected_code = detected_error or any(re.search(pattern, text, re.IGNORECASE | re.DOTALL) for pattern in CODE_PATTERNS)

    tags: list[str] = []
    # Keep a kind the user already set unless a heuristic below overrides it.
    # Starting point is the record's current kind (so a manual label survives when no
    # heuristic fires); the branches below override it only on a positive match.
    # ImageAnalysisThread only adopts the returned kind when the record is still
    # "unknown" or the heuristic found an error/code screenshot.
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

    # Source tags describe provenance, independent of content: pasted screenshots are
    # usually debugging captures, which helps later filtering.
    if record.source == "clipboard":
        tags.append("clipboard")
    if text.strip():
        tags.append("ocr")

    # Layout is a short human-readable phrase (not an enum) because it is shown in
    # the metadata panel and passed into prompts as-is.
    layout = _layout_hint(record, text)
    return AnalysisSuggestion(
        kind=kind,
        # Existing tags come first so a user's own tags keep their position; new
        # heuristic tags are appended and duplicates (case-insensitive) dropped.
        tags=_dedupe([*record.tags, *tags]),
        detected_code=detected_code,
        detected_error=detected_error,
        layout=layout,
    )


def _layout_hint(record: ImageRecord, text: str) -> str:
    # Guard against height 0 to avoid ZeroDivisionError.
    """Describe image shape (wide/tall/balanced) and how much text OCR found.

    More than 12 non-empty lines counts as "dense".
    """
    # Width/height can be 0 for records created before size detection; 1.0 yields
    # "balanced frame" instead of a ZeroDivisionError.
    aspect = record.width / record.height if record.height else 1.0
    lines = [line for line in text.splitlines() if line.strip()]
    # Thresholds: >1.6 (wider than 16:10) is landscape, <0.75 (taller than 3:4) is
    # portrait. A typical desktop screenshot (16:9 = 1.78) is therefore "wide".
    if aspect > 1.6:
        shape = "wide landscape"
    elif aspect < 0.75:
        shape = "tall portrait"
    else:
        shape = "balanced frame"
    # 12 non-blank OCR lines is the dense/sparse boundary: roughly a full terminal
    # or code window versus a dialog with a few labels.
    if len(lines) > 12:
        return f"{shape}; dense text layout with {len(lines)} OCR lines"
    if lines:
        return f"{shape}; sparse text layout with {len(lines)} OCR lines"
    return f"{shape}; no OCR text layout signal"


def _dedupe(values: list[str]) -> list[str]:
    """Lower-case, strip and de-duplicate tags while keeping first-seen order."""
    seen = set()
    out = []
    for value in values:
        # Normalised before the membership test so "Error" and "error " collapse into
        # one tag, and the stored form is always lower-case.
        clean = value.strip().lower()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out
