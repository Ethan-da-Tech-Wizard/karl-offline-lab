# ==============================================================================
# Program: ocr_engine.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Core utility providing ocr engine capabilities.
#          Implements data structures and algorithms supporting offline
#          introspection and model execution.
# ==============================================================================

"""
OCR via the locally installed Tesseract command-line program.

Nothing is bundled or downloaded: if the ``tesseract`` executable is not on
PATH the engine reports itself unavailable and analyze() returns an empty
result carrying an error box instead of raising.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
from io import StringIO

from app.vision.schemas import OcrResult


class TesseractOcrEngine:
    """Thin wrapper that runs the tesseract CLI and converts its TSV output to OcrResult."""
    def __init__(self, executable: str = "tesseract"):
        # Step 1: Validate input parameters, operational preconditions, and state
        """``executable`` may be a bare name (looked up on PATH) or a full path."""
        self.executable = executable

    def available(self) -> bool:
        # Step 1: Validate input parameters, operational preconditions, and state
        """True if the tesseract executable can be found."""
        return shutil.which(self.executable) is not None

    def version(self) -> str | None:
        # Step 1: Validate input parameters, operational preconditions, and state
        """Return the first line of ``tesseract --version`` or None if unavailable/failing."""
        if not self.available():
            return None
        try:
            # Execute guarded operations and establish recovery boundaries
            result = subprocess.run(
                [self.executable, "--version"],
                capture_output=True,
                text=True,
                check=False,
                timeout=8,
            )
        except Exception:
            return None
        # Some builds print the version banner on stderr, so fall back to it.
        first = (result.stdout or result.stderr or "").splitlines()
        return first[0].strip() if first else None

    def analyze(self, image_path: str, lang: str = "eng") -> OcrResult:
        # Step 1: Validate input parameters, operational preconditions, and state
        """Run OCR on ``image_path`` in language ``lang`` and return an OcrResult.

        Failures (missing binary, non-zero exit) are reported as an empty result with
        an {"error": ...} entry in boxes so callers can show the reason.
        """
        if not self.available():
            return OcrResult(
                engine="tesseract",
                language=lang,
                text="",
                confidence=0.0,
                boxes=[{"error": "tesseract executable not found"}],
            )

        # "tsv" output gives per-word boxes and confidences, not just plain text.
        # A 60s timeout bounds very large images; TimeoutExpired is not caught here.
        result = subprocess.run(
            [self.executable, image_path, "stdout", "-l", lang, "tsv"],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "unknown OCR error").strip()
            return OcrResult(
                engine="tesseract",
                language=lang,
                text="",
                confidence=0.0,
                boxes=[{"error": message}],
            )

        return self._parse_tsv(result.stdout, lang)

    def _parse_tsv(self, raw: str, lang: str) -> OcrResult:
        # Step 1: Validate input parameters, operational preconditions, and state
        """Convert tesseract TSV text into an OcrResult.

        Rows with empty text (layout rows) are skipped. Confidence is the mean of the
        non-negative word confidences, scaled from 0-100 to 0-1.
        """
        reader = csv.DictReader(StringIO(raw), delimiter="\t")
        words = []
        boxes = []
        confidences = []

        for row in reader:
            # Iterate over items, evaluate loop invariant, and update progress
            text = (row.get("text") or "").strip()
            if not text:
                continue
            try:
                # Execute guarded operations and establish recovery boundaries
                # Tesseract uses -1 for rows without a confidence; treat bad values the same.
                conf = float(row.get("conf", "-1"))
            except ValueError:
                conf = -1.0
            # Exclude -1 sentinel values from the average.
            if conf >= 0:
                confidences.append(conf)
            words.append(text)
            boxes.append({
                "text": text,
                "confidence": conf,
                "left": int(float(row.get("left", 0) or 0)),
                "top": int(float(row.get("top", 0) or 0)),
                "width": int(float(row.get("width", 0) or 0)),
                "height": int(float(row.get("height", 0) or 0)),
                "block": int(float(row.get("block_num", 0) or 0)),
                "line": int(float(row.get("line_num", 0) or 0)),
            })

        # Average over words that had a real confidence score.
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return OcrResult(
            engine="tesseract",
            language=lang,
            text=" ".join(words),
            confidence=round(confidence / 100.0, 4),
            boxes=boxes,
        )

