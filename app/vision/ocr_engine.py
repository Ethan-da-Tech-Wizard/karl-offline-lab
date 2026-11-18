# Program: karl-offline-lab - ocr_engine.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Wraps the Tesseract command-line tool to run OCR on an image and parse its TSV output into an OcrResult.
"""Tesseract OCR engine wrapper."""
from __future__ import annotations

import csv
import shutil
import subprocess
from io import StringIO

from app.vision.schemas import OcrResult


# === Tesseract OCR wrapper ===
class TesseractOcrEngine:
    def __init__(self, executable: str = "tesseract"):
        """Remember which tesseract executable name or path to run."""
        self.executable = executable

    def available(self) -> bool:
        """Return True if the executable can be found on PATH."""
        return shutil.which(self.executable) is not None

    def version(self) -> str | None:
        """Return the first line of the tesseract version output, or None if unavailable."""
        if not self.available():
            return None
        try:
            result = subprocess.run(
                [self.executable, "--version"],
                capture_output=True,
                text=True,
                check=False,
                timeout=8,
            )
        except Exception:
            return None
        first = (result.stdout or result.stderr or "").splitlines()
        return first[0].strip() if first else None

    def analyze(self, image_path: str, lang: str = "eng") -> OcrResult:
        """Run tesseract on an image and return the parsed result.

        Problems (missing binary, non-zero exit) are returned as an empty result with an error box instead of raising.
        """
        if not self.available():
            return OcrResult(
                engine="tesseract",
                language=lang,
                text="",
                confidence=0.0,
                boxes=[{"error": "tesseract executable not found"}],
            )

        result = subprocess.run(
            # Ask tesseract to print word-level TSV (with confidences and boxes) to stdout.
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
        """Convert tesseract TSV text into an OcrResult with joined text, boxes and average confidence."""
        reader = csv.DictReader(StringIO(raw), delimiter="\t")
        words = []
        boxes = []
        confidences = []

        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            try:
                conf = float(row.get("conf", "-1"))
            except ValueError:
                conf = -1.0
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

        # Tesseract reports 0-100; the average is scaled to 0-1 below.
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return OcrResult(
            engine="tesseract",
            language=lang,
            text=" ".join(words),
            confidence=round(confidence / 100.0, 4),
            boxes=boxes,
        )

