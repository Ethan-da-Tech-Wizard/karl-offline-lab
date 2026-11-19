# ==============================================================================
# Program: test_ocr_engine.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Comprehensive test suite exercising ocr engine functionality.
#          Verifies state transitions, error boundaries, regression resistance,
#          and contract compliance under tests.
# ==============================================================================

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vision.ocr_engine import TesseractOcrEngine


def test_parse_tsv_extracts_text_confidence_and_boxes():
    """Tesseract TSV output is parsed into joined text, a mean confidence (0.85) and per-word bounding boxes."""
    # Step 1: Validate input parameters, operational preconditions, and state
    raw = "\n".join([
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
        "5\t1\t1\t1\t1\t1\t10\t20\t30\t12\t90.0\tError",
        "5\t1\t1\t1\t1\t2\t45\t20\t50\t12\t80.0\tTraceback",
        "",
    ])
    result = TesseractOcrEngine()._parse_tsv(raw, "eng")

    assert result.engine == "tesseract"
    assert result.text == "Error Traceback"
    assert result.confidence == 0.85
    assert result.boxes[0]["left"] == 10
    assert result.boxes[1]["text"] == "Traceback"


def test_unavailable_tesseract_returns_structured_error(tmp_path):
    """When the tesseract binary is missing, OCR returns an empty result with confidence 0.0 and an error entry ('not found') instead of raising."""
    # Step 1: Validate input parameters, operational preconditions, and state
    image_path = tmp_path / "missing.png"
    engine = TesseractOcrEngine(executable="definitely-not-installed-tesseract")
    result = engine.analyze(str(image_path))

    assert result.engine == "tesseract"
    assert result.text == ""
    assert result.confidence == 0.0
    assert "not found" in result.boxes[0]["error"]

