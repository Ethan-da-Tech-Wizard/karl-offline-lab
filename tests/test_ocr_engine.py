# Program: karl-offline-lab - test_ocr_engine.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Tests TesseractOcrEngine TSV parsing and its structured error when Tesseract is unavailable.

"""Tests TesseractOcrEngine TSV parsing and its structured error when Tesseract is unavailable."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vision.ocr_engine import TesseractOcrEngine


def test_parse_tsv_extracts_text_confidence_and_boxes():
    """Scenario: parse tsv extracts text confidence and boxes. Expected: the assertions in this test hold."""
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
    """Scenario: unavailable tesseract returns structured error. Expected: the assertions in this test hold."""
    image_path = tmp_path / "missing.png"
    engine = TesseractOcrEngine(executable="definitely-not-installed-tesseract")
    result = engine.analyze(str(image_path))

    assert result.engine == "tesseract"
    assert result.text == ""
    assert result.confidence == 0.0
    assert "not found" in result.boxes[0]["error"]

