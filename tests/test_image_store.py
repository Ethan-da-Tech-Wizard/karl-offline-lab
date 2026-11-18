# Program: karl-offline-lab - test_image_store.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Tests for the vision image store: saving, importing, analysis updates and metadata corrections.
"""Tests for app.vision.image_store.ImageStore."""

import os
import sys

# Make the repository root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtGui import QColor, QImage

from app.vision.schemas import OcrResult
from app.vision.image_store import ImageStore


def _sample_image(width=80, height=48):
    """Create a small solid-colour image for tests."""
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor("#00C2FF"))
    return image


def test_save_qimage_creates_record_and_files(tmp_path):
    """Scenario: save an in-memory image. Expected: a record with id, size, type and hash, three files on disk, and it can be loaded back."""
    store = ImageStore(base_dir=str(tmp_path / "images"))
    record = store.save_qimage(_sample_image(), source="test")

    assert record.id
    assert record.source == "test"
    assert record.width == 80
    assert record.height == 48
    assert record.mime == "image/png"
    assert len(record.sha256) == 64
    assert os.path.exists(record.original_path)
    assert os.path.exists(record.thumbnail_path)
    assert os.path.exists(record.processed_path)

    loaded = store.get(record.id)
    assert loaded.id == record.id
    assert loaded.original_path == record.original_path


def test_import_file_copies_image(tmp_path):
    """Scenario: import a PNG from disk. Expected: the record has the right size and points at a copy, not the source."""
    source = tmp_path / "source.png"
    assert _sample_image(32, 24).save(str(source), "PNG")

    store = ImageStore(base_dir=str(tmp_path / "images"))
    record = store.import_file(str(source))

    assert record.width == 32
    assert record.height == 24
    assert record.original_path != str(source)
    assert os.path.exists(record.original_path)


def test_update_analysis_round_trip(tmp_path):
    """Scenario: update kind and tags. Expected: the returned record and the stored one both show them."""
    store = ImageStore(base_dir=str(tmp_path / "images"))
    record = store.save_qimage(_sample_image(), source="test")

    updated = store.update_analysis(record.id, kind="code_screenshot", tags=["code", "error"])

    assert updated.kind == "code_screenshot"
    assert updated.tags == ["code", "error"]
    assert store.get(record.id).kind == "code_screenshot"


def test_update_analysis_accepts_ocr_dict(tmp_path):
    """Scenario: update with OCR results as a dict. Expected: text, confidence and engine are stored."""
    store = ImageStore(base_dir=str(tmp_path / "images"))
    record = store.save_qimage(_sample_image(), source="test")
    ocr = OcrResult(engine="tesseract", text="NameError at line 3", confidence=0.9)

    updated = store.update_analysis(record.id, ocr=ocr.to_dict())

    assert updated.ocr.text == "NameError at line 3"
    assert updated.ocr.confidence == 0.9
    assert store.get(record.id).ocr.engine == "tesseract"


def test_image_metadata_and_corrections_round_trip(tmp_path):
    """Scenario: update metadata and save OCR and caption corrections. Expected: the reloaded record shows all of them."""
    store = ImageStore(base_dir=str(tmp_path / "images"))
    record = store.save_qimage(_sample_image(), source="test")

    store.update_metadata(record.id, kind="code_screenshot", tags=["code", "traceback"])
    store.save_ocr_correction(record.id, "Traceback line 10")
    store.save_caption_correction(record.id, "A code editor showing a traceback.")
    loaded = store.get(record.id)

    assert loaded.kind == "code_screenshot"
    assert loaded.tags == ["code", "traceback"]
    assert loaded.ocr.text == "Traceback line 10"
    assert loaded.corrections["ocr_text"] == "Traceback line 10"
    assert loaded.vision.caption == "A code editor showing a traceback."
    assert loaded.corrections["caption"] == "A code editor showing a traceback."
