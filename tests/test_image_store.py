# ==============================================================================
# Program: test_image_store.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Comprehensive test suite exercising image store functionality.
#          Verifies state transitions, error boundaries, regression resistance,
#          and contract compliance under tests.
# ==============================================================================

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtGui import QColor, QImage

from app.vision.schemas import OcrResult
from app.vision.image_store import ImageStore


def _sample_image(width=80, height=48):
    """Create a small solid-colour QImage for tests."""
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor("#00C2FF"))
    return image


def test_save_qimage_creates_record_and_files(tmp_path):
    """Saving a QImage returns a record with id, source, size, PNG mime and SHA-256, writes original, thumbnail and processed files and can be loaded back."""
    # Every test uses a store rooted in tmp_path so image files and the index are discarded with the test.
    store = ImageStore(base_dir=str(tmp_path / "images"))
    record = store.save_qimage(_sample_image(), source="test")

    assert record.id
    assert record.source == "test"
    assert record.width == 80
    assert record.height == 48
    assert record.mime == "image/png"
    # SHA-256 hex digest is 64 characters; the hash lets the store recognise identical content.
    assert len(record.sha256) == 64
    assert os.path.exists(record.original_path)
    # Three derived files exist per image: original, a thumbnail for the library grid and a processed copy used as OCR/vision input.
    assert os.path.exists(record.thumbnail_path)
    assert os.path.exists(record.processed_path)

    loaded = store.get(record.id)
    assert loaded.id == record.id
    assert loaded.original_path == record.original_path


def test_import_file_copies_image(tmp_path):
    """Importing an image file copies it into the store (the stored original path differs from the source) and records its size."""
    source = tmp_path / "source.png"
    assert _sample_image(32, 24).save(str(source), "PNG")

    store = ImageStore(base_dir=str(tmp_path / "images"))
    record = store.import_file(str(source))

    assert record.width == 32
    assert record.height == 24
    # import_file must copy into the store; if it only referenced the source path, deleting or moving the user's file would silently break the record.
    assert record.original_path != str(source)
    assert os.path.exists(record.original_path)


def test_update_analysis_round_trip(tmp_path):
    """update_analysis stores kind and tags that are returned by later get() calls."""
    store = ImageStore(base_dir=str(tmp_path / "images"))
    record = store.save_qimage(_sample_image(), source="test")

    updated = store.update_analysis(record.id, kind="code_screenshot", tags=["code", "error"])

    assert updated.kind == "code_screenshot"
    assert updated.tags == ["code", "error"]
    assert store.get(record.id).kind == "code_screenshot"


def test_update_analysis_accepts_ocr_dict(tmp_path):
    """An OCR result given as a plain dict is converted and persisted (text, confidence, engine)."""
    store = ImageStore(base_dir=str(tmp_path / "images"))
    record = store.save_qimage(_sample_image(), source="test")
    # OCR results arrive as dicts over RPC/JSON, so update_analysis has to rebuild the dataclass from a dict rather than require the object.
    ocr = OcrResult(engine="tesseract", text="NameError at line 3", confidence=0.9)

    updated = store.update_analysis(record.id, ocr=ocr.to_dict())

    assert updated.ocr.text == "NameError at line 3"
    assert updated.ocr.confidence == 0.9
    assert store.get(record.id).ocr.engine == "tesseract"


def test_image_metadata_and_corrections_round_trip(tmp_path):
    """Kind, tags, OCR, vision caption and user corrections all survive saving and loading."""
    store = ImageStore(base_dir=str(tmp_path / "images"))
    record = store.save_qimage(_sample_image(), source="test")

    store.update_metadata(record.id, kind="code_screenshot", tags=["code", "traceback"])
    store.save_ocr_correction(record.id, "Traceback line 10")
    store.save_caption_correction(record.id, "A code editor showing a traceback.")
    loaded = store.get(record.id)

    assert loaded.kind == "code_screenshot"
    assert loaded.tags == ["code", "traceback"]
    assert loaded.ocr.text == "Traceback line 10"
    # Corrections are stored separately from the machine output so the original OCR/caption is preserved for later comparison with the human fix.
    assert loaded.corrections["ocr_text"] == "Traceback line 10"
    assert loaded.vision.caption == "A code editor showing a traceback."
    assert loaded.corrections["caption"] == "A code editor showing a traceback."
