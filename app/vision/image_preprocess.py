# Program: karl-offline-lab - image_preprocess.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Small Qt-based image helpers: read dimensions, make thumbnails, and normalize images for OCR.
"""Image preprocessing helpers (info, thumbnails, OCR-ready copies) built on PyQt6."""
from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage


def image_info(path: str) -> dict:
    """Return width, height and file size in bytes for the image at path.

    Raises ValueError if Qt cannot read the file.
    """
    image = QImage(path)
    if image.isNull():
        raise ValueError(f"Could not read image: {path}")
    return {
        "width": image.width(),
        "height": image.height(),
        "bytes": os.path.getsize(path),
    }


def make_thumbnail(src: str, dst: str, max_size: int = 512) -> None:
    """Write a smooth, aspect-ratio-preserving thumbnail of src to dst (fits in max_size x max_size)."""
    image = QImage(src)
    if image.isNull():
        raise ValueError(f"Could not read image: {src}")
    # Make sure the output folder exists before saving.
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    thumb = image.scaled(
        max_size,
        max_size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    if not thumb.save(dst):
        raise OSError(f"Could not write thumbnail: {dst}")


def prepare_for_ocr(src: str, dst: str) -> dict:
    """Convert src to a plain 32-bit RGB image saved at dst and return its size and path."""
    image = QImage(src)
    if image.isNull():
        raise ValueError(f"Could not read image: {src}")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    # A uniform pixel format avoids alpha/palette surprises in the OCR engine.
    normalized = image.convertToFormat(QImage.Format.Format_RGB32)
    if not normalized.save(dst):
        raise OSError(f"Could not write OCR image: {dst}")
    return {
        "width": normalized.width(),
        "height": normalized.height(),
        "path": dst,
    }

