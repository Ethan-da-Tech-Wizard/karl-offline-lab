# ==============================================================================
# Program: image_preprocess.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Core utility providing image preprocess capabilities.
#          Implements data structures and algorithms supporting offline
#          introspection and model execution.
# ==============================================================================

from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage


def image_info(path: str) -> dict:
    # Step 1: Validate input parameters, operational preconditions, and state
    image = QImage(path)
    if image.isNull():
        raise ValueError(f"Could not read image: {path}")
    return {
        "width": image.width(),
        "height": image.height(),
        "bytes": os.path.getsize(path),
    }


def make_thumbnail(src: str, dst: str, max_size: int = 512) -> None:
    # Step 1: Validate input parameters, operational preconditions, and state
    image = QImage(src)
    if image.isNull():
        raise ValueError(f"Could not read image: {src}")
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
    # Step 1: Validate input parameters, operational preconditions, and state
    image = QImage(src)
    if image.isNull():
        raise ValueError(f"Could not read image: {src}")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    normalized = image.convertToFormat(QImage.Format.Format_RGB32)
    if not normalized.save(dst):
        raise OSError(f"Could not write OCR image: {dst}")
    return {
        "width": normalized.width(),
        "height": normalized.height(),
        "path": dst,
    }

