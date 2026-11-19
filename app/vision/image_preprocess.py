# ==============================================================================
# Program: image_preprocess.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Core utility providing image preprocess capabilities.
#          Implements data structures and algorithms supporting offline
#          introspection and model execution.
# ==============================================================================

"""
Image preprocessing helpers for the offline vision pipeline.

All helpers use Qt's QImage (already a dependency of the desktop app) so no
extra imaging library such as Pillow is needed. Each helper decodes the file
itself and fails loudly if Qt cannot read it.
"""

from __future__ import annotations

import os
from pathlib import Path

# Hard import (not lazy): these helpers only run inside the PyQt6 app.
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage


def image_info(path: str) -> dict:
    """Return the pixel width, height and on-disk byte size of the image at ``path``.

    Raises ValueError if Qt cannot decode the file (QImage.isNull() is Qt's
    only failure signal, it does not raise on its own).
    """
    image = QImage(path)
    if image.isNull():
        raise ValueError(f"Could not read image: {path}")
    # Size comes from the filesystem, not the decoded image, so it reflects the
    # compressed file the user actually stored.
    return {
        "width": image.width(),
        "height": image.height(),
        "bytes": os.path.getsize(path),
    }


def make_thumbnail(src: str, dst: str, max_size: int = 512) -> None:
    """Write a downscaled copy of ``src`` to ``dst`` that fits inside max_size x max_size.

    Aspect ratio is preserved (KeepAspectRatio) and SmoothTransformation is used
    for quality. The destination directory is created on demand. Raises
    ValueError for unreadable input and OSError if the thumbnail cannot be saved.
    """
    image = QImage(src)
    if image.isNull():
        raise ValueError(f"Could not read image: {src}")
    # Create the output folder first; QImage.save() will not create directories.
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    thumb = image.scaled(
        max_size,
        max_size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    # save() returns False rather than raising, so convert that to an exception.
    if not thumb.save(dst):
        raise OSError(f"Could not write thumbnail: {dst}")


def prepare_for_ocr(src: str, dst: str) -> dict:
    """Normalize ``src`` to 32-bit RGB and save it to ``dst`` for the OCR engine.

    Converting to Format_RGB32 flattens exotic pixel formats (palette, alpha,
    16-bit) into one predictable layout. Returns the new width, height and path.
    """
    image = QImage(src)
    if image.isNull():
        raise ValueError(f"Could not read image: {src}")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    # Force a single pixel format so OCR behaves the same for every input type.
    normalized = image.convertToFormat(QImage.Format.Format_RGB32)
    if not normalized.save(dst):
        raise OSError(f"Could not write OCR image: {dst}")
    return {
        "width": normalized.width(),
        "height": normalized.height(),
        "path": dst,
    }

