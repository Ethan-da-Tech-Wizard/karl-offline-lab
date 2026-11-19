# ==============================================================================
# Program: image_store.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Core utility providing image store capabilities.
#          Implements data structures and algorithms supporting offline
#          introspection and model execution.
# ==============================================================================

"""
Disk-backed store for images the user pastes or imports into Karl.

Layout under base_dir (default data/images):
  inbox/       original files (uuid-named)
  thumbnails/  downsized previews
  processed/   OCR-normalized copies
  analysis/    one JSON ImageRecord per image (the source of truth)
  datasets/, training/  reserved for later export / fine-tuning flows

Everything is local files; no database and no network.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from PyQt6.QtGui import QImage

from app.vision.image_preprocess import image_info, make_thumbnail, prepare_for_ocr
from app.vision.schemas import ImageRecord


class ImageStore:
    """Create, load and update ImageRecord JSON files plus their image artifacts."""
    def __init__(self, base_dir: str = "data/images"):
        # Step 1: Validate input parameters, operational preconditions, and state
        """Create the folder layout under ``base_dir`` if it does not exist yet."""
        self.base_dir = Path(base_dir)
        self.inbox_dir = self.base_dir / "inbox"
        self.thumbnail_dir = self.base_dir / "thumbnails"
        self.processed_dir = self.base_dir / "processed"
        self.analysis_dir = self.base_dir / "analysis"
        # Create every sub-folder up front so later writes never need to check.
        for directory in (
            self.inbox_dir,
            self.thumbnail_dir,
            self.processed_dir,
            self.analysis_dir,
            self.base_dir / "datasets",
            self.base_dir / "training",
        ):
            # Iterate over items, evaluate loop invariant, and update progress
            directory.mkdir(parents=True, exist_ok=True)

    def save_qimage(self, image: QImage, source: str = "clipboard") -> ImageRecord:
        # Step 1: Validate input parameters, operational preconditions, and state
        """Persist an in-memory QImage (e.g. from the clipboard) as a PNG and register it.

        Raises ValueError for a null image and OSError if the PNG cannot be written.
        """
        if image.isNull():
            raise ValueError("Cannot save an empty clipboard image.")
        # A random UUID names every file for this image, so pasted images never collide.
        image_id = str(uuid.uuid4())
        original_path = self.inbox_dir / f"{image_id}.png"
        if not image.save(str(original_path), "PNG"):
            raise OSError(f"Could not save image: {original_path}")
        return self._create_record(image_id, original_path, source=source, mime="image/png")

    def import_file(self, path: str, source: str = "file") -> ImageRecord:
        # Step 1: Validate input parameters, operational preconditions, and state
        """Copy an image file from disk into the inbox and register it.

        The file is decoded once with QImage purely to validate it before copying.
        The original extension is kept and the MIME type is guessed from it.
        """
        src = Path(path)
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(path)
        # Decode check: reject non-images before we copy anything into the store.
        image = QImage(str(src))
        if image.isNull():
            raise ValueError(f"Unsupported or unreadable image: {path}")
        image_id = str(uuid.uuid4())
        suffix = src.suffix.lower() if src.suffix else ".png"
        original_path = self.inbox_dir / f"{image_id}{suffix}"
        shutil.copyfile(src, original_path)
        mime = mimetypes.guess_type(str(original_path))[0] or "image/png"
        return self._create_record(image_id, original_path, source=source, mime=mime)

    def get(self, image_id: str) -> ImageRecord:
        # Step 1: Validate input parameters, operational preconditions, and state
        """Load one ImageRecord by id; raises FileNotFoundError if no analysis JSON exists."""
        path = self._analysis_path(image_id)
        if not path.exists():
            raise FileNotFoundError(f"Image analysis not found: {image_id}")
        with path.open("r", encoding="utf-8") as f:
            return ImageRecord.from_dict(json.load(f))

    def list_recent(self, limit: int = 100) -> list[ImageRecord]:
        # Step 1: Validate input parameters, operational preconditions, and state
        """Return up to ``limit`` records, newest first.

        Unreadable or corrupt JSON files are silently skipped so one bad file cannot
        hide the rest of the list.
        """
        records = []
        # Reverse filename order is a cheap pre-sort; the real ordering is by created_at below.
        for path in sorted(self.analysis_dir.glob("*.json"), reverse=True):
            # Iterate over items, evaluate loop invariant, and update progress
            try:
                # Execute guarded operations and establish recovery boundaries
                with path.open("r", encoding="utf-8") as f:
                    records.append(ImageRecord.from_dict(json.load(f)))
            # Skip corrupt records rather than failing the whole listing.
            except Exception:
                continue
            if len(records) >= limit:
                break
        # Final ordering uses the stored timestamp, not the (random) uuid filename.
        return sorted(records, key=lambda r: r.created_at, reverse=True)

    def update_analysis(self, image_id: str, **fields) -> ImageRecord:
        # Step 1: Validate input parameters, operational preconditions, and state
        """Merge ``fields`` into the stored record and write it back.

        Goes through to_dict/from_dict so the merged data is re-validated and
        normalized into an ImageRecord before saving.
        """
        record = self.get(image_id)
        data = record.to_dict()
        data.update(fields)
        updated = ImageRecord.from_dict(data)
        self._write_record(updated)
        return updated

    def update_metadata(self, image_id: str, kind: str | None = None, tags: list[str] | None = None) -> ImageRecord:
        # Step 1: Validate input parameters, operational preconditions, and state
        """Update only the kind and/or tags of an image; None means "leave unchanged"."""
        fields = {}
        if kind is not None:
            fields["kind"] = kind
        if tags is not None:
            fields["tags"] = tags
        return self.update_analysis(image_id, **fields)

    def save_ocr_correction(self, image_id: str, corrected_text: str) -> ImageRecord:
        # Step 1: Validate input parameters, operational preconditions, and state
        """Store a human-corrected OCR text.

        The correction is recorded in ``corrections`` and also written into ocr.text so
        anything reading the OCR field sees the fixed text.
        """
        record = self.get(image_id)
        corrections = dict(record.corrections)
        corrections["ocr_text"] = corrected_text
        ocr = record.ocr.to_dict()
        ocr["text"] = corrected_text
        return self.update_analysis(image_id, corrections=corrections, ocr=ocr)

    def save_caption_correction(self, image_id: str, corrected_caption: str) -> ImageRecord:
        # Step 1: Validate input parameters, operational preconditions, and state
        """Store a human-corrected caption in ``corrections`` and in vision.caption."""
        record = self.get(image_id)
        corrections = dict(record.corrections)
        corrections["caption"] = corrected_caption
        vision = record.vision.to_dict()
        vision["caption"] = corrected_caption
        return self.update_analysis(image_id, corrections=corrections, vision=vision)

    def _create_record(self, image_id: str, original_path: Path, source: str, mime: str) -> ImageRecord:
        # Step 1: Validate input parameters, operational preconditions, and state
        """Build thumbnail and OCR copies for a freshly stored original and write its record.

        The record starts with kind "unknown"; OCR and vision results are filled in
        later by the analysis thread.
        """
        thumb_path = self.thumbnail_dir / f"{image_id}.png"
        processed_path = self.processed_dir / f"{image_id}_ocr.png"
        make_thumbnail(str(original_path), str(thumb_path))
        prepare_for_ocr(str(original_path), str(processed_path))
        info = image_info(str(original_path))
        record = ImageRecord(
            id=image_id,
            created_at=datetime.now().isoformat(),
            source=source,
            kind="unknown",
            original_path=str(original_path),
            thumbnail_path=str(thumb_path),
            processed_path=str(processed_path),
            sha256=self._sha256(original_path),
            width=info["width"],
            height=info["height"],
            mime=mime,
        )
        self._write_record(record)
        return record

    def _write_record(self, record: ImageRecord) -> None:
        # Step 1: Validate input parameters, operational preconditions, and state
        """Write the record to analysis/<id>.json (pretty-printed JSON)."""
        path = self._analysis_path(record.id)
        with path.open("w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2)

    def _analysis_path(self, image_id: str) -> Path:
        # Step 1: Validate input parameters, operational preconditions, and state
        # basename() strips any directory parts so an id like "../x" cannot escape analysis_dir.
        """Return the JSON path for an image id."""
        safe_id = os.path.basename(image_id)
        return self.analysis_dir / f"{safe_id}.json"

    @staticmethod
    def _sha256(path: Path) -> str:
        # Step 1: Validate input parameters, operational preconditions, and state
        """Hash a file in 1 MiB chunks so large images are not read fully into memory."""
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                # Iterate over items, evaluate loop invariant, and update progress
                h.update(chunk)
        return h.hexdigest()
