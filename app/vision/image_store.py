# Program: karl-offline-lab - image_store.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Stores imported images on disk (original, thumbnail, OCR copy) and keeps one JSON analysis record per image.
"""Disk-backed store for imported images and their analysis records."""
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


# === ImageStore: folder layout and CRUD for image records ===
class ImageStore:
    def __init__(self, base_dir: str = "data/images"):
        """Set up the folder layout under base_dir, creating any missing folders."""
        self.base_dir = Path(base_dir)
        self.inbox_dir = self.base_dir / "inbox"
        self.thumbnail_dir = self.base_dir / "thumbnails"
        self.processed_dir = self.base_dir / "processed"
        self.analysis_dir = self.base_dir / "analysis"
        for directory in (
            self.inbox_dir,
            self.thumbnail_dir,
            self.processed_dir,
            self.analysis_dir,
            self.base_dir / "datasets",
            self.base_dir / "training",
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def save_qimage(self, image: QImage, source: str = "clipboard") -> ImageRecord:
        """Save an in-memory image (for example from the clipboard) as a PNG and create its record."""
        if image.isNull():
            raise ValueError("Cannot save an empty clipboard image.")
        image_id = str(uuid.uuid4())
        original_path = self.inbox_dir / f"{image_id}.png"
        if not image.save(str(original_path), "PNG"):
            raise OSError(f"Could not save image: {original_path}")
        return self._create_record(image_id, original_path, source=source, mime="image/png")

    def import_file(self, path: str, source: str = "file") -> ImageRecord:
        """Copy an image file into the inbox after checking Qt can read it, then create its record."""
        src = Path(path)
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(path)
        image = QImage(str(src))
        if image.isNull():
            raise ValueError(f"Unsupported or unreadable image: {path}")
        # A random id keeps stored file names unique and free of user-controlled text.
        image_id = str(uuid.uuid4())
        suffix = src.suffix.lower() if src.suffix else ".png"
        original_path = self.inbox_dir / f"{image_id}{suffix}"
        shutil.copyfile(src, original_path)
        mime = mimetypes.guess_type(str(original_path))[0] or "image/png"
        return self._create_record(image_id, original_path, source=source, mime=mime)

    def get(self, image_id: str) -> ImageRecord:
        """Load the saved record for image_id; raises FileNotFoundError if none exists."""
        path = self._analysis_path(image_id)
        if not path.exists():
            raise FileNotFoundError(f"Image analysis not found: {image_id}")
        with path.open("r", encoding="utf-8") as f:
            return ImageRecord.from_dict(json.load(f))

    def list_recent(self, limit: int = 100) -> list[ImageRecord]:
        """Return up to limit records, newest first, skipping unreadable JSON files."""
        records = []
        for path in sorted(self.analysis_dir.glob("*.json"), reverse=True):
            try:
                with path.open("r", encoding="utf-8") as f:
                    records.append(ImageRecord.from_dict(json.load(f)))
            except Exception:
                continue
            if len(records) >= limit:
                break
        return sorted(records, key=lambda r: r.created_at, reverse=True)

    def update_analysis(self, image_id: str, **fields) -> ImageRecord:
        """Merge the given fields into an existing record, save it and return the updated record."""
        record = self.get(image_id)
        data = record.to_dict()
        data.update(fields)
        updated = ImageRecord.from_dict(data)
        self._write_record(updated)
        return updated

    def update_metadata(self, image_id: str, kind: str | None = None, tags: list[str] | None = None) -> ImageRecord:
        """Update only the kind and/or tags of a record; arguments left as None are unchanged."""
        fields = {}
        if kind is not None:
            fields["kind"] = kind
        if tags is not None:
            fields["tags"] = tags
        return self.update_analysis(image_id, **fields)

    def save_ocr_correction(self, image_id: str, corrected_text: str) -> ImageRecord:
        """Store a human-corrected OCR text both in the corrections map and in the OCR result."""
        record = self.get(image_id)
        corrections = dict(record.corrections)
        corrections["ocr_text"] = corrected_text
        ocr = record.ocr.to_dict()
        ocr["text"] = corrected_text
        return self.update_analysis(image_id, corrections=corrections, ocr=ocr)

    def save_caption_correction(self, image_id: str, corrected_caption: str) -> ImageRecord:
        """Store a human-corrected caption both in the corrections map and in the vision result."""
        record = self.get(image_id)
        corrections = dict(record.corrections)
        corrections["caption"] = corrected_caption
        vision = record.vision.to_dict()
        vision["caption"] = corrected_caption
        return self.update_analysis(image_id, corrections=corrections, vision=vision)

    def _create_record(self, image_id: str, original_path: Path, source: str, mime: str) -> ImageRecord:
        """Build the thumbnail and OCR copy for a new image, then write its initial record."""
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
        """Write the record as indented JSON to its analysis file."""
        path = self._analysis_path(record.id)
        with path.open("w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2)

    def _analysis_path(self, image_id: str) -> Path:
        """Return the JSON path for an id."""
        # Strip any directory part so an id cannot escape the analysis folder.
        safe_id = os.path.basename(image_id)
        return self.analysis_dir / f"{safe_id}.json"

    @staticmethod
    def _sha256(path: Path) -> str:
        """Return the SHA-256 hex digest of a file, read in 1 MB chunks."""
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
