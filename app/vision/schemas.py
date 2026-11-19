# ==============================================================================
# Program: schemas.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Core utility providing schemas capabilities.
#          Implements data structures and algorithms supporting offline
#          introspection and model execution.
# ==============================================================================

"""
Plain dataclasses describing the records the vision subsystem stores.

These are the on-disk / over-the-wire shapes for OCR output, vision-model
output and the combined per-image record kept by image_store. Each class has
from_dict (tolerant of missing keys, so older saved records still load) and
to_dict (JSON-serializable) helpers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class OcrResult:
    """Text extracted from an image by an OCR engine, plus confidence and word boxes."""
    engine: str = "none"
    language: str = "eng"
    text: str = ""
    confidence: float = 0.0
    boxes: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "OcrResult":
        # Accept None so callers can pass a record that has no OCR section yet.
        """Build an OcrResult from a possibly-None dict, filling defaults for missing keys."""
        data = data or {}
        return cls(
            engine=data.get("engine", "none"),
            language=data.get("language", "eng"),
            text=data.get("text", ""),
            # "or 0.0" guards against an explicit None stored in the JSON.
            confidence=float(data.get("confidence", 0.0) or 0.0),
            boxes=list(data.get("boxes", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain nested dict suitable for json.dumps."""
        return asdict(self)


@dataclass
class VisionResult:
    """Output of a vision/caption model: caption, layout summary, flags and detected objects."""
    engine: str = "none"
    model: str | None = None
    caption: str = ""
    layout: str = ""
    detected_code: bool = False
    detected_error: bool = False
    objects: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VisionResult":
        """Build a VisionResult from a possibly-None dict, filling defaults for missing keys."""
        data = data or {}
        return cls(
            engine=data.get("engine", "none"),
            model=data.get("model"),
            caption=data.get("caption", ""),
            layout=data.get("layout", ""),
            detected_code=bool(data.get("detected_code", False)),
            detected_error=bool(data.get("detected_error", False)),
            objects=list(data.get("objects", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain nested dict suitable for json.dumps."""
        return asdict(self)


@dataclass
class ImageRecord:
    """One stored image: file paths, hashes, OCR, vision output, user corrections and RAG index state.

    The ``corrections`` dict holds human overrides (None means "not corrected")
    so that the original model output is never lost. ``rag`` tracks whether the
    image text has been added to the retrieval index.
    """
    id: str
    created_at: str
    source: str
    kind: str
    original_path: str
    thumbnail_path: str
    processed_path: str | None
    sha256: str
    width: int
    height: int
    mime: str
    tags: list[str] = field(default_factory=list)
    ocr: OcrResult = field(default_factory=OcrResult)
    vision: VisionResult = field(default_factory=VisionResult)
    # default_factory with lambda gives each record its own fresh dict (a shared
    # mutable default would leak edits between records).
    corrections: dict[str, Any] = field(default_factory=lambda: {
        "ocr_text": None,
        "caption": None,
        "answer": None,
    })
    rag: dict[str, Any] = field(default_factory=lambda: {
        "indexed": False,
        "indexed_at": None,
        "text": "",
    })

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageRecord":
        """Rebuild an ImageRecord from stored JSON; only ``id`` is mandatory.

        Nested ocr/vision dicts are converted back into their dataclasses.
        """
        return cls(
            id=data["id"],
            created_at=data.get("created_at", ""),
            source=data.get("source", "file"),
            kind=data.get("kind", "unknown"),
            original_path=data.get("original_path", ""),
            thumbnail_path=data.get("thumbnail_path", ""),
            processed_path=data.get("processed_path"),
            sha256=data.get("sha256", ""),
            width=int(data.get("width", 0) or 0),
            height=int(data.get("height", 0) or 0),
            mime=data.get("mime", "image/png"),
            tags=list(data.get("tags", [])),
            ocr=OcrResult.from_dict(data.get("ocr")),
            vision=VisionResult.from_dict(data.get("vision")),
            corrections=dict(data.get("corrections", {})),
            rag=dict(data.get("rag", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-ready dict."""
        data = asdict(self)
        # asdict() already recurses, but we call to_dict explicitly so any custom
        # to_dict logic on the nested types stays authoritative.
        data["ocr"] = self.ocr.to_dict()
        data["vision"] = self.vision.to_dict()
        return data

