# ==============================================================================
# Program: vision_model_loader.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Core utility providing vision model loader capabilities.
#          Implements data structures and algorithms supporting offline
#          introspection and model execution.
# ==============================================================================

"""
Loads and runs a local multimodal (image + text) GGUF model through llama-cpp-python.

A JSON registry (data/vision_model_registry.json) lists the known vision
models; each needs a model GGUF and a "projector" (CLIP/mmproj) GGUF on disk.
The last-used model id is remembered in data/active_vision_model.json.
Everything is local; if llama_cpp or the files are missing, describe_image()
returns a VisionResult explaining why rather than raising.
"""

from __future__ import annotations

import base64
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.vision.schemas import VisionResult


# Relative paths: resolved against the process working directory (the app root).
# Layout: the registry JSON lists models; the weights live in two sibling folders so
# that a model GGUF and its projector (mmproj, the CLIP-style image encoder that
# turns pixels into embeddings the language model can read) are downloaded and
# replaced independently. All four are relative to the working directory.
REGISTRY_PATH = Path("data/vision_model_registry.json")
ACTIVE_PATH = Path("data/active_vision_model.json")
MODEL_DIR = Path("data/vision_models")
PROJECTOR_DIR = Path("data/vision_projectors")


# Immutable registry row; model_path()/projector_path() derive file locations.
@dataclass(frozen=True)
# Frozen so entries read from the registry cannot drift from the file. The
# values min_ram_gb / min_vram_gb are advisory hints shown in the UI; load() does
# not enforce them, a too-large model fails at construction instead.
class VisionModelEntry:
    """One vision model from the registry: file names, RAM/VRAM hints and chat-handler family."""
    id: str
    name: str
    # Selects the llama-cpp chat handler class (see _handler_class); the handler
    # knows how to splice image embeddings into that model family's prompt template.
    # A wrong family loads fine but produces garbage, so it must match the model.
    family: str
    model_filename: str
    projector_filename: str
    min_ram_gb: float = 0.0
    min_vram_gb: float = 0.0
    # Context window in tokens. An image consumes many of these (the projector emits
    # dozens to hundreds of image tokens), so very small values leave no room for the
    # answer; 4096 is the default when the registry omits it.
    n_ctx: int = 4096
    strengths: str = ""
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisionModelEntry":
        # "or 0.0" / "or 4096" repair explicit nulls in hand-edited JSON.
        """Build an entry from a registry dict; only ``id`` is required, others get defaults."""
        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            # "llava15" is the most widely supported handler, hence the default for
            # registry rows that predate the family field.
            family=data.get("family", "llava15"),
            model_filename=data.get("model_filename", ""),
            projector_filename=data.get("projector_filename", ""),
            min_ram_gb=float(data.get("min_ram_gb", 0.0) or 0.0),
            min_vram_gb=float(data.get("min_vram_gb", 0.0) or 0.0),
            n_ctx=int(data.get("n_ctx", 4096) or 4096),
            strengths=data.get("strengths", ""),
            notes=data.get("notes", ""),
        )

    def model_path(self) -> Path:
        """Path of the main model GGUF under data/vision_models."""
        return MODEL_DIR / self.model_filename

    def projector_path(self) -> Path:
        """Path of the multimodal projector GGUF under data/vision_projectors."""
        return PROJECTOR_DIR / self.projector_filename

    def to_dict(self) -> dict[str, Any]:
        """Return the entry plus resolved paths and whether each file is installed, for the UI."""
        return {
            "id": self.id,
            "name": self.name,
            "family": self.family,
            "model_filename": self.model_filename,
            "projector_filename": self.projector_filename,
            "min_ram_gb": self.min_ram_gb,
            "min_vram_gb": self.min_vram_gb,
            "n_ctx": self.n_ctx,
            "strengths": self.strengths,
            "notes": self.notes,
            "model_path": str(self.model_path()),
            "projector_path": str(self.projector_path()),
            "model_installed": self.model_path().exists(),
            "projector_installed": self.projector_path().exists(),
        }


def read_vision_registry(path: Path = REGISTRY_PATH) -> list[VisionModelEntry]:
    # Imported inside the function to avoid import-time coupling/cycles between vision and engine.
    """Read the registry JSON into entries; a missing or non-list file yields []."""
    from app.engine.config_store import read_json

    raw = read_json(str(path), default=[])
    if not isinstance(raw, list):
        return []
    # A row without "id" raises KeyError here (from_dict requires it), which propagates
    # to the caller: the registry is treated as a trusted, developer-edited file.
    return [VisionModelEntry.from_dict(item) for item in raw]


def active_vision_model_id(path: Path = ACTIVE_PATH) -> str | None:
    """Return the remembered active model id, or None. Accepts the legacy key "vision_model_id"."""
    from app.engine.config_store import read_json

    data = read_json(str(path), default=None)
    if not isinstance(data, dict):
        return None
    # Two spellings are accepted for backwards compatibility with an earlier file
    # format; only "id" is ever written (see set_active_vision_model).
    return data.get("id") or data.get("vision_model_id")


def set_active_vision_model(model_id: str, path: Path = ACTIVE_PATH) -> None:
    """Persist the active model id using an atomic write so a crash cannot leave a half-written file."""
    from app.engine.config_store import write_json_atomic

    # Atomic (temp file + rename) because this runs right after a successful load: a
    # crash mid-write must not leave a truncated file that makes the next launch fall
    # back to the first registry entry.
    write_json_atomic(str(path), {"id": model_id}, indent=2)


def installed_vision_models() -> list[dict[str, Any]]:
    """List every registry entry as a UI dict (including installed flags)."""
    return [entry.to_dict() for entry in read_vision_registry()]


# Class-level state instead of an instance because the whole process shares one
# vision model: loading a second multi-GB copy would exhaust memory. There is no
# __init__; callers (and tests) use the class methods directly or inject the
# class itself, as VisionAnalyzer does.
class VisionModelLoader:
    # Serializes load()/reset(); describe_image() reads _llm without the lock.
    """Process-wide singleton-style holder for the loaded vision model.

    All state is class-level and guarded by ``_lock`` so the GUI thread and worker
    threads see one consistent model.
    """
    # Plain Lock (not RLock): load() must not call anything that takes it again.
    # Holding it for the whole load means a concurrent reset() waits for a load to
    # finish rather than yanking the model out mid-construction.
    _lock = threading.Lock()
    _entry: VisionModelEntry | None = None
    _llm = None
    # Human-readable reason of the most recent failure, surfaced verbatim in the
    # Vision tab and the Workbench warning. Cleared by load() and reset().
    _last_error = ""

    @classmethod
    def status(cls) -> dict[str, Any]:
        """Report loaded state, active model, backend availability and the last error."""
        # Before any load, status still reports the model that WOULD load (active id,
        # else first registry row), so the UI can show a meaningful name while "not loaded".
        entry = cls._entry or cls._default_entry()
        # Recomputed on each call by importing probes; Python caches imports, so after
        # the first call this is cheap, but the first call can take noticeable time.
        backend = cls._backend_status()
        return {
            "loaded": cls._llm is not None,
            "active_id": entry.id if entry else None,
            "active_name": entry.name if entry else None,
            "backend_available": backend["available"],
            "backend_error": backend["error"],
            "last_error": cls._last_error,
            "registry": installed_vision_models(),
        }

    @classmethod
    def reset(cls) -> None:
        """Forget the loaded model and error so the next call reloads from scratch."""
        with cls._lock:
            cls._llm = None
            cls._entry = None
            cls._last_error = ""

    @classmethod
    def load(cls, model_id: str | None = None):
        """Load the requested (or active/default) model and return the Llama object, or None.

        Every failure path stores a readable reason in _last_error instead of
        raising: missing registry entry, missing model or projector file, llama_cpp
        not importable, unsupported family, or an exception during construction.
        """
        with cls._lock:
            # Entry resolution happens under the lock so two racing load() calls cannot
            # interleave their _entry / _llm updates.
            entry = cls._resolve_entry(model_id)
            cls._entry = entry
            cls._llm = None
            cls._last_error = ""

            if entry is None:
                cls._last_error = "no vision model registry entries found"
                return None
            if not entry.model_path().exists():
                cls._last_error = f"vision model file is missing: {entry.model_path()}"
                return None
            if not entry.projector_path().exists():
                cls._last_error = f"vision projector file is missing: {entry.projector_path()}"
                return None

            try:
                # Imported lazily so the app starts even when llama-cpp-python is not installed.
                # Import failure (package missing, or compiled without the needed backend) is a
                # normal state on many machines, so it is reported as an error string, not raised.
                from llama_cpp import Llama
            except Exception as exc:
                cls._last_error = f"llama_cpp import failed: {exc}"
                return None

            # Checked before constructing the model so an unsupported family fails fast,
            # without reading gigabytes of weights first.
            handler_cls, handler_error = cls._handler_class(entry.family)
            if handler_cls is None:
                cls._last_error = handler_error
                return None

            try:
                # verbose=False on both objects silences llama.cpp's very chatty stderr logging.
                # The handler loads the projector; the Llama object loads the language model.
                handler = handler_cls(clip_model_path=str(entry.projector_path()), verbose=False)
                cls._llm = Llama(
                    model_path=str(entry.model_path()),
                    chat_handler=handler,
                    # Taken from the registry entry so each model runs with the window it was
                    # tested with.
                    n_ctx=entry.n_ctx,
                    # -1 offloads all layers to the GPU when one is available.
                    n_gpu_layers=-1,
                    verbose=False,
                )
                # Remember the successfully loaded model for next launch.
                # Persisted only after success, so a model that fails to load never becomes the
                # remembered default.
                set_active_vision_model(entry.id)
                return cls._llm
            except Exception as exc:
                cls._last_error = f"vision model load failed: {exc}"
                cls._llm = None
                return None

    @classmethod
    def describe_image(
        cls,
        image_path: str,
        prompt: str | None = None,
        model_id: str | None = None,
        max_tokens: int = 512,
    ) -> VisionResult:
        """Ask the vision model to describe ``image_path``; always returns a VisionResult.

        The image is sent inline as a base64 data URL in an OpenAI-style chat
        message. Temperature is low (0.1) to reduce invented details. Errors become
        an "unavailable"/"error" result whose caption holds the message.
        """
        # No lock here (see the note on _lock): the worst case is two threads both seeing
        # None and both calling load(), which serialises on the lock; the second load
        # replaces the first. Inference itself is not serialised by this class.
        llm = cls._llm or cls.load(model_id)
        entry = cls._entry or cls._resolve_entry(model_id)
        # Reported even on failure so the result says which model was attempted.
        model_name = entry.name if entry else None
        if llm is None:
            message = cls._last_error or "local vision model is not loaded"
            return VisionResult(
                engine="llama-cpp-unavailable",
                model=model_name,
                caption=f"Local image model unavailable: {message}",
            )

        try:
            # The whole file is read and base64-encoded in memory (about 1.33x its size),
            # which is fine for screenshots but would be heavy for very large photos.
            data_url = cls._image_data_url(image_path)
            # Default prompt asks for accuracy and explicitly forbids inventing unseen details.
            user_prompt = prompt or (
                "Describe this image accurately. If it is a code screenshot, identify visible errors, "
                "important filenames, relevant code, and likely fixes. Do not invent details you cannot see."
            )
            response = llm.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                # 512 default caps the reply length: enough for a few paragraphs while bounding
                # the generation time on CPU-only machines.
                max_tokens=max_tokens,
                # Low temperature: a description should be repeatable and stay close to what is
                # visible rather than creative.
                temperature=0.1,
            )
            # OpenAI-style response shape, as returned by llama-cpp-python's
            # create_chat_completion.
            caption = response["choices"][0]["message"]["content"].strip()
            return VisionResult(
                engine="llama-cpp-vision",
                model=model_name,
                caption=caption,
                # Flags are derived from the model's OWN words, not from pixels, so they
                # describe what the caption mentions; VisionAnalyzer ORs them with the OCR text
                # heuristics.
                detected_code=_looks_like_code(caption),
                detected_error=_looks_like_error(caption),
            )
        except Exception as exc:
            # Inference errors (out of memory, bad image bytes, handler mismatch) are
            # swallowed into the result so one bad image cannot crash the analysis thread; the
            # model stays loaded for the next image.
            cls._last_error = f"vision inference failed: {exc}"
            return VisionResult(
                engine="llama-cpp-error",
                model=model_name,
                caption=cls._last_error,
            )

    @classmethod
    # Used only for display (status); loading goes through _resolve_entry, which
    # adds the explicit model_id argument. Both fall back to the first registry row,
    # so an unknown active id silently selects the first model.
    def _default_entry(cls) -> VisionModelEntry | None:
        """Pick the registry entry matching the active id, else the first entry, else None."""
        entries = read_vision_registry()
        if not entries:
            return None
        active_id = active_vision_model_id()
        if active_id:
            for entry in entries:
                if entry.id == active_id:
                    return entry
        return entries[0]

    @classmethod
    def _resolve_entry(cls, model_id: str | None = None) -> VisionModelEntry | None:
        """Pick the entry for ``model_id`` (or the active id); fall back to the first entry."""
        entries = read_vision_registry()
        if not entries:
            return None
        wanted = model_id or active_vision_model_id()
        if wanted:
            for entry in entries:
                if entry.id == wanted:
                    return entry
        return entries[0]

    @staticmethod
    def _handler_class(family: str):
        """Map a model family name to its llama-cpp chat handler class.

        Returns (handler_class, "") on success or (None, error_message) on failure.
        """
        try:
            # Handler classes only exist in newer llama-cpp-python builds, hence the guarded import.
            from llama_cpp.llama_chat_format import (
                Llava15ChatHandler,
                Llava16ChatHandler,
                MiniCPMv26ChatHandler,
                MoondreamChatHandler,
                NanoLlavaChatHandler,
                Qwen25VLChatHandler,
            )
        except Exception as exc:
            return None, f"llama-cpp multimodal handlers unavailable: {exc}"

        # Keys are the registry "family" strings; adding a model family means adding its
        # handler here and to the import above, and listing it in the registry.
        handlers = {
            "llava15": Llava15ChatHandler,
            "llava16": Llava16ChatHandler,
            "nanollava": NanoLlavaChatHandler,
            "minicpmv26": MiniCPMv26ChatHandler,
            "moondream": MoondreamChatHandler,
            "qwen25vl": Qwen25VLChatHandler,
        }
        handler = handlers.get(family)
        if handler is None:
            return None, f"unsupported vision model family: {family}"
        return handler, ""

    @staticmethod
    def _backend_status() -> dict[str, Any]:
        """Check, step by step, that llama_cpp, its multimodal handlers and the native mtmd library import."""
        try:
            import llama_cpp  # noqa: F401
        except Exception as exc:
            return {"available": False, "error": f"llama_cpp unavailable: {exc}"}
        try:
            from llama_cpp.llama_chat_format import Llava15ChatHandler  # noqa: F401
        except Exception as exc:
            return {"available": False, "error": f"multimodal handlers unavailable: {exc}"}
        try:
            import llama_cpp.mtmd_cpp  # noqa: F401
        except Exception as exc:
            return {"available": False, "error": f"native multimodal mtmd library unavailable: {exc}"}
        return {"available": True, "error": ""}

    @staticmethod
    def _image_data_url(image_path: str) -> str:
        """Read an image file and return it as a ``data:image/<type>;base64,...`` URL."""
        path = Path(image_path)
        # A file without an extension is assumed to be PNG (the format the Workbench
        # paste path writes).
        suffix = path.suffix.lower().lstrip(".") or "png"
        # "jpg" is not a valid image MIME subtype; the correct one is "jpeg".
        mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
        with path.open("rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/{mime};base64,{encoded}"


def _looks_like_code(text: str) -> bool:
    """Cheap keyword check on model output for code-related words."""
    lowered = text.lower()
    # Words a model uses when it recognises source code or a terminal in the image.
    # Very loose by design (a caption saying "classroom" matches "class"), because
    # the flag is only a hint merged with OCR evidence.
    markers = ("code", "function", "class", "import", "traceback", "compiler", "terminal")
    return any(marker in lowered for marker in markers)


def _looks_like_error(text: str) -> bool:
    """Cheap keyword check on model output for error-related words."""
    lowered = text.lower()
    # Same trade-off as above: favours recall; a false flag only adds an "error" tag.
    markers = ("error", "exception", "traceback", "failed", "warning", "stack trace")
    return any(marker in lowered for marker in markers)
