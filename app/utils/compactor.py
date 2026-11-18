# Program: karl-offline-lab - compactor.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Shrinks trace records into a dense "compact" JSON form (short keys, stripped boilerplate and stop words, code as diffs) and expands them back.

"""Trace compaction for model-facing "Machine Speak" datasets.

The compact format is intentionally denser than the canonical TraceLogger JSONL
schema. It preserves trace identity and training-critical fields, shortens
standard keys, strips repeated prompt boilerplate, and represents code blocks as
unified diffs so repeated coder iterations are cheaper to store and replay.
"""

from __future__ import annotations

# Standard-library imports: diffs, hashing, JSON, regex and typing.
import difflib
import hashlib
import json
import re
from typing import Any


# ── Constants ────────────────────────────────────────────────────────────
# Long field names mapped to short ones used in the compact format.
KEY_MAP = {
    "id": "i",
    "session_id": "sid",
    "timestamp": "ts",
    "timing": "tm",
    "total_seconds": "tt",
    "prefill_seconds": "ps",
    "generation_seconds": "gs",
    "prefill_tps": "pt",
    "generation_tps": "gt",
    "total_tps": "tp",
    "prompt_tokens": "pk",
    "generation_tokens": "gk",
    "gpu_temp_c": "gc",
    "throttle_reasons": "tr",
    "cooling_duration_sec": "cd",
    "model": "m",
    "adapter": "ad",
    "workflow": "w",
    "template": "te",
    "hyperparams": "hp",
    "system_prompt": "sp",
    "compiled_prompt": "p",
    "thinking": "tk",
    "response": "r",
    "raw_output": "ro",
    "rag_chunks": "rg",
    "feedback": "fb",
    "corrected_response": "cr",
    "warning": "wa",
}
# Inverse mapping, used when expanding a compact trace.
REVERSE_KEY_MAP = {v: k for k, v in KEY_MAP.items()}

# Fields whose text values get boilerplate/stop-word stripping and code-block diffing.
TEXT_KEYS = {"compiled_prompt", "system_prompt", "thinking", "response", "raw_output", "corrected_response"}

# Common words dropped from prose to save space (this loss is intentional).
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "if", "in", "into", "is", "it", "its", "of", "on",
    "or", "that", "the", "their", "then", "there", "this", "to", "was",
    "were", "with", "you", "your",
}

# Patterns for repeated prompt scaffolding (special tokens, think tags, stock system prompts) that are removed.
BOILERPLATE_PATTERNS = [
    re.compile(r"(?is)<\|begin_of_text\|>|<\|end_of_text\|>|<\|eot_id\|>"),
    re.compile(r"(?is)<think>\s*|\s*</think>"),
    re.compile(r"(?is)you are karl,?.*?(?=\n\n|\nuser:|\nassistant:|$)"),
    re.compile(r"(?is)you are a helpful assistant\.?"),
    re.compile(r"(?is)system:\s*(?:you are .*?)(?=\n(?:user|assistant):|$)"),
    re.compile(r"(?is)###\s*system\s*.*?(?=###\s*(?:user|assistant)|$)"),
]

# Finds fenced code blocks: captures the language tag and the code body.
CODE_FENCE_RE = re.compile(r"```([A-Za-z0-9_+.\-#]*)\n(.*?)```", re.DOTALL)


# ── Public API ───────────────────────────────────────────────────────────
def compact_trace_for_ai(trace: dict) -> str:
    """Return a dense JSON string for a TraceLogger entry."""
    # Wrap the data with a version marker so decompact_trace can recognise the format.
    compacted = {
        "$": "karl.trace.compact.v1",
        "d": _compact_value(trace),
    }
    return json.dumps(compacted, ensure_ascii=False, separators=(",", ":"))


def decompact_trace(compacted: str) -> dict:
    """Expand a compact trace string back into the canonical key names.

    Text fields are normalized, not byte-for-byte originals: boilerplate and
    stop words stripped during compaction are intentionally unrecoverable.
    """
    # Accept both marked (versioned) and bare compact data.
    data = json.loads(compacted)
    if isinstance(data, dict) and data.get("$") == "karl.trace.compact.v1":
        expanded = _decompact_value(data.get("d", {}))
    else:
        expanded = _decompact_value(data)
    return expanded if isinstance(expanded, dict) else {}


# ── Recursive key/value compaction ───────────────────────────────────────
def _compact_value(value: Any, key: str | None = None) -> Any:
    """Recursively shorten keys and compact text fields in a nested dict/list value."""
    if isinstance(value, dict):
        # Step/iteration lists get their own dense encoding.
        transitions = _compact_transition_dict(value)
        if transitions is not None:
            return transitions
        out: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            # Unknown keys are kept unchanged.
            short_key = KEY_MAP.get(raw_key, raw_key)
            out[short_key] = _compact_value(raw_value, raw_key)
        return out
    if isinstance(value, list):
        return [_compact_value(item, key) for item in value]
    if isinstance(value, str):
        # Only known text fields are compacted; other strings pass through.
        if key in TEXT_KEYS:
            return _compact_text(value)
        return value
    return value


def _decompact_value(value: Any, key: str | None = None) -> Any:
    """Reverse _compact_value: restore long key names and rebuild text, code and step data."""
    if isinstance(value, dict):
        # Marker keys identify the special encodings produced by compaction.
        if "$parts" in value:
            return _decompact_parts(value["$parts"])
        if "$diff" in value:
            return _new_text_from_unified_diff(value.get("$diff", []))
        if "$tr" in value:
            return _decompact_transition_array(value)
        out: dict[str, Any] = {}
        for short_key, raw_value in value.items():
            long_key = REVERSE_KEY_MAP.get(short_key, short_key)
            out[long_key] = _decompact_value(raw_value, long_key)
        return out
    if isinstance(value, list):
        return [_decompact_value(item, key) for item in value]
    return value


# ── Text and code handling ───────────────────────────────────────────────
def _compact_text(text: str) -> Any:
    """Compact prose by removing boilerplate and stop words; code blocks become diffs against the previous block."""
    cleaned = _strip_boilerplate(text)
    matches = list(CODE_FENCE_RE.finditer(cleaned))
    # No code blocks: just strip stop words from the whole text.
    if not matches:
        return _remove_stop_words(cleaned)

    parts: list[Any] = []
    cursor = 0
    # Each code block is diffed against the one before it, so repeated iterations store only changes.
    previous_code = ""
    for index, match in enumerate(matches):
        prefix = cleaned[cursor:match.start()]
        if prefix.strip():
            parts.append(_remove_stop_words(prefix))
        lang = (match.group(1) or "text").strip()
        code = match.group(2)
        diff = _unified_diff(previous_code, code, fromfile=f"code_{index - 1}", tofile=f"code_{index}")
        parts.append({
            "$diff": diff,
            "l": lang,
            # Short content hash identifies the code block.
            "h": hashlib.sha256(code.encode("utf-8")).hexdigest()[:12],
        })
        previous_code = code
        cursor = match.end()

    # Keep any text after the last code block.
    suffix = cleaned[cursor:]
    if suffix.strip():
        parts.append(_remove_stop_words(suffix))
    return {"$parts": parts}


def _strip_boilerplate(text: str) -> str:
    """Remove repeated prompt boilerplate and collapse extra whitespace."""
    out = text
    for pattern in BOILERPLATE_PATTERNS:
        out = pattern.sub(" ", out)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _remove_stop_words(text: str) -> str:
    # Replacement callback: blank out stop words, keep every other word.
    """Drop stop words from the text and tidy leftover spacing."""
    def repl(match: re.Match[str]) -> str:
        word = match.group(0)
        return "" if word.lower() in STOP_WORDS else word

    out = re.sub(r"\b[A-Za-z]+\b", repl, text)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r" *\n *", "\n", out)
    return out.strip()


def _unified_diff(old: str, new: str, fromfile: str = "prev", tofile: str = "next") -> list[str]:
    """Return a unified diff (as a list of lines) turning old text into new text."""
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    return list(difflib.unified_diff(old_lines, new_lines, fromfile=fromfile, tofile=tofile, lineterm=""))


def _new_text_from_unified_diff(diff_lines: list[str]) -> str:
    """Rebuild the new text from a unified diff by keeping added and unchanged lines."""
    new_lines: list[str] = []
    for line in diff_lines:
        # Skip empty lines and diff headers.
        if not line or line.startswith(("---", "+++", "@@")):
            continue
        if line.startswith("+"):
            new_lines.append(line[1:])
        elif line.startswith(" "):
            new_lines.append(line[1:])
    return "\n".join(new_lines)


def _decompact_parts(parts: list[Any]) -> str:
    """Rejoin prose pieces and diff-encoded code blocks into one text with code fences restored."""
    restored: list[str] = []
    for part in parts:
        if isinstance(part, str):
            restored.append(part)
        elif isinstance(part, dict) and "$diff" in part:
            lang = part.get("l", "")
            code = _new_text_from_unified_diff(part.get("$diff", []))
            restored.append(f"```{lang}\n{code}\n```")
    return "\n".join(chunk for chunk in restored if chunk)


# ── Step/transition handling ─────────────────────────────────────────────
def _compact_transition_dict(value: dict) -> dict | None:
    """Encode a list of steps as [index, state, output] rows, or return None when no step list is present."""
    steps = value.get("transitions") or value.get("iterations") or value.get("steps")
    if not isinstance(steps, list) or not steps:
        return None

    dense: list[list[Any]] = []
    previous = ""
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        # Accept several common field names for state and output.
        state = step.get("state") or step.get("status") or step.get("phase") or ""
        output = step.get("output") or step.get("response") or step.get("content") or ""
        # Outputs with code use text compaction; plain outputs are stored as diffs from the previous step.
        if isinstance(output, str) and "```" in output:
            compacted = _compact_text(output)
        elif isinstance(output, str):
            compacted = _unified_diff(previous, output, fromfile=f"step_{index - 1}", tofile=f"step_{index}")
            previous = output
        else:
            compacted = output
        dense.append([index, state, compacted])
    return {"$tr": dense}


def _decompact_transition_array(value: dict) -> dict:
    """Expand [index, state, output] rows back into a transitions list of dicts."""
    steps = []
    for row in value.get("$tr", []):
        if not isinstance(row, list) or len(row) < 3:
            continue
        output = row[2]
        if isinstance(output, list):
            output = _new_text_from_unified_diff(output)
        else:
            output = _decompact_value(output)
        steps.append({"index": row[0], "state": row[1], "output": output})
    return {"transitions": steps}
