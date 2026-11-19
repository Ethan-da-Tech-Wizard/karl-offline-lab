# ==============================================================================
# Program: kv_cache.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Key-Value cache governance and context allocation manager.
#          Monitors KV memory usage, detects cache fragmentation, and optimizes
#          prompt prefix caching across turns.
# ==============================================================================

"""
KV-cache statistics and logging helpers.

Shared by LLMThread and AgenticThread to avoid duplication.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("karl.kv_cache")

# Relative to the process working directory; one JSON object per generation.
_KV_CACHE_LOG = "data/logs/traces/kv_cache.jsonl"


def kv_cache_stats(llm: Any, prompt: str) -> dict[str, Any]:
    """
    Compute pre-call cache hit statistics for *prompt* on *llm*.

    Inspects the LlamaRAMCache attached to *llm* (if any) and returns how
    many tokens will be served from the KV-cache versus freshly evaluated.

    Returns a dict that is always safe to pass to log_cache_stats(); if
    anything goes wrong the dict will contain ``{"cache_enabled": True,
    "cache_hit": None, "error": <message>}``.
    """
    try:
        # No cache attached means every prompt is evaluated from scratch.
        cache = getattr(llm, "cache", None)
        if cache is None:
            return {"cache_enabled": False}

        # Lazy import: llama_cpp is only present when a local model is in use.
        from llama_cpp import Llama as _Llama

        # Tokenize exactly as generation will, so the prefix comparison matches the real cache keys.
        prompt_tokens = tuple(llm.tokenize(prompt.encode("utf-8")))
        # Uses a private llama-cpp cache method; it may break if llama-cpp-python changes, which is why
        # the whole function is wrapped in a try/except below.
        prefix_key = cache._find_longest_prefix_key(prompt_tokens)
        if prefix_key is None:
            return {
                "cache_enabled": True,
                "cache_hit": False,
                "tokens_from_cache": 0,
                "tokens_to_eval": len(prompt_tokens),
            }
        # Number of leading tokens shared with the cached sequence; those need no re-evaluation.
        hit_len = _Llama.longest_token_prefix(prefix_key, prompt_tokens)
        return {
            "cache_enabled": True,
            "cache_hit": hit_len > 0,
            "tokens_from_cache": hit_len,
            "tokens_to_eval": len(prompt_tokens) - hit_len,
        }
    except Exception as exc:
        return {"cache_enabled": True, "cache_hit": None, "error": str(exc)}


def log_cache_stats(stats: dict[str, Any], ts: str) -> None:
    """
    Append one JSONL line to _KV_CACHE_LOG with *stats* and the generation
    *ts* (ISO timestamp string from the raw-token log filename).

    Silently swallows all I/O errors so it never disrupts inference.
    """
    try:
        # Create the log folder on first use.
        os.makedirs(os.path.dirname(_KV_CACHE_LOG), exist_ok=True)
        entry: dict[str, Any] = {"ts": ts, **stats}
        # Append mode keeps history across runs; JSONL lets tools tail/grep it.
        with open(_KV_CACHE_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
