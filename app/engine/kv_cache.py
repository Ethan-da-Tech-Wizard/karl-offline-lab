# Program: karl-offline-lab - kv_cache.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Computes how much of a prompt the llama.cpp KV-cache can reuse and appends those statistics to a JSONL log.
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
        cache = getattr(llm, "cache", None)
        if cache is None:
            return {"cache_enabled": False}

        from llama_cpp import Llama as _Llama

        prompt_tokens = tuple(llm.tokenize(prompt.encode("utf-8")))
        # Ask the cache for the stored token sequence that shares the longest prefix with this prompt.
        prefix_key = cache._find_longest_prefix_key(prompt_tokens)
        if prefix_key is None:
            return {
                "cache_enabled": True,
                "cache_hit": False,
                "tokens_from_cache": 0,
                "tokens_to_eval": len(prompt_tokens),
            }
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
        # Make sure the log folder exists before appending.
        os.makedirs(os.path.dirname(_KV_CACHE_LOG), exist_ok=True)
        entry: dict[str, Any] = {"ts": ts, **stats}
        with open(_KV_CACHE_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
