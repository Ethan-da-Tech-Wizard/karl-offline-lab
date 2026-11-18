# Program: karl-offline-lab - swarm_agent_profiles.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Stores and loads the swarm agent profiles (architect, coder, tester and custom ones) in a JSON file, validating and clamping their settings.

"""Persistent specialist profile registry for Karl's swarm agents."""

from __future__ import annotations

# Standard-library imports plus the default system prompts for built-in agents.
import copy
import json
import os
import re
from typing import Any

from core.default_prompts import SWARM_ARCHITECT_SYSTEM_PROMPT, SWARM_CODER_SYSTEM_PROMPT


# ── Constants ────────────────────────────────────────────────────────────
# Where custom profiles are saved, relative to the working directory.
PROFILE_PATH = os.path.join("data", "agent_profiles.json")
# The only tool names a profile may switch on or off.
ALLOWED_TOOLS = ("read_files", "write_files", "execute_sandbox", "query_rag")
# Profile ids must start with a letter, then letters, digits, underscore or hyphen (max 64 characters).
PROFILE_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]{0,63}$")


# Built-in profiles: each agent role has a prompt, sampling temperature, context size and tool permissions.
DEFAULT_SWARM_AGENT_PROFILES: dict[str, dict[str, Any]] = {
    "architect": {
        "name": "Architect",
        "icon": "A",
        "system_prompt": SWARM_ARCHITECT_SYSTEM_PROMPT,
        "temperature": 0.1,
        "context_limit": 1536,
        "tools": {
            "read_files": True,
            "write_files": False,
            "execute_sandbox": False,
            "query_rag": True,
        },
        "builtin": True,
    },
    "coder": {
        "name": "Coder",
        "icon": "C",
        "system_prompt": SWARM_CODER_SYSTEM_PROMPT,
        "temperature": 0.2,
        "context_limit": 2048,
        "tools": {
            "read_files": True,
            "write_files": True,
            "execute_sandbox": False,
            "query_rag": True,
        },
        "builtin": True,
    },
    "tester": {
        "name": "Tester",
        "icon": "T",
        "system_prompt": "Run the configured verification command and report exact failures.",
        "temperature": 0.0,
        "context_limit": 1024,
        "tools": {
            "read_files": True,
            "write_files": False,
            "execute_sandbox": True,
            "query_rag": False,
        },
        "builtin": True,
    },
}


# ── Helpers ──────────────────────────────────────────────────────────────
def _default_profiles() -> dict[str, dict[str, Any]]:
    """Return a fresh deep copy of the built-in profiles so callers cannot change the defaults."""
    return copy.deepcopy(DEFAULT_SWARM_AGENT_PROFILES)


def _normalize_profile(profile_id: str, raw: dict[str, Any], builtin: bool = False) -> dict[str, Any]:
    """Turn loosely-typed profile data into a clean profile dict with defaults, accepted aliases and clamped values."""
    profile = {
        "name": str(raw.get("name") or raw.get("label") or profile_id).strip() or profile_id,
        "icon": str(raw.get("icon") or profile_id[:1].upper()).strip()[:4] or profile_id[:1].upper(),
        "system_prompt": str(raw.get("system_prompt") or raw.get("prompt") or ""),
        "temperature": float(raw.get("temperature", 0.2)),
        "context_limit": int(raw.get("context_limit", raw.get("max_tokens", 2048))),
        "tools": {},
        "builtin": bool(raw.get("builtin", builtin)),
    }
    # Keep numbers inside safe ranges: temperature 0-2, context limit 256-32768.
    profile["temperature"] = min(2.0, max(0.0, profile["temperature"]))
    profile["context_limit"] = min(32768, max(256, profile["context_limit"]))
    # Every allowed tool is present as an explicit True/False; unknown tools are discarded.
    raw_tools = raw.get("tools") if isinstance(raw.get("tools"), dict) else {}
    for tool in ALLOWED_TOOLS:
        profile["tools"][tool] = bool(raw_tools.get(tool, False))
    return profile


# ── Load and save ────────────────────────────────────────────────────────
def load_agent_profiles() -> dict[str, dict[str, Any]]:
    """Load built-in profiles and overlay any saved ones from disk; unreadable files are ignored."""
    profiles = _default_profiles()
    if os.path.exists(PROFILE_PATH):
        try:
            with open(PROFILE_PATH, "r", encoding="utf-8") as fh:
                disk = json.load(fh)
            if isinstance(disk, dict):
                for profile_id, raw in disk.items():
                    if isinstance(profile_id, str) and isinstance(raw, dict):
                        profiles[profile_id] = _normalize_profile(
                            profile_id,
                            raw,
                            builtin=profile_id in DEFAULT_SWARM_AGENT_PROFILES,
                        )
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    # Normalize everything once more so built-ins and loaded profiles share one shape.
    for profile_id, raw in list(profiles.items()):
        profiles[profile_id] = _normalize_profile(
            profile_id,
            raw,
            builtin=profile_id in DEFAULT_SWARM_AGENT_PROFILES,
        )
    return profiles


def save_agent_profiles(profiles: dict[str, dict[str, Any]]) -> bool:
    """Validate and write all profiles to disk (via a temporary file); raises ValueError for a bad id or entry."""
    os.makedirs(os.path.dirname(PROFILE_PATH), exist_ok=True)
    normalized: dict[str, dict[str, Any]] = {}
    for profile_id, raw in profiles.items():
        if not isinstance(profile_id, str) or not PROFILE_ID_RE.match(profile_id):
            raise ValueError(f"Invalid profile id: {profile_id!r}")
        if not isinstance(raw, dict):
            raise ValueError(f"Profile {profile_id!r} must be an object.")
        normalized[profile_id] = _normalize_profile(
            profile_id,
            raw,
            builtin=profile_id in DEFAULT_SWARM_AGENT_PROFILES,
        )
    # Write to a temporary file then rename, so a crash cannot leave a partial file.
    tmp = PROFILE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(normalized, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, PROFILE_PATH)
    return True


def save_agent_profile(profile_id: str, profile: dict[str, Any]) -> dict[str, Any]:
    """Validate and save a single profile, keeping the others; returns the normalized profile."""
    if not PROFILE_ID_RE.match(profile_id):
        raise ValueError("Profile id must start with a letter and contain only letters, numbers, '_' or '-'.")
    profiles = load_agent_profiles()
    normalized = _normalize_profile(profile_id, profile, builtin=profile_id in DEFAULT_SWARM_AGENT_PROFILES)
    profiles[profile_id] = normalized
    save_agent_profiles(profiles)
    return normalized


def active_profile_map(overrides: dict[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """Return role -> profile mapping for architect/coder/tester."""
    profiles = load_agent_profiles()
    overrides = overrides or {}
    result: dict[str, dict[str, Any]] = {}
    for role in ("architect", "coder", "tester"):
        selected = overrides.get(role) or role
        result[role] = profiles.get(selected, profiles[role])
    return result
