#!/usr/bin/env python3
# ==============================================================================
# Program: setup_speculative_decoding.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Speculative decoding pairing and configuration tool.
#          Configures lightweight draft models alongside primary models to
#          accelerate generation throughput.
# ==============================================================================

"""Download and configure Karl's speculative-decoding draft model."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path


# Small Qwen2.5-0.5B Q8 GGUF used as the speculative-decoding draft model: fast enough to propose
# tokens that the main model then verifies.
# Draft and main model must share a tokenizer family for speculative decoding to accept
# proposed tokens; Qwen2.5-0.5B pairs with the Qwen-family models Karl ships. Pairing it with
# an unrelated family would load fine but get almost no accepted tokens.
URL = (
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/"
    "qwen2.5-0.5b-instruct-q8_0.gguf"
)
# Relative paths: run from the repo root.
TARGET = Path("data/models/qwen2.5-0.5b-instruct-q8_0.gguf")
CONFIG = Path("data/draft_model.json")
# 1 MiB read chunks keep memory flat while still updating progress often.
CHUNK_SIZE = 1024 * 1024


def _format_bytes(value: float) -> str:
    """Human-readable size (B..TB, one decimal) for download progress lines."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _format_eta(seconds: float | None) -> str:
    """Format seconds as MM:SS or H:MM:SS; returns '--:--' when unknown or negative."""
    if seconds is None or seconds < 0:
        return "--:--"
    seconds = int(seconds)
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours:d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def download(url: str, target: Path, force: bool = False) -> None:
    """Download url to target with a progress line. Skips if a non-empty file exists (unless force). Data is written to a .part file and renamed on success so an interrupted download never leaves a truncated model in place."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0 and not force:
        print(f"Draft model already exists: {target}")
        return

    # Download to a temp name; atomically replaced below.
    tmp = target.with_suffix(target.suffix + ".part")
    # Identify the tool explicitly rather than sending the default Python-urllib agent, which
    # some hosts throttle or block.
    req = urllib.request.Request(url, headers={"User-Agent": "Karl/SpeculativeSetup"})
    start = time.monotonic()
    downloaded = 0

    with urllib.request.urlopen(req) as response, tmp.open("wb") as out:
        # Content-Length may be missing (chunked transfer); progress then omits percent and ETA.
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header and total_header.isdigit() else None
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            # Floor at 1 ms to avoid division by zero on the first chunk.
            elapsed = max(time.monotonic() - start, 0.001)
            rate = downloaded / elapsed
            remaining = ((total - downloaded) / rate) if total and rate > 0 else None
            if total:
                pct = downloaded / total * 100.0
                status = (
                    f"\r{pct:6.2f}%  {_format_bytes(downloaded)} / {_format_bytes(total)}  "
                    f"{_format_bytes(rate)}/s  ETA {_format_eta(remaining)}"
                )
            else:
                status = f"\r{_format_bytes(downloaded)}  {_format_bytes(rate)}/s  ETA {_format_eta(None)}"
            print(status, end="", flush=True)

    print()
    # Rename only after the loop ends cleanly; a dropped connection raises before this line, so the
    # target path never holds a truncated model. (A stale .part file may remain; it is overwritten
    # on the next attempt.)
    tmp.replace(target)
    print(f"Saved draft model to {target}")


def write_config(target: Path, config_path: Path) -> None:
    """Write data/draft_model.json telling the inference engine where the draft model is and how to load it (n_gpu_layers=-1 offloads all layers)."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "draft_model_path": str(target),
        # Kept modest: the draft model only needs enough context to follow the main model, and its
        # KV cache competes with the main model for memory.
        "n_ctx": 4096,
        # -1 offloads every layer; at 0.5B parameters the whole draft model costs well under 1 GB.
        "n_gpu_layers": -1,
    }
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(f"Wrote speculative decoding config to {config_path}")


def main() -> int:
    """CLI entry. Returns 0 on success, 130 on Ctrl-C, 1 on any other failure. This is the only tool here that touches the network, and only when run explicitly."""
    parser = argparse.ArgumentParser(description="Download Karl's Qwen 0.5B draft GGUF.")
    parser.add_argument("--url", default=URL)
    parser.add_argument("--target", default=str(TARGET))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--force", action="store_true", help="Re-download even if the target exists")
    args = parser.parse_args()

    target = Path(args.target)
    try:
        download(args.url, target, force=args.force)
        write_config(target, Path(args.config))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Speculative setup failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
