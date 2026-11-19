#!/usr/bin/env python3
# ==============================================================================
# Program: download_public_sft.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Downloader for public instruction-tuning and fine-tuning datasets.
#          Retrieves validated open-source datasets and prepares them for local
#          model training.
# ==============================================================================

"""Download and format a public programming instruction dataset for Karl fine-tuning."""

from __future__ import annotations

import ast
import json
import logging
import re
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("karl.download_public_sft")

# Repo root derived from this file's location.
ROOT = Path(__file__).resolve().parents[1]
# Consumed by tools/curate_code_datasets.py (merge step) as the 'public' dataset.
OUT_PATH = ROOT / "data/training/code/public_code_sft.jsonl"


def is_python_code(text: str) -> bool:
    """Return True if the text contains syntactically valid Python code."""
    # Judge only fenced code if there is any; otherwise parse the entire answer.
    # Find code blocks if present
    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    snippets = code_blocks if code_blocks else [text]
    for snippet in snippets:
        # Iterate over items, evaluate loop invariant, and update progress
        try:
            # Execute guarded operations and establish recovery boundaries
            ast.parse(snippet)
            return True
        except SyntaxError:
            continue
    return False


def main() -> int:
    """Download two public CodeAlpaca-style datasets via HuggingFace 'datasets', keep only Python-related, de-duplicated rows and write them as system/user/assistant SFT JSONL. Needs network access; returns 1 if 'datasets' is missing."""
    # Step 1: Validate input parameters, operational preconditions, and state
    try:
        # Execute guarded operations and establish recovery boundaries
        from datasets import load_dataset
    except ImportError:
        logger.error("HuggingFace 'datasets' library is not installed. Install requirements.txt first.")
        return 1

    # (dataset id, split, provenance label written to each row's 'source').
    datasets_to_load = [
        ("sahil2801/CodeAlpaca-20k", "train", "public_codealpaca"),
        ("iamtarun/python_code_instructions_18k_alpaca", "train", "python_instructions_18k")
    ]

    formatted_count = 0
    seen_prompts = set()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for ds_name, split_name, source_label in datasets_to_load:
            # Iterate over items, evaluate loop invariant, and update progress
            logger.info("Downloading %s from HuggingFace...", ds_name)
            try:
                # Execute guarded operations and establish recovery boundaries
                dataset = load_dataset(ds_name, split=split_name)
            except Exception as exc:
                # One unreachable dataset should not abort the other; skip it.
                logger.warning("Failed to download dataset %s: %s", ds_name, exc)
                continue

            logger.info("Filtering and formatting programming instructions from %s...", ds_name)
            for item in dataset:
                # Iterate over items, evaluate loop invariant, and update progress
                instruction = item.get("instruction", "")
                input_context = item.get("input", "")
                output = item.get("output", "")

                if not instruction.strip() or not output.strip():
                    continue

                # Ensure we are capturing Python programming queries
                is_python = (
                    "python" in instruction.lower()
                    or "python" in input_context.lower()
                    or is_python_code(output)
                )
                if not is_python:
                    continue

                user_content = instruction
                if input_context:
                    user_content += f"\n\nInput Context:\n{input_context}"

                # The two datasets overlap heavily; first occurrence wins.
                # Deduplicate by prompt
                prompt_key = user_content.strip()
                if prompt_key in seen_prompts:
                    continue
                seen_prompts.add(prompt_key)

                # Format to conversational system/user/assistant block
                sft_entry = {
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are an expert Python software engineer. Prioritise correctness, "
                                "idiomatic style, and minimal complexity."
                            )
                        },
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": output}
                    ],
                    "source": source_label
                }

                fh.write(json.dumps(sft_entry, ensure_ascii=False) + "\n")
                formatted_count += 1

    logger.info("Successfully wrote %d SFT examples to %s", formatted_count, OUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
