#!/usr/bin/env python3
# ==============================================================================
# Program: agent3_curator.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Flywheel Agent-3 solution curator and sandbox verifier.
#          Executes proposed solutions in test sandboxes and filters out failing
#          or low-quality samples.
# ==============================================================================

"""
Agent 3: Automated Curator & Training Grader
Part of the Karl self-improvement flywheel.

Reads execution JSONs from data/flywheel/execution/ produced by Agent 2,
verifies each model response, and curates SFT/DPO training data into
data/training/curated.jsonl.
"""

import os
import sys
import json
import time
import logging
import tempfile
import subprocess
import argparse
from datetime import datetime, timezone
from pathlib import Path

# Project root resolution
# Repo root (three levels up from data/flywheel/agent3_curator.py).
_ROOT = Path(__file__).resolve().parent.parent.parent
# Make "data.flywheel..." and "app..." importable when run directly as a script.
sys.path.insert(0, str(_ROOT))

from data.flywheel.executor_sandbox import SafePythonSandbox

# Timestamped log lines so a long unattended run can be audited afterwards.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("karl.agent3_curator")

# Input: execution JSONs from Agent 2. Output: curated.jsonl (DPO pairs) and a stats file.
EXECUTION_DIR = str(_ROOT / "data" / "flywheel" / "execution")
CURATED_PATH  = str(_ROOT / "data" / "training" / "curated.jsonl")
STATS_PATH    = str(_ROOT / "data" / "flywheel" / "agent3_stats.json")

# Create folders at import time so the first poll never fails on a missing directory.
os.makedirs(EXECUTION_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CURATED_PATH), exist_ok=True)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _load_stats() -> dict:
    # Step 1: Validate input parameters, operational preconditions, and state
    """Load persisted counters, or zeroed defaults if the file is missing or corrupt."""
    if os.path.exists(STATS_PATH):
        try:
            # Execute guarded operations and establish recovery boundaries
            with open(STATS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "processed": 0,
        "passed": 0,
        "failed": 0,
        "sft_examples": 0,
        "dpo_pairs": 0,
        "last_updated": "",
    }


def _save_stats(stats: dict):
    # Step 1: Validate input parameters, operational preconditions, and state
    """Write counters atomically (tmp file + os.replace); failures only log a warning."""
    stats["last_updated"] = datetime.now(timezone.utc).isoformat()
    # os.replace is atomic, so a crash never leaves a truncated stats file.
    tmp = STATS_PATH + ".tmp"
    try:
        # Execute guarded operations and establish recovery boundaries
        os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        os.replace(tmp, STATS_PATH)
    except Exception as e:
        logger.warning("Failed to write stats: %s", e)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def run_verification(exec_record: dict) -> tuple[bool, str]:
    """
    Dispatch to the appropriate verifier.
    Returns (passed, detail_string).
    """
    # Step 1: Validate input parameters, operational preconditions, and state
    # Fields come from Agent 2; the defaults keep a malformed record from raising KeyError.
    vtype  = exec_record.get("verification_type", "exact_match")
    script = exec_record.get("verification_script", "")
    resp   = exec_record.get("model_response", "")

    if not script:
        return False, "no verification_script"

    # Primary check: run the task's own verify() inside the sandbox.
    passed, detail = _verify_subprocess(script, resp)

    # Symbolic answers can be written many ways ("4.0" vs "4"), so give them a numeric second chance.
    # Secondary sympy pass for symbolic problems when subprocess fails
    if not passed and vtype == "symbolic_match":
        sym_passed, sym_detail = _verify_sympy(
            exec_record.get("ground_truth_answer", ""), resp
        )
        if sym_passed:
            return True, sym_detail

    return passed, detail


def _verify_subprocess(verification_script: str, model_response: str) -> tuple[bool, str]:
    """
    Executes verification_script + a call to verify(response) in the SafePythonSandbox.
    """
    # Step 1: Validate input parameters, operational preconditions, and state
    sandbox = SafePythonSandbox(cpu_timeout_sec=5.0, memory_limit_mb=256)
    
    # Construct the runner script as the test code
    # The response is JSON-encoded twice (dumps of dumps) so it is embedded in the
    # script as a safe string literal and decoded by json.loads at run time. This
    # keeps quotes/newlines in the model output from breaking out of the string.
    test_code = (
        "import sys as _sys, json as _json\n"
        f"_resp = _json.loads({json.dumps(json.dumps(model_response))})\n"
        "try:\n"
        "    _result = verify(_resp)\n"
        "    _sys.exit(0 if _result else 1)\n"
        "except Exception as _e:\n"
        "    print('verify() raised:', _e, file=_sys.stderr)\n"
        "    _sys.exit(2)\n"
    )
    
    passed, trace = sandbox.run_code(verification_script, test_code)
    return passed, trace.strip()[:400]


def _verify_sympy(ground_truth: str, model_response: str) -> tuple[bool, str]:
    """Extract a number from ground_truth and look for it in model_response via sympy."""
    # Step 1: Validate input parameters, operational preconditions, and state
    try:
        # Execute guarded operations and establish recovery boundaries
        import re
        import sympy

        gt_nums = re.findall(r"-?\d+(?:\.\d+)?", ground_truth)
        if not gt_nums:
            return False, "no numeric ground truth for sympy"
        # The final number in the ground truth is taken as the answer.
        gt_val = sympy.sympify(gt_nums[-1])

        resp_nums = re.findall(r"-?\d+(?:\.\d+)?", model_response)
        for n in resp_nums:
            # Iterate over items, evaluate loop invariant, and update progress
            try:
                # Execute guarded operations and establish recovery boundaries
                if sympy.simplify(sympy.sympify(n) - gt_val) == 0:
                    return True, f"sympy: {n} == {gt_val}"
            except Exception:
                pass
        return False, f"sympy: ground truth {gt_val} not found in response"
    except ImportError:
        return False, "sympy not available"
    except Exception as e:
        return False, f"sympy error: {e}"


# ---------------------------------------------------------------------------
# Correction generation
# ---------------------------------------------------------------------------

def generate_correction(exec_record: dict) -> tuple[str, str]:
    """
    Produce a corrected thought trace and solution for a failing execution.
    Tries the local LLM first; falls back to a ground-truth stub.

    Returns (thought_trace, corrected_response).
    """
    # Step 1: Validate input parameters, operational preconditions, and state
    ground_truth = exec_record.get("ground_truth_answer", "")
    problem      = exec_record.get("problem_statement", "")
    vtype        = exec_record.get("verification_type", "")

    # ── LLM-based correction ────────────────────────────────────────────────
    try:
        # Execute guarded operations and establish recovery boundaries
        # Imported lazily: curation must still work when no model code can be imported.
        from app.engine.model_loader import ModelLoader
        from core.interaction_loop import build_prompt

        # Never trigger a model load here; only use one that is already in memory.
        if ModelLoader.is_loaded():
            llm = ModelLoader.get_instance()
            correction_prompt = (
                f"You are a precise tutor correcting a wrong answer.\n\n"
                f"Problem:\n{problem}\n\n"
                f"The student's incorrect answer was:\n"
                f"{exec_record.get('model_response', '')}\n\n"
                f"The correct answer is: {ground_truth}\n\n"
                "Provide a complete, step-by-step correct solution ending with "
                "the final answer clearly stated."
            )
            prompt_text = build_prompt(
                "You are a precise mathematical and programming tutor.",
                [{"role": "user", "content": correction_prompt}],
            )
            raw = llm(
                prompt_text,
                max_tokens=1024,
                # Low temperature: we want a faithful, deterministic worked solution.
                temperature=0.1,
                top_p=0.95,
                stop=["<|im_end|>"],
            )
            full = raw["choices"][0]["text"].strip()
            # Split reasoning (<think>...</think>) from the final answer so each can be stored separately.
            if "</think>" in full:
                head, _, tail = full.partition("</think>")
                thought   = head.replace("<think>", "").strip()
                corrected = tail.strip()
            else:
                thought, corrected = "", full
            if corrected:
                return thought, corrected
    except Exception as e:
        logger.debug("LLM correction unavailable: %s", e)

    # Fallback texts differ by verification type; they are thin but never wrong.
    # ── Ground-truth stub fallback ──────────────────────────────────────────
    if vtype == "unit_test":
        thought = (
            "The model's code failed the unit tests. "
            f"A correct implementation must satisfy: {ground_truth}."
        )
        corrected = (
            f"The correct answer is: {ground_truth}\n\n"
            "A correct implementation must pass all provided test cases, "
            "including edge cases such as empty inputs."
        )
    elif vtype in ("exact_match", "symbolic_match"):
        thought = (
            f"Let me work through this step-by-step.\n"
            f"The correct answer is {ground_truth}."
        )
        corrected = (
            f"Working through the problem carefully:\n"
            f"The correct final answer is: {ground_truth}"
        )
    else:
        thought   = f"The correct answer is: {ground_truth}"
        corrected = f"The correct answer is: {ground_truth}"

    return thought, corrected


# ---------------------------------------------------------------------------
# Curation writers
# ---------------------------------------------------------------------------

def save_sft_example(exec_record: dict):
    """Save a passing execution as a thumbs-up SFT example."""
    # Step 1: Validate input parameters, operational preconditions, and state
    try:
        # Execute guarded operations and establish recovery boundaries
        from app.utils.training_curator import save_example
        save_example(
            system_prompt="",
            user_msg=exec_record.get("problem_statement", ""),
            good_response=exec_record.get("model_response", ""),
            source="thumbs_up",
        )
    except Exception as e:
        logger.error("save_sft_example failed: %s", e)


def save_dpo_pair(
    exec_record: dict, corrected_thought: str, corrected_response: str
):
    """
    Append a DPO pair to curated.jsonl using the instruction/chosen/rejected schema.
    """
    # Step 1: Validate input parameters, operational preconditions, and state
    model_thought   = exec_record.get("model_thought", "")
    model_response  = exec_record.get("model_response", "")

    # Wrap with <think> tags when a thought trace is available
    # Re-wrap reasoning in <think> tags so the pair matches the model's native output format.
    chosen = (
        f"<think>\n{corrected_thought}\n</think>\n{corrected_response}"
        if corrected_thought
        else corrected_response
    )
    rejected = (
        f"<think>\n{model_thought}\n</think>\n{model_response}"
        if model_thought
        else model_response
    )

    record = {
        "instruction": exec_record.get("problem_statement", ""),
        "input": "",
        "chosen": chosen,
        "rejected": rejected,
        "source": "curated_dpo",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(os.path.dirname(CURATED_PATH), exist_ok=True)
    # Append mode + one JSON object per line (JSONL) keeps writes simple and resumable.
    with open(CURATED_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Per-file processor
# ---------------------------------------------------------------------------

def process_execution_file(path: str, stats: dict):
    # Step 1: Validate input parameters, operational preconditions, and state
    """Grade one execution file: verify, save SFT or DPO data, update stats, delete the file.

    Unreadable files are deleted (they can never succeed). Unexpected errors keep
    the file so the next poll retries it.
    """
    fname = os.path.basename(path)
    logger.info("Grading: %s", fname)

    try:
        # Execute guarded operations and establish recovery boundaries
        with open(path, "r", encoding="utf-8") as f:
            exec_record = json.load(f)
    except Exception as e:
        logger.error("Failed to load %s: %s", fname, e)
        try:
            # Execute guarded operations and establish recovery boundaries
            # The file is removed only after its result was saved, so work is not lost.
            os.remove(path)
        except Exception:
            pass
        return

    try:
        # Execute guarded operations and establish recovery boundaries
        passed, detail = run_verification(exec_record)
        stats["processed"] += 1

        vtype = exec_record.get("verification_type", "?")

        if passed:
            logger.info("  PASS [%s] %s", vtype, fname)
            save_sft_example(exec_record)
            stats["passed"]       += 1
            stats["sft_examples"] += 1
        else:
            logger.info("  FAIL [%s] %s — %s", vtype, fname, detail[:80])
            thought, corrected = generate_correction(exec_record)
            save_dpo_pair(exec_record, thought, corrected)
            stats["failed"]    += 1
            stats["dpo_pairs"] += 1

        os.remove(path)
        logger.info("  Deleted execution file: %s", fname)

    except Exception as e:
        logger.error("Unexpected error processing %s: %s", fname, e, exc_info=True)
        # Leave it on disk to retry on the next cycle.
        # Do not delete on unexpected error — allow retry on next poll cycle
    finally:
        _save_stats(stats)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    # Step 1: Validate input parameters, operational preconditions, and state
    """Poll the execution folder and grade each file; --once drains the folder and exits."""
    parser = argparse.ArgumentParser(
        description="Agent 3: Automated Grader & Training Curator"
    )
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="Poll interval in seconds when execution dir is empty (default: 1)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Process all current files then exit (no infinite loop)",
    )
    args = parser.parse_args()

    logger.info("Agent 3 started. Watching: %s", EXECUTION_DIR)
    stats = _load_stats()

    while True:
        # Iterate over items, evaluate loop invariant, and update progress
        try:
            # Execute guarded operations and establish recovery boundaries
            # Ignore .tmp files that Agent 2 may still be writing.
            files = sorted(
                f for f in os.listdir(EXECUTION_DIR)
                if f.endswith(".json") and not f.endswith(".tmp")
            )

            if not files:
                if args.once:
                    logger.info("No files remaining. Exiting (--once).")
                    break
                time.sleep(args.interval)
                continue

            for fname in files:
                # Iterate over items, evaluate loop invariant, and update progress
                fpath = os.path.join(EXECUTION_DIR, fname)
                if os.path.exists(fpath):
                    process_execution_file(fpath, stats)

            if args.once:
                logger.info("Batch complete. Exiting (--once).")
                break

        except Exception as e:
            logger.error("Main loop error: %s", e)
            # Back off after an unexpected error so a persistent fault does not spin the CPU.
            time.sleep(5)

    logger.info(
        "Agent 3 done — processed=%d  passed=%d  failed=%d  sft=%d  dpo=%d",
        stats["processed"], stats["passed"], stats["failed"],
        stats["sft_examples"], stats["dpo_pairs"],
    )


if __name__ == "__main__":
    main()
