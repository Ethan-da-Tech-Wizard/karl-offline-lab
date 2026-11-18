#!/usr/bin/env python3
# Program: karl-offline-lab - generate_self_correction_dataset.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Command-line tool that has the local model solve coding problems, verifies answers in a sandbox, and saves problems that were fixed on the second or third attempt as self-correction SFT examples.
"""Compile self-correction (STaR) trajectories into high-value code SFT data."""

from __future__ import annotations

import argparse
import difflib
import glob
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Repository root, added to the import path so project modules can be imported.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.default_prompts import SWARM_CODER_SYSTEM_PROMPT
from core.cognitive_parser import parse_thought_stream
from data.flywheel.executor_sandbox import SafePythonSandbox


DEFAULT_OUTPUT = ROOT / "data" / "training" / "code" / "self_correction_sft.jsonl"
# Patterns for fenced code blocks and for thinking sections.
CODE_FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)
THINK_RE = re.compile(r"<think>\s*([\s\S]*?)\s*</think>", re.IGNORECASE)


@dataclass
class CodingProblem:
    """A coding problem: an id, the prompt text, a verification script that defines verify(), and optional context code."""
    problem_id: str
    prompt: str
    verification_script: str
    context: str = ""


def extract_code_block(text: str) -> str:
    """Return the first fenced code block in the text, or the whole text when there is none."""
    match = CODE_FENCE_RE.search(text or "")
    return (match.group(1) if match else (text or "")).strip()


def extract_thought_block(text: str) -> str:
    """Return the text inside the first think tags, or an empty string."""
    match = THINK_RE.search(text or "")
    return (match.group(1) if match else "").strip()


def code_diff(before: str, after: str) -> str:
    """Return a unified diff between the code of a failed answer and the corrected answer."""
    before_code = extract_code_block(before).splitlines()
    after_code = extract_code_block(after).splitlines()
    return "\n".join(
        difflib.unified_diff(
            before_code,
            after_code,
            fromfile="failed_attempt.py",
            tofile="corrected_attempt.py",
            lineterm="",
        )
    )


def normalize_problem_record(record: dict, fallback_id: str = "") -> CodingProblem | None:
    """Normalize JSONL/flywheel records into executable coding problems."""
    verification_script = str(record.get("verification_script") or "")
    if not verification_script.strip():
        return None

    # The prompt field may be stored under several names.
    prompt = (
        record.get("problem_statement")
        or record.get("prompt")
        or record.get("instruction")
        or record.get("question")
        or ""
    )
    context = str(record.get("context") or record.get("code") or "")
    # Append extra context code to the prompt unless it is already included.
    if context and context not in prompt:
        prompt = f"{prompt}\n\nContext/code:\n{context}"
    prompt = str(prompt).strip()
    if not prompt:
        return None

    return CodingProblem(
        problem_id=str(record.get("id") or record.get("case_id") or fallback_id),
        prompt=prompt,
        verification_script=verification_script,
        context=context,
    )


def iter_problem_records(paths: Iterable[str]) -> Iterable[CodingProblem]:
    # Running counter used to build fallback ids.
    """Yield problems from JSON and JSONL files matched by the given paths or glob patterns, skipping unreadable files and records without a verification script."""
    index = 0
    for pattern in paths:
        for raw_path in sorted(glob.glob(pattern)):
            path = Path(raw_path)
            if not path.exists():
                continue
            # JSONL files hold one record per line; other files hold a list or a single object.
            if path.suffix == ".jsonl":
                with path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        index += 1
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        problem = normalize_problem_record(record, fallback_id=f"{path.name}:{index}")
                        if problem:
                            yield problem
            else:
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                rows = data if isinstance(data, list) else [data]
                for record in rows:
                    if not isinstance(record, dict):
                        continue
                    index += 1
                    problem = normalize_problem_record(record, fallback_id=f"{path.name}:{index}")
                    if problem:
                        yield problem


def build_initial_prompt(problem: CodingProblem) -> str:
    """Build the first request asking for an explanation and a complete Python solution."""
    return (
        "Solve this coding problem. Return a concise explanation and a complete Python solution "
        "inside a ```python code block.\n\n"
        f"Problem:\n{problem.prompt}"
    )


def build_correction_prompt(problem: CodingProblem, failed_response: str, traceback_text: str) -> str:
    """Build a follow-up request that includes the problem, the failed answer and the verifier error, asking for a fix."""
    return (
        "Your previous solution failed verification. Use the traceback to correct the bug. "
        "Return the corrected complete Python solution in a ```python code block and include "
        "your reasoning inside <think>...</think>.\n\n"
        f"Problem:\n{problem.prompt}\n\n"
        f"Failed response:\n{failed_response}\n\n"
        f"Verifier traceback/error:\n{traceback_text}"
    )


def verify_response(problem: CodingProblem, response: str, timeout: float = 5.0) -> tuple[bool, str]:
    # Run the verification with CPU and memory limits.
    """Run the problem's verify() function on a response inside a resource-limited sandbox.
    Returns (passed, trace text).
    """
    sandbox = SafePythonSandbox(cpu_timeout_sec=timeout, memory_limit_mb=256)
    test_code = (
        "import sys as _sys, json as _json\n"
        f"_resp = _json.loads({json.dumps(json.dumps(response))})\n"
        "try:\n"
        "    _result = verify(_resp)\n"
        "    _sys.exit(0 if _result else 1)\n"
        "except Exception as _e:\n"
        "    print('verify() raised:', _e, file=_sys.stderr)\n"
        "    _sys.exit(2)\n"
    )
    # The sandbox returns whether the check passed and its output.
    passed, trace = sandbox.run_code(problem.verification_script, test_code)
    return passed, trace.strip()


def query_local_model(llm, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> tuple[str, str]:
    """Stream a reply from the local model for a system and user prompt and split it into the thought part and the answer part."""
    from core.interaction_loop import build_prompt

    # Build the full chat prompt in the model's expected format.
    prompt = build_prompt(system_prompt, [{"role": "user", "content": user_prompt}])
    chunks: list[str] = []
    for chunk in llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        stream=True,
        stop=["<|im_end|>", "<|endoftext|>", "<|end_of_text|>", "<|im_start|>"],
        echo=False,
    ):
        if "choices" in chunk and chunk["choices"]:
            chunks.append(chunk["choices"][0].get("text", ""))
    # Split the raw text into thinking and answer, starting in thought mode when the prompt ends with an open think tag.
    raw = "".join(chunks)
    thought, response = parse_thought_stream(raw, start_in_thought=prompt.rstrip().endswith("<think>"))
    return thought, response or raw


def build_self_correction_row(problem: CodingProblem, attempts: list[dict]) -> dict:
    """Create a HuggingFace messages row from a failed-then-fixed trajectory."""
    # Pick the last failed attempt as the "before" state for the correction diff.
    failed_attempts = [a for a in attempts[:-1] if not a["passed"]]
    final = attempts[-1]
    previous = failed_attempts[-1] if failed_attempts else attempts[0]
    diff = code_diff(previous["response"], final["response"])
    thought = extract_thought_block(final["raw"]) or final.get("thought", "")
    # Summary of what failed, what fixed it and the code difference.
    transition = {
        "failed_iteration": previous["iteration"],
        "fixed_iteration": final["iteration"],
        "verification_error": previous.get("trace", ""),
        "correction_diff": diff,
    }
    assistant = (
        f"<think>\n{thought.strip()}\n</think>\n\n"
        "Self-correction transition:\n"
        f"```json\n{json.dumps(transition, ensure_ascii=False, indent=2)}\n```\n\n"
        "Corrected solution:\n"
        f"{final['response'].strip()}"
    )
    return {
        "messages": [
            {"role": "system", "content": SWARM_CODER_SYSTEM_PROMPT},
            {"role": "user", "content": build_initial_prompt(problem)},
            {"role": "assistant", "content": assistant},
        ],
        "origin": "self_correction_star",
        "problem_id": problem.problem_id,
        "iterations": len(attempts),
    }


def run_problem(llm, problem: CodingProblem, max_attempts: int, max_tokens: int) -> dict | None:
    """Ask the model up to max_attempts times, feeding failures back, and return a training row only when it passes on attempt 2 or 3 (otherwise None)."""
    attempts: list[dict] = []
    user_prompt = build_initial_prompt(problem)
    for iteration in range(1, max_attempts + 1):
        thought, response = query_local_model(
            llm,
            SWARM_CODER_SYSTEM_PROMPT,
            user_prompt,
            max_tokens=max_tokens,
            # Use a higher temperature for the first try and a lower one when correcting.
            temperature=0.2 if iteration > 1 else 0.7,
        )
        raw = f"<think>\n{thought}\n</think>\n{response}" if thought else response
        passed, trace = verify_response(problem, response)
        attempts.append({
            "iteration": iteration,
            "thought": thought,
            "response": response,
            "raw": raw,
            "passed": passed,
            "trace": trace,
        })
        if passed:
            # Only keep real corrections: a first-try pass has nothing to learn from and a later pass is excluded.
            if iteration in (2, 3):
                return build_self_correction_row(problem, attempts)
            return None
        user_prompt = build_correction_prompt(problem, response, trace)
    return None


def main() -> int:
    """Parse options, load problems and the local model, build trajectories, write them as JSON lines and return an exit code (1 when no problems are found)."""
    parser = argparse.ArgumentParser(description="Generate STaR self-correction SFT trajectories.")
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["data/flywheel/queue/*.json", "eval/datasets/*.jsonl"],
        help="JSON/JSONL problem files or glob patterns",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0, help="Maximum normalized problems to process; 0 means all")
    args = parser.parse_args()

    # Load problems, optionally limited to a maximum count.
    problems = list(iter_problem_records(args.dataset))
    if args.limit > 0:
        problems = problems[:args.limit]
    if not problems:
        print("No executable coding problems found. Records need a verification_script.", file=sys.stderr)
        return 1

    # Import lazily so the model code loads only when generation really runs.
    from app.engine.model_loader import ModelLoader

    llm = ModelLoader.get_instance()
    rows: list[dict] = []
    for idx, problem in enumerate(problems, 1):
        print(f"[{idx}/{len(problems)}] {problem.problem_id}")
        try:
            row = run_problem(llm, problem, args.max_attempts, args.max_tokens)
        except Exception as exc:
            print(f"  skipped: {exc}")
            continue
        if row:
            rows.append(row)
            print("  saved self-correction trajectory")
        else:
            print("  no iteration-2/3 fix")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} STaR SFT row(s) to {out_path}")
    return 0


# Run main() and use its return value as the exit code.
if __name__ == "__main__":
    raise SystemExit(main())
