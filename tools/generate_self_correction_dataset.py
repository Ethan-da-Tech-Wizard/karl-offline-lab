#!/usr/bin/env python3
# ==============================================================================
# Program: generate_self_correction_dataset.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Synthetic dataset generator for self-healing code models.
#          Produces problem pairs featuring buggy code, diagnostic reasoning,
#          and corrected solutions.
# ==============================================================================

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


# Repo root (tools/ is one level down); added to sys.path below so 'core' and 'data' import when run
# as a plain script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Imports must come after the sys.path tweak above, hence not at the top of the file.
from core.default_prompts import SWARM_CODER_SYSTEM_PROMPT
from core.cognitive_parser import parse_thought_stream
from data.flywheel.executor_sandbox import SafePythonSandbox


DEFAULT_OUTPUT = ROOT / "data" / "training" / "code" / "self_correction_sft.jsonl"
# Matches ```python ... ``` (or bare ```) fences; non-greedy so only the first block is captured.
# Only the first fence is used on purpose: models sometimes add a second block with example
# usage, and including it would make the verified "solution" differ from what was checked.
CODE_FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)
# Matches a complete <think> block; DOTALL-style [\s\S] is used so reasoning may span lines.
THINK_RE = re.compile(r"<think>\s*([\s\S]*?)\s*</think>", re.IGNORECASE)


@dataclass
class CodingProblem:
    """One executable coding task: prompt text, a verification script that defines verify(response) -> bool, and optional extra context."""
    problem_id: str
    prompt: str
    verification_script: str
    context: str = ""


def extract_code_block(text: str) -> str:
    """Return the first fenced code block in text (python tag optional); if there is no fence, return the whole text stripped."""
    match = CODE_FENCE_RE.search(text or "")
    return (match.group(1) if match else (text or "")).strip()


def extract_thought_block(text: str) -> str:
    """Return the content of the first <think>...</think> block, or '' when the model produced none."""
    match = THINK_RE.search(text or "")
    return (match.group(1) if match else "").strip()


def code_diff(before: str, after: str) -> str:
    """Unified diff between the code blocks of a failed and a corrected response; this is the 'what changed' signal stored in the training row."""
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
    # Records come from two shapes: flywheel queue tasks (problem_statement + verification_script)
    # and eval datasets (instruction/question variants). Accept the union so both feed one pipeline.
    verification_script = str(record.get("verification_script") or "")
    # Without a verifier a problem cannot be checked, so it can never yield a trustworthy trajectory;
    # drop it.
    if not verification_script.strip():
        return None

    prompt = (
        record.get("problem_statement")
        or record.get("prompt")
        or record.get("instruction")
        or record.get("question")
        or ""
    )
    context = str(record.get("context") or record.get("code") or "")
    # Append context/code only if the prompt does not already contain it, to avoid duplicating text.
    if context and context not in prompt:
        prompt = f"{prompt}\n\nContext/code:\n{context}"
    prompt = str(prompt).strip()
    if not prompt:
        return None

    return CodingProblem(
        # Prefer an explicit id, then case_id, then the file:index fallback so ids stay traceable.
        problem_id=str(record.get("id") or record.get("case_id") or fallback_id),
        prompt=prompt,
        verification_script=verification_script,
        context=context,
    )


def iter_problem_records(paths: Iterable[str]) -> Iterable[CodingProblem]:
    """Yield CodingProblem objects from JSON/JSONL files matched by the given glob patterns. Unreadable files and malformed lines are skipped silently so one bad file cannot abort a batch run."""
    index = 0
    for pattern in paths:
        for raw_path in sorted(glob.glob(pattern)):
            path = Path(raw_path)
            if not path.exists():
                continue
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
    """First-attempt user prompt asking for an explanation plus a complete python solution."""
    return (
        "Solve this coding problem. Return a concise explanation and a complete Python solution "
        "inside a ```python code block.\n\n"
        f"Problem:\n{problem.prompt}"
    )


def build_correction_prompt(problem: CodingProblem, failed_response: str, traceback_text: str) -> str:
    """Follow-up prompt that shows the model its failed answer and the verifier traceback and asks for a fix with reasoning in <think> tags."""
    return (
        "Your previous solution failed verification. Use the traceback to correct the bug. "
        "Return the corrected complete Python solution in a ```python code block and include "
        "your reasoning inside <think>...</think>.\n\n"
        f"Problem:\n{problem.prompt}\n\n"
        f"Failed response:\n{failed_response}\n\n"
        f"Verifier traceback/error:\n{traceback_text}"
    )


def verify_response(problem: CodingProblem, response: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Run problem.verification_script in the restricted SafePythonSandbox with the model response passed to verify(). Returns (passed, trace). Exit code 0 = pass, 1 = verify returned falsy, 2 = verify raised."""
    # The verifier is model-adjacent code, so it runs in the restricted sandbox with tight CPU and
    # memory limits rather than being exec()d in this process.
    sandbox = SafePythonSandbox(cpu_timeout_sec=timeout, memory_limit_mb=256)
    # Harness script run next to the verification script. The response is double json.dumps'd so it
    # embeds as a safe Python string literal regardless of quotes/newlines.
    test_code = (
        "import sys as _sys, json as _json\n"
        f"_resp = _json.loads({json.dumps(json.dumps(response))})\n"
        "try:\n"
        "    _result = verify(_resp)\n"
        # Three distinct exit codes (0 pass, 1 fail, 2 verifier crashed) let the traceback shown to
        # the model distinguish "wrong answer" from "your verification script is broken".
        "    _sys.exit(0 if _result else 1)\n"
        "except Exception as _e:\n"
        "    print('verify() raised:', _e, file=_sys.stderr)\n"
        "    _sys.exit(2)\n"
    )
    passed, trace = sandbox.run_code(problem.verification_script, test_code)
    return passed, trace.strip()


def query_local_model(llm, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> tuple[str, str]:
    """Stream a completion from the local llama.cpp-style model and split it into (thought, response) using the R1 think parser. Falls back to the raw text when no response part is found."""
    from core.interaction_loop import build_prompt

    # Imported lazily to keep this module importable without the heavy interaction_loop dependencies.
    prompt = build_prompt(system_prompt, [{"role": "user", "content": user_prompt}])
    chunks: list[str] = []
    for chunk in llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        stream=True,
        # Chat-template end tokens for the model families used here so generation halts at turn end.
        stop=["<|im_end|>", "<|endoftext|>", "<|end_of_text|>", "<|im_start|>"],
        echo=False,
    ):
        if "choices" in chunk and chunk["choices"]:
            chunks.append(chunk["choices"][0].get("text", ""))
    raw = "".join(chunks)
    # If the template pre-seeded <think>, the model output starts inside a thought and the parser must
    # know.
    thought, response = parse_thought_stream(raw, start_in_thought=prompt.rstrip().endswith("<think>"))
    return thought, response or raw


def build_self_correction_row(problem: CodingProblem, attempts: list[dict]) -> dict:
    """Create a HuggingFace messages row from a failed-then-fixed trajectory."""
    # The 'previous' attempt is the last failed one; if none failed, fall back to attempt 0.
    failed_attempts = [a for a in attempts[:-1] if not a["passed"]]
    final = attempts[-1]
    previous = failed_attempts[-1] if failed_attempts else attempts[0]
    # The diff is computed on code blocks only, not prose: the explanation text always differs
    # between attempts and would drown out the real fix in the training signal.
    diff = code_diff(previous["response"], final["response"])
    # Prefer the <think> text embedded in the raw response; fall back to the parser-split thought
    # for models whose think block was pre-seeded and so has no opening tag in the raw text.
    thought = extract_thought_block(final["raw"]) or final.get("thought", "")
    # Structured summary of the failure->fix hop; embedded as JSON in the assistant turn so the
    # student learns to articulate the correction.
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
        # Kept so downstream filtering can weigh a 3-try repair differently from a 2-try one.
        "iterations": len(attempts),
    }


def run_problem(llm, problem: CodingProblem, max_attempts: int, max_tokens: int) -> dict | None:
    """Try a problem up to max_attempts times, feeding each failure back to the model. Only a pass on iteration 2 or 3 yields a row: a first-try pass has nothing to correct, and later fixes are treated as too noisy. Returns the row or None."""
    attempts: list[dict] = []
    user_prompt = build_initial_prompt(problem)
    # Each failure feeds the previous response and its traceback into the next prompt, so the
    # model always repairs its own latest attempt rather than starting over.
    for iteration in range(1, max_attempts + 1):
        thought, response = query_local_model(
            llm,
            SWARM_CODER_SYSTEM_PROMPT,
            user_prompt,
            max_tokens=max_tokens,
            # Explore at higher temperature on the first try, then be conservative while repairing.
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
            # Keep only genuine corrections (fixed on 2nd/3rd attempt).
            # Iteration 1 passes carry no correction signal, so they are dropped; iterations beyond 3
            # are also dropped to keep the dataset to short, learnable repairs.
            if iteration in (2, 3):
                return build_self_correction_row(problem, attempts)
            return None
        user_prompt = build_correction_prompt(problem, response, trace)
    return None


def main() -> int:
    """CLI entry: load problems, run each through the local model, and write the self-correction rows to a JSONL file. Returns 1 when no executable problems were found."""
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

    problems = list(iter_problem_records(args.dataset))
    if args.limit > 0:
        # Applied after normalisation so --limit counts runnable problems, not raw records.
        problems = problems[:args.limit]
    if not problems:
        print("No executable coding problems found. Records need a verification_script.", file=sys.stderr)
        return 1

    # Lazy import: loading the model stack is slow and only needed for real runs, not for --help or
    # unit tests.
    from app.engine.model_loader import ModelLoader

    # get_instance() is used (not acquire_instance) because this is a single-process batch job
    # with no UI generation competing for the model.
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
    # Rows are only written after the whole run, so an interrupted run leaves the previous file
    # untouched until this point; a crash during the loop above loses that run's rows.
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} STaR SFT row(s) to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
