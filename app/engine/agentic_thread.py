# ==============================================================================
# Program: agentic_thread.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Asynchronous QThread executing autonomous multi-turn loops.
#          Coordinates repeated model invocations, applies hot-reloaded agentic
#          loop logic, and streams intermediate reasoning to the UI.
# ==============================================================================

"""Autonomous multi-iteration generation worker (AgenticThread).

Each iteration runs one full generation (same streaming/parsing machinery as
llm_thread.LLMThread), then asks the hot-reloadable core.agentic_loop whether to
continue (should_continue) and what prompt to feed next (build_next_prompt),
until it says stop, the iteration cap is hit, or the user cancels. Many helper
functions here intentionally mirror llm_thread.py (see Key Gotchas in AGENTS.md:
a streaming-parser fix must be applied to both files).
"""
import json
import logging
import time
import re
import threading
from difflib import SequenceMatcher
from datetime import datetime, timezone
from PyQt6.QtCore import QThread, pyqtSignal
from app.engine.hot_reload import compile_and_reload
from app.engine.model_loader import CircuitBreakerOpenException, ModelLoader
from app.engine.kv_cache import kv_cache_stats, log_cache_stats
from app.engine.event_broker import EventBroker
from app.engine.task_supervisor import TaskSupervisor
from app.engine.streaming_parser import StreamingThoughtParser
from app.utils.compactor import decompact_trace
from app.utils.trace_logger import TraceLogger
import core.interaction_loop
import core.agentic_loop

# Raw per-token archive directory (same location as llm_thread).
RAW_LOG_DIR = "data/logs/raw"

# Reserve this many tokens for the next generation's output.
# The rest of the budget is used for history.
_RESPONSE_RESERVE = 1024  # max_tokens headroom
# Stall watchdog: abort if no token arrives for this long.
_WATCHDOG_TIMEOUT_SECONDS = 30.0
_WATCHDOG_ERROR = (
    "Inference Watchdog Timeout: Token generation froze for more than 30s. "
    "Inference terminated safely."
)

# Module logger.
logger = logging.getLogger("karl.agentic_thread")

# Set to True on Stage 3 emergency; cleared when GPU cools below 85°C.
_thermal_suspended: bool = False


def _clamp_max_tokens_to_context(
    configured_max_tokens: int, current_prompt_tokens: int, context_budget: int,
    safety_margin: int = 8,
) -> int:
    """Identical to app/engine/llm_thread.py's helper of the same name -- see
    that file for the full rationale. Kept duplicated per this file's
    existing convention of mirroring llm_thread.py's per-generation helpers."""
    # Step 1: Validate input parameters, operational preconditions, and state
    remaining = context_budget - current_prompt_tokens - safety_margin
    return max(0, min(configured_max_tokens, remaining))


def _trim_memory() -> None:
    """Return freed glibc heap arenas to the OS after a generation.

    Identical fix to app/engine/llm_thread.py's _trim_memory() -- see that
    docstring for the full rationale (fixed-size llama.cpp buffers, glibc
    arena fragmentation from per-call token/numpy churn, and a live OOM kill
    this was reproduced against). Kept duplicated per this file's existing
    convention of mirroring llm_thread.py's per-generation helpers.
    """
    # Step 1: Validate input parameters, operational preconditions, and state
    try:
        # Execute guarded operations and establish recovery boundaries
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _get_gpu_temp() -> float | None:
    """Return the hottest GPU temp across all NVML-monitored GPUs, or None."""
    # Step 1: Validate input parameters, operational preconditions, and state
    try:
        # Execute guarded operations and establish recovery boundaries
        from core.hardware_scout import get_hardware_profile
        temps = [
            g["temperature_c"]
            for g in get_hardware_profile().get("gpu_list", [])
            if "temperature_c" in g
        ]
        return max(temps) if temps else None
    except Exception:
        return None


# True when `text` starts with "{" and its braces balance (string/escape aware):
# detects the end of a compact "Machine Speak" JSON trace while streaming.
def _json_object_complete(text: str) -> bool:
    # Step 1: Validate input parameters, operational preconditions, and state
    depth = 0
    in_string = False
    escaped = False
    started = False
    for ch in text.lstrip():
        # Iterate over items, evaluate loop invariant, and update progress
        if not started:
            if ch != "{":
                return False
            started = True
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return True
    return False


# Schema marker for the compact trace format (keys: "tk" thinking, "r" response).
_COMPACT_TRACE_MARKER = "karl.trace.compact.v1"


def _salvage_compact_trace_fields(buffer: str) -> tuple[str, str]:
    """Best-effort extraction of thinking/response text from a buffer that
    looked like a complete karl.trace.compact.v1 JSON object (balanced
    braces) but failed strict json.loads() -- typically because a small
    model wrote prose inside the JSON string values without escaping
    embedded quotes. Regex-extracts the compact ("tk"/"r") or canonical
    ("thinking"/"response") field values directly so a malformed trace
    doesn't leak raw JSON scaffolding to the user as the chat response."""

    # Step 1: Validate input parameters, operational preconditions, and state
    # Local helper: regex-extract one JSON string value, decoding escapes when possible.
    def _extract(key: str) -> str:
        # Step 1: Validate input parameters, operational preconditions, and state
        match = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"', buffer)
        if not match:
            return ""
        raw = match.group(1)
        try:
            # Execute guarded operations and establish recovery boundaries
            return json.loads(f'"{raw}"')
        except Exception:
            return raw.replace('\\"', '"').replace("\\n", "\n")

    response = _extract("r") or _extract("response")
    thought = _extract("tk") or _extract("thinking")
    return thought, response


# Decode a compact trace and push its thought/response to the parser's emitters;
# returns False when the buffer isn't a trace (caller falls back to raw text).
def _emit_compacted_generation(buffer: str, parser: StreamingThoughtParser) -> bool:
    # Step 1: Validate input parameters, operational preconditions, and state
    stripped = buffer.strip()
    try:
        # Execute guarded operations and establish recovery boundaries
        trace = decompact_trace(stripped)
    except Exception as exc:
        if _COMPACT_TRACE_MARKER not in stripped:
            # Not actually an attempted compact trace, just coincidental
            # JSON-like output (e.g. a real JSON answer the user asked for).
            # Let the caller fall back to its normal raw passthrough.
            return False
        logger.warning(
            "Machine Speak: failed to decompact a %s buffer (%s) -- "
            "attempting field salvage instead of leaking raw JSON to the "
            "chat response.", _COMPACT_TRACE_MARKER, exc,
        )
        thought, response = _salvage_compact_trace_fields(stripped)
        if not (thought or response):
            logger.warning(
                "Machine Speak: salvage found no usable thinking/response "
                "fields in a malformed compact trace buffer; falling back "
                "to raw passthrough."
            )
            return False
        if thought:
            parser._emit_thought(thought)
        if response:
            parser._emit_chat(response)
        parser.in_thought = False
        parser.buffer = ""
        return True

    thought = str(trace.get("thinking") or trace.get("tk") or "")
    response = str(trace.get("response") or trace.get("r") or "")
    if thought:
        parser._emit_thought(thought)
    if response:
        parser._emit_chat(response)
    parser.in_thought = False
    parser.buffer = ""
    return bool(thought or response)


class AgenticThread(QThread):
    """
    Runs Karl in autonomous agentic mode.
    It loops: generate → parse → check stop condition → inject next prompt → repeat.
    The hackable stop condition and next-prompt logic live in core/agentic_loop.py.
    """
    # Signals mirror LLMThread's, plus per-iteration progress. All are emitted from
    # the worker thread and delivered to the GUI thread by Qt.
    new_thought_token = pyqtSignal(str)
    new_chat_token = pyqtSignal(str)
    new_raw_token = pyqtSignal(str)                  # M7: every character pre-parser
    # iteration_index, thought, response, diagnostics
    iteration_finished = pyqtSignal(int, str, str, dict)
    live_stats = pyqtSignal(int, float)
    loop_finished = pyqtSignal(int)
    reload_notice = pyqtSignal(str)   # module name that was hot-reloaded
    error_occurred = pyqtSignal(str)
    context_stats = pyqtSignal(int, int, int, int)  # prompt_tokens, history_tokens, rag_tokens, budget
    status_update = pyqtSignal(str, bool)           # (text, active)

    def __init__(self, system_prompt, initial_history, hyperparams,
                 retrieved_chunks=None, workflow="general_chat",
                 template="reasoning_minimal", adapter_name=None, model_name=None):
        """Create an autonomous generation loop worker.

        Args:
            system_prompt: Base system prompt for each iteration.
            initial_history: Starting role/content messages copied into the loop.
            hyperparams: Inference parameters and optional watchdog timeout.
            retrieved_chunks: Optional RAG chunks injected into prompts.
            workflow: Trace workflow name.
            template: Trace prompt template name.
            adapter_name: Optional LoRA adapter for ModelLoader.
            model_name: Optional display/trace model name override.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__()
        self.system_prompt = system_prompt
        # Copy so appending assistant turns / injected prompts never mutates the caller's list.
        self.chat_history = list(initial_history)   # copy so we can mutate safely
        self.hyperparams = hyperparams
        self.retrieved_chunks = retrieved_chunks or []
        self.workflow = workflow
        self.template = template
        self.adapter_name = adapter_name
        self.model_name = model_name
        self.logger = TraceLogger()
        self._stop_requested = False
        self.watchdog_timeout_seconds = float(
            self.hyperparams.get("watchdog_timeout_seconds", _WATCHDOG_TIMEOUT_SECONDS)
        )
        self.last_token_timestamp = time.time()
        self._watchdog_timed_out = False
        self._watchdog_error_emitted = False
        self._watchdog_stop = threading.Event()
        self._active_response_generator = None
        self.task_id: str | None = None

    def request_stop(self):
        """Request cooperative cancellation and stop the watchdog."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._stop_requested = True
        self._watchdog_stop.set()

    # Watchdog and token-counting helpers below behave exactly like their llm_thread twins.
    def _mark_token_activity(self):
        # Step 1: Validate input parameters, operational preconditions, and state
        self.last_token_timestamp = time.time()

    def _emit_watchdog_timeout(self):
        # Step 1: Validate input parameters, operational preconditions, and state
        if self._watchdog_error_emitted:
            return
        self._watchdog_error_emitted = True
        self.error_occurred.emit(_WATCHDOG_ERROR)

    def _cleanup_after_watchdog_timeout(self, llm=None):
        # Step 1: Validate input parameters, operational preconditions, and state
        self._stop_requested = True
        self._watchdog_timed_out = True
        self._emit_watchdog_timeout()
        if self.task_id:
            try:
                # Execute guarded operations and establish recovery boundaries
                TaskSupervisor.instance().fail(self.task_id, _WATCHDOG_ERROR)
            except Exception:
                pass
        generator = self._active_response_generator
        close_fn = getattr(generator, "close", None)
        if callable(close_fn):
            try:
                # Execute guarded operations and establish recovery boundaries
                close_fn()
            except Exception:
                pass
        reset_fn = getattr(llm, "reset", None)
        if callable(reset_fn):
            try:
                # Execute guarded operations and establish recovery boundaries
                reset_fn()
            except Exception:
                pass

    def _start_watchdog(self, llm=None) -> threading.Thread:
        # Step 1: Validate input parameters, operational preconditions, and state
        self._watchdog_stop.clear()
        self._watchdog_timed_out = False
        self._watchdog_error_emitted = False
        self._mark_token_activity()

        def _monitor():
            # Step 1: Validate input parameters, operational preconditions, and state
            while not self._watchdog_stop.wait(0.25):
                # Iterate over items, evaluate loop invariant, and update progress
                if self._stop_requested:
                    return
                if time.time() - self.last_token_timestamp > self.watchdog_timeout_seconds:
                    self._cleanup_after_watchdog_timeout(llm)
                    return

        thread = threading.Thread(target=_monitor, name="karl-agentic-watchdog", daemon=True)
        thread.start()
        return thread

    def _stop_watchdog(self, thread: threading.Thread | None):
        # Step 1: Validate input parameters, operational preconditions, and state
        self._watchdog_stop.set()
        if thread and thread.is_alive():
            thread.join(timeout=1.0)

    def _token_count(self, llm, text: str) -> int:
        # Step 1: Validate input parameters, operational preconditions, and state
        if not text:
            return 0
        try:
            # Execute guarded operations and establish recovery boundaries
            return len(llm.tokenize(text.encode("utf-8"), add_bos=False))
        except TypeError:
            return len(llm.tokenize(text.encode("utf-8")))
        except Exception as exc:
            logger.warning("Tokenizer count failed; falling back to character estimate: %s", exc)
            return max(1, len(text) // 3)

    def _message_token_count(self, llm, msg) -> int:
        # Step 1: Validate input parameters, operational preconditions, and state
        role = msg.get("role", "")
        content = msg.get("content", "")
        return self._token_count(llm, f"<|im_start|>{role}\n{content}<|im_end|>\n")

    # Remove <think> blocks from old assistant turns (never the newest) until the
    # history fits the token budget.
    def _strip_historical_thoughts(self, history, llm, budget):
        """
        Iterates through history (oldest to newest) and strips <think> blocks
        from assistant turns until the total token count fits the budget.
        Always preserves the very last assistant turn's thoughts if it exists.
        """
        # Find all assistant turns with think blocks, except the most recent one
        assistant_indices = []
        for i, msg in enumerate(history):
            # Iterate over items, evaluate loop invariant, and update progress
            if msg.get("role") == "assistant" and "<think>" in msg.get("content", "") and "</think>" in msg.get("content", ""):
                assistant_indices.append(i)
        
        if not assistant_indices:
            return history
            
        # Don't strip the most recent one to maintain active reasoning continuity
        targets = assistant_indices[:-1]
        
        optimized_history = list(history)
        
        for idx in targets:
            # Check current total
            total = sum(self._message_token_count(llm, m) for m in optimized_history)
            if total <= budget:
                break
                
            content = optimized_history[idx]["content"]
            # Strip <think>...</think> including the tags
            new_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            optimized_history[idx] = {**optimized_history[idx], "content": new_content}
            
        return optimized_history

    def _trim_history(self, history, system_prompt, llm):
        """
        Trims the chat history so the compiled prompt stays inside the context budget.
        
        1. Cognitive Context Pruning: strips <think> blocks from old assistant turns
        2. Standard Trimming: drops oldest messages until history fits in the budget
        
        Always keeps the first user message (the seed) and the most recent turns.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        # History budget = window - reply reserve - system prompt tokens (floor 256).
        context_budget = ModelLoader.context_limit()
        base_tokens = self._token_count(llm, system_prompt)
        budget = max(256, context_budget - _RESPONSE_RESERVE - base_tokens)

        # ── Phase 1: Cognitive Context Pruning ───────────────────────────────
        # Strip older thoughts first to reclaim space while keeping dialogue turns
        processed = self._strip_historical_thoughts(history, llm, budget)

        # ── Phase 2: Standard Trimming ───────────────────────────────────────
        # Walk backwards through history accumulating tokenizer-measured size.
        # Always keep at least the first message (index 0) as the seed.
        kept = []
        running = 0
        for msg in reversed(processed):
            # Iterate over items, evaluate loop invariant, and update progress
            entry_len = self._message_token_count(llm, msg)
            if running + entry_len > budget and kept:
                break  # Stop adding older messages
            kept.insert(0, msg)
            running += entry_len

        # Always preserve seed (first message)
        if history and history[0] not in kept:
            kept.insert(0, history[0])

        # Tell the user in the thought stream when older turns were dropped.
        if len(kept) < len(history):
            self.new_thought_token.emit(
                f"\n[Context Trim: kept {len(kept)}/{len(history)} messages to fit context window]\n"
            )
        return kept

    def _response_reserve(self) -> int:
        # Step 1: Validate input parameters, operational preconditions, and state
        try:
            # Execute guarded operations and establish recovery boundaries
            max_tokens = int(self.hyperparams.get("max_tokens", _RESPONSE_RESERVE))
        except Exception:
            max_tokens = _RESPONSE_RESERVE
        return min(max(128, max_tokens), _RESPONSE_RESERVE)

    def _build_prompt_with_context_budget(self, llm, system_prompt: str, history: list[dict]):
        """Compile the prompt and enforce a final tokenizer-measured budget."""
        # Step 1: Validate input parameters, operational preconditions, and state
        budget = max(256, ModelLoader.context_limit() - self._response_reserve())
        working_history = [dict(m) for m in history]
        working_system = system_prompt
        changed = False

        def _compile():
            # Step 1: Validate input parameters, operational preconditions, and state
            core.interaction_loop.ACTIVE_HYPERPARAMS = self.hyperparams
            prompt_text = core.interaction_loop.build_prompt(working_system, working_history)
            return prompt_text, self._token_count(llm, prompt_text)

        prompt, tokens = _compile()

        while working_history and tokens > budget:
            # Iterate over items, evaluate loop invariant, and update progress
            if len(working_history) > 1:
                working_history.pop(0)
                changed = True
            else:
                content = working_history[0].get("content", "")
                if len(content) <= 256:
                    break
                keep = max(256, len(content) // 2)
                working_history[0] = {
                    **working_history[0],
                    "content": "[...truncated to fit context...]\n" + content[-keep:],
                }
                changed = True
            prompt, tokens = _compile()

        while tokens > budget and len(working_system) > 512:
            # Iterate over items, evaluate loop invariant, and update progress
            keep = max(512, len(working_system) // 2)
            working_system = (
                working_system[:keep]
                + "\n\n[System/RAG context truncated to fit model context.]"
            )
            changed = True
            prompt, tokens = _compile()

        if changed:
            self.new_thought_token.emit(
                "\n[Context budget: prompt was reduced to fit the loaded model window.]\n"
            )

        return prompt, working_history, working_system, tokens, budget

    def _is_repetitive_response(self, previous: str, current: str) -> bool:
        # Step 1: Validate input parameters, operational preconditions, and state
        # Normalize whitespace/case, then treat two long (>=80 chars) responses that are
        # >=96% similar as a loop: the model is repeating itself, so the loop should end.
        prev = " ".join((previous or "").split()).lower()
        curr = " ".join((current or "").split()).lower()
        if len(prev) < 80 or len(curr) < 80:
            return False
        return SequenceMatcher(None, prev, curr).ratio() >= 0.96

    def _run_single_generation(self, llm, prompt, raw_file, thermal_enabled: bool = False):
        """Runs streaming generation, continuing automatically if truncated."""
        # Step 1: Validate input parameters, operational preconditions, and state
        global _thermal_suspended
        raw_output = ""
        # Determine starting mode for the streaming parser.
        # If the compiled prompt ends with <think> (ignoring trailing whitespace/newlines),
        # we pre-seeded it, so we start in thought mode.
        in_thought = prompt.rstrip().endswith("<think>")
        parser = StreamingThoughtParser(
            in_thought=in_thought,
            thought_cb=self.new_thought_token.emit,
            chat_cb=self.new_chat_token.emit,
        )
        first_token_time = None
        gen_token_count = 0
        compact_stream_mode: bool | None = None
        compact_buffer = ""
        compact_emitted = False

        # Same continuation / compression guards as LLMThread (see that file for details).
        continuation_count = 0
        max_continuations = 5
        compression_reset_count = 0
        max_compression_resets = 2
        state_transitioned = False

        # Snapshot cache state and wall-clock start before the first generation.
        _init_cache_stats = kv_cache_stats(llm, prompt)
        _gen_start = time.perf_counter()

        # One streaming llama.cpp call per pass; passes repeat on length stops (continuation)
        # or when the sampling profile switches after </think>.
        while continuation_count <= max_continuations:
            # Stage 3 emergency check at each continuation boundary
            if thermal_enabled:
                _ct = _get_gpu_temp()
                if _ct is not None and _ct >= 100.0:
                    _thermal_suspended = True
                    raise RuntimeError(
                        f"Emergency Thermal Suspension: GPU reached {_ct:.1f}°C."
                    )

            # Dynamic Parameter Scheduling: exploratory temperature while thinking, near-greedy
            # while answering.
            # Dynamic Parameter Scheduling
            current_temp = self.hyperparams.get("temperature", 0.7)
            current_top_p = self.hyperparams.get("top_p", 0.95)
            
            if self.hyperparams.get("enable_dynamic_scheduling", True):
                if parser.in_thought:
                    current_temp = self.hyperparams.get("thinking_temperature", 0.8)
                else:
                    current_temp = self.hyperparams.get("answering_temperature", 0.1)
                    current_top_p = 0.1 # High precision for answering
            speculative_kwargs = ModelLoader.speculative_generation_kwargs()

            # Clamp max_tokens to what's actually left in the context window.
            # Identical fix to app/engine/llm_thread.py's mirrored loop -- see
            # that comment for the full rationale (growing raw_output across
            # continuation passes vs. a fixed requested max_tokens used to
            # let llama.cpp hard-fail with "Requested tokens (N) exceed
            # context window of N" instead of finishing gracefully).
            configured_max_tokens = self.hyperparams.get("max_tokens", 2048)
            current_prompt_tokens = len(llm.tokenize((prompt + raw_output).encode('utf-8')))
            effective_max_tokens = _clamp_max_tokens_to_context(
                configured_max_tokens, current_prompt_tokens, ModelLoader.context_limit(),
            )
            if effective_max_tokens < configured_max_tokens and effective_max_tokens < 16:
                logger.info(
                    "Context budget exhausted after %d continuation pass(es) "
                    "(%d/%d tokens used); stopping instead of overflowing.",
                    continuation_count, current_prompt_tokens, ModelLoader.context_limit(),
                )
                break

            response_gen = llm(
                prompt + raw_output,
                max_tokens=effective_max_tokens,
                temperature=current_temp,
                top_p=current_top_p,
                repeat_penalty=1.1,
                stream=True,
                stop=["<|im_end|>", "<|endoftext|>", "<|end_of_text|>", "<|im_start|>"],
                echo=False,
                **speculative_kwargs,
            )
            self._active_response_generator = response_gen
            watchdog_thread = self._start_watchdog(llm)

            finish_reason = "stop"
            has_tokens = False
            state_transitioned = False

            try:
                # Execute guarded operations and establish recovery boundaries
                for chunk in response_gen:
                    # Iterate over items, evaluate loop invariant, and update progress
                    if self._watchdog_timed_out or self._stop_requested:
                        break
                    if 'choices' not in chunk or not chunk['choices']:
                        continue

                    choice = chunk['choices'][0]
                    fr = choice.get('finish_reason')
                    if fr is not None:
                        finish_reason = fr

                    text = choice.get('text', '')
                    if not text:
                        continue

                    self._mark_token_activity()

                    if first_token_time is None:
                        first_token_time = time.perf_counter()

                    chunk_tokens = len(llm.tokenize(text.encode('utf-8'), add_bos=False))
                    gen_token_count += chunk_tokens
                    elapsed = time.perf_counter() - first_token_time
                    speed = gen_token_count / elapsed if elapsed > 0 else 0.0
                    self.live_stats.emit(gen_token_count, speed)

                    has_tokens = True
                    raw_output += text

                    # M7: write raw token before parsing (raw archive survives parser bugs/crashes).
                    # M7: write raw token before parsing
                    micro_ts = f"{time.time():.6f}"
                    raw_file.write(f"{micro_ts}\t{text}\n")
                    raw_file.flush()
                    self.new_raw_token.emit(text)
                    EventBroker.get_instance().publish("tokens:raw", {"token": text})

                    # Compact-trace ("Machine Speak") detection, as in LLMThread.
                    if compact_stream_mode is None:
                        stripped_raw = raw_output.lstrip()
                        if not stripped_raw:
                            continue
                        if stripped_raw.startswith("{"):
                            compact_stream_mode = True
                            compact_buffer = raw_output
                        else:
                            compact_stream_mode = False
                    elif compact_stream_mode:
                        compact_buffer += text

                    if compact_stream_mode:
                        if _json_object_complete(compact_buffer):
                            compact_emitted = _emit_compacted_generation(compact_buffer, parser)
                            if not compact_emitted:
                                parser.feed(compact_buffer, defer_after_think_close=False)
                            compact_buffer = ""
                            compact_stream_mode = False
                        continue

                    dynamic_scheduling = self.hyperparams.get("enable_dynamic_scheduling", True)
                    parse_result = parser.feed(
                        text,
                        defer_after_think_close=dynamic_scheduling,
                    )
                    if parse_result.closed_think and dynamic_scheduling:
                        logger.info("Agentic Dynamic Scheduler: detected </think>, switching to ANSWERING profile")
                        state_transitioned = True
                        break
            finally:
                self._stop_watchdog(watchdog_thread)
                _close_fn = getattr(response_gen, "close", None)
                if callable(_close_fn):
                    _close_fn()
                self._active_response_generator = None

            # After the stream: watchdog abort or user stop ends this generation; a profile
            # switch re-enters the loop; only a length stop with output continues.
            if self._watchdog_timed_out:
                break

            if self._stop_requested:
                break

            if state_transitioned:
                continue

            if finish_reason != "length" or not has_tokens:
                break

            # Check if we need to do cognitive roll-up compression
            current_tokens_count = len(llm.tokenize((prompt + raw_output).encode('utf-8')))
            context_budget = ModelLoader.context_limit()
            
            compressed = False
            if current_tokens_count > 0.8 * context_budget and parser.parsed_thought:
                self.new_thought_token.emit("\n[Context full — compressing cognitive state...]\n")
                try:
                    # Execute guarded operations and establish recovery boundaries
                    compress_prompt = (
                        "<|im_start|>system\n"
                        "You are a cognitive compression tool. Summarize the reasoning steps, calculations, "
                        "and findings established in the thinking process below. Write a dense, concise summary "
                        "in one short paragraph so that the reasoning can proceed from that point. "
                        "Do not solve the problem yourself, just summarize the current train of thought.\n"
                        f"Thinking process to summarize:\n{parser.parsed_thought}\n<|im_end|>\n"
                        "<|im_start|>assistant\n"
                        "<think>\n"
                    )
                    comp_res = llm(
                        compress_prompt,
                        max_tokens=300,
                        temperature=0.1,
                        stop=["</think>", "<|im_end|>"],
                        echo=False
                    )
                    summary = comp_res['choices'][0]['text'].strip()
                    summary = summary.replace("<think>", "").replace("</think>", "")
                    
                    parser.parsed_thought = f"[Summary of thoughts so far: {summary}]"
                    parser.buffer = ""
                    raw_output = f"<think>\n{parser.parsed_thought}\n"
                    parser.in_thought = True # Resume in thought mode
                    compressed = True
                    self.new_thought_token.emit(f"\n[Compressed state: {parser.parsed_thought}]\n")
                except Exception as ce:
                    logger.warning("Cognitive compression failed: %s", ce)
                    self.new_thought_token.emit("\n[continuing...]\n")
            else:
                self.new_thought_token.emit("\n[continuing...]\n")
            
            if compressed and compression_reset_count < max_compression_resets:
                compression_reset_count += 1
                continuation_count = 0
            else:
                if compressed:
                    self.new_thought_token.emit("\n[Compression reset limit reached — continuing with normal continuation budget.]\n")
                continuation_count += 1

        # flush remainder
        if compact_stream_mode:
            compact_emitted = _emit_compacted_generation(compact_buffer, parser)
            if not compact_emitted:
                parser.feed(compact_buffer, defer_after_think_close=False)
        if not compact_emitted:
            parser.flush()

        # Log KV-cache hit statistics now that we have TTFT.
        _ttft_ms = (
            round((first_token_time - _gen_start) * 1000, 2)
            if first_token_time is not None
            else None
        )
        _ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        log_cache_stats({**_init_cache_stats, "ttft_ms": _ttft_ms}, _ts)

        return raw_output, parser.parsed_thought, parser.parsed_response, first_token_time

    # Returns (raw_output, thought, response, first_token_time) for one full iteration.
    def run(self):
        """Run iterative generation until stop condition, max iterations, or error.

        CircuitBreakerOpenException is converted into error_occurred with the
        operator-facing circuit-breaker message.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        # Register with the supervisor so the loop can be listed and cancelled.
        supervisor = TaskSupervisor.instance()
        if self.task_id is None:
            self.task_id = supervisor.register("Agentic loop", cancellable=self)

        # ── CPU Core Pinning ──────────────────────────────────────────────────
        # Pin inference to physical cores to optimize TPS. os.sched_setaffinity(0, ...)
        # targets the calling thread specifically, preventing process-wide races.
        original_affinity = None
        lock_held = False  # set once acquire_instance() has taken the inference lock
        try:
            # Execute guarded operations and establish recovery boundaries
            import os
            import psutil
            if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
                original_affinity = os.sched_getaffinity(0)
                p_count = psutil.cpu_count(logical=False)
                if p_count:
                    physical_cores = list(range(p_count))
                    os.sched_setaffinity(0, physical_cores)
                    logger.debug(f"AgenticThread pinned to physical cores: {physical_cores}")
        except Exception as e:
            logger.warning(f"Could not set CPU affinity: {e}")

        # Hot-reload the user-editable prompt/loop modules so edits apply to this run.
        try:
            # Execute guarded operations and establish recovery boundaries
            core.interaction_loop = compile_and_reload(
                core.interaction_loop,
                "core/interaction_loop.py",
                self.reload_notice.emit,
                logger,
            )
            core.agentic_loop = compile_and_reload(
                core.agentic_loop,
                "core/agentic_loop.py",
                self.reload_notice.emit,
                logger,
            )

            # Optional per-run base model (custom agent) overrides the active selection.
            model_path = None
            if self.model_name:
                model_path = os.path.join("data", "models", self.model_name)
            llm = ModelLoader.acquire_instance(model_path=model_path, adapter_name=self.adapter_name)
            lock_held = True
            
            # Fetch the actual (potentially fallback-scaled) context limit
            actual_context_budget = ModelLoader.context_limit()

            # Main loop: one iteration = generate -> log -> store response -> ask
            # core.agentic_loop.should_continue() -> inject build_next_prompt() as the next user turn.
            iteration = 0
            # Only the previous iteration is remembered: the repetition check below looks for a loop of
            # period one (the model answering the same thing twice), not longer cycles.
            previous_response = ""

            global _thermal_suspended
            while not self._stop_requested:
                # Thermal gate is re-checked every iteration (config may change mid-run).
                # ── Thermal config & hysteresis gate ─────────────────────────────
                from app.engine import config_store
                cfg = config_store.get_ui_config()
                thermal_enabled = cfg.get("thermal_protection_enabled", True)

                if thermal_enabled and _thermal_suspended:
                    t = _get_gpu_temp()
                    if t is None or t >= 85.0:
                        raise RuntimeError(
                            f"GPU thermal suspension active ({t}°C). "
                            "Generation blocked until GPU cools below 85°C."
                        )
                    _thermal_suspended = False
                # ─────────────────────────────────────────────────────────────────

                # Build and emit iteration header
                # Iteration numbers shown to the user are 1-based, while the `iteration` counter passed to
                # signals and the loop module stays 0-based, matching the raw-log file names.
                self.new_thought_token.emit(f"\n{'='*40}\n[AGENTIC LOOP — Iteration {iteration + 1}]\n{'='*40}\n")
                self.new_chat_token.emit(f"\n[Iteration {iteration + 1}]\n")

                # Inject RAG context into system prompt if present
                # Rebuilt from the original every iteration so the RAG text is injected exactly once and
                # never accumulates across loop turns.
                system_prompt = self.system_prompt
                context_str = "(No context retrieved.)"
                if self.retrieved_chunks:
                    context_str = "\n".join(self.retrieved_chunks)
                    if "{rag_context}" in system_prompt:
                        system_prompt = system_prompt.replace("{rag_context}", context_str)
                    else:
                        system_prompt += "\n\nRetrieved Context:\n" + context_str

                # Trim history to fit context before building prompt
                # Trimming returns a copy: self.chat_history keeps the full transcript for the UI/logs while
                # only the trimmed view is sent to the model.
                trimmed_history = self._trim_history(self.chat_history, system_prompt, llm)
                prompt, trimmed_history, system_prompt, prompt_tokens, _prompt_budget = (
                    self._build_prompt_with_context_budget(llm, system_prompt, trimmed_history)
                )

                # Emit context budget stats for the HUD
                try:
                    # Execute guarded operations and establish recovery boundaries
                    # The HUD stats are best-effort; a tokenizer error here must never abort generation, hence
                    # the broad except below.
                    hist_tokens = sum(self._message_token_count(llm, m) for m in trimmed_history)
                    rag_tokens  = self._token_count(llm, context_str) if self.retrieved_chunks else 0
                    sys_tokens  = self._token_count(llm, system_prompt)
                    self.context_stats.emit(sys_tokens + hist_tokens + rag_tokens, hist_tokens, rag_tokens, actual_context_budget)
                except Exception:
                    pass

                # Per-iteration timing: prefill = until the first token, generation = after it.
                # perf_counter is monotonic, so these timings survive wall-clock changes (NTP, sleep).
                start = time.perf_counter()
                # M7: open raw archive file for this iteration
                # One raw token file per iteration, named with a microsecond timestamp, so concurrent or
                # rapid iterations never overwrite each other and a crash still leaves the partial output.
                os.makedirs(RAW_LOG_DIR, exist_ok=True)
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                raw_log_path = os.path.join(RAW_LOG_DIR, f"agentic_{ts}_iter{iteration}.tokens")
                with open(raw_log_path, "w", encoding="utf-8") as raw_file:
                    raw, thought, response, first_token_time = self._run_single_generation(
                        llm, prompt, raw_file, thermal_enabled=thermal_enabled
                    )
                end = time.perf_counter()

                prefill_time = (first_token_time - start) if first_token_time is not None else (end - start)
                # Floor avoids ZeroDivisionError when the first token arrives in the same clock tick.
                if prefill_time <= 0:
                    prefill_time = 0.001
                prefill_tps = prompt_tokens / prefill_time

                generation_time = (end - first_token_time) if first_token_time is not None else 0.0
                # Counted by re-tokenizing the raw text rather than counting streamed chunks, because a
                # chunk is not always exactly one token.
                generation_tokens = len(llm.tokenize(raw.encode('utf-8'), add_bos=False))
                if generation_time <= 0:
                    generation_tps = generation_tokens / 0.001 if generation_tokens > 0 else 0.0
                else:
                    generation_tps = generation_tokens / generation_time

                total_time = end - start
                total_tokens = prompt_tokens + generation_tokens
                total_tps = total_tokens / total_time if total_time > 0 else 0.0

                # Per-iteration diagnostics, forwarded with iteration_finished and into the trace log.
                diagnostics = {
                    "prompt_tokens": prompt_tokens,
                    "prefill_time": prefill_time,
                    "prefill_tps": prefill_tps,
                    "generation_tokens": generation_tokens,
                    "generation_time": generation_time,
                    "generation_tps": generation_tps,
                    "total_time": total_time,
                    "total_tps": total_tps,
                }

                # A stop request discards this iteration's results (nothing is logged or emitted).
                if self._stop_requested:
                    break

                # Temperature is sampled after generation, so it reflects the load this iteration caused.
                gpu_temp = _get_gpu_temp()
                throttle_reasons = []
                try:
                    # Execute guarded operations and establish recovery boundaries
                    from core.hardware_scout import get_hardware_profile
                    for gpu in get_hardware_profile().get("gpu_list", []):
                        # Iterate over items, evaluate loop invariant, and update progress
                        for alert in gpu.get("alerts", []):
                            # Iterate over items, evaluate loop invariant, and update progress
                            # De-duplicated so a multi-GPU box with the same alert on every card logs it once.
                            if alert not in throttle_reasons:
                                throttle_reasons.append(alert)
                except Exception:
                    pass

                # Recorded in the trace log so slow iterations can be explained by deliberate cooling pauses
                # rather than a slow model.
                cooling_duration = 0.0
                # Post-iteration cooldown: >=98 C waits 15 s, >=95 C waits 5 s; >=100 C aborts.
                # ── Multi-Stage Thermal Cooldown ──────────────────────────────────
                if thermal_enabled and gpu_temp is not None:
                    cooling_start = time.perf_counter()
                    if gpu_temp >= 100.0:
                        _thermal_suspended = True
                        raise RuntimeError(
                            f"Emergency Thermal Suspension: GPU reached {gpu_temp:.1f}°C."
                        )
                    elif gpu_temp >= 98.0:
                        self.status_update.emit(
                            f"GPU temperature critical ({gpu_temp:.1f}°C). Cooling down for 15s...", True
                        )
                        # 150 x 0.1 s = the 15 s cooldown, in short sleeps so a stop request is honoured within
                        # ~0.1 s instead of after the whole wait.
                        for _ in range(150):
                            # Iterate over items, evaluate loop invariant, and update progress
                            if self._stop_requested:
                                break
                            time.sleep(0.1)
                        cooling_duration = time.perf_counter() - cooling_start
                        self.status_update.emit("idle", False)
                    elif gpu_temp >= 95.0:
                        self.status_update.emit(
                            f"GPU temperature hot ({gpu_temp:.1f}°C). Cooling down for 5s...", True
                        )
                        # 50 x 0.1 s = 5 s, again polling the stop flag.
                        for _ in range(50):
                            # Iterate over items, evaluate loop invariant, and update progress
                            if self._stop_requested:
                                break
                            time.sleep(0.1)
                        cooling_duration = time.perf_counter() - cooling_start
                        self.status_update.emit("idle", False)
                # ─────────────────────────────────────────────────────────────────

                # Log this iteration
                self.logger.log_generation(
                    compiled_prompt=prompt,
                    hyperparams=self.hyperparams,
                    raw_output=raw,
                    parsed_thought=thought,
                    parsed_response=response,
                    execution_time=total_time,
                    rag_context=self.retrieved_chunks,
                    model_name=ModelLoader.model_name(),
                    adapter_name=getattr(ModelLoader, '_active_adapter', None),
                    workflow=self.workflow,
                    template=self.template,
                    diagnostics=diagnostics,
                    gpu_temp_c=gpu_temp,
                    throttle_reasons=throttle_reasons,
                    cooling_duration_sec=cooling_duration,
                )

                # Store only the response — not the think block — to keep context lean
                # Only the visible response is kept: replaying earlier <think> blocks would burn context on
                # reasoning the model does not need to see again.
                self.chat_history.append({"role": "assistant", "content": response})

                self.iteration_finished.emit(iteration, thought, response, diagnostics)
                # Duplicate of the Qt signal above for non-UI consumers (websocket clients, watchers) that
                # cannot connect to Qt signals.
                EventBroker.get_instance().publish("iteration:finished", {
                    "iteration": iteration,
                    "thought": thought,
                    "response": response,
                    "diagnostics": diagnostics
                })

                # Incremented before the stop checks so should_continue() receives the number of COMPLETED
                # iterations, and loop_finished reports the same count.
                iteration += 1

                # Hot-reload stop condition and check it
                # Stop conditions, in order: should_continue() False, then a near-duplicate
                # response (the model is looping).
                # Hot-reload stop condition and check it
                core.agentic_loop = compile_and_reload(
                    core.agentic_loop,
                    "core/agentic_loop.py",
                    self.reload_notice.emit,
                    logger,
                )
                if not core.agentic_loop.should_continue(iteration, response):
                    break

                # Guards against a model that ignores build_next_prompt() and repeats itself until the
                # iteration cap; this ends the loop earlier and saves compute.
                if self._is_repetitive_response(previous_response, response):
                    self.new_thought_token.emit(
                        "\n[Agentic loop stopped: latest response repeated the previous iteration.]\n"
                    )
                    break
                previous_response = response

                # Build next user turn and inject it
                next_prompt_content = core.agentic_loop.build_next_prompt(response, iteration)
                # The follow-up is appended as a user turn so the next iteration sees a normal chat
                # alternation (user, assistant, user, ...).
                self.chat_history.append({"role": "user", "content": next_prompt_content})

            # Reached on normal completion and on a user stop; errors skip this and go to the except
            # handlers, which emit error_occurred and mark the supervisor task as failed instead.
            self.loop_finished.emit(iteration)
            EventBroker.get_instance().publish("loop:finished", {
                "total_iterations": iteration
            })
            supervisor.finish(self.task_id)

        except CircuitBreakerOpenException as e:
            self.error_occurred.emit(str(e))
            supervisor.fail(self.task_id, str(e))
        except Exception as e:
            self.error_occurred.emit(f"Agentic Error: {str(e)}")
            supervisor.fail(self.task_id, str(e))
        # Always release the inference lock, restore CPU affinity and trim heap memory.
        finally:
            # Skip the release if acquire_instance() raised before taking the lock.
            if lock_held:
                ModelLoader.unlock_instance()
            # Restore original CPU affinity
            if original_affinity:
                try:
                    # Execute guarded operations and establish recovery boundaries
                    import os
                    if hasattr(os, "sched_setaffinity"):
                        os.sched_setaffinity(0, original_affinity)
                except Exception as e:
                    logger.debug(f"Could not restore CPU affinity: {e}")
            _trim_memory()
