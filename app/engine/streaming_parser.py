# ==============================================================================
# Program: streaming_parser.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Real-time streaming token parser for conversational generation.
#          Separates <think> tags from main answer streams incrementally without
#          buffering entire responses.
# ==============================================================================

"""
Incremental parser that splits a streamed model reply into "thought" and "chat" text.

Reasoning models wrap their chain of thought in <think>...</think>. Tokens arrive
in tiny, arbitrarily split pieces, so StreamingThoughtParser buffers just enough
text to recognise those tags (and ChatML control tokens such as <|im_end|>) even
when a tag is split across several chunks, then routes each piece to the right
callback and, optionally, to the EventBroker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from app.engine.event_broker import EventBroker


# The reasoning delimiters used by DeepSeek-style models.
_THINK_TAGS = ["<think>", "</think>"]

# ChatML control tokens (stop sequences passed to llm(..., stop=[...]) in
# llm_thread.py/agentic_thread.py). llama.cpp streams generation a few bytes
# at a time, and these can arrive fragmented across many separate chunks --
# e.g. "<|im_end|>" observed live as five separate feed() calls: "<", "|",
# "im", "_end", ">\n". A regex over each individual chunk can never catch
# that (no single chunk contains the full pattern), so the guard mechanism
# below has to hold back anything that *might still become* one of these
# tokens, the same way it already holds back a partial "<think>"/"</think>",
# until the full token (or a clear divergence from it) is visible.
# Control tokens are never shown to the user; see the fragmentation note above.
_CHATML_CONTROL_TOKENS = ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "<|end_of_text|>"]

# Everything that must not be emitted half-seen.
_PROTECTED_SEQUENCES = _THINK_TAGS + _CHATML_CONTROL_TOKENS

# Every proper prefix of every protected sequence, e.g. for "<|im_end|>":
# "<", "<|", "<|i", "<|im", "<|im_", "<|im_e", "<|im_en", "<|im_end", "<|im_end|".
# If the buffer's tail is one of these, more incoming text could still turn
# it into a full protected sequence, so emission must wait.
# Sorted set of all proper prefixes; the sort only makes the value deterministic.
OPEN_GUARDS = sorted({seq[:i] for seq in _PROTECTED_SEQUENCES for i in range(1, len(seq))})
# Same set: the check is identical whether we are inside or outside a think block.
CLOSE_GUARDS = OPEN_GUARDS

# Complete (or truncated-by-one -- see the leak this was written for, which
# dropped exactly one character) forms of the control tokens, stripped from
# whatever's about to be emitted once the buffer has grown enough to no
# longer be a mere prefix.
# The optional "|" and ">" cover tokens truncated by a character, so a partial token is still removed.
_SPECIAL_TOKEN_LEAK_RE = re.compile(
    r"<\|(?:im_end|im_start|endoftext|end_of_text)\|?>?"
)


def _strip_special_token_leaks(text: str) -> str:
    """Remove any ChatML control token (complete or one char short) from ``text``."""
    return _SPECIAL_TOKEN_LEAK_RE.sub("", text)


@dataclass
class ParserFeedResult:
    """Result of one feed() call; closed_think is True if </think> was seen in that call."""
    closed_think: bool = False


class StreamingThoughtParser:
    """Incremental router for DeepSeek-style <think> streams."""

    def __init__(
        self,
        *,
        in_thought: bool,
        thought_cb: Callable[[str], None],
        chat_cb: Callable[[str], None],
        publish_events: bool = True,
    ):
        """``in_thought`` sets the starting mode (True when the prompt already opened <think>).

        ``thought_cb`` / ``chat_cb`` receive cleaned text. ``publish_events`` also mirrors
        each piece to EventBroker topics "tokens:thought" and "tokens:chat".
        """
        self.in_thought = in_thought
        self.buffer = ""
        self.parsed_thought = ""
        self.parsed_response = ""
        self._thought_cb = thought_cb
        self._chat_cb = chat_cb
        self._publish_events = publish_events

    def feed(self, text: str, defer_after_think_close: bool = False) -> ParserFeedResult:
        """Add a chunk of model output and emit whatever is safe to emit.

        If ``defer_after_think_close`` is True, return right after </think> without
        emitting the remainder, so the caller can act on the end of the thought first.
        """
        self.buffer += text
        closed_think = False

        # Entering a think block: text before the tag is normal chat output.
        if "<think>" in self.buffer and not self.in_thought:
            self.in_thought = True
            pre_think = self.buffer.split("<think>")[0]
            if pre_think:
                self._emit_chat(pre_think)
            self.buffer = self.buffer.split("<think>", 1)[1]

        # Leaving a think block: text before the tag is the last of the thought.
        if self.in_thought and "</think>" in self.buffer:
            self.in_thought = False
            thought, remainder = self.buffer.split("</think>", 1)
            if thought:
                self._emit_thought(thought)
            self.buffer = remainder
            closed_think = True
            if defer_after_think_close:
                return ParserFeedResult(closed_think=True)

        # Hold back the buffer only if it ends in something that could become a tag;
        # otherwise it is safe to emit now and clear.
        if self.in_thought:
            if not any(self.buffer.endswith(s) for s in CLOSE_GUARDS):
                self._emit_thought(self.buffer)
                self.buffer = ""
        else:
            if not any(self.buffer.endswith(s) for s in OPEN_GUARDS):
                self._emit_chat(self.buffer)
                self.buffer = ""

        return ParserFeedResult(closed_think=closed_think)

    def flush(self) -> None:
        """Emit any buffered text (generation is over, so no tag can still complete)."""
        if not self.buffer:
            return
        if self.in_thought:
            self._emit_thought(self.buffer)
        else:
            self._emit_chat(self.buffer)
        self.buffer = ""

    def _emit_thought(self, token: str) -> None:
        """Clean and deliver a piece of thought text; also accumulates parsed_thought."""
        token = _strip_special_token_leaks(token)
        if not token:
            return
        self._thought_cb(token)
        if self._publish_events:
            EventBroker.get_instance().publish("tokens:thought", {"token": token})
        self.parsed_thought += token

    def _emit_chat(self, token: str) -> None:
        """Clean and deliver a piece of answer text; also accumulates parsed_response."""
        token = _strip_special_token_leaks(token)
        if not token:
            return
        self._chat_cb(token)
        if self._publish_events:
            EventBroker.get_instance().publish("tokens:chat", {"token": token})
        self.parsed_response += token
