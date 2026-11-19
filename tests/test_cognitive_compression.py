# ==============================================================================
# Program: test_cognitive_compression.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Comprehensive test suite exercising cognitive compression functionality.
#          Verifies state transitions, error boundaries, regression resistance,
#          and contract compliance under tests.
# ==============================================================================

import tests.qt_test_helper  # noqa: F401

import os
import sys
from PyQt6.QtCore import QCoreApplication

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch
from app.engine.model_loader import ModelLoader
from app.engine.llm_thread import LLMThread
from app.engine.agentic_thread import AgenticThread
import core.agentic_loop

# Ensure QCoreApplication exists so Qt signals/threads function correctly in test environment
app = QCoreApplication.instance()
if not app:
    app = QCoreApplication([])


class MockLlama:
    """Scripted llama double: the first stream stops with a 'length' finish once past 80% of the 4096-token context, then the compression summary call, then a resumed stream with the final response. Records whether compression was requested."""
    def __init__(self):
        """Initialise call counters and the compress_called flag."""
        self.call_count = 0
        self.compress_called = False

    def tokenize(self, text_bytes, add_bos=True):
        """Fake tokenizer that reports a token count large enough to exceed the 80% context budget when the history is long."""
        text = text_bytes.decode("utf-8")
        if "You are a cognitive compression tool" in text:
            return [0] * 50
        if "trigger_compression" in text:
            # The compression trigger compares the prompt token count with a fraction of n_ctx; this value is just over the threshold, so the history is judged too long.
            # Exceed 80% context budget (80% of 4096 is 3276)
            return [0] * 3500
        return [0] * (len(text) // 4 + 1)

    def __call__(self, prompt, **kwargs):
        """Dispatch by call type: non-streaming compression request, first streamed generation (hits the length limit) or the resumed generation after compression."""
        self.call_count += 1

        # Check if this is the compression call (non-streaming, returning dict)
        if "You are a cognitive compression tool" in prompt:
            self.compress_called = True
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "text": "Compressed summary of thoughts."
                    }
                ]
            }

        # finish_reason='length' mimics a generation cut off by the context budget mid-<think>, which is the condition that makes the thread compress and resume.
        # First call: return length continuation trigger
        if "trigger_compression" in prompt and self.call_count == 1:
            return [
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "text": "<think>Reasoning step before compression."
                        }
                    ]
                }
            ]
        elif "trigger_compression" in prompt:
            # Resuming after compression
            return [
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "text": " Continued reasoning step.</think>Response."
                        }
                    ]
                }
            ]

        # Default fallback
        return [
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "text": "<think>Thought.</think>Response."
                    }
                ]
            }
        ]


def test_cognitive_compression_and_live_stats():
    """When generation hits the context budget LLMThread asks the model to compress its reasoning, resumes, emits live stats, and finishes once with the compressed summary in the thought, the final response, not truncated, and diagnostics with positive token counts and time."""
    print("Testing cognitive roll-up compression and live stats signals (LLMThread)...")

    # Save original ModelLoader attributes to restore later
    orig_get_instance = ModelLoader.get_instance
    orig_model_name = ModelLoader.model_name
    orig_n_ctx = ModelLoader.n_ctx

    mock_llama = MockLlama()
    ModelLoader.get_instance = lambda *args, **kwargs: mock_llama
    ModelLoader.model_name = lambda: "mock-r1-model.gguf"
    # A fixed small context makes the 80% threshold (3276 tokens) a known number that MockLlama.tokenize can exceed deterministically.
    ModelLoader.n_ctx = lambda: 4096

    try:
        # Prepare slots for signals
        stats_emitted = []
        thought_emitted = []
        chat_emitted = []
        done_payload = []

        def on_live_stats(count, speed):
            stats_emitted.append((count, speed))

        def on_thought(token):
            thought_emitted.append(token)

        def on_chat(token):
            chat_emitted.append(token)

        def on_done(thought, response, truncated, ended_in_thought, diagnostics):
            done_payload.append((thought, response, truncated, ended_in_thought, diagnostics))

        # Instantiate LLMThread with a prompt targeting compression
        thread = LLMThread(
            system_prompt="System Prompt",
            chat_history=[{"role": "user", "content": "trigger_compression"}],
            hyperparams={"max_tokens": 100, "temperature": 0.7},
            workflow="general_chat",
            template="reasoning_minimal"
        )

        thread.live_stats.connect(on_live_stats)
        thread.new_thought_token.connect(on_thought)
        thread.new_chat_token.connect(on_chat)
        thread.generation_finished.connect(on_done)

        # Run thread logic synchronously
        # Calling run() directly executes the QThread body synchronously on the test thread, so no event loop is needed and signals are delivered immediately.
        thread.run()

        # Assertions
        assert mock_llama.compress_called, "Cognitive compression should have been triggered"
        assert len(stats_emitted) > 0, "live_stats signal should have emitted multiple times"
        
        # Verify live stats values
        for count, speed in stats_emitted:
            assert count >= 0
            assert speed >= 0.0

        assert len(done_payload) == 1, "done signal should emit exactly once"
        thought, response, truncated, ended_in_thought, diagnostics = done_payload[0]

        assert "Compressed summary of thoughts" in thought, "Thought should contain the compressed summary"
        assert "Response" in response, "Response should be retrieved correctly after resuming"
        # truncated is reserved for a final 'length' stop; the intermediate length stop was handled by compression and must not leak into the result.
        assert not truncated, "Final run finished with stop reason, should not be marked truncated"
        assert diagnostics is not None, "Diagnostics payload must be returned"
        assert diagnostics["prompt_tokens"] > 0
        assert diagnostics["generation_tokens"] > 0
        assert diagnostics["total_time"] > 0

        print("LLMThread cognitive compression and live stats unit tests passed successfully!")

    finally:
        # Restore ModelLoader state
        ModelLoader.get_instance = orig_get_instance
        ModelLoader.model_name = orig_model_name
        ModelLoader.n_ctx = orig_n_ctx


def test_agentic_cognitive_compression_and_live_stats():
    """The same compression/resume behaviour inside AgenticThread: one iteration completes, per-iteration results carry the compressed thought and diagnostics, and the loop finishes after 1 iteration."""
    print("Testing cognitive roll-up compression and live stats signals (AgenticThread)...")

    # Save original ModelLoader attributes to restore later
    orig_get_instance = ModelLoader.get_instance
    orig_model_name = ModelLoader.model_name
    orig_n_ctx = ModelLoader.n_ctx

    orig_should_continue = core.agentic_loop.should_continue
    orig_build_next_prompt = core.agentic_loop.build_next_prompt

    mock_llama = MockLlama()
    ModelLoader.get_instance = lambda *args, **kwargs: mock_llama
    ModelLoader.model_name = lambda: "mock-r1-model.gguf"
    ModelLoader.n_ctx = lambda: 4096

    # Force Agentic Loop to stop after the first generation turn
    # Without this the agentic loop would request further iterations, and the mock's call-count-based script would not match; stopping after one turn isolates the compression behaviour.
    core.agentic_loop.should_continue = lambda iteration, response: False
    core.agentic_loop.build_next_prompt = lambda response, iteration: "next"

    try:
        stats_emitted = []
        iteration_finished_payload = []
        loop_finished_payload = []

        def on_live_stats(count, speed):
            stats_emitted.append((count, speed))

        def on_iteration_finished(iteration, thought, response, diagnostics):
            iteration_finished_payload.append((iteration, thought, response, diagnostics))

        def on_loop_finished(total):
            loop_finished_payload.append(total)

        thread = AgenticThread(
            system_prompt="System Prompt",
            initial_history=[{"role": "user", "content": "trigger_compression"}],
            hyperparams={"max_tokens": 100, "temperature": 0.7},
            workflow="general_chat",
            template="reasoning_minimal"
        )

        thread.live_stats.connect(on_live_stats)
        thread.iteration_finished.connect(on_iteration_finished)
        thread.loop_finished.connect(on_loop_finished)

        # Run thread logic synchronously with compile_and_reload patched to avoid mock eviction
        with patch("app.engine.agentic_thread.compile_and_reload", side_effect=lambda module, *args, **kwargs: module):
            thread.run()

        # Assertions
        assert mock_llama.compress_called, "Cognitive compression should have been triggered in AgenticThread"
        assert len(stats_emitted) > 0, "live_stats signal should have emitted multiple times in AgenticThread"
        
        for count, speed in stats_emitted:
            assert count >= 0
            assert speed >= 0.0

        assert len(iteration_finished_payload) == 1, "Agentic loop should complete exactly 1 iteration in this test"
        iteration, thought, response, diagnostics = iteration_finished_payload[0]

        assert "Compressed summary of thoughts" in thought, "Thought should contain compressed summary"
        assert "Response" in response, "Response should be processed"
        assert diagnostics is not None
        assert diagnostics["prompt_tokens"] > 0
        assert diagnostics["generation_tokens"] > 0

        assert len(loop_finished_payload) == 1
        assert loop_finished_payload[0] == 1

        print("AgenticThread cognitive compression and live stats unit tests passed successfully!")

    finally:
        # Restore original objects
        ModelLoader.get_instance = orig_get_instance
        ModelLoader.model_name = orig_model_name
        ModelLoader.n_ctx = orig_n_ctx
        core.agentic_loop.should_continue = orig_should_continue
        core.agentic_loop.build_next_prompt = orig_build_next_prompt


if __name__ == "__main__":
    test_cognitive_compression_and_live_stats()
    test_agentic_cognitive_compression_and_live_stats()
