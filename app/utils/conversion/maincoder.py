# ==============================================================================
# Program: maincoder.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for MAINCODER.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

from .base import ModelBase, TextModel, gguf


@ModelBase.register("MaincoderForCausalLM")
class MaincoderModel(TextModel):
    """Converter for Maincoder (a Llama-like code model)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.MAINCODER

    def set_gguf_parameters(self):
        """Write base parameters and the RoPE dimension when head_dim is configured."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()

        # Optional key: without it the base class derives the RoPE size from the hidden size.
        if (head_dim := self.hparams.get("head_dim")) is not None:
            self.gguf_writer.add_rope_dimension_count(head_dim)
