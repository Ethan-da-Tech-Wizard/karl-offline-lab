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
    model_arch = gguf.MODEL_ARCH.MAINCODER

    def set_gguf_parameters(self):
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()

        if (head_dim := self.hparams.get("head_dim")) is not None:
            self.gguf_writer.add_rope_dimension_count(head_dim)
