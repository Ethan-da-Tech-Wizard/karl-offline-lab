# ==============================================================================
# Program: command_r.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for COMMAND_R.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("CohereForCausalLM")
class CommandR2Model(TextModel):
    """Converter for Cohere Command-R / Aya-23 models (no RoPE scaling)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.COMMAND_R

    def __init__(self, *args, **kwargs):
        """Fix up max_position_embeddings from the real trained context length."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(*args, **kwargs)

        # (see note) the config under-reports context, so prefer model_max_length when present.
        # max_position_embeddings = 8192 in config.json but model was actually
        # trained on 128k context length
        # aya-23 models don't have model_max_length specified
        self.hparams["max_position_embeddings"] = self.find_hparam(["model_max_length", "max_position_embeddings"])

    def set_gguf_parameters(self):
        """Write base parameters, the logit scale, and disable RoPE scaling."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()
        self.gguf_writer.add_logit_scale(self.hparams["logit_scale"])
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)


@ModelBase.register("Cohere2ForCausalLM")
class Cohere2Model(TextModel):
    """Converter for Cohere Command-R2 (sliding-window layers, partial rotary)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.COHERE2

    def set_gguf_parameters(self):
        """Write base parameters plus logit scale, sliding window, vocab size and RoPE size."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()

        self.gguf_writer.add_logit_scale(self.hparams["logit_scale"])
        self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])
        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])

        # RoPE dimension = rotary_pct * head size (only a fraction of each head is rotated).
        rotary_pct = self.hparams["rotary_pct"]
        hidden_size = self.hparams["hidden_size"]
        num_attention_heads = self.hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(int(rotary_pct * (hidden_size // num_attention_heads)))
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Skip all-zero bias tensors; refuse conversion if a bias is non-zero."""
        # (see note) raising on non-zero bias protects against silently dropping real weights.
        # Cohere2 runtime in llama.cpp expects no bias tensors;
        # the actual weight only contains 0-value tensors as bias, we can skip them
        if name.endswith(".bias"):
            if torch.any(data_torch != 0):
                raise ValueError(f"Bias tensor {name!r} is not zero.")
            logger.debug(f"Skipping bias tensor {name!r} for Cohere2 conversion.")
            return

        yield from super().modify_tensors(data_torch, name, bid)
