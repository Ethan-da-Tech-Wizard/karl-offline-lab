# ==============================================================================
# Program: dots1.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for DOTS1.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf

# dots.llm1 is a Qwen2-MoE style model, so reuse its MoE handling.
from .qwen import Qwen2MoeModel


@ModelBase.register("Dots1ForCausalLM")
class Dots1Model(Qwen2MoeModel):
    """Converter for rednote dots.llm1 (MoE with shared experts)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.DOTS1

    def __init__(self, *args, **kwargs):
        """Alias the expert count under the name the Qwen2-MoE parent expects."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(*args, **kwargs)
        # Qwen2MoeModel reads `num_experts`; dots1 calls it `n_routed_experts`.
        self.hparams["num_experts"] = self.hparams["n_routed_experts"]

    def set_gguf_parameters(self):
        """Add dots1 MoE routing metadata to the Qwen2-MoE parameters."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()
        self.gguf_writer.add_leading_dense_block_count(self.hparams["first_k_dense_replace"])
        self.gguf_writer.add_expert_shared_count(self.hparams["n_shared_experts"])
        self.gguf_writer.add_expert_weights_scale(self.hparams["routed_scaling_factor"])
        self.gguf_writer.add_expert_weights_norm(self.hparams["norm_topk_prob"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
        """Bypass the Qwen2-MoE merge logic for shared-expert tensors."""
        # Step 1: Validate input parameters, operational preconditions, and state
        # Shared experts are ordinary dense tensors (one set per layer), not part of the
        # routed-expert stack, so skip the parent's expert-merging and map them directly.
        if "shared_experts" in name:
            yield from ModelBase.modify_tensors(self, data_torch, name, bid)
        else:
            yield from super().modify_tensors(data_torch, name, bid)
