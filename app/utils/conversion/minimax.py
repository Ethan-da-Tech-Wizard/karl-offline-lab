# ==============================================================================
# Program: minimax.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for MINIMAX.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("MiniMaxM2ForCausalLM")
class MiniMaxM2Model(TextModel):
    """Converter for MiniMax-M2 MoE models; stacks per-expert weights into 3D tensors."""
    model_arch = gguf.MODEL_ARCH.MINIMAXM2
    # Per-layer buffer of expert tensors awaiting merge. Note: class-level dict is shared by
    # all instances, which is fine for one conversion per process.
    _experts_cache: dict[int, dict[str, Tensor]] = {}

    def set_gguf_parameters(self):
        """Add expert FFN size and rotary dimension from the MiniMax config keys."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()

        self.gguf_writer.add_expert_feed_forward_length(self.find_hparam(["intermediate_size"]))
        self.gguf_writer.add_rope_dimension_count(self.find_hparam(["rotary_dim"]))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
        """Buffer expert tensors per layer; once complete, stack w1/w2/w3 across experts and emit them.
        """
        # merge expert weights
        if 'experts' in name:
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None

            expert_cache = self._experts_cache.setdefault(bid, {})
            expert_cache[name] = data_torch
            expert_weights = ["w1", "w2", "w3"]

            # not enough expert weights to merge
            # Not all experts of this layer have arrived yet; wait for more.
            if len(expert_cache) < n_experts * len(expert_weights):
                return

            for w_name in expert_weights:
                # Iterate over items, evaluate loop invariant, and update progress
                datas: list[Tensor] = []

                for xid in range(n_experts):
                    # Iterate over items, evaluate loop invariant, and update progress
                    ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{w_name}.weight"
                    datas.append(expert_cache[ename])
                    del expert_cache[ename]

                data_torch = torch.stack(datas, dim=0)
                merged_name = f"model.layers.{bid}.block_sparse_moe.experts.{w_name}.weight"
                new_name = self.map_tensor_name(merged_name)
                yield from super().modify_tensors(data_torch, new_name, bid)

            del self._experts_cache[bid]
            return

        yield from super().modify_tensors(data_torch, name, bid)
