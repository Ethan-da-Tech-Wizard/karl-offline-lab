# ==============================================================================
# Program: afmoe.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for AFMOE.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

# ModelBase provides the @register decorator and the GGUF writer plumbing;
# `gguf` is the llama.cpp gguf-py module used for architecture ids.
from .base import ModelBase, gguf

# AFMOE is a Llama-style decoder, so it reuses LlamaModel for the shared
# attention/embedding handling and overrides only the MoE specifics.
from .llama import LlamaModel


@ModelBase.register("AfmoeForCausalLM")
class AfmoeModel(LlamaModel):
    """Converter for Arcee AFMoE (Mixture-of-Experts) checkpoints.

    Extends LlamaModel with MoE metadata (shared experts, dense-layer prefix,
    routing normalisation/scale) and with merging per-expert weights.
    """
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.AFMOE

    def set_gguf_parameters(self):
        """Write the Llama parameters plus AFMoE-specific MoE and attention metadata.

        Each key is optional in config.json, so a value is only written when present.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()

        # Each hparam lookup uses the walrus operator so absent keys are skipped
        # rather than written as None.
        # MoE parameters
        if (n_shared_experts := self.hparams.get("num_shared_experts")) is not None:
            self.gguf_writer.add_expert_shared_count(n_shared_experts)
        if (moe_intermediate_size := self.hparams.get("moe_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)
        if (n_dense_layers := self.hparams.get("num_dense_layers")) is not None:
            self.gguf_writer.add_leading_dense_block_count(n_dense_layers)

        # Route normalization and scaling
        if (route_norm := self.hparams.get("route_norm")) is not None:
            self.gguf_writer.add_expert_weights_norm(route_norm)
        if (route_scale := self.hparams.get("route_scale")) is not None:
            self.gguf_writer.add_expert_weights_scale(route_scale)

        # Sliding window attention
        if (sliding_window := self.hparams.get("sliding_window")) is not None:
            self.gguf_writer.add_sliding_window(sliding_window)

    # Runs on every (name, lazy-tensor) pair before conversion; may rename or drop it.
    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Rename the router bias tensor so it maps onto the GGUF tensor name table.

        HF stores `...expert_bias`; the tensor map expects `...expert_bias.bias`.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        name, gen = item

        # Rewrite before delegating so the base class's name mapping recognises it.
        if name.endswith(".expert_bias"):
            name = name.replace(".expert_bias", ".expert_bias.bias")

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Merge per-expert weights into stacked 3D tensors; pass others through.

        This is a generator: it yields nothing while experts are still being
        collected, and yields the merged tensors once a layer is complete.
        """
        # Expert tensors are identified by name; everything else falls to the
        # base implementation at the bottom of this method.
        # Handle expert weights - they're already merged in the HF format
        # process the experts separately
        if name.find("mlp.experts") != -1:
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None

            # Experts arrive as separate per-expert tensors; buffer them per layer
            # (block id) until the whole set for that layer has been seen.
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            # 3 projections (gate/up/down) x n_experts tensors means the layer is complete.
            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for w_name in ["gate_proj", "up_proj", "down_proj"]:
                    # Iterate over items, evaluate loop invariant, and update progress
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        # Iterate over items, evaluate loop invariant, and update progress
                        ename_to_retrieve = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename_to_retrieve])
                        del self._experts[bid][ename_to_retrieve]

                    # Stack the per-expert 2D matrices into [n_experts, ...].
                    data_torch = torch.stack(datas, dim=0)
                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"
                    # Call ModelBase.modify_tensors directly, bypassing LlamaModel's version:
                    # that one un-permutes Q/K weights and merges Mixtral-named
                    # (block_sparse_moe) experts, neither of which applies here.
                    yield from ModelBase.modify_tensors(self, data_torch, merged_name, bid)

                return
            else:
                return

        yield from ModelBase.modify_tensors(self, data_torch, name, bid)
