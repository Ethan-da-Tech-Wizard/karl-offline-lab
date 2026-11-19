# ==============================================================================
# Program: olmo.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for OLMO.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

"""HuggingFace-to-GGUF conversion support for the 'olmo' model family.

Converter classes here are generally registered with ModelBase under HF architecture names;
they write GGUF metadata (set_gguf_parameters), the vocabulary (set_vocab)
and remap or reshape tensors (modify_tensors) for llama.cpp.

Classes:
  OlmoModel - Converter for OLMo (v1) models.
  SeedOssModel - Converter for ByteDance Seed-OSS; the base class handles everything.
  Olmo2Model - Converter for OLMo2/OLMo3 models.
  OlmoeModel - Converter for OLMoE (mixture-of-experts) models.
"""

from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf

from .llama import LlamaModel


@ModelBase.register("OlmoForCausalLM")
@ModelBase.register("OLMoForCausalLM")
class OlmoModel(TextModel):
    """Converter for OLMo (v1) models."""
    model_arch = gguf.MODEL_ARCH.OLMO

    def set_gguf_parameters(self):
        """Write layer-norm eps and optional QKV clamp."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()
        self.gguf_writer.add_layer_norm_eps(1e-5)
        clip_qkv = self.hparams.get("clip_qkv")
        if clip_qkv is not None:
            self.gguf_writer.add_clamp_kqv(clip_qkv)

    # Same as super class, but permuting q_proj, k_proj
    # Copied from: LlamaModel
    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Undo the HF q/k permutation, as the llama converter does."""
        # Step 1: Validate input parameters, operational preconditions, and state
        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams.get("num_key_value_heads")

        if name.endswith("q_proj.weight"):
            data_torch = LlamaModel.permute(data_torch, n_head, n_head)
        if name.endswith("k_proj.weight"):
            data_torch = LlamaModel.permute(data_torch, n_head, n_kv_head)

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("SeedOssForCausalLM")
class SeedOssModel(TextModel):
    """Converter for ByteDance Seed-OSS; the base class handles everything."""
    model_arch = gguf.MODEL_ARCH.SEED_OSS


@ModelBase.register("Olmo2ForCausalLM")
@ModelBase.register("Olmo3ForCausalLM")
class Olmo2Model(TextModel):
    """Converter for OLMo2/OLMo3 models."""
    model_arch = gguf.MODEL_ARCH.OLMO2

    def set_gguf_parameters(self):
        """Write the sliding-window size and a per-layer sliding-window pattern."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()

        if "sliding_window" in self.hparams:
            self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])

            sliding_window_pattern = []
            if "layer_types" in self.hparams:
                sliding_window_pattern = [t == "sliding_attention" for t in self.hparams["layer_types"]]
            else:
                # Olmo2 has no sliding window attention.
                # Olmo3 defaults to sliding window on all layers except every 4th.
                # Olmo2 does not use sliding window attention.
                # Olmo3 defaults to using sliding window for all layers except every 4th.
                for i in range(self.hparams["num_hidden_layers"]):
                    # Iterate over items, evaluate loop invariant, and update progress
                    sliding_window_pattern.append((i + 1) % 4 != 0)

            self.gguf_writer.add_sliding_window_pattern(sliding_window_pattern)


@ModelBase.register("OlmoeForCausalLM")
class OlmoeModel(TextModel):
    """Converter for OLMoE (mixture-of-experts) models."""
    model_arch = gguf.MODEL_ARCH.OLMOE

    def set_gguf_parameters(self):
        """Write RMS norm epsilon."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()
        self.gguf_writer.add_layer_norm_rms_eps(1e-5)

    # Per-layer buffer for expert tensors until they can be merged.
    _experts: list[dict[str, Tensor]] | None = None

    # Copied from: Qwen2MoeModel
    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Buffer per-expert weights per layer and emit merged 3D tensors once complete."""
        # process the experts separately
        if name.find("experts") != -1:
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    # Iterate over items, evaluate loop invariant, and update progress
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        # Iterate over items, evaluate loop invariant, and update progress
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    yield from super().modify_tensors(data_torch, merged_name, bid)
                return
            else:
                return

        yield from super().modify_tensors(data_torch, name, bid)

    # Copied from: Qwen2MoeModel
    def prepare_tensors(self):
        """Run base preparation and fail if any expert tensors were left unmerged."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().prepare_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")
