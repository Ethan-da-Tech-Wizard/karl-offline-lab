# ==============================================================================
# Program: gpt2.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for GPT2.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("GPT2LMHeadModel")
class GPT2Model(TextModel):
    """Converter for OpenAI GPT-2 style checkpoints (learned positions, LayerNorm, GELU)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.GPT2

    def set_gguf_parameters(self):
        """Write GPT-2 hyperparameters using its legacy key names (n_ctx, n_embd, n_head)."""
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_context_length(self.hparams["n_ctx"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])
        # GPT-2's FFN width is fixed at 4x the embedding size.
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["n_embd"])
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Transpose Conv1D-style weights (stored [in, out]) to the [out, in] layout expected."""
        # The causal-mask buffers are not weights; the base class filters/ignores them.
        # we don't need these
        if name.endswith((".attn.bias", ".attn.masked_bias")):
            yield from super().modify_tensors(data_torch, name, bid)
            return

        # HF GPT-2 uses Conv1D layers whose weight is the transpose of nn.Linear's.
        if name.endswith((".c_attn.weight", ".c_proj.weight", ".c_fc.weight", ".c_proj.weight")):
            data_torch = data_torch.transpose(1, 0)

        new_name = self.map_tensor_name(name)

        yield from super().modify_tensors(data_torch, new_name, bid)


@ModelBase.register("RuGPT3XLForCausalLM")
class RuGPT3XLModel(TextModel):
    """Converter for sberbank RuGPT-3 XL (GPT-2 architecture with separate Q/K/V).

    Separate q/k/v projections are fused back into one QKV tensor per layer.
    """
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.GPT2

    # Per-layer buffers holding q/k/v tensors until all three of a kind are present.
    _qkv_parts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Fuse Q, K and V (weights, then biases) into a single ATTN_QKV tensor.

        Generator: yields nothing until all three parts of one layer/kind have arrived.
        """
        # Fuse separate Q, K, V projections into a single QKV tensor
        if ".self_attn.q_proj." in name or ".self_attn.k_proj." in name or ".self_attn.v_proj." in name:
            # Weights and biases are fused independently, keyed by suffix.
            suffix = "weight" if name.endswith(".weight") else "bias"
            part = "q" if ".q_proj." in name else ("k" if ".k_proj." in name else "v")
            key = f"{part}.{suffix}"

            assert bid is not None
            if self._qkv_parts is None:
                self._qkv_parts = [{} for _ in range(self.block_count)]
            self._qkv_parts[bid][key] = data_torch

            q_key, k_key, v_key = f"q.{suffix}", f"k.{suffix}", f"v.{suffix}"
            # All three parts present: concatenate along the output dimension in q, k, v order.
            if all(k in self._qkv_parts[bid] for k in [q_key, k_key, v_key]):
                q = self._qkv_parts[bid].pop(q_key)
                k = self._qkv_parts[bid].pop(k_key)
                v = self._qkv_parts[bid].pop(v_key)
                data_torch = torch.cat([q, k, v], dim=0)
                name = self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_QKV, bid, f".{suffix}")
                logger.debug(f"Fused Q/K/V {suffix} for layer {bid} -> {name}")
            else:
                return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        """Run the normal conversion, then verify no Q/K/V part was left unfused."""
        super().prepare_tensors()

        # Leftovers mean a q/k/v tensor was missing from the checkpoint.
        if self._qkv_parts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            parts = [f"({i}){k}" for i, d in enumerate(self._qkv_parts) for k in d.keys()]
            if len(parts) > 0:
                raise ValueError(f"Unprocessed Q/K/V parts: {parts}")
