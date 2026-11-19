# ==============================================================================
# Program: baichuan.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for BAICHUAN.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("BaichuanForCausalLM", "BaiChuanForCausalLM")
class BaichuanModel(TextModel):
    """Converter for Baichuan / BaiChuan causal LMs (7B/13B).

    Baichuan packs Q, K and V into one fused W_pack matrix, which this class
    splits and un-permutes into separate GGUF attention tensors.
    """
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.BAICHUAN

    def set_vocab(self):
        """Baichuan ships a SentencePiece tokenizer.model; use the standard loader."""
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        """Write base parameters plus tensor layout and RoPE dimension count."""
        super().set_gguf_parameters()

        # Marks the weights as stored in the original Meta/Llama pth layout so the
        # loader applies the matching RoPE convention.
        self.gguf_writer.add_tensor_data_layout("Meta AI original pth")
        self.gguf_writer.add_rope_dimension_count(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Split the fused W_pack tensor into Q/K/V; delegate all other tensors."""
        head_count = self.hparams["num_attention_heads"]
        # Falls back to the full head count when the config has no GQA setting.
        head_count_kv = self.hparams.get("num_key_value_heads", head_count)

        # W_pack stacks Q, K and V along dim 0 in three equal parts (index 0/1/2).
        # Q and K must additionally be un-permuted from the HF rotary layout; V is a plain slice.
        if bid is not None and name == f"model.layers.{bid}.self_attn.W_pack.weight":
            logger.info(f"Unpacking and permuting layer {bid}")
            yield from [
                (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q, bid),
                    self._reverse_hf_permute_part(data_torch, 0, head_count, head_count)),
                (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K, bid),
                    self._reverse_hf_permute_part(data_torch, 1, head_count, head_count_kv)),
                (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V, bid),
                    self._reverse_hf_part(data_torch, 2)),
            ]
        else:
            # Delegate to the base class; calling self.modify_tensors here would recurse forever.
            yield from super().modify_tensors(data_torch, name, bid)

    def _reverse_hf_permute(self, weights: Tensor, n_head: int, n_kv_head: int | None = None) -> Tensor:
        """Undo the HF rotary-embedding head permutation (inverse of the permute done by HF).

        When K/V use fewer heads (GQA) the head count is divided by n_kv_head.
        """
        if n_kv_head is not None and n_head != n_kv_head:
            n_head //= n_kv_head

        return (
            weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape)
        )

    def _reverse_hf_permute_part(
        self, weights: Tensor, n_part: int, n_head: int, n_head_kv: int | None = None,
    ) -> Tensor:
        """Slice one third (Q, K or V by n_part) out of a fused matrix and un-permute it."""
        # Size of one third of the fused first dimension.
        r = weights.shape[0] // 3
        return self._reverse_hf_permute(weights[r * n_part:r * n_part + r, ...], n_head, n_head_kv)

    def _reverse_hf_part(self, weights: Tensor, n_part: int) -> Tensor:
        """Slice one third (Q, K or V by n_part) out of a fused matrix without permuting."""
        r = weights.shape[0] // 3
        return weights[r * n_part:r * n_part + r, ...]
