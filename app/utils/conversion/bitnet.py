# ==============================================================================
# Program: bitnet.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for BITNET.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("BitnetForCausalLM")
class BitnetModel(TextModel):
    """Converter for BitNet b1.58 (ternary-weight) checkpoints.

    Projection weights are re-quantised to {-1, 0, +1} times a per-tensor scale at
    conversion time, mirroring what the HF model does on every forward pass.
    """
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.BITNET

    def set_vocab(self):
        """BitNet ships a SentencePiece tokenizer.model."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        """Write base parameters plus a linear RoPE scaling of 1.0 (i.e. no scaling)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
        self.gguf_writer.add_rope_scaling_factor(1.0)

    def weight_quant(self, weight: Tensor) -> Tensor:
        """Ternarise a weight tensor: round(w / mean|w|) clamped to [-1, 1], then rescale.

        Returns a tensor of the original dtype whose values are only 0 and +/- mean|w|.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        # Remember the original dtype so the result can be cast back at the end.
        dtype = weight.dtype
        weight = weight.float()
        # Per-tensor absmean scale; the clamp avoids dividing by ~0 for all-zero tensors.
        scale = weight.abs().mean().clamp(min=1e-5)
        iscale = 1 / scale
        # TODO: multiply by the scale directly instead of inverting it twice
        # (this is also unnecessarily doubly inverted upstream)
        # ref: https://huggingface.co/1bitLLM/bitnet_b1_58-3B/blob/af89e318d78a70802061246bf037199d2fb97020/utils_quant.py#L10
        # Quantise to {-1,0,1} then multiply back by the scale (iscale is 1/scale).
        result = (weight * iscale).round().clamp(-1, 1) / iscale
        return result.type(dtype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Ternarise attention and FFN projection weights, leave other tensors untouched."""
        # Step 1: Validate input parameters, operational preconditions, and state
        # Map first so the check below can use the canonical GGUF tensor identity.
        new_name = self.map_tensor_name(name)

        # Only the attention Q/K/V/output and FFN up/down/gate matrices are BitLinear
        # layers; embeddings and norms stay full precision.
        if any(self.match_model_tensor_name(new_name, key, bid) for key in [
            gguf.MODEL_TENSOR.ATTN_Q,
            gguf.MODEL_TENSOR.ATTN_K,
            gguf.MODEL_TENSOR.ATTN_V,
            gguf.MODEL_TENSOR.ATTN_OUT,
            gguf.MODEL_TENSOR.FFN_UP,
            gguf.MODEL_TENSOR.FFN_DOWN,
            gguf.MODEL_TENSOR.FFN_GATE,
        ]):
            # transform weight into 1/0/-1 (in fp32)
            data_torch = self.weight_quant(data_torch)

        yield from super().modify_tensors(data_torch, name, bid)
