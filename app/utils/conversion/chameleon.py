# ==============================================================================
# Program: chameleon.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for CHAMELEON.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf

from .llama import LlamaModel


@ModelBase.register("ChameleonForConditionalGeneration")
@ModelBase.register("ChameleonForCausalLM")  # obsolete
class ChameleonModel(TextModel):
    """Converter for Meta Chameleon (early-fusion text+image) models, text side only.

    Image tokens are in the vocab, but the VQ image tokenizer is not converted.
    """
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.CHAMELEON

    def set_gguf_parameters(self):
        """Write base parameters plus the swin_norm flag (norm placement variant)."""
        super().set_gguf_parameters()
        self.gguf_writer.add_swin_norm(self.hparams.get("swin_norm", False))

    def set_vocab(self):
        """Chameleon uses a GPT-2 style BPE tokenizer."""
        self._set_vocab_gpt2()

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Drop the VQ-VAE image tokenizer weights; keep the language model."""
        name, gen = item

        # The image tokenizer (vqmodel) is not used for text inference.
        # ignore image tokenizer for now
        # TODO: image support for Chameleon
        if name.startswith("model.vqmodel"):
            return None

        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Un-permute Q/K projections and reshape the per-head Q/K norm parameters."""
        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams.get("num_key_value_heads")
        hidden_dim = self.hparams.get("hidden_size")

        # Same Q/K rotary un-permutation as Llama (K uses the KV head count).
        if name.endswith(("q_proj.weight", "q_proj.bias")):
            data_torch = LlamaModel.permute(data_torch, n_head, n_head)
        if name.endswith(("k_proj.weight", "k_proj.bias")):
            data_torch = LlamaModel.permute(data_torch, n_head, n_kv_head)
        # Chameleon normalises Q and K per head; HF stores those norm params in the
        # rotary-permuted order, so they need the matching reverse permutation.
        if name.endswith(("q_norm.weight", "q_norm.bias")):
            data_torch = ChameleonModel._reverse_hf_permute(data_torch, n_head, hidden_dim)
        if name.endswith(("k_norm.weight", "k_norm.bias")):
            data_torch = ChameleonModel._reverse_hf_permute(data_torch, n_kv_head, hidden_dim)

        yield from super().modify_tensors(data_torch, name, bid)

    # see: https://github.com/huggingface/transformers/blob/72fb02c47dbbe1999ae105319f24631cad6e2e00/src/transformers/models/chameleon/convert_chameleon_weights_to_hf.py#L176-L203
    # Reverse of the HF conversion script's permutation for per-head norm vectors.
    @staticmethod
    def _reverse_hf_permute(data_torch, n_heads, hidden_dim):
        """Undo the HF rotary permutation on a per-head norm parameter.

        Takes the first head's vector, de-interleaves its two halves and repeats it
        across all heads (the norm is stored once, not per head).
        """
        head_dim = hidden_dim // n_heads
        data_torch = data_torch[0].view(2, head_dim // 2).t().reshape(1, -1)
        data_torch = data_torch.repeat_interleave(n_heads, 0)
        return data_torch
