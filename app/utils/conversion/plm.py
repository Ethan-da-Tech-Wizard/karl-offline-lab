# ==============================================================================
# Program: plm.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for PLM.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

"""HuggingFace-to-GGUF conversion support for the 'plm' model family.

Converter classes here are generally registered with ModelBase under HF architecture names;
they write GGUF metadata (set_gguf_parameters), the vocabulary (set_vocab)
and remap or reshape tensors (modify_tensors) for llama.cpp.

Classes:
  PLMModel - Converter for PLM (MLA-style latent-attention) models.
"""

from __future__ import annotations

from .base import ModelBase, TextModel, gguf


@ModelBase.register("PLMForCausalLM")
class PLMModel(TextModel):
    """Converter for PLM (MLA-style latent-attention) models."""
    model_arch = gguf.MODEL_ARCH.PLM

    def set_vocab(self):
        """PLM uses a GPT-2 style BPE vocab."""
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        """Write vocab size and the latent-attention (MLA) ranks and head dimensions."""
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_kv_lora_rank(hparams["kv_lora_rank"])
        self.gguf_writer.add_key_length(hparams["qk_nope_head_dim"] + hparams["qk_rope_head_dim"])
        self.gguf_writer.add_value_length(hparams["v_head_dim"])
        self.gguf_writer.add_rope_dimension_count(hparams["qk_rope_head_dim"])

    def prepare_tensors(self):
        """Run base tensor preparation unchanged (explicit hook kept for symmetry with sibling converters).
        """
        super().prepare_tensors()
