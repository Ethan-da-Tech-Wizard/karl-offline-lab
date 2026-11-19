# ==============================================================================
# Program: starcoder.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for STARCODER.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

"""HuggingFace-to-GGUF conversion support for the 'starcoder' model family.

Converter classes here are generally registered with ModelBase under HF architecture names;
they write GGUF metadata (set_gguf_parameters), the vocabulary (set_vocab)
and remap or reshape tensors (modify_tensors) for llama.cpp.

Classes:
  StarCoderModel - Converter for StarCoder (GPT-BigCode, multi-query attention).
  StarCoder2Model - Converter for StarCoder2; the base class handles everything.
"""

from __future__ import annotations

from .base import ModelBase, TextModel, gguf


@ModelBase.register("GPTBigCodeForCausalLM")
class StarCoderModel(TextModel):
    """Converter for StarCoder (GPT-BigCode, multi-query attention)."""
    model_arch = gguf.MODEL_ARCH.STARCODER

    def set_gguf_parameters(self):
        """Write StarCoder parameters; KV head count is 1 (multi-query) and the FFN is 4x the embedding.
        """
        self.gguf_writer.add_context_length(self.hparams["n_positions"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["n_embd"])
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_head_count_kv(1)
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)


@ModelBase.register("Starcoder2ForCausalLM")
class StarCoder2Model(TextModel):
    """Converter for StarCoder2; the base class handles everything."""
    model_arch = gguf.MODEL_ARCH.STARCODER2
