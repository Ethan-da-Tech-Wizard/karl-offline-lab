# ==============================================================================
# Program: wavtokenizer.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for WAVTOKENIZER.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

"""HuggingFace-to-GGUF conversion support for the 'wavtokenizer' model family.

Converter classes here are generally registered with ModelBase under HF architecture names;
they write GGUF metadata (set_gguf_parameters), the vocabulary (set_vocab)
and remap or reshape tensors (modify_tensors) for llama.cpp.

Classes:
  WavTokenizerDecModel - Converter for the WavTokenizer decoder (audio codec: PosNet + ConvNeXt blocks).
"""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("WavTokenizerDec")
class WavTokenizerDecModel(TextModel):
    """Converter for the WavTokenizer decoder (audio codec: PosNet + ConvNeXt blocks)."""
    model_arch = gguf.MODEL_ARCH.WAVTOKENIZER_DEC

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Drop codebook training statistics that inference does not need."""
        name, gen = item

        if \
                name.endswith("codebook.cluster_size") or \
                name.endswith("codebook.embed_avg") or \
                name.endswith("codebook.inited"):
            logger.debug(f"Skipping {name!r}")
            return None

        return super().filter_tensors(item)

    def set_vocab(self):
        """The codec has no text vocabulary."""
        self._set_vocab_none()

    def set_gguf_parameters(self):
        """Write feature widths, group-norm settings and PosNet/ConvNeXt block sizes; attention is non-causal.
        """
        super().set_gguf_parameters()
        self.gguf_writer.add_vocab_size         (self.hparams["vocab_size"])
        self.gguf_writer.add_features_length    (self.hparams["n_embd_features"])
        self.gguf_writer.add_feed_forward_length(self.hparams["n_ff"])
        self.gguf_writer.add_group_norm_eps     (self.hparams["group_norm_epsilon"])
        self.gguf_writer.add_group_norm_groups  (self.hparams["group_norm_groups"])

        self.gguf_writer.add_posnet_embedding_length(self.hparams["posnet"]["n_embd"])
        self.gguf_writer.add_posnet_block_count     (self.hparams["posnet"]["n_layer"])

        self.gguf_writer.add_convnext_embedding_length(self.hparams["convnext"]["n_embd"])
        self.gguf_writer.add_convnext_block_count     (self.hparams["convnext"]["n_layer"])

        self.gguf_writer.add_causal_attention(False)
