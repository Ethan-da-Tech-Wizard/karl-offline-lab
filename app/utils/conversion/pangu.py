# ==============================================================================
# Program: pangu.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for PANGU.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

import json

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("PanguEmbeddedForCausalLM")
class PanguEmbeddedModel(TextModel):
    """Converter for Huawei openPangu Embedded models."""
    model_arch = gguf.MODEL_ARCH.PANGU_EMBED

    def set_vocab(self):
        """Use SentencePiece and copy add_prefix_space from tokenizer_config.json if present.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        self._set_vocab_sentencepiece()

        tokenizer_config_file = self.dir_model / 'tokenizer_config.json'
        if tokenizer_config_file.is_file():
            with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                tokenizer_config_json = json.load(f)
                if "add_prefix_space" in tokenizer_config_json:
                    self.gguf_writer.add_add_space_prefix(tokenizer_config_json["add_prefix_space"])

    def set_gguf_parameters(self):
        """Write vocab size and RoPE dimension (derived when head_dim is missing)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])

        # PanguEmbedded's hparam loaded from config.json without head_dim
        if (rope_dim := hparams.get("head_dim")) is None:
            rope_dim = hparams["hidden_size"] // hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(rope_dim)

        if hparams.get("head_dim") is None:
            self.gguf_writer.add_key_length(rope_dim)
            self.gguf_writer.add_value_length(rope_dim)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Skip lm_head when embeddings are tied; map everything else normally."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if name == "lm_head.weight":
            if self.hparams.get("tie_word_embeddings", False):
                logger.info("Skipping tied output layer 'lm_head.weight'")
                return
        yield from super().modify_tensors(data_torch, name, bid)
