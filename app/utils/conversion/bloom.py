# ==============================================================================
# Program: bloom.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for BLOOM.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

import re

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("BloomForCausalLM", "BloomModel")
class BloomModel(TextModel):
    """Converter for BigScience BLOOM models (ALiBi attention, fused QKV)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.BLOOM

    def set_gguf_parameters(self):
        """Write BLOOM hyperparameters; accepts both HF (`hidden_size`) and legacy (`n_embed`) key names."""
        # Step 1: Validate input parameters, operational preconditions, and state
        # Config files use either naming scheme depending on how the model was saved.
        n_embed = self.hparams.get("hidden_size", self.hparams.get("n_embed"))
        n_head = self.hparams.get("n_head", self.hparams.get("num_attention_heads"))
        assert n_head is not None
        assert n_embed is not None
        # BLOOM uses ALiBi, so seq_length is informational; fall back to the embedding size if absent.
        self.gguf_writer.add_context_length(self.hparams.get("seq_length", n_embed))
        self.gguf_writer.add_embedding_length(n_embed)
        # BLOOM's FFN width is always 4x the embedding size.
        self.gguf_writer.add_feed_forward_length(4 * n_embed)
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_head_count(n_head)
        # Plain multi-head attention: as many KV heads as query heads.
        self.gguf_writer.add_head_count_kv(n_head)
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Re-order the fused query_key_value tensors into [all Q | all K | all V].

        BLOOM interleaves Q,K,V per head; the GPT-style layout llama.cpp expects
        stores all Q rows, then all K rows, then all V rows.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        n_head = self.hparams.get("n_head", self.hparams.get("num_attention_heads"))
        n_embed = self.hparams.get("hidden_size", self.hparams.get("n_embed"))
        assert n_head is not None
        assert n_embed is not None

        # Strip the HF "transformer." prefix so names match the tensor map.
        name = re.sub(r'transformer\.', '', name)

        # Bloom stores QKV as [n_head, 3, head_dim, n_embd]. Pull out the Q/K/V slices
        # for every head and concatenate them along the output dimension.
        if re.match(r"h\.\d+\.self_attention\.query_key_value\.weight", name):
            # Map bloom-style qkv_linear to gpt-style qkv_linear
            # bloom: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/modeling_bloom.py#L238-L252  # noqa
            # gpt-2: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L312  # noqa
            qkv_weights = data_torch.reshape((n_head, 3, n_embed // n_head, n_embed))
            data_torch = torch.cat(
                (
                    qkv_weights[:, 0, :, :].reshape((-1, n_embed)),
                    qkv_weights[:, 1, :, :].reshape((-1, n_embed)),
                    qkv_weights[:, 2, :, :].reshape((-1, n_embed)),
                ),
                dim=0,
            )
            logger.info("re-format attention.linear_qkv.weight")
        # The bias follows the same per-head Q/K/V interleaving, with no n_embd axis.
        elif re.match(r"h\.\d+\.self_attention\.query_key_value\.bias", name):
            qkv_bias = data_torch.reshape((n_head, 3, n_embed // n_head))
            data_torch = torch.cat(
                (
                    qkv_bias[:, 0, :].reshape((n_embed,)),
                    qkv_bias[:, 1, :].reshape((n_embed,)),
                    qkv_bias[:, 2, :].reshape((n_embed,)),
                ),
                dim=0,
            )
            logger.info("re-format attention.linear_qkv.bias")

        yield from super().modify_tensors(data_torch, name, bid)
