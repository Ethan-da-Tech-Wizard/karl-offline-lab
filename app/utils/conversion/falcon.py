# ==============================================================================
# Program: falcon.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for FALCON.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("FalconForCausalLM", "RWForCausalLM")
class FalconModel(TextModel):
    """Converter for TII Falcon 7B/40B/180B (multi-query attention with fused QKV)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.FALCON

    def set_gguf_parameters(self):
        """Write Falcon hyperparameters, accepting both new and old config key names."""
        # Older Falcon configs use n_head / n_head_kv instead of the HF names.
        n_head = self.hparams.get("num_attention_heads")
        if n_head is None:
            n_head = self.hparams["n_head"]  # old name

        n_head_kv = self.hparams.get("num_kv_heads")
        if n_head_kv is None:
            n_head_kv = self.hparams.get("n_head_kv", 1)  # old name

        # Falcon's config.json does not store the context length; 2048 is its trained length.
        self.gguf_writer.add_context_length(2048)  # not in config.json
        # Marks the QKV layout produced by modify_tensors (the jploski ggml convention).
        self.gguf_writer.add_tensor_data_layout("jploski")  # qkv tensor transform
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head_kv)
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Re-arrange the fused query_key_value tensor into [all Q | all K | all V]."""
        # the original layout interleaves per KV group, which is awkward in ggml.
        # QKV tensor transform
        # The original query_key_value tensor contains n_head_kv "kv groups",
        # each consisting of n_head/n_head_kv query weights followed by one key
        # and one value weight (shared by all query heads in the kv group).
        # This layout makes it a big pain to work with in GGML.
        # So we rearrange them here,, so that we have n_head query weights
        # followed by n_head_kv key weights followed by n_head_kv value weights,
        # in contiguous fashion.
        # ref: https://github.com/jploski/ggml/blob/falcon40b/examples/falcon/convert-hf-to-ggml.py

        if "query_key_value" in name:
            n_head = self.find_hparam(["num_attention_heads", "n_head"])
            n_head_kv = self.find_hparam(["num_kv_heads", "n_head_kv"], optional=True) or 1
            head_dim = self.hparams["hidden_size"] // n_head

            # View as [kv groups, (q per group + k + v), head_dim, hidden]; the last two
            # slots of every group are that group's K and V, the rest are its Q heads.
            qkv = data_torch.view(n_head_kv, n_head // n_head_kv + 2, head_dim, head_dim * n_head)
            q = qkv[:, :-2].reshape(n_head * head_dim, head_dim * n_head)
            k = qkv[:, [-2]].reshape(n_head_kv * head_dim, head_dim * n_head)
            v = qkv[:, [-1]].reshape(n_head_kv * head_dim, head_dim * n_head)
            # Concatenate contiguous Q, K, V blocks and restore the original tensor shape.
            data_torch = torch.cat((q, k, v)).reshape_as(data_torch)

        yield from super().modify_tensors(data_torch, name, bid)
