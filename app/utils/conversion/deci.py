# ==============================================================================
# Program: deci.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for DECI.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

import math

from typing import Any, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("DeciLMForCausalLM")
class DeciModel(TextModel):
    """Converter for Deci / Nemotron-51B style NAS-derived Llama models.

    DeciLM-7B varies the KV head count per layer; Llama-3.1-Nemotron-51B/253B
    describe each layer via `block_configs` (attention type, FFN multiplier),
    including linear-attention, attention-free and dummy layers.
    """
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.DECI

    @staticmethod
    def _ffn_mult_to_intermediate_size(ffn_mult: float, n_embd: int) -> int:
        """Convert a DeciLM FFN multiplier to an intermediate size (rounded up to 256)."""
        # DeciLM-specific code
        intermediate_size = int(2 * ffn_mult * n_embd / 3)
        return DeciModel._find_multiple(intermediate_size, 256)

    @staticmethod
    def _find_multiple(n: int, k: int) -> int:
        """Round `n` up to the next multiple of `k`."""
        # DeciLM-specific code
        if n % k == 0:
            return n
        return n + k - (n % k)

    def __init__(self, *args, **kwargs):
        """Pre-compute per-layer head counts and FFN widths from `block_configs`."""
        super().__init__(*args, **kwargs)

        # Only the Nemotron-51B/253B style configs carry per-layer block configs.
        if "block_configs" in self.hparams: # Llama-3_1-Nemotron-51B
            _block_configs: list[dict[str,Any]] = self.hparams["block_configs"]
            assert self.block_count == len(_block_configs)
            self._num_kv_heads = list()
            self._num_heads = list()
            _ffn_multipliers = list()
            # Per-layer attention kind encoded as (kv heads, heads):
            # ***linear attention layer***
            # if n_heads_in_group is None and replace_with_linear is True
            # then _num_kv_heads[il] is 0 and _num_heads[il] is num_attention_heads
            # ***attention-free layer***
            # if n_heads_in_group is None and replace_with_linear is False
            # then _num_kv_heads[il] is 0 and _num_heads[il] is 0
            # ***normal attention-layer***
            # if n_heads_in_group is not None, then
            # _num_kv_heads[il] is num_attention_head // n_heads_in_group and
            # _num_heads[il] is num_attention_head
            # ***dummy layer*** for nemotron 253B
            # if n_heads_in_group is None and ffn_mult is None
            # then _num_kv_heads[il] is 0 and _num_heads[il] is 0 and _ffn_dims is 0
            # il = layer index; derive per-layer KV heads, heads and FFN multiplier.
            for il in range(len(_block_configs)):
                if _block_configs[il]["attention"]["n_heads_in_group"] is None:
                    if _block_configs[il]["attention"]["replace_with_linear"] is True:
                        self._num_kv_heads.append(0)
                        self._num_heads.append(self.hparams["num_attention_heads"])
                    else:
                        self._num_kv_heads.append(0)
                        self._num_heads.append(0)
                else:
                    self._num_kv_heads.append(self.hparams["num_attention_heads"] // _block_configs[il]["attention"]["n_heads_in_group"])
                    self._num_heads.append(self.hparams["num_attention_heads"])
                if _block_configs[il]["ffn"]["ffn_mult"] is None: # dummy layer
                    _ffn_multipliers.append(0.0)
                else:
                    _ffn_multipliers.append(_block_configs[il]["ffn"]["ffn_mult"])
            # Sanity checks: one entry per layer, and the expected element types.
            assert self.block_count == len(self._num_kv_heads)
            assert self.block_count == len(self._num_heads)
            assert self.block_count == len(_ffn_multipliers)
            assert isinstance(self._num_kv_heads, list) and isinstance(self._num_kv_heads[0], int)
            assert isinstance(self._num_heads, list) and isinstance(self._num_heads[0], int)
            assert isinstance(_ffn_multipliers, list) and isinstance(_ffn_multipliers[0], float)
            # Final FFN width per layer; a multiplier of 0 gives width 0 (dummy layer).
            self._ffn_dims: list[int] = [
                DeciModel._ffn_mult_to_intermediate_size(multiplier, self.hparams["hidden_size"])
                for multiplier in _ffn_multipliers
            ]

    def set_vocab(self):
        """Use the GPT-2 style BPE vocab for Llama-3 sized vocabs, else the HF Llama loader."""
        # the eos token in tokenizer_config.json must be fixed up by the user first.
        # Please change tokenizer_config.json of Llama-3_1-Nemotron-51B's
        # eos_token from '|eot_id|' to '|end_of_text|'
        if self.hparams.get("vocab_size", 128256) == 128256:
            tokens, toktypes, tokpre = self.get_vocab_base()
            self.gguf_writer.add_tokenizer_model("gpt2")
            self.gguf_writer.add_tokenizer_pre(tokpre)
            self.gguf_writer.add_token_list(tokens)
            self.gguf_writer.add_token_types(toktypes)

            special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
            special_vocab.add_to_gguf(self.gguf_writer)
        else:
            # DeciLM-7B
            self._set_vocab_llama_hf()

    def set_gguf_parameters(self):
        """Write per-layer arrays (heads, KV heads, FFN) for the NAS variants, or
        the standard parameters (plus per-layer KV heads) for DeciLM-7B.
        """
        if "block_configs" in self.hparams: # Llama-3_1-Nemotron-51B
            assert self.block_count == len(self._num_kv_heads)
            assert self.block_count == len(self._num_heads)
            assert self.block_count == len(self._ffn_dims)
            if (rope_theta := self.rope_parameters.get("rope_theta")) is not None:
                self.gguf_writer.add_rope_freq_base(rope_theta)
            self.gguf_writer.add_head_count_kv(self._num_kv_heads)
            self.gguf_writer.add_head_count(self._num_heads)
            self.gguf_writer.add_feed_forward_length(self._ffn_dims)
            self.gguf_writer.add_block_count(self.block_count)
            self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
            self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
            self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])
            self.gguf_writer.add_key_length(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])
            self.gguf_writer.add_value_length(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])
            self.gguf_writer.add_file_type(self.ftype)
        # Classic DeciLM-7B: standard parameters, optionally with per-layer KV heads.
        else: # DeciLM-7B
            super().set_gguf_parameters()
            if "num_key_value_heads_per_layer" in self.hparams: # DeciLM-7B
                self._num_kv_heads: list[int] = self.hparams["num_key_value_heads_per_layer"]
                assert self.block_count == len(self._num_kv_heads)
                self.gguf_writer.add_head_count_kv(self._num_kv_heads)
        # Common to both layouts: vocab size and RoPE dimension count.
        hparams = self.hparams
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])

        if (rope_dim := hparams.get("head_dim")) is None:
            rope_dim = hparams["hidden_size"] // hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(rope_dim)

    @staticmethod
    def permute(weights: Tensor, n_head: int, n_head_kv: int | None):
        """Reorder Q/K weights from the HF rotary layout to the llama.cpp layout."""
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Un-permute Q/K using the per-layer head counts for this block."""
        n_head = self.hparams["num_attention_heads"]
        # Head counts can differ per layer (DeciLM-7B list or Nemotron block configs),
        # so look them up for this block id before permuting.
        if bid is not None:
            if "num_key_value_heads_per_layer" in self.hparams:
                n_kv_head = self.hparams["num_key_value_heads_per_layer"][bid]
            elif "block_configs" in self.hparams:
                n_kv_head = self._num_kv_heads[bid]
                n_head = self._num_heads[bid]
            else:
                n_kv_head = self.hparams.get("num_key_value_heads")
        else:
            n_kv_head = self.hparams.get("num_key_value_heads")

        if name.endswith(("q_proj.weight", "q_proj.bias")):
            data_torch = DeciModel.permute(data_torch, n_head, n_head)
        if name.endswith(("k_proj.weight", "k_proj.bias")):
            data_torch = DeciModel.permute(data_torch, n_head, n_kv_head)
        yield from super().modify_tensors(data_torch, name, bid)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        """Emit the precomputed Llama-3 RoPE frequency factors, if the config asks for them.

        Implements Llama 3's long-context scaling: high-frequency components are left
        alone, low-frequency ones are divided by `factor`, and the band in between is
        smoothly interpolated.
        """
        if rope_params := self.rope_parameters.get("full_attention", self.rope_parameters):
            if rope_params.get("rope_type", '').lower() == "llama3":
                base = rope_params.get("rope_theta", 10000.0)
                if (dim := self.hparams.get("head_dim")) is None:
                    dim = self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
                # Standard RoPE inverse frequencies for each pair of dimensions.
                freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

                factor = rope_params.get("factor", 8.0)
                low_freq_factor = rope_params.get("low_freq_factor", 1.0)
                high_freq_factor = rope_params.get("high_freq_factor", 4.0)
                old_context_len = self.hparams.get("original_max_position_embeddings", 8192)

                low_freq_wavelen = old_context_len / low_freq_factor
                high_freq_wavelen = old_context_len / high_freq_factor
                # The smoothing term below divides by (high - low), so they must differ.
                assert low_freq_wavelen != high_freq_wavelen

                rope_factors = []
                # Per-frequency scale factor: 1 (no change), `factor` (full), or a smooth blend.
                for freq in freqs:
                    wavelen = 2 * math.pi / freq
                    if wavelen < high_freq_wavelen:
                        rope_factors.append(1)
                    elif wavelen > low_freq_wavelen:
                        rope_factors.append(factor)
                    else:
                        smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                        rope_factors.append(1 / ((1 - smooth) / factor + smooth))

                yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FREQS), torch.tensor(rope_factors, dtype=torch.float32))

    def prepare_tensors(self):
        """Run the base conversion (no extra post-processing needed for Deci)."""
        super().prepare_tensors()
