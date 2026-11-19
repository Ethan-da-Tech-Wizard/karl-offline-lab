# ==============================================================================
# Program: exaone.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for EXAONE.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

import math

from pathlib import Path
from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("ExaoneForCausalLM")
class ExaoneModel(TextModel):
    """Converter for LG EXAONE 3.x (Llama-like, SiLU, optional partial rotary)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.EXAONE

    def set_gguf_parameters(self):
        """Write base parameters and the RoPE dimension (partial rotary supported)."""
        super().set_gguf_parameters()
        hparams = self.hparams

        # The runtime graph only implements the SiLU-gated FFN for this architecture.
        assert (hparams["activation_function"] == "silu")

        # Fraction of each head that is rotated; the key name differs between releases
        # and the default (None) means the whole head.
        rotary_factor = self.find_hparam(["partial_rotary_factor", "rope_pct"], optional=True)
        rotary_factor = rotary_factor if rotary_factor is not None else 1.0
        self.gguf_writer.add_rope_dimension_count(int(rotary_factor * (hparams["hidden_size"] // hparams["num_attention_heads"])))

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        """Emit the precomputed Llama-3 RoPE frequency factors, if the config asks for them.

        Implements Llama 3's long-context scaling: high-frequency components are left
        alone, low-frequency ones are divided by `factor`, and the band in between is
        smoothly interpolated.
        """
        if rope_params := self.rope_parameters.get("full_attention", self.rope_parameters):
            if rope_params.get("rope_type", '').lower() == "llama3":
                base = self.rope_parameters.get("rope_theta", 10000.0)
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
                # The smoothing term divides by (high - low), so the two factors must differ.
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


@ModelBase.register("Exaone4ForCausalLM")
class Exaone4Model(TextModel):
    """Converter for LG EXAONE 4.0 (sliding-window attention layers, BPE vocab)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.EXAONE4

    def set_vocab(self):
        """Write a GPT-2 style BPE vocab with merges."""
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        """Write base parameters, vocab size and the per-layer sliding-window pattern."""
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])

        if hparams.get("sliding_window") is not None:
            self.gguf_writer.add_sliding_window(hparams["sliding_window"])
            # Newer configs list the type of each layer explicitly.
            if "layer_types" in hparams:
                self.gguf_writer.add_sliding_window_pattern([t == "sliding_attention" for t in hparams["layer_types"]])
            # Older configs encode a repeating pattern: a string like "LLLG" (L = local,
            # G = global) or an integer N meaning every Nth layer is global.
            elif "sliding_window_pattern" in hparams:
                sliding_window_pattern = []
                if isinstance(hparams["sliding_window_pattern"], str):  # e.g. LLLG
                    for i in range(hparams["num_hidden_layers"]):
                        sliding_window_pattern.append(hparams["sliding_window_pattern"][i % len(hparams["sliding_window_pattern"])] == "L")
                if isinstance(hparams["sliding_window_pattern"], int):  # e.g. 4
                    for i in range(hparams["num_hidden_layers"]):
                        sliding_window_pattern.append((i + 1) % hparams["sliding_window_pattern"] != 0)
                # Only write the pattern if it covers every layer (i.e. the format was understood).
                if len(sliding_window_pattern) == hparams["num_hidden_layers"]:
                    self.gguf_writer.add_sliding_window_pattern(sliding_window_pattern)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        """Emit the precomputed Llama-3 RoPE frequency factors, if the config asks for them.

        Implements Llama 3's long-context scaling: high-frequency components are left
        alone, low-frequency ones are divided by `factor`, and the band in between is
        smoothly interpolated.

        EXAONE 4 defaults the scaling factor to 16 rather than Llama's 8.
        """
        if rope_params := self.rope_parameters.get("full_attention", self.rope_parameters):
            if rope_params.get("rope_type", '').lower() == "llama3":
                base = rope_params.get("rope_theta", 10_000.0)
                if (dim := self.hparams.get("head_dim")) is None:
                    dim = self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
                freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

                factor = rope_params.get("factor", 16.0)
                low_freq_factor = rope_params.get("low_freq_factor", 1.0)
                high_freq_factor = rope_params.get("high_freq_factor", 4.0)
                old_context_len = self.hparams.get("original_max_position_embeddings", 8192)

                low_freq_wavelen = old_context_len / low_freq_factor
                high_freq_wavelen = old_context_len / high_freq_factor

                rope_factors = []
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


@ModelBase.register("ExaoneMoEForCausalLM")
class ExaoneMoEModel(Exaone4Model):
    """Converter for LG EXAONE MoE, which adds routed/shared experts and NextN (MTP) layers."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.EXAONE_MOE

    def __init__(self, *args, **kwargs):
        """Extend the block count with NextN layers and rebuild the tensor name map for it."""
        super().__init__(*args, **kwargs)
        self.block_count = self.hparams["num_hidden_layers"] + self.hparams.get("num_nextn_predict_layers", 0)
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def set_gguf_parameters(self):
        """Add MoE sizing/routing metadata and the NextN layer count."""
        super().set_gguf_parameters()
        moe_intermediate_size = self.hparams["moe_intermediate_size"]
        num_shared_experts = self.hparams["num_shared_experts"]
        self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)
        self.gguf_writer.add_expert_shared_count(num_shared_experts)
        self.gguf_writer.add_expert_shared_feed_forward_length(moe_intermediate_size * num_shared_experts)
        self.gguf_writer.add_expert_weights_scale(self.hparams["routed_scaling_factor"])
        self.gguf_writer.add_expert_weights_norm(self.hparams["norm_topk_prob"])
        # The leading-dense-layer count appears under two different key names.
        n_dense_layer = self.hparams.get("first_k_dense_replace", self.hparams.get("first_last_k_dense_replace", 0))
        self.gguf_writer.add_leading_dense_block_count(n_dense_layer)
        self.gguf_writer.add_nextn_predict_layers(self.hparams.get("num_nextn_predict_layers", 0))

        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)

    # Per-layer buffers for expert tensors awaiting merge.
    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Remap MTP tensors onto extra layers, merge routed experts, pass others through.

        Generator: yields nothing while a layer's experts are still being collected.
        """
        # MTP (multi-token prediction) weights live under "mtp." and are stored as
        # extra decoder layers numbered after the real ones.
        if name.startswith("mtp."):
            if name.find("layers.") != -1:
                # per-layer MTP weights: shift the index by num_hidden_layers.
                # `mtp.layers.0.[module_name]` format
                name = name.replace(f"mtp.layers.{bid}", f"model.layers.{bid + self.hparams['num_hidden_layers']}")
            else:
                # mtp fc/norm weights
                # Standalone MTP projection/norm weights map onto the standard NextN tensor names.
                remapper = {
                    "mtp.fc": "model.layers.{bid}.eh_proj",
                    "mtp.pre_fc_norm_embedding": "model.layers.{bid}.enorm",
                    "mtp.pre_fc_norm_hidden": "model.layers.{bid}.hnorm",
                    "mtp.norm": "model.layers.{bid}.shared_head.norm",
                }
                _n = Path(name)
                new_name = remapper[_n.stem] + _n.suffix

                # These weights are shared across all NextN layers, so emit one copy per layer
                # (note: the loop variable `bid` deliberately shadows the argument).
                # set shared weights for all NextN/MTP layers
                for bid in range(self.hparams['num_hidden_layers'], self.block_count):
                    yield from super().modify_tensors(data_torch, new_name.format(bid=bid), bid)
                return

        # Routed experts: buffer per layer and merge when the layer is complete.
        if name.find("mlp.experts") != -1:
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None

            # Experts arrive as separate per-expert tensors; buffer them per layer
            # (block id) until the whole set for that layer has been seen.
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            # Layer complete once all {n_experts} experts have delivered their 3 weights
            # (gate/up/down), i.e. n_experts * 3 buffered tensors.
            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    # Stack the per-expert 2D matrices into [n_experts, ...].
                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    new_name = self.map_tensor_name(merged_name)

                    yield from super().modify_tensors(data_torch, new_name, bid)
                return
            else:
                return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        """Run the normal conversion, then verify every expert tensor was merged."""
        super().prepare_tensors()
        # Anything still buffered here means the checkpoint was missing expert
        # tensors, so fail loudly instead of writing a broken GGUF.
        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")
