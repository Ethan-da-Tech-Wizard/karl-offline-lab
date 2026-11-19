# ==============================================================================
# Program: jamba.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for JAMBA.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("JambaForCausalLM")
class JambaModel(TextModel):
    """Converter for AI21 Jamba (hybrid Mamba + attention + MoE)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.JAMBA

    def set_vocab(self):
        """Use tokenizer.model if present, otherwise the HF Llama-style tokenizer."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if (self.dir_model / "tokenizer.model").is_file():
            self._set_vocab_sentencepiece()
        else:
            self._set_vocab_llama_hf()
            self.gguf_writer.add_add_space_prefix(False)

    def set_gguf_parameters(self):
        """Write hybrid SSM/attention parameters, including per-layer KV head counts."""
        # Step 1: Validate input parameters, operational preconditions, and state
        d_model = self.find_hparam(["hidden_size", "mamba_d_model"])
        # Mamba defaults used when the config omits them: conv kernel 4, state size 16.
        d_conv  = self.find_hparam(["mamba_d_conv"],  optional=True) or 4
        d_inner = self.hparams["mamba_expand"] * d_model
        d_state = self.find_hparam(["mamba_d_state"], optional=True) or 16
        # (see links) default dt_rank = ceil(d_model / 16), written as -(d_model // -16).
        # ceiling division
        # ref: https://stackoverflow.com/a/17511341/22827863
        # ref: https://github.com/state-spaces/mamba/blob/ce59daea3a090d011d6476c6e5b97f6d58ddad8b/mamba_ssm/modules/mamba_simple.py#L58
        dt_rank      = self.find_hparam(["mamba_dt_rank"], optional=True) or -(d_model // -16)
        rms_norm_eps = self.find_hparam(["layer_norm_epsilon", "rms_norm_eps"], optional=True) or 1e-6
        n_kv_head = self.hparams["num_key_value_heads"]
        attn_offset = self.hparams["attn_layer_offset"]
        attn_period = self.hparams["attn_layer_period"]
        # Per-layer KV heads: only layers at offset + k*period are attention layers; the
        # rest are Mamba layers and get 0 (no KV cache).
        n_kv_vec = [0 for _ in range(attn_offset)] + [
            n_kv_head if (i - attn_offset) % attn_period == 0 else 0 for i in range(attn_offset, self.block_count)
        ]

        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_context_length(self.find_hparam(["max_position_embeddings", "n_ctx"]))
        self.gguf_writer.add_embedding_length(d_model)
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(n_kv_vec)
        self.gguf_writer.add_ssm_conv_kernel(d_conv)
        self.gguf_writer.add_ssm_inner_size(d_inner)
        self.gguf_writer.add_ssm_state_size(d_state)
        self.gguf_writer.add_ssm_time_step_rank(dt_rank)
        self.gguf_writer.add_layer_norm_rms_eps(rms_norm_eps)
        self.gguf_writer.add_expert_count(self.find_hparam(["num_local_experts", "num_experts"]))
        self.gguf_writer.add_expert_used_count(self.find_hparam(["num_experts_per_tok", "num_experts_per_token"]))
        self.gguf_writer.add_file_type(self.ftype)

    # Per-layer buffers for expert tensors awaiting merge.
    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Normalise MoE names, merge experts, and adapt Mamba tensors (conv1d, A_log).

        Generator: yields nothing while a layer's experts are still being collected.
        """

        # Mini-Jamba names the MoE block ".moe."; unify with the ".feed_forward." naming.
        # Mini-Jamba
        name = name.replace(".moe.", ".feed_forward.")
        if bid is not None:
            moe_offset = self.hparams["expert_layer_offset"]
            moe_period = self.hparams["expert_layer_period"]

            # Layers that are not MoE layers have one plain FFN stored as "experts.0"; strip the
            # expert index so it maps like a normal FFN.
            if not (bid >= moe_offset and (bid - moe_offset) % moe_period == 0):
                name = name.replace(".experts.0.", ".")

        # MoE layers: buffer expert tensors per layer and merge once complete.
        # process the experts separately
        if ".feed_forward.experts." in name:
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
                for wid in ["down_proj", "gate_proj", "up_proj"]:
                    # Iterate over items, evaluate loop invariant, and update progress
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        # Iterate over items, evaluate loop invariant, and update progress
                        ename = f"model.layers.{bid}.feed_forward.experts.{xid}.{wid}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    # Stack the per-expert 2D matrices into [n_experts, ...].
                    data_torch = torch.stack(datas, dim=0)

                    # (see note) reuse Qwen2-MoE's merged name so the standard tensor map applies.
                    # using the same merged name as qwen2moe
                    merged_name = f"model.layers.{bid}.mlp.experts.{wid}.weight"

                    new_name = self.map_tensor_name(merged_name)

                    yield new_name, data_torch
            return

        new_name = self.map_tensor_name(name)

        # The depthwise conv1d weight has a singleton axis that GGUF does not store.
        if self.match_model_tensor_name(new_name, gguf.MODEL_TENSOR.SSM_CONV1D, bid):
            data_torch = data_torch.squeeze()

        # Mamba stores A in log-space with the sign dropped; the runtime wants A = -exp(A_log).
        if name.endswith(".A_log"):
            logger.debug("A_log --> A ==> " + new_name)
            data_torch = -torch.exp(data_torch)

        yield (new_name, data_torch)

    def prepare_tensors(self):
        """Run the normal conversion, then verify every expert tensor was merged."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().prepare_tensors()

        # Anything still buffered here means the checkpoint was missing expert
        # tensors, so fail loudly instead of writing a broken GGUF.
        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")
