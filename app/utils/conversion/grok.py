# ==============================================================================
# Program: grok.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for GROK.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

import sys

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("GrokForCausalLM", "Grok1ForCausalLM")
class GrokModel(TextModel):
    """Converter for xAI Grok-1 / Grok-2 (MoE with logit soft-capping and YaRN)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.GROK

    def set_vocab(self):
        """Use SentencePiece if available, otherwise require the Grok-2 tokenizer files."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if (self.dir_model / 'tokenizer.model').is_file():
            self._set_vocab_sentencepiece()
            return

        # Grok-2 weights ship without a tokenizer; the user must supply tokenizer.json and
        # chat_template.jinja from the linked community repo.
        if not (self.dir_model / 'tokenizer.json').is_file() or not (self.dir_model / 'chat_template.jinja').is_file():
            logger.error('Error: Missing vocab and chat template, download files from https://huggingface.co/alvarobartt/grok-2-tokenizer')
            sys.exit(1)

        self._set_vocab_gpt2()

    def __init__(self, *args, **kwargs):
        """No Grok-specific initialisation; defers to the base class."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(*args, **kwargs)

    def set_gguf_parameters(self):
        """Write soft-capping, MoE, YaRN RoPE scaling and scaling multipliers."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()

        self.gguf_writer.add_attn_logit_softcapping(self.hparams.get("attn_logit_softcapping", 30.0))
        self.gguf_writer.add_router_logit_softcapping(self.hparams.get("router_logit_softcapping", 30.0))
        if (final_logit_softcap := self.hparams.get("final_logit_softcapping")):
            self.gguf_writer.add_final_logit_softcapping(final_logit_softcap)

        if (rope_dim := self.hparams.get("head_dim")) is None:
            rope_dim = self.hparams["hidden_size"] // self.hparams["num_attention_heads"]

        if (moe_intermediate_size := self.hparams.get("moe_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)

        # (see note) configs that say "original" are handled as YaRN.
        # Treat "original" as "yarn", seems to have been a mistake
        if self.hparams.get("rope_type") in ("yarn", "original"):
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
            self.gguf_writer.add_rope_scaling_factor(self.hparams["scaling_factor"])
            self.gguf_writer.add_rope_scaling_orig_ctx_len(self.hparams["original_max_position_embeddings"])
            self.gguf_writer.add_rope_scaling_yarn_ext_factor(self.hparams["extrapolation_factor"])
            self.gguf_writer.add_rope_scaling_yarn_attn_factor(self.hparams["attn_factor"])
            self.gguf_writer.add_rope_scaling_yarn_beta_fast(self.hparams["beta_fast"])
            self.gguf_writer.add_rope_scaling_yarn_beta_slow(self.hparams["beta_slow"])

        if temp_len := self.hparams.get("attn_temperature_len"):
            self.gguf_writer.add_attn_temperature_length(temp_len)

        # Attention output scale defaults to 1/sqrt(head size) when not configured.
        self.gguf_writer.add_attn_output_scale(self.hparams.get("attn_output_multiplier", rope_dim**-0.5))
        self.gguf_writer.add_embedding_scale(self.hparams["embedding_multiplier_scale"])
        self.gguf_writer.add_logit_scale(self.hparams["output_multiplier_scale"])

    # Per-layer buffers: each expert weight name maps to a list of pieces, because
    # some checkpoints split one expert tensor across several files.
    _experts: list[dict[str, list[Tensor]]] | None = None
    # Name of the expert tensor currently being accumulated ("" when not inside a run).
    _cur_expert = ""

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Collect (possibly split) expert tensors and merge them once a layer is complete.

        Non-expert tensors are "deferred" and emitted after any merged experts, so
        merging happens when the first non-expert tensor after an expert run arrives.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        # The current tensor, if it is not an expert, is emitted at the very end.
        deferred: list[tuple[Tensor, str, int | None]] = []
        is_expert = ".moe." in name or ".block_sparse_moe.experts." in name

        if not is_expert:
            deferred.append((data_torch, name, bid))

        # Enter the expert path for expert tensors, or for the first tensor after an
        # expert run (that is what triggers the merge step).
        # process the experts separately
        if is_expert or self._cur_expert:
            n_experts = self.hparams["num_local_experts"]

            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            # A name seen again is another piece of the same tensor: keep appending.
            # concatenate split tensors
            if name in self._experts[bid]:
                self._cur_expert = name
                self._experts[bid][name].append(data_torch)
                return
            elif is_expert:
                self._cur_expert = name
                self._experts[bid][name] = [data_torch]
                return
            else:
                self._cur_expert = ""

            # Check every layer: a layer is complete when all n_experts * 3 weights are held.
            for bid in range(self.block_count):
                # Iterate over items, evaluate loop invariant, and update progress
                if len(self._experts[bid]) >= n_experts * 3:
                    # merge the experts into a single 3d tensor
                    # (Grok name, Mixtral-style name, concat dim): the pieces of w1/w3 are joined on
                    # dim 0 and w2 on dim 1 when they were split across files.
                    for wid in [("linear", "w1", 0), ("linear_1", "w2", 1), ("linear_v", "w3", 0)]:
                        # Iterate over items, evaluate loop invariant, and update progress
                        datas: list[Tensor] = []

                        for xid in range(n_experts):
                            # Iterate over items, evaluate loop invariant, and update progress
                            # Support both the original Grok checkpoint names and the HF Mixtral-style names.
                            ename = f"transformer.decoder_layer.{bid}.moe.{xid}.{wid[0]}.weight"
                            if ename not in self._experts[bid]:
                                ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{wid[1]}.weight"
                            tensor_list = self._experts[bid][ename]
                            datas.append(torch.cat(tensor_list, dim=wid[2]) if len(tensor_list) > 1 else tensor_list[0])
                            del self._experts[bid][ename]

                        data_torch = torch.stack(datas, dim=0)

                        merged_name = f"transformer.decoder_layer.{bid}.moe.{wid[0]}.weight"

                        yield from super().modify_tensors(data_torch, merged_name, bid)

        # Emit the pending non-expert tensor after merged experts.
        for t in deferred:
            # Iterate over items, evaluate loop invariant, and update progress
            yield from super().modify_tensors(*t)
