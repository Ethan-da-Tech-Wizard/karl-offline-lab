# ==============================================================================
# Program: bailingmoe.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for BAILINGMOE.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("BailingMoeForCausalLM")
class BailingMoeModel(TextModel):
    """Converter for Ant Group Bailing MoE (Ling) models, version 1.

    Handles the fused query_key_value attention matrix, MoE expert merging and
    the optional L2-normalised output head.
    """
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.BAILINGMOE

    def set_vocab(self):
        """Bailing uses a GPT-2 style BPE tokenizer."""
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        """Write base parameters plus MoE routing / expert sizing metadata."""
        super().set_gguf_parameters()
        hparams = self.hparams
        if (rope_dim := hparams.get("head_dim")) is None:
            rope_dim = hparams["hidden_size"] // hparams["num_attention_heads"]

        self.gguf_writer.add_rope_dimension_count(rope_dim)
        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        # v1 applies no extra scale to routed-expert outputs, so the factor is fixed at 1.
        self.gguf_writer.add_expert_weights_scale(1.0)
        self.gguf_writer.add_expert_shared_count(hparams["num_shared_experts"])
        self.gguf_writer.add_expert_weights_norm(hparams["norm_topk_prob"])

    # Per-layer buffers for expert tensors awaiting merge (see modify_tensors).
    _experts: list[dict[str, Tensor]] | None = None

    @staticmethod
    def permute(weights: Tensor, n_head: int, n_head_kv: int | None):
        """Reorder Q/K weights from the HF rotary layout to the llama.cpp layout.

        (Same transform as LlamaModel.permute; K uses the KV head count under GQA.)
        """
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Rename/split attention tensors, merge experts, and normalise the output head.

        Generator: yields nothing while a layer's experts are still being collected.
        """
        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams.get("num_key_value_heads")
        n_embd = self.hparams["hidden_size"]
        # Prefer explicit head_dim; otherwise hidden size / heads.
        if (head_dim := self.hparams.get("head_dim")) is None:
            head_dim = n_embd // n_head

        # Needed later to recognise the output head tensor for norm_head handling.
        output_name = self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT)

        # Attention output projection: just map to the GGUF ATTN_OUT name.
        if name.endswith("attention.dense.weight"):
            yield from super().modify_tensors(data_torch, self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_OUT, bid), bid)
            return
        # Fused QKV matrix: dim -2 holds Q (n_head*head_dim) then K and V
        # (n_kv_head*head_dim each). Split it, un-permute Q and K, emit three tensors.
        elif name.endswith("query_key_value.weight"):
            q, k, v = data_torch.split([n_head * head_dim, n_kv_head * head_dim, n_kv_head * head_dim], dim=-2)

            yield from super().modify_tensors(BailingMoeModel.permute(q, n_head, n_head), self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q, bid), bid)
            yield from super().modify_tensors(BailingMoeModel.permute(k, n_head, n_kv_head), self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K, bid), bid)
            yield from super().modify_tensors(v,self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V, bid), bid)
            return
        # Routed experts: buffer per layer and merge into stacked tensors when complete.
        elif name.find("mlp.experts") != -1:
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

        # Everything else is renamed via the standard tensor map.
        new_name = self.map_tensor_name(name)

        # Some Bailing models (norm_head) L2-normalise the output embedding rows at
        # runtime in HF; bake that normalisation into the stored weights (1e-7 avoids /0).
        if new_name == output_name and self.hparams.get("norm_head"):
            data_torch = data_torch.float()
            data_torch /= torch.norm(data_torch, p=2, dim=0, keepdim=True) + 1e-7

        yield from super().modify_tensors(data_torch, new_name, bid)

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


@ModelBase.register("BailingMoeV2ForCausalLM")
class BailingMoeV2Model(TextModel):
    """Converter for Bailing MoE V2 (Ling 2.x), which adds shared experts and optional
    next-token-prediction (MTP) layers.
    """
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.BAILINGMOE2

    def __init__(self, *args, **kwargs):
        """Extend the block count with any NextN (multi-token prediction) layers."""
        super().__init__(*args, **kwargs)
        # NextN layers are stored after the regular layers, so the tensor name map must
        # cover block_count + nextn_layers blocks.
        if nextn_layers := self.hparams.get("num_nextn_predict_layers", 0):
            self.block_count = self.hparams["num_hidden_layers"] + nextn_layers
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def set_vocab(self):
        """Bailing uses a GPT-2 style BPE tokenizer."""
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        """Write base parameters, partial-rotary RoPE size, MoE metadata and NextN count."""
        super().set_gguf_parameters()
        hparams = self.hparams
        if (rope_dim := hparams.get("head_dim")) is None:
            rope_dim = hparams["hidden_size"] // hparams["num_attention_heads"]

        # V2 rotates only a fraction of each head (partial_rotary_factor, default 0.5).
        self.gguf_writer.add_rope_dimension_count(int(rope_dim * self.hparams.get("partial_rotary_factor", 0.5)))
        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        # Shared-expert FFN width; defaults to moe_intermediate_size * num_shared_experts.
        self.gguf_writer.add_expert_shared_feed_forward_length(hparams.get("moe_shared_expert_intermediate_size", hparams["moe_intermediate_size"] * hparams["num_shared_experts"]))
        self.gguf_writer.add_expert_weights_scale(hparams["routed_scaling_factor"])
        self.gguf_writer.add_expert_shared_count(hparams["num_shared_experts"])
        self.gguf_writer.add_expert_weights_norm(hparams["norm_topk_prob"])

        if (nextn_layers := self.hparams.get("num_nextn_predict_layers")) is not None:
            self.gguf_writer.add_nextn_predict_layers(nextn_layers)

    # Per-layer buffers for expert tensors awaiting merge.
    _experts: list[dict[str, Tensor]] | None = None

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Rename the router bias so it matches the GGUF tensor name table."""
        name, gen = item

        if name.endswith(".expert_bias"):
            name = name.replace(".expert_bias", ".expert_bias.bias")

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Merge per-expert weights into stacked tensors; pass others through.

        Generator: yields nothing while a layer's experts are still being collected.
        """
        # Routed experts: buffer per layer and merge into stacked tensors when complete.
        if "mlp.experts" in name:
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

                    yield from super().modify_tensors(data_torch, merged_name, bid)
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


@ModelBase.register("SarvamMoEForCausalLM", "modeling_sarvam_moe.SarvamMoEForCausalLM")
class SarvamMoEModel(BailingMoeV2Model):
    """Sarvam-MoE: BailingMoeV2 layout with full rotary and zero-mean expert bias."""
    model_arch = gguf.MODEL_ARCH.BAILINGMOE2
    # Sarvam-MoE shares the BailingMoeV2 architecture; only differences:
    #  - full rotary (no partial_rotary_factor)
    #  - expert bias is zero-mean normalized at load time

    def set_gguf_parameters(self):
        """Override the partial rotary dimension with the full rotary dimension."""
        super().set_gguf_parameters()
        hparams = self.hparams
        if (rope_dim := hparams.get("head_dim")) is None:
            rope_dim = hparams["hidden_size"] // hparams["num_attention_heads"]
        # Override the partial-rotary value written by BailingMoeV2 with the full rotary dim
        self.gguf_writer.add_rope_dimension_count(rope_dim)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Zero-centre the expert bias, then defer to the parent filter."""
        name, gen = item
        if name.endswith(".expert_bias"):
            # Sarvam normalizes expert bias to zero mean
            # Wrap the lazy generator: keep the original callable and post-process its result.
            inner = gen

            def gen():
                """Load the bias tensor and subtract its mean (zero-mean normalisation)."""
                t = inner()
                return t - t.mean()
        return super().filter_tensors((name, gen))
