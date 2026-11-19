# ==============================================================================
# Program: glm.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for GLM.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger

# GLM-4.7-Flash and GLM-5 style models reuse DeepSeek-V2's MLA/MoE converter.
from .deepseek import DeepseekV2Model


@ModelBase.register("Glm4ForCausalLM", "Glm4vForConditionalGeneration")
class Glm4Model(TextModel):
    """Converter for Zhipu GLM-4 (dense) and the text part of GLM-4V.

    GLM-4V uses M-RoPE, which the runtime only supports in NeoX ordering, so Q/K
    weights are re-ordered when `use_mrope` is set.
    """
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.GLM4
    # Class defaults; __init__ overrides them from the model's rope parameters.
    use_mrope = False
    partial_rotary_factor = 0.5

    def __init__(self, *args, **kwargs):
        """Read the partial rotary factor and detect M-RoPE from rope_parameters."""
        super().__init__(*args, **kwargs)
        self.partial_rotary_factor = self.rope_parameters.get("partial_rotary_factor", 0.5)
        if "mrope_section" in self.rope_parameters:
            self.use_mrope = True
            logger.info("Q/K weight will need to be permuted for M-RoPE")

    def set_vocab(self):
        """Write a GPT-2 style BPE vocab and patch the special token ids GLM does not
        store in config.json (eos/eot/unk/bos).
        """
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        # Token list and pre-tokenizer id come from the shared BPE helper.
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        # GLM uses <|endoftext|> for eos/unk/bos and <|user|> as the end-of-turn marker.
        special_vocab._set_special_token("eos", tokenizer.get_added_vocab()["<|endoftext|>"])  # ty: ignore[unresolved-attribute]
        special_vocab._set_special_token("eot", tokenizer.get_added_vocab()["<|user|>"])  # ty: ignore[unresolved-attribute]
        special_vocab._set_special_token("unk", tokenizer.get_added_vocab()["<|endoftext|>"])  # ty: ignore[unresolved-attribute]
        special_vocab._set_special_token("bos", tokenizer.get_added_vocab()["<|endoftext|>"])  # ty: ignore[unresolved-attribute]
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        """Write base parameters plus the (partial) RoPE dimension count."""
        super().set_gguf_parameters()
        if (rope_dim := self.hparams.get("head_dim")) is None:
            rope_dim = self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(int(rope_dim * self.partial_rotary_factor))

    @staticmethod
    def normal_to_neox(weights: Tensor, n_head: int, n_head_kv: int, head_dim: int, partial_rotary_factor: float) -> Tensor:
        """Re-order rotary dimensions from interleaved ("normal") to split-half (NeoX) layout.

        Only the first `partial_rotary_factor` fraction of each head is rotated; within
        that part even-indexed rows are placed first, then odd-indexed ones. 1-D
        tensors (biases) are handled by temporarily adding a trailing axis.
        """
        orig_shape = weights.shape
        # Treat biases as [out_dim, 1] matrices so one code path handles both.
        if len(orig_shape) == 1:
            weights = weights.unsqueeze(1)  # [out_dim, 1]
        if len(weights.shape) != 2:
            raise ValueError("Only 1D and 2D tensors are supported.")
        # Q has n_head heads, K/V have n_head_kv; infer which one this tensor is.
        n_effective_heads = weights.shape[0] // head_dim
        if n_head_kv is not None and n_effective_heads != n_head:
            if n_effective_heads != n_head_kv:
                raise AssertionError(f"Mismatch in effective heads: computed {n_effective_heads}, expected {n_head} or {n_head_kv}")
        # Number of rows per head that take part in rotation (must be even: cos/sin pairs).
        rotary_dim = int(head_dim * partial_rotary_factor)
        if rotary_dim % 2 != 0:
            raise ValueError("rotary_dim must be even.")
        reshaped = weights.reshape(n_effective_heads, head_dim, -1)
        rot_part = reshaped[:, :rotary_dim, :]
        non_rot_part = reshaped[:, rotary_dim:, :]
        # De-interleave: [x0,x1,x2,x3,...] -> [x0,x2,...,x1,x3,...].
        permuted_rot = torch.cat((rot_part[:, ::2, :], rot_part[:, 1::2, :]), dim=1)
        combined = torch.cat((permuted_rot, non_rot_part), dim=1)
        result = combined.reshape(weights.shape)
        return result if len(orig_shape) != 1 else result.squeeze(1)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Permute Q/K to NeoX ordering when M-RoPE is in use."""
        if self.use_mrope:
            n_head = self.hparams["num_attention_heads"]
            n_kv_head = self.hparams["num_key_value_heads"]
            n_embd = self.hparams["hidden_size"]
            head_dim = self.hparams.get("head_dim", n_embd // n_head)
            # llama.cpp's M-RoPE kernel only supports NeoX ordering.
            # because llama.cpp M-RoPE kernel only supports Neox ordering, we have to permute the weights here
            if name.endswith(("q_proj.weight", "q_proj.bias")):
                data_torch = Glm4Model.normal_to_neox(data_torch, n_head, n_head, head_dim, self.partial_rotary_factor)
            if name.endswith(("k_proj.weight", "k_proj.bias")):
                data_torch = Glm4Model.normal_to_neox(data_torch, n_head, n_kv_head, head_dim, self.partial_rotary_factor)
        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("GlmOcrForConditionalGeneration")
class GlmOCRModel(Glm4Model):
    """GLM-OCR text model: identical to GLM-4 plus one NextN (MTP) layer."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.GLM4
    use_mrope = False
    partial_rotary_factor = 0.5

    # Note: GLM-OCR is the same as GLM4, but with an extra NextN/MTP prediction layer

    def __init__(self, *args, **kwargs):
        """Count the NextN layer as an extra block and rebuild the tensor name map."""
        super().__init__(*args, **kwargs)
        # GLM-OCR has num_hidden_layers + 1 actual layers (including NextN layer)
        self.block_count = self.hparams["num_hidden_layers"] + self.hparams.get("num_nextn_predict_layers", 0)
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def set_gguf_parameters(self):
        """Record how many NextN (multi-token-prediction) layers the checkpoint has."""
        super().set_gguf_parameters()
        # NextN/MTP prediction layers
        if (num_nextn_predict_layers := self.hparams.get("num_nextn_predict_layers")) is not None:
            self.gguf_writer.add_nextn_predict_layers(num_nextn_predict_layers)


@ModelBase.register("Glm4MoeForCausalLM", "Glm4vMoeForConditionalGeneration")
class Glm4MoeModel(TextModel):
    """Converter for GLM-4.5 / 4.6 MoE models (sigmoid-gated experts, NextN layers)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.GLM4_MOE

    def __init__(self, *args, **kwargs):
        """Count NextN layers as extra blocks and rebuild the tensor name map."""
        super().__init__(*args, **kwargs)
        # GLM4_MOE has num_hidden_layers + 1 actual layers (including NextN layer)
        self.block_count = self.hparams["num_hidden_layers"] + self.hparams.get("num_nextn_predict_layers", 0)
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def set_vocab(self):
        """Use the shared GLM vocab loader."""
        return self._set_vocab_glm()

    def set_gguf_parameters(self):
        """Write RoPE size, MoE sizing/routing metadata and the NextN layer count."""
        super().set_gguf_parameters()
        if (rope_dim := self.hparams.get("head_dim")) is None:
            rope_dim = (
                self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
            )
        self.gguf_writer.add_rope_dimension_count(
            int(rope_dim * self.hparams.get("partial_rotary_factor", 0.5))
        )

        # MoE parameters - Use only routed expert count (shared experts handled separately)
        if (n_routed_experts := self.hparams.get("n_routed_experts")) is not None:
            self.gguf_writer.add_expert_count(n_routed_experts)
        if (moe_intermediate_size := self.hparams.get("moe_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)
        if (n_shared_experts := self.hparams.get("n_shared_experts")) is not None:
            self.gguf_writer.add_expert_shared_count(n_shared_experts)
        if (first_k_dense_replace := self.hparams.get("first_k_dense_replace")) is not None:
            self.gguf_writer.add_leading_dense_block_count(first_k_dense_replace)

        # GLM-4 MoE routes with a sigmoid gate (not softmax).
        # Expert gating function (sigmoid for GLM4_MOE)
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)

        # Routed scaling factor
        if (routed_scaling_factor := self.hparams.get("routed_scaling_factor")) is not None:
            self.gguf_writer.add_expert_weights_scale(routed_scaling_factor)

        # Normalise topk probabilities
        if (norm_topk_prob := self.hparams.get("norm_topk_prob")) is not None:
            self.gguf_writer.add_expert_weights_norm(norm_topk_prob)

        # NextN/MTP prediction layers
        if (num_nextn_predict_layers := self.hparams.get("num_nextn_predict_layers")) is not None:
            self.gguf_writer.add_nextn_predict_layers(num_nextn_predict_layers)

    # Per-layer buffers for expert tensors awaiting merge.
    _experts: list[dict[str, Tensor]] | None = None

    # note: unlike GLM4V non-MoE, we don't need to permute Q/K here since GLM4V_MOE uses Neox ordering already
    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Map the main embedding, merge routed experts, pass the rest through.

        Generator: yields nothing while a layer's experts are still being collected.
        """
        # Only the top-level embedding is renamed; NextN layers have their own embeddings
        # under ".layers." that must keep their per-layer names.
        # Handle main token embedding (but not layer-specific NextN embeddings)
        if name == "model.embed_tokens.weight" and ".layers." not in name:
            yield from super().modify_tensors(data_torch, "token_embd.weight", bid)
            return

        # Routed experts: buffer per layer and merge when the layer is complete.
        # Handle routed experts
        if name.find("mlp.experts") != -1:
            n_experts = self.hparams["n_routed_experts"]
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


@ModelBase.register("Glm4MoeLiteForCausalLM")
class Glm4MoeLiteModel(DeepseekV2Model):
    """GLM-4.7-Flash style MoE: DeepSeek-V2 architecture with the GLM tokenizer."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.DEEPSEEK2

    def set_vocab(self):
        """Use the shared GLM vocab loader."""
        return self._set_vocab_glm()


@ModelBase.register("GlmMoeDsaForCausalLM")
class GlmMoeDsaModel(DeepseekV2Model):
    """GLM-5 (MoE with DeepSeek Sparse Attention): DeepSeek-V2 layout plus NextN layers
    and a lightning indexer. NextN layers are kept (skip_mtp = False).
    """
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.GLM_DSA
    # Keep the multi-token-prediction layers in the output.
    skip_mtp = False

    def __init__(self, *args, **kwargs):
        """Count NextN layers as extra blocks and rebuild the tensor name map."""
        super().__init__(*args, **kwargs)
        self.block_count = self.hparams["num_hidden_layers"] + self.hparams.get("num_nextn_predict_layers", 0)
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def set_vocab(self):
        """Use the shared GLM vocab loader."""
        return self._set_vocab_glm()

    def set_gguf_parameters(self):
        """Write the partial-rotary RoPE size, NextN layer count and DSA indexer parameters."""
        super().set_gguf_parameters()

        rope_dim = self.hparams["qk_rope_head_dim"]
        partial_rotary_factor = self.hparams.get("partial_rotary_factor", 1.0)
        self.gguf_writer.add_rope_dimension_count(int(rope_dim * partial_rotary_factor))

        # NextN/MTP prediction layers
        if (num_nextn_predict_layers := self.hparams.get("num_nextn_predict_layers")) is not None:
            self.gguf_writer.add_nextn_predict_layers(num_nextn_predict_layers)

        # DSA indexer parameters
        self.gguf_writer.add_indexer_head_count(self.hparams["index_n_heads"])
        self.gguf_writer.add_indexer_key_length(self.hparams["index_head_dim"])
        self.gguf_writer.add_indexer_top_k(self.hparams["index_topk"])


@ModelBase.register("SolarOpenForCausalLM")
class SolarOpenModel(Glm4MoeModel):
    """Upstage Solar-Open: GLM-4 MoE layout with a different set of special tokens."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.GLM4_MOE

    def set_vocab(self):
        """Write the BPE vocab and set Solar-Open's special token ids."""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        # Solar-Open uses <|startoftext|> as BOS and <|endoftext|> for eos/eot.
        special_vocab._set_special_token("eos", tokenizer.get_added_vocab()["<|endoftext|>"])  # ty: ignore[unresolved-attribute]
        special_vocab._set_special_token("eot", tokenizer.get_added_vocab()["<|endoftext|>"])  # ty: ignore[unresolved-attribute]
        special_vocab._set_special_token("unk", tokenizer.get_added_vocab()["<unk>"])  # ty: ignore[unresolved-attribute]
        special_vocab._set_special_token("bos", tokenizer.get_added_vocab()["<|startoftext|>"])  # ty: ignore[unresolved-attribute]
        special_vocab.add_to_gguf(self.gguf_writer)
