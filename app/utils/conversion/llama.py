# ==============================================================================
# Program: llama.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for LLAMA.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

import json
import math

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register(
    # One converter serves all these HF architectures: they share the Llama
    # decoder layout (Mistral, Mixtral, Llava text tower, Voxtral, ...).
    "LLaMAForCausalLM",
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "VLlama3ForCausalLM",
    "LlavaForConditionalGeneration",
    "VoxtralForConditionalGeneration",
    "IQuestCoderForCausalLM",
    "LlamaModel")
class LlamaModel(TextModel):
    """Converter for Llama-family decoder models (also base class of many others).

    Class attributes: model_arch selects the GGUF architecture;
    undo_permute controls whether Q/K weights are un-permuted from the HF
    rotary layout back to the original layout llama.cpp expects.
    """
    model_arch = gguf.MODEL_ARCH.LLAMA
    # Subclasses whose checkpoints are not stored in the HF-permuted layout set this False.
    undo_permute = True

    def __init__(self, *args, **kwargs):
        """Initialise the base converter and record the original HF architecture name."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(*args, **kwargs)
        # SmolVLM2 configs omit this key but later code requires it; 32 is the default used here.
        # fix for SmolVLM2, missing `num_attention_heads` in config.json
        if self.hf_arch == "VLlama3ForCausalLM":
            self.hparams["num_attention_heads"] = self.hparams.get("num_attention_heads", 32)
        # The Mistral-native format ships no config.json, so there is no HF arch to read.
        # Mistral consolidated format has no config.json; origin_hf_arch is HF-only.
        if self.is_mistral_format:
            self.origin_hf_arch = None
        else:
            hparams = ModelBase.load_hparams(self.dir_model, is_mistral_format=False)
            self.origin_hf_arch = hparams.get('architectures', [None])[0]

    def set_vocab(self):
        """Pick the right vocab loader for this checkpoint.

        Tries Mistral/tekken, then SentencePiece, then the HF Llama tokenizer, and
        finally GPT-2 style BPE (what Llama 3 uses).
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        # Special-cased architectures return early with their own vocab loaders.
        if self.origin_hf_arch == "GlmasrModel":
            return self._set_vocab_glmedge()

        if self.is_mistral_format:
            return self._set_vocab_mistral()

        # A tekken.json with no tokenizer.json indicates a Mistral "tekken" tokenizer.
        path_tekken_json = self.dir_model / "tekken.json"
        path_tokenizer_json = self.dir_model / "tokenizer.json"
        if path_tekken_json.is_file() and not path_tokenizer_json.is_file():
            self._set_vocab_mistral()

        # Read add_prefix_space and detect the HybridDNA tokenizer from tokenizer_config.json.
        tokenizer_config_file = self.dir_model / 'tokenizer_config.json'
        if tokenizer_config_file.is_file():
            with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                tokenizer_config_json = json.load(f)
                if (add_prefix_space := tokenizer_config_json.get("add_prefix_space")) is not None:
                    self.gguf_writer.add_add_space_prefix(add_prefix_space)
                if tokenizer_config_json.get("tokenizer_class") == "HybridDNATokenizer":
                    return self._set_vocab_hybriddna()

        try:
            # Execute guarded operations and establish recovery boundaries
            # Fallback chain: SentencePiece -> HF llama tokenizer -> GPT-2 BPE (Llama 3).
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            try:
                # Execute guarded operations and establish recovery boundaries
                self._set_vocab_llama_hf()
            except (FileNotFoundError, TypeError):
                # Llama 3
                self._set_vocab_gpt2()

        # CodeLlama has vocab 32016 and needs its fill-in-the-middle token ids registered
        # (the ids are fixed by the original model release).
        # Apply to CodeLlama only (and ignore for Llama 3 with a vocab size of 128256)
        if self.hparams.get("vocab_size", 32000) == 32016:
            special_vocab = gguf.SpecialVocab(
                self.dir_model, load_merges=False,
                special_token_types = ['prefix', 'suffix', 'middle', 'eot']
            )
            special_vocab._set_special_token("prefix", 32007)
            special_vocab._set_special_token("suffix", 32008)
            special_vocab._set_special_token("middle", 32009)
            special_vocab._set_special_token("eot",    32010)
            special_vocab.add_to_gguf(self.gguf_writer)

        # Vocab size 49152 identifies the small Granite models, which do not add BOS.
        # Apply to granite small models only
        if self.hparams.get("vocab_size", 32000) == 49152:
            self.gguf_writer.add_add_bos_token(False)

    def set_gguf_parameters(self):
        """Write base parameters, vocab size and the RoPE dimension count."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()
        hparams = self.hparams

        if not self.is_mistral_format:
            self.gguf_writer.add_vocab_size(hparams["vocab_size"])

        # Prefer an explicit head_dim; otherwise derive it from hidden size / heads.
        if (rope_dim := hparams.get("head_dim")) is None:
            rope_dim = hparams["hidden_size"] // hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(rope_dim)

    @staticmethod
    def permute(weights: Tensor, n_head: int, n_head_kv: int | None):
        """Reorder Q/K weights between the HF rotary layout and the llama.cpp layout.

        With grouped-query attention the K projection has n_head_kv heads, so that
        count is used for the reshape.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        # K/V use fewer heads than Q under GQA; reshape with the KV head count.
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))

    def _repack_nvfp4(self, name: str, weight: Tensor, scale: Tensor, scale2: Tensor, input_scale: Tensor):
        """Apply the Q/K permutation to NVFP4-quantised weights before repacking.

        Both the packed weight and its per-block scale must be permuted together.
        """
        # Mirror the BF16 Q/K RoPE permutation site in modify_tensors; the NVFP4 path bypasses it.
        if self.undo_permute:
            n_head = self.find_hparam(["n_heads", "num_attention_heads"], optional=True)
            n_kv_head = self.find_hparam(["n_kv_heads", "num_key_value_heads"], optional=True)
            if n_head is not None:
                if name.endswith("q_proj.weight"):
                    weight = LlamaModel.permute(weight, n_head, n_head)
                    scale  = LlamaModel.permute(scale, n_head, n_head)
                elif name.endswith("k_proj.weight"):
                    weight = LlamaModel.permute(weight, n_head, n_kv_head)
                    scale  = LlamaModel.permute(scale, n_head, n_kv_head)
        super()._repack_nvfp4(name, weight, scale, scale2, input_scale)

    # Per-layer expert buffers used by modify_tensors/prepare_tensors.
    _experts: list[dict[str, Tensor]] | None = None

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Strip the SmolVLM "text_model." prefix before the base class sees the name."""
        # Step 1: Validate input parameters, operational preconditions, and state
        name, gen = item

        if "text_model." in name:
            name = name.replace("text_model.", "") # for SmolVLM

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Un-permute Q/K and merge Mixtral-style experts; pass other tensors through.

        Generator: yields nothing while a layer's experts are still being collected.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        n_head = self.find_hparam(["n_heads", "num_attention_heads"])
        n_kv_head = self.find_hparam(["n_kv_heads", "num_key_value_heads"])

        # Bare LlamaModel checkpoints omit the "model." prefix the tensor map expects.
        if self.hf_arch == "LlamaModel":
            name = "model." + name

        # Un-permute Q/K weights and biases from the HF rotary layout; K uses the KV head count.
        if self.undo_permute:
            if name.endswith(("q_proj.weight", "q_proj.bias")):
                data_torch = LlamaModel.permute(data_torch, n_head, n_head)
            if name.endswith(("k_proj.weight", "k_proj.bias")):
                data_torch = LlamaModel.permute(data_torch, n_head, n_kv_head)

        # Mixtral experts are recognised by name; all other tensors fall through.
        # process the experts separately
        if name.find("block_sparse_moe.experts") != -1:
            n_experts = self.hparams["num_local_experts"]

            assert bid is not None

            # Experts arrive as separate per-expert tensors; buffer them per layer
            # (block id) until the whole set for that layer has been seen.
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            # 3 weights (w1/w2/w3) x n_experts tensors present: the layer is complete.
            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for wid in ["w1", "w2", "w3"]:
                    # Iterate over items, evaluate loop invariant, and update progress
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        # Iterate over items, evaluate loop invariant, and update progress
                        ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{wid}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    # Stack the per-expert 2D matrices into [n_experts, ...].
                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"layers.{bid}.feed_forward.experts.{wid}.weight"

                    yield from super().modify_tensors(data_torch, merged_name, bid)
                return
            else:
                return

        yield from super().modify_tensors(data_torch, name, bid)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        """Emit the precomputed Llama-3 RoPE frequency factors, if the config asks for them.

        The factors implement Llama 3's long-context scaling: high-frequency
        components are left alone, low-frequency ones are divided by `factor`, and
        the band in between is smoothly interpolated.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        if rope_params := self.rope_parameters.get("full_attention", self.rope_parameters):
            if rope_params.get("rope_type", '').lower() == "llama3":
                base = rope_params.get("rope_theta", 10000.0)
                if (dim := self.hparams.get("head_dim")) is None:
                    dim = self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
                # Standard RoPE inverse frequencies for each pair of dimensions.
                freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

                # Defaults match the published Llama 3.1 scaling parameters.
                factor = rope_params.get("factor", 8.0)
                low_freq_factor = rope_params.get("low_freq_factor", 1.0)
                high_freq_factor = rope_params.get("high_freq_factor", 4.0)
                old_context_len = self.hparams.get("original_max_position_embeddings", 8192)

                # Wavelength cut-offs: shorter than "high" -> unchanged, longer than "low" -> fully scaled.
                low_freq_wavelen = old_context_len / low_freq_factor
                high_freq_wavelen = old_context_len / high_freq_factor
                # assert low_freq_wavelen != high_freq_wavelen # Errors for Llama4

                rope_factors = []
                # Per-frequency scale factor: 1 (no change), `factor` (full), or a smooth blend.
                for freq in freqs:
                    # Iterate over items, evaluate loop invariant, and update progress
                    wavelen = 2 * math.pi / freq
                    if wavelen < high_freq_wavelen:
                        rope_factors.append(1)
                    elif wavelen > low_freq_wavelen:
                        rope_factors.append(factor)
                    else:
                        smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                        rope_factors.append(1 / ((1 - smooth) / factor + smooth))

                # Stored as a tensor so llama.cpp can apply it without recomputing.
                yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FREQS), torch.tensor(rope_factors, dtype=torch.float32))

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


@ModelBase.register("ArceeForCausalLM")
class ArceeModel(LlamaModel):
    """Arcee (AFM) dense model: Llama layout with its own GGUF architecture id."""
    model_arch = gguf.MODEL_ARCH.ARCEE

    def set_gguf_parameters(self):
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()
        # Writes a pooling type only if the model config defines one (embedding use).
        self._try_set_pooling_type()


@ModelBase.register(
    "Llama4ForConditionalGeneration",
    "Llama4ForCausalLM",
)
class Llama4Model(LlamaModel):
    """Llama 4 text model (MoE). Weights are not stored permuted, hence undo_permute=False."""
    model_arch = gguf.MODEL_ARCH.LLAMA4
    undo_permute = False

    def __init__(self, *args, **kwargs):
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(*args, **kwargs)
        # IMPORTANT: the normal "intermediate_size" is renamed to "intermediate_size_mlp", we need to undo this
        self.hparams["intermediate_size_moe"] = self.hparams["intermediate_size"]
        self.hparams["intermediate_size"] = self.hparams["intermediate_size_mlp"]

    def set_vocab(self):
        """Llama 4 uses a GPT-2 style BPE tokenizer."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        """Add Llama 4 MoE layout parameters and disable SWA when every layer is full attention."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()
        self.gguf_writer.add_interleave_moe_layer_step(self.hparams["interleave_moe_layer_step"])
        self.gguf_writer.add_expert_feed_forward_length(self.hparams["intermediate_size_moe"])
        if "layer_types" in self.hparams:
            if all(lt == "full_attention" for lt in self.hparams["layer_types"]):
                # all layers are full attention (for MobileLLM), disable swa
                self.gguf_writer.add_sliding_window(0)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
        """Split the fused gate_up projection and transpose weights stored as [in, out].

        Llama 4 stores experts fused and transposed relative to llama.cpp.
        """
        # gate_up_proj holds gate and up side by side on the last axis; after the
        # transpose they are on dim -2, so split there into two equal halves.
        # split the gate_up into gate and up
        if "gate_up_proj" in name:
            name_up = name.replace("gate_up_proj", "up_proj.weight")
            name_gate = name.replace("gate_up_proj", "gate_proj.weight")
            dim_half = data_torch.shape[-1] // 2
            gate_proj_weight, up_proj_weight = data_torch.transpose(-1, -2).split(dim_half, dim=-2)
            yield from super().modify_tensors(gate_proj_weight, name_gate, bid)
            yield from super().modify_tensors(up_proj_weight, name_up, bid)
            return

        # down_proj has no ".weight" suffix in this format; add it and transpose.
        if name.endswith("down_proj"):
            name += ".weight"
            data_torch = data_torch.transpose(-1, -2)

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("LlamaBidirectionalModel")
class LlamaEmbedNemotronModel(LlamaModel):
    """Bidirectional Llama used as an embedding model (separate GGUF arch)."""
    model_arch = gguf.MODEL_ARCH.LLAMA_EMBED


@ModelBase.register("SmolLM3ForCausalLM")
class SmolLM3Model(LlamaModel):
    """SmolLM3: Llama layout with its own GGUF architecture id."""
    model_arch = gguf.MODEL_ARCH.SMOLLM3


@ModelBase.register("ApertusForCausalLM")
class ApertusModel(LlamaModel):
    """Swiss AI Apertus model, which uses the learnable xIELU activation.

    The per-layer xIELU scalars (alpha_n, alpha_p, beta, eps) are collected as they
    stream past and written to GGUF as per-layer arrays once all layers are seen.
    """
    model_arch = gguf.MODEL_ARCH.APERTUS
    undo_permute = False

    # Class-level dicts (shared across instances) keyed by block id, filled as tensors arrive.
    _alpha_n = {}
    _alpha_p = {}
    _beta = {}
    _eps = {}

    def modify_tensors(self, data_torch, name, bid):
        """Capture xIELU scalars into GGUF metadata; pass everything else through."""
        # Each branch stores the scalar for this block and writes the metadata array
        # only when every layer has reported, in layer order (sorted by block id).
        # The tensor itself is consumed, not emitted (hence the bare return).
        # Handle xIELU activation parameters
        n_layers = self.hparams["num_hidden_layers"]
        if name.endswith(".act_fn.alpha_n"):
            self._alpha_n[bid] = data_torch.to("cpu").float().item()
            if (len(self._alpha_n) == n_layers):
                self.gguf_writer.add_xielu_alpha_n([self._alpha_n[k] for k in sorted(self._alpha_n)])
            return
        if name.endswith(".act_fn.alpha_p"):
            self._alpha_p[bid] = data_torch.to("cpu").float().item()
            if (len(self._alpha_p) == n_layers):
                self.gguf_writer.add_xielu_alpha_p([self._alpha_p[k] for k in sorted(self._alpha_p)])
            return
        if name.endswith(".act_fn.beta"):
            self._beta[bid] = data_torch.to("cpu").float().item()
            if (len(self._beta) == n_layers):
                self.gguf_writer.add_xielu_beta([self._beta[k] for k in sorted(self._beta)])
            return
        if name.endswith(".act_fn.eps"):
            self._eps[bid] = data_torch.to("cpu").float().item()
            if (len(self._eps) == n_layers):
                self.gguf_writer.add_xielu_eps([self._eps[k] for k in sorted(self._eps)])
            return

        yield from super().modify_tensors(data_torch, name, bid)
