# ==============================================================================
# Program: lfm2.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for LFM2.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, TextModel, gguf

# Reuses the Conformer audio helpers (is_audio_tensor, batch-norm folding).
from .gemma import ConformerAudioModel


@ModelBase.register("Lfm2ForCausalLM", "LFM2ForCausalLM")
class LFM2Model(TextModel):
    """Converter for Liquid AI LFM2 (hybrid short-convolution + attention layers)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.LFM2

    def _add_feed_forward_length(self):
        """Compute and write the FFN width, reproducing LFM2's "auto adjust" rule.

        With block_auto_adjust_ff_dim the width is 2/3 of ff_dim, optionally scaled by
        block_ffn_dim_multiplier, then rounded up to a multiple of block_multiple_of.
        """
        ff_dim = self.find_hparam(["block_ff_dim", "intermediate_size"])
        auto_adjust_ff_dim = self.hparams["block_auto_adjust_ff_dim"]
        ffn_dim_multiplier = self.hparams["block_ffn_dim_multiplier"]
        multiple_of = self.hparams["block_multiple_of"]

        if auto_adjust_ff_dim:
            # SwiGLU-style adjustment: only 2/3 of the nominal width is used.
            ff_dim = int(2 * ff_dim / 3)
            # custom dim factor multiplier
            if ffn_dim_multiplier is not None:
                ff_dim = int(ffn_dim_multiplier * ff_dim)
            ff_dim = multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)

        self.gguf_writer.add_feed_forward_length(ff_dim)

    def set_gguf_parameters(self):
        """Write base parameters, with KV heads set only for attention layers."""
        # convolution layers have no KV cache, so their KV head count is 0.
        # set num_key_value_heads only for attention layers
        self.hparams["num_key_value_heads"] = [
            self.hparams["num_key_value_heads"] if layer_type != "conv" else 0
            for layer_type in self.hparams["layer_types"]
        ]

        super().set_gguf_parameters()
        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])
        self.gguf_writer.add_shortconv_l_cache(self.hparams["conv_L_cache"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["norm_eps"])
        self._add_feed_forward_length()

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Drop multimodal audio tensors and map "lfm." prefixes to "model."."""
        name, gen = item

        if ConformerAudioModel.is_audio_tensor(name):
            # skip multimodal tensors
            return None

        name = name.replace("lfm.", "model.")      # audio

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Squeeze the depthwise conv weight to 2-D."""
        # the stored kernel has a singleton channel axis at dim 1.
        # conv op requires 2d tensor
        if 'conv.conv' in name:
            data_torch = data_torch.squeeze(1)

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Lfm2Model")
class LFM2ColBertModel(LFM2Model):
    """LFM2 ColBERT retrieval model: LFM2 backbone plus an extra dense output projection."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.LFM2
    # Name of the extra projection tensor appended after the backbone.
    dense_tensor_name = "dense_2"

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Prefix backbone tensor names with "model." (the dense projection is kept as is)."""
        if not name.startswith(self.dense_tensor_name):
            name = "model." + name

        yield from super().modify_tensors(data_torch, name, bid)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        """Load the dense projection from its separate safetensors file and emit it."""
        # lives in 1_Dense/model.safetensors per the sentence-transformers layout.
        # dense tensor is stored in a separate safetensors file
        from safetensors.torch import load_file
        tensors_file = self.dir_model / "1_Dense" / "model.safetensors"
        assert tensors_file.is_file()
        tensor = load_file(tensors_file)["linear.weight"]
        self.gguf_writer.add_embedding_length_out(tensor.shape[0])
        yield f"{self.dense_tensor_name}.weight", tensor.clone()


@ModelBase.register("Lfm2MoeForCausalLM")
class LFM2MoeModel(TextModel):
    """Converter for LFM2-MoE (hybrid conv/attention with sigmoid-gated experts)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.LFM2MOE

    def set_gguf_parameters(self):
        """Write base parameters, MoE sizing, sigmoid gating and the conv cache size."""
        # set num_key_value_heads only for attention layers
        self.hparams["num_key_value_heads"] = [
            self.hparams["num_key_value_heads"] if layer_type == "full_attention" else 0
            for layer_type in self.hparams["layer_types"]
        ]

        super().set_gguf_parameters()

        self.gguf_writer.add_expert_feed_forward_length(self.hparams["moe_intermediate_size"])
        self.gguf_writer.add_leading_dense_block_count(self.hparams["num_dense_layers"])
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)

        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])
        self.gguf_writer.add_shortconv_l_cache(self.hparams["conv_L_cache"])

    # keyed by layer (block id); a layer's entry is deleted once merged.
    # cache for experts weights for merging
    _experts_cache: dict[int, dict[str, Tensor]] = {}

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Rename the router bias so it matches the GGUF tensor name table."""
        name, gen = item

        if name.endswith(".expert_bias"):
            name = name.replace(".expert_bias", ".expert_bias.bias")

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Merge routed experts per layer (w1/w2/w3) and squeeze conv weights.

        Generator: yields nothing until the whole layer's experts have been collected.
        """
        # conv op requires 2d tensor
        if 'conv.conv' in name:
            data_torch = data_torch.squeeze(1)

        # All experts of a layer are merged in one go once every w1/w2/w3 tensor is present.
        # merge expert weights
        if 'experts' in name:
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None

            expert_cache = self._experts_cache.setdefault(bid, {})
            expert_cache[name] = data_torch
            expert_weights = ["w1", "w2", "w3"]

            # Layer incomplete: wait for more tensors.
            # not enough expert weights to merge
            if len(expert_cache) < n_experts * len(expert_weights):
                return

            for w_name in expert_weights:
                datas: list[Tensor] = []

                for xid in range(n_experts):
                    ename = f"model.layers.{bid}.feed_forward.experts.{xid}.{w_name}.weight"
                    datas.append(expert_cache[ename])
                    del expert_cache[ename]

                data_torch = torch.stack(datas, dim=0)
                merged_name = f"layers.{bid}.feed_forward.experts.{w_name}.weight"

                yield from super().modify_tensors(data_torch, merged_name, bid)

            del self._experts_cache[bid]
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        """Run the normal conversion, then assert no expert tensors remain cached."""
        super().prepare_tensors()
        assert not self._experts_cache


@ModelBase.register("Lfm2VlForConditionalGeneration")
class LFM2VLModel(MmprojModel):
    """Converter for the LFM2-VL vision encoder and projector (mmproj GGUF)."""
    def __init__(self, *args, **kwargs):
        """Set a nominal image size (dynamic-resolution encoder)."""
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        # TODO(tarek): for dynamic resolution image_size is not specified, setting here for compatibility
        self.hparams_vision["image_size"] = 256

    def set_gguf_parameters(self):
        """Write the LFM2 projector type, norm epsilon, downsample factor and layer count."""
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.LFM2)
        self.gguf_writer.add_vision_attention_layernorm_eps(self.find_vparam(["layer_norm_eps"]))
        self.gguf_writer.add_vision_projector_scale_factor(self.global_config.get("downsample_factor", 2))
        self.gguf_writer.add_vision_use_gelu(True)
        # a feature layer of -1 means "last layer", i.e. drop 0 trailing layers,
        # so the number of layers kept = total - dropped.
        # python notation, e.g. for vision_feature_layer == -1, we pick last layer -> vision_feature_layers_to_drop = 0
        vision_feature_layers_to_drop = -(self.global_config.get("vision_feature_layer", -1) + 1)
        self.gguf_writer.add_vision_block_count(self.find_vparam(self.n_block_keys) - vision_feature_layers_to_drop)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Strip the leading "model." from vision tower and projector names."""
        name, gen = item

        name = name.replace("model.vision_tower.", "vision_tower.")
        name = name.replace("model.multi_modal_projector.", "multi_modal_projector.")

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Reshape the flattened patch embedding to a [out, 3, 16, 16] convolution kernel."""
        if "patch_embedding.weight" in name:
            data_torch = data_torch.view(data_torch.shape[0], 16, 16, 3).permute(0, 3, 1, 2)

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Lfm2AudioForConditionalGeneration")
class LFM2AudioModel(ConformerAudioModel):
    """Converter for the LFM2-Audio Conformer encoder (mmproj GGUF)."""
    has_vision_encoder = False
    has_audio_encoder = True
    # Class-level default for the model name that ModelBase passes into the GGUF
    # metadata loader (normally supplied by the caller or derived from the directory).
    model_name = "Lfm2AudioEncoder"

    def get_audio_config(self) -> dict[str, Any] | None:
        """The encoder's hyperparameters live under "encoder" in the config."""
        return self.global_config.get("encoder")

    def set_gguf_parameters(self):
        """Map the encoder config onto generic key names, then write audio parameters."""
        assert self.hparams_audio is not None
        self.hparams_audio["hidden_size"] = self.hparams_audio["d_model"]
        self.hparams_audio["intermediate_size"] = self.hparams_audio["d_model"]
        self.hparams_audio["num_attention_heads"] = self.hparams_audio["n_heads"]
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.LFM2A)
        self.gguf_writer.add_audio_num_mel_bins(self.hparams_audio["feat_in"])
        self.gguf_writer.add_audio_attention_layernorm_eps(1e-5)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Drop language-model, training-only and audio-output tensors."""
        name, gen = item

        # skip language model tensors
        if name.startswith("lfm."):
            return None

        # for training only
        if any(p in name for p in ["audio_loss_weight"]):
            return None

        # for audio output
        if any(p in name for p in ["codebook_offsets", "depth_embeddings", "depth_linear", "depthformer"]):
            return None

        return super().filter_tensors(item)


@ModelBase.register("Lfm25AudioTokenizer")
class LFM25AudioTokenizer(LFM2Model):
    """Converter for the LFM2.5 audio tokenizer/detokenizer (LFM2 layout, no vocab)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.LFM2

    def set_vocab(self):
        """No vocabulary for this model."""
        self._set_vocab_none()

    def set_gguf_parameters(self):
        """Write LFM2 parameters plus sliding window and output embedding size."""
        super().set_gguf_parameters()
        self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])
        self.gguf_writer.add_embedding_length_out(self.hparams["output_size"])

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Drop the iSTFT window and embedding tensors; rename "lin*" to "dense_2_out*"."""
        name, gen = item

        # skip language model tensors
        if name == "istft.window" or name.startswith("emb.emb"):
            return None

        if name.startswith("lin"):
            name = name.replace("lin", "dense_2_out")

        return super().filter_tensors((name, gen))
