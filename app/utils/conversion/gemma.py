# ==============================================================================
# Program: gemma.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for GEMMA.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

import json
import re

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, TextModel, gguf, logger


@ModelBase.register("GemmaForCausalLM")
class GemmaModel(TextModel):
    """Converter for Google Gemma 1 (2B/7B) and CodeGemma."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.GEMMA

    def set_vocab(self):
        """SentencePiece vocab plus the CodeGemma fill-in-the-middle special tokens."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._set_vocab_sentencepiece()

        # Ids are fixed by the Gemma tokenizer: prefix/suffix/middle/fsep/eot for
        # fill-in-the-middle prompting.
        # TODO: these special tokens should be exported only for the CodeGemma family
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=False,
                                          special_token_types = ['prefix', 'suffix', 'middle', 'fsep', 'eot'])
        special_vocab._set_special_token("prefix", 67)
        special_vocab._set_special_token("suffix", 69)
        special_vocab._set_special_token("middle", 68)
        special_vocab._set_special_token("fsep",   70)
        special_vocab._set_special_token("eot",    107)
        # The chat template is already emitted elsewhere; avoid writing it twice.
        special_vocab.chat_template = None  # do not add it twice
        special_vocab.add_to_gguf(self.gguf_writer)

        self.gguf_writer.add_add_space_prefix(False)

    def set_gguf_parameters(self):
        """Write Gemma's hyperparameters (all keys are required in config.json)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        hparams = self.hparams

        self.gguf_writer.add_context_length(hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
        self.gguf_writer.add_head_count(hparams["num_attention_heads"])
        # Gemma 2B uses multi-query attention; fall back to full head count if the key is absent.
        self.gguf_writer.add_head_count_kv(self.hparams["num_key_value_heads"] if "num_key_value_heads" in hparams else hparams["num_attention_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])
        self.gguf_writer.add_key_length(hparams["head_dim"])
        self.gguf_writer.add_value_length(hparams["head_dim"])
        self.gguf_writer.add_file_type(self.ftype)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Skip lm_head.weight (tied to the embeddings)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        name, gen = item

        # Drop lm_head.weight: llama.cpp ties the output matrix to the token
        # embeddings, and AutoAWQ-style exports still ship the tensor.
        # lm_head is not used in llama.cpp, while autoawq will include this tensor in model
        # To prevent errors, skip loading lm_head.weight.
        if name == "lm_head.weight":
            logger.debug(f"Skipping get tensor {name!r} in safetensors so that convert can end normally.")
            return None

        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Add 1 to RMSNorm weights: HF computes x * (1 + weight) but llama.cpp computes x * weight."""
        # ref: https://github.com/huggingface/transformers/blob/fc37f38915372c15992b540dfcbbe00a916d4fc6/src/transformers/models/gemma/modeling_gemma.py#L89
        # Bake the "+1" into the stored weights so the runtime uses the plain RMSNorm.
        if name.endswith("norm.weight"):
            data_torch = data_torch + 1

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Gemma2ForCausalLM")
class Gemma2Model(TextModel):
    """Converter for Google Gemma 2 (adds logit soft-capping and sliding-window attention)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.GEMMA2

    def set_vocab(self):
        """SentencePiece vocab without an automatic leading space."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._set_vocab_sentencepiece()

        self.gguf_writer.add_add_space_prefix(False)

    def set_gguf_parameters(self):
        """Write Gemma 2 parameters including attention/final logit soft-capping and window size."""
        # Step 1: Validate input parameters, operational preconditions, and state
        hparams = self.hparams

        self.gguf_writer.add_context_length(hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
        self.gguf_writer.add_head_count(hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(self.hparams["num_key_value_heads"] if "num_key_value_heads" in hparams else hparams["num_attention_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])
        self.gguf_writer.add_key_length(hparams["head_dim"])
        self.gguf_writer.add_value_length(hparams["head_dim"])
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_attn_logit_softcapping(
            self.hparams["attn_logit_softcapping"]
        )
        self.gguf_writer.add_final_logit_softcapping(
            self.hparams["final_logit_softcapping"]
        )
        self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Skip lm_head.weight (tied to the embeddings)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        name, gen = item

        # Drop lm_head.weight: llama.cpp ties the output matrix to the token
        # embeddings, and AutoAWQ-style exports still ship the tensor.
        # lm_head is not used in llama.cpp, while autoawq will include this tensor in model
        # To prevent errors, skip loading lm_head.weight.
        if name == "lm_head.weight":
            logger.debug(f"Skipping get tensor {name!r} in safetensors so that convert can end normally.")
            return None

        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Add 1 to RMSNorm weights (HF computes x * (1 + weight))."""
        # ref: https://github.com/huggingface/transformers/blob/fc37f38915372c15992b540dfcbbe00a916d4fc6/src/transformers/models/gemma/modeling_gemma.py#L89
        if name.endswith("norm.weight"):
            data_torch = data_torch + 1

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Gemma3ForCausalLM", "Gemma3ForConditionalGeneration")
class Gemma3Model(TextModel):
    """Converter for Google Gemma 3 text models (also the text tower of multimodal Gemma 3).

    Subclasses customise the RMSNorm shift via norm_shift().
    """
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.GEMMA3

    def norm_shift(self, name: str) -> float:
        """Constant added to each norm weight (1.0 for Gemma3RMSNorm, which computes x * (1 + w))."""
        # Step 1: Validate input parameters, operational preconditions, and state
        return 1.0 if name.endswith("norm.weight") else 0.0  # Gemma3RMSNorm adds 1.0 to the norm value

    def set_vocab(self):
        """SentencePiece vocab when tokenizer.model exists, otherwise GPT-2 style BPE."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if (self.dir_model / "tokenizer.model").is_file():
            self._set_vocab_sentencepiece()
            self.gguf_writer.add_add_space_prefix(False)
        else:
            self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        """Write base parameters with the Gemma 3 defaults for keys missing from config.json."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()
        hparams = self.hparams

        # (see note) configs of released checkpoints omit several keys.
        # some default values are not specified in the hparams
        self.gguf_writer.add_context_length(hparams.get("max_position_embeddings", 131072))
        self.gguf_writer.add_head_count(hparams.get("num_attention_heads", 8))
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams.get("rms_norm_eps", 1e-6))
        self.gguf_writer.add_key_length(hparams.get("head_dim", 256))
        self.gguf_writer.add_value_length(hparams.get("head_dim", 256))
        # Base frequency for the global (full-attention) layers; defaults to 1e6.
        self.gguf_writer.add_rope_freq_base(self.rope_parameters.get("full_attention", self.rope_parameters).get("rope_theta", 1_000_000.0)) # for global layers
        # Gemma 3 dropped attention soft-capping; fail if a config still sets it.
        # attn_logit_softcapping is removed in Gemma3
        assert hparams.get("attn_logit_softcapping") is None
        if (final_logit_softcap := hparams.get("final_logit_softcapping")):
            self.gguf_writer.add_final_logit_softcapping(final_logit_softcap)
        # A pattern of 1 means every layer is global, so there is no window to record.
        if hparams.get("sliding_window_pattern") != 1:
            self.gguf_writer.add_sliding_window(hparams["sliding_window"])
        self.gguf_writer.add_head_count_kv(hparams.get("num_key_value_heads", 4))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Trim padded embedding rows and apply the RMSNorm shift."""
        # (see note) the embedding matrix may be larger than the real vocabulary; keep only
        # the rows that correspond to real tokens.
        # remove OOV (out-of-vocabulary) rows in token_embd
        if "embed_tokens.weight" in name:
            n_vocab_real = -1
            if (self.dir_model / "tokenizer.model").is_file():
                tokens = self._create_vocab_sentencepiece()[0]
                n_vocab_real = len(tokens)
            else:
                with open(self.dir_model / "tokenizer.json", "r", encoding="utf-8") as f:
                    tokenizer_json = json.load(f)
                    n_vocab_real = len(tokenizer_json["model"]["vocab"]) + len(tokenizer_json["added_tokens"])
            data_torch = data_torch[:n_vocab_real]

        # (see note) subclasses (Gemma 3n, 4) override norm_shift() because their norms differ.
        # ref code in Gemma3RMSNorm
        # output = output * (1.0 + self.weight.float())
        # note: this is not the case on gemma3n
        f_shift = self.norm_shift(name)
        if f_shift != 0.0:
            data_torch = data_torch + f_shift

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Gemma3TextModel")
class EmbeddingGemma(Gemma3Model):
    """EmbeddingGemma: Gemma 3 text encoder with optional sentence-transformers Dense layers."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.GEMMA_EMBEDDING
    # Class-level state shared across instances: paths of Dense sub-modules found in
    # modules.json and their (in, out) feature sizes keyed by tensor prefix.
    module_paths = []
    dense_features_dims = {}

    def __init__(self, *args, **kwargs):
        """Discover sentence-transformers Dense modules and record their feature dimensions."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(*args, **kwargs)
        if self.sentence_transformers_dense_modules:
            # (see note) only modules of type "...Dense" that ship their own model.safetensors are used.
            # read modules.json to determine if model has Dense layers
            modules_file = self.dir_model / "modules.json"
            if modules_file.is_file():
                with open(modules_file, encoding="utf-8") as modules_json_file:
                    mods = json.load(modules_json_file)
                for mod in mods:
                    # Iterate over items, evaluate loop invariant, and update progress
                    if mod["type"].endswith("Dense"):
                        mod_path = mod["path"]
                        # check if model.safetensors file for Dense layer exists
                        model_tensors_file = self.dir_model / mod_path / "model.safetensors"
                        if model_tensors_file.is_file():
                            self.module_paths.append(mod_path)
                            # read config.json of the Dense layer to get in/out features
                            mod_conf_file = self.dir_model / mod_path / "config.json"
                            if mod_conf_file.is_file():
                                with open(mod_conf_file, encoding="utf-8") as mod_conf_json_file:
                                    mod_conf = json.load(mod_conf_json_file)
                                    # hparams dense_2_feat_out and dense_3_feat_in are required when loading model's dense weights
                                    prefix = self._get_dense_prefix(mod_path)
                                    if mod_conf["in_features"] is not None and mod_conf["out_features"] is not None:
                                        self.dense_features_dims[prefix] = (mod_conf["in_features"], mod_conf["out_features"])

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        """Emit the weights of the Dense sub-modules as extra tensors (dense_2 / dense_3)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        # Dense modules live in separate safetensors files next to the main model.
        from safetensors.torch import load_file
        module_paths = list(self.module_paths)
        for i, module_path in enumerate(module_paths):
            # Iterate over items, evaluate loop invariant, and update progress
            tensors_file = self.dir_model / module_path / "model.safetensors"
            local_tensors = load_file(tensors_file)
            tensor_name = self._get_dense_prefix(module_path)
            for name, local_tensor in local_tensors.items():
                # Iterate over items, evaluate loop invariant, and update progress
                if not name.endswith(".weight"):
                    continue
                # Rename the generic "linear" parameter to this module's prefix before mapping.
                orig_name = name.replace("linear", tensor_name)
                name = self.map_tensor_name(orig_name)
                yield name, local_tensor.clone()

    @staticmethod
    # Maps the sentence-transformers module directory name onto a GGUF tensor prefix.
    def _get_dense_prefix(module_path) -> str:
        """Get the tensor name prefix for the Dense layer from module path."""
        # Step 1: Validate input parameters, operational preconditions, and state
        tensor_name = "dense_2" if module_path == "2_Dense" else "dense_3"
        return tensor_name

    def set_gguf_parameters(self):
        """Use the original sliding_window, record Dense layer sizes and the pooling type."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()

        # (see note) Gemma3TextConfig rewrites the value, so read the raw config.json instead.
        # Override the sliding window size as it gets adjusted by the Gemma3TextConfig
        # constructor. We want to use the value from the original model's config.json.
        # ref: https://github.com/huggingface/transformers/pull/40700
        with open(self.dir_model / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
            orig_sliding_window = config.get("sliding_window")
            if orig_sliding_window is None:
                raise ValueError("sliding_window not found in model config - this is required for the model")

            logger.info(f"Using original sliding_window from config: {orig_sliding_window} "
                        f"instead of {self.hparams['sliding_window']}")
            self.gguf_writer.add_sliding_window(orig_sliding_window)
        if self.sentence_transformers_dense_modules:
            for dense, dims in self.dense_features_dims.items():
                # Iterate over items, evaluate loop invariant, and update progress
                logger.info(f"Setting dense layer {dense} in/out features to {dims}")
                self.gguf_writer.add_dense_features_dims(dense, dims[0], dims[1])

        self._try_set_pooling_type()


@ModelBase.register("Gemma3ForConditionalGeneration")
class Gemma3VisionModel(MmprojModel):
    """Converter for the Gemma 3 vision tower (SigLIP) and multimodal projector."""
    def set_gguf_parameters(self):
        """Write vision parameters and the projector scale factor."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.GEMMA3)
        # default values below are taken from HF transformers code
        self.gguf_writer.add_vision_attention_layernorm_eps(hparams.get("layer_norm_eps", 1e-6))
        self.gguf_writer.add_vision_use_gelu(True)
        # (see note) the factor is derived from patches-per-side vs. output tokens per side.
        # calculate proj_scale_factor (used by tinygemma3 test model)
        image_seq_length = self.preprocessor_config.get("image_seq_length", 256)
        n_per_side = int(image_seq_length ** 0.5)
        image_size = self.hparams["image_size"]
        patch_size = self.hparams["patch_size"]
        proj_scale_factor = (image_size // patch_size) // n_per_side
        if proj_scale_factor > 0 and proj_scale_factor != 4:
            # we only need to write this if it's not the default value
            # in this case, we are converting a test model
            self.gguf_writer.add_vision_projector_scale_factor(proj_scale_factor)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        """Keep the input projection in F16 and embeddings in F32 (quantising them hurts quality)."""
        # related to https://github.com/ggml-org/llama.cpp/issues/13025
        if "input_projection" in name:
            return gguf.GGMLQuantizationType.F16
        if ".embeddings." in name:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Keep only vision/projector tensors and normalise "_weight" names to ".weight"."""
        # Step 1: Validate input parameters, operational preconditions, and state
        name, gen = item

        if "vision_model.head." in name:
            # skip redundant tensors for tinygemma3
            return None

        if not name.startswith(("multi_modal_projector.", "vision_tower.", "multimodal_projector.", "vision_model.")):
            return None

        name = name.replace("_weight", ".weight")

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Fix the projector's soft_emb_norm (needs +1); the SigLIP norms are already correct."""
        # correct norm value ; only this "soft_emb_norm" need to be corrected as it's part of Gemma projector
        # the other norm values are part of SigLIP model, and they are already correct
        # ref code: Gemma3RMSNorm
        if "soft_emb_norm.weight" in name:
            logger.info(f"Correcting norm value for '{name}'")
            data_torch = data_torch + 1

        yield from super().modify_tensors(data_torch, name, bid)


class ConformerAudioModel(MmprojModel):
    """Shared audio (Conformer) handling for multimodal Gemma-family models.

    Folds batch-norm statistics into scale/shift and reshapes convolution weights.
    """
    # Per-layer buffer of the 5 batch-norm tensors (weight, bias, running_mean,
    # running_var, num_batches_tracked) until a layer's set is complete.
    _batch_norm_tensors: list[dict[str, Tensor]] | None = None

    @staticmethod
    def is_audio_tensor(name: str):
        """True if the tensor name belongs to the audio/Conformer part of the model."""
        # Step 1: Validate input parameters, operational preconditions, and state
        return any(p in name for p in ["audio", "codebook", "conformer", "depth_embedding", "depthformer", "depth_linear"])

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        """Keep Conformer convolution weights in F32."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if ConformerAudioModel.is_audio_tensor(name):
            if ".conv" in name or "_conv" in name and ".weight" in name:
                return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Fold batch-norm into a per-channel scale/shift and reshape conv weights.

        Generator: yields nothing until all batch-norm tensors for a layer arrived.
        """
        # Inference-time batch-norm is y = a*x + b with a = w/sqrt(var+eps) and
        # b = bias - mean*a, so it can be stored as a fixed scale and shift.
        # fold running_mean, running_var and eps into weight and bias for batch_norm
        if "batch_norm" in name:
            if self._batch_norm_tensors is None:
                self._batch_norm_tensors = [{} for _ in range(self.block_count)]
            assert bid is not None
            self._batch_norm_tensors[bid][name] = data_torch

            # Wait until weight, bias, running_mean, running_var and the counter are all present.
            if len(self._batch_norm_tensors[bid]) < 5:
                return

            weight = self._batch_norm_tensors[bid][f"conformer.layers.{bid}.conv.batch_norm.weight"]
            bias = self._batch_norm_tensors[bid][f"conformer.layers.{bid}.conv.batch_norm.bias"]
            running_mean = self._batch_norm_tensors[bid][f"conformer.layers.{bid}.conv.batch_norm.running_mean"]
            running_var = self._batch_norm_tensors[bid][f"conformer.layers.{bid}.conv.batch_norm.running_var"]
            # The Conformer batch-norm epsilon (PyTorch default).
            eps = 1e-5 # default value

            a = weight / torch.sqrt(running_var + eps)
            b = bias - running_mean * a
            yield from super().modify_tensors(a, f"conformer.layers.{bid}.conv.batch_norm.weight", bid)
            yield from super().modify_tensors(b, f"conformer.layers.{bid}.conv.batch_norm.bias", bid)
            return

        # Convolutions are stored as 2-D matrices (kernel width 1 dims are squeezed away).
        # reshape conv weights
        if name.startswith("conformer.pre_encode.conv.") and name.endswith(".bias"):
            data_torch = data_torch[:, None, None]
        if "conv.depthwise_conv" in name and name.endswith(".weight"):
            assert data_torch.shape[1] == 1
            data_torch = data_torch.reshape(data_torch.shape[0], data_torch.shape[2])
        if "conv.pointwise_conv" in name and name.endswith(".weight"):
            assert data_torch.shape[2] == 1
            data_torch = data_torch.reshape(data_torch.shape[0], data_torch.shape[1])

        mapped_name = self.map_tensor_name(name, (".weight", ".bias", ".input_max", ".input_min", ".output_max", ".output_min"))
        yield (mapped_name, data_torch)


@ModelBase.register("Gemma3nForConditionalGeneration")
class Gemma3nVisionAudioModel(ConformerAudioModel):
    """Converter for Gemma 3n's vision (MobileNetV5) and audio (Conformer) encoders."""
    has_audio_encoder = True
    has_vision_encoder = True

    # (see note) MobileNetV5 blocks are indexed by (block, sub-block), so the usual
    # {bid}-only tensor map cannot express them; this table does.
    # Double indexed mapping for MobileNetV5 blocks (not supported by tensor_mapping.py)
    # This is the only known model having this, so we prefer implementing it outside of tensor_mapping.py
    # HF template name -> GGUF template name, with {bid} (block) and {sid} (sub-block).
    block_tensor_mapping = {
        "model.vision_tower.timm_model.blocks.{bid}.{sid}.conv_exp.weight":             "v.blk.{bid}.{sid}.conv_exp.weight",
        "model.vision_tower.timm_model.blocks.{bid}.{sid}.bn1.weight":                  "v.blk.{bid}.{sid}.bn1.weight",
        "model.vision_tower.timm_model.blocks.{bid}.{sid}.conv_pwl.weight":             "v.blk.{bid}.{sid}.conv_pwl.weight",
        "model.vision_tower.timm_model.blocks.{bid}.{sid}.bn2.weight":                  "v.blk.{bid}.{sid}.bn2.weight",
        "model.vision_tower.timm_model.blocks.{bid}.{sid}.dw_start.conv.weight":        "v.blk.{bid}.{sid}.dw_start.conv.weight",
        "model.vision_tower.timm_model.blocks.{bid}.{sid}.dw_start.bn.weight":          "v.blk.{bid}.{sid}.dw_start.bn.weight",
        "model.vision_tower.timm_model.blocks.{bid}.{sid}.dw_mid.conv.weight":          "v.blk.{bid}.{sid}.dw_mid.conv.weight",
        "model.vision_tower.timm_model.blocks.{bid}.{sid}.dw_mid.bn.weight":            "v.blk.{bid}.{sid}.dw_mid.bn.weight",
        "model.vision_tower.timm_model.blocks.{bid}.{sid}.pw_exp.conv.weight":          "v.blk.{bid}.{sid}.pw_exp.conv.weight",
        "model.vision_tower.timm_model.blocks.{bid}.{sid}.pw_exp.bn.weight":            "v.blk.{bid}.{sid}.pw_exp.bn.weight",
        "model.vision_tower.timm_model.blocks.{bid}.{sid}.pw_proj.conv.weight":         "v.blk.{bid}.{sid}.pw_proj.conv.weight",
        "model.vision_tower.timm_model.blocks.{bid}.{sid}.pw_proj.bn.weight":           "v.blk.{bid}.{sid}.pw_proj.bn.weight",
        "model.vision_tower.timm_model.blocks.{bid}.{sid}.layer_scale.gamma":           "v.blk.{bid}.{sid}.layer_scale.gamma",
        "model.vision_tower.timm_model.blocks.{bid}.{sid}.attn.query.proj.weight":      "v.blk.{bid}.{sid}.attn.query.proj.weight",
        "model.vision_tower.timm_model.blocks.{bid}.{sid}.attn.key.proj.weight":        "v.blk.{bid}.{sid}.attn.key.proj.weight",
        "model.vision_tower.timm_model.blocks.{bid}.{sid}.attn.value.proj.weight":      "v.blk.{bid}.{sid}.attn.value.proj.weight",
        "model.vision_tower.timm_model.blocks.{bid}.{sid}.attn.output.proj.weight":     "v.blk.{bid}.{sid}.attn.output.proj.weight",
        "model.vision_tower.timm_model.blocks.{bid}.{sid}.attn.key.down_conv.weight":   "v.blk.{bid}.{sid}.attn.key.down_conv.weight",
        "model.vision_tower.timm_model.blocks.{bid}.{sid}.attn.key.norm.weight":        "v.blk.{bid}.{sid}.attn.key.norm.weight",
        "model.vision_tower.timm_model.blocks.{bid}.{sid}.attn.value.down_conv.weight": "v.blk.{bid}.{sid}.attn.value.down_conv.weight",
        "model.vision_tower.timm_model.blocks.{bid}.{sid}.attn.value.norm.weight":      "v.blk.{bid}.{sid}.attn.value.norm.weight",
        "model.vision_tower.timm_model.blocks.{bid}.{sid}.norm.weight":                 "v.blk.{bid}.{sid}.norm.weight",
    }

    def __init__(self, *args, **kwargs):
        """Fill in fake/derived vision and audio hyperparameters the base class requires."""
        # (see note) the base class reads these through find_hparam.
        # Parent init will call find_hparam which now returns 0 for empty keys
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        # Placeholder: the shared header code needs a layer count even though the
        # vision encoder here is not a plain transformer.
        self.hparams_vision["n_layers"] = 128 # fake value for audio encoder, vision encoder doesn't use it
        self.hparams_vision["intermediate_size"] = self.hparams_vision.get("intermediate_size", 2048) * 4
        self.hparams_vision["num_attention_heads"] = self.hparams_vision.get("num_attention_heads", 8)

        # Identity normalisation (mean 0, std 1) because MobileNetV5 normalises internally.
        # MobileNetV5 does not use image_mean/std
        self.preprocessor_config["image_mean"] = [0.0 ,0.0 , 0.0]
        self.preprocessor_config["image_std"] = [1.0 ,1.0 ,1.0]
        self.hparams_vision["image_size"] = self.preprocessor_config.get(
            "size", {"height": 768, "width": 768}
        )["height"]

        # Gemma 3n emits 256 image tokens (16x16).
        # Image sequence length (256 tokens = 16x16 for Gemma3n)
        image_seq_length = self.preprocessor_config.get("image_seq_length", 256)
        image_size = self.hparams_vision["image_size"]
        self.hparams_vision["patch_size"] = image_size // image_seq_length

        # Map Conformer config names onto the generic keys used by MmprojModel.
        # remap audio hparams
        assert self.hparams_audio is not None
        self.hparams_audio["n_layers"] = self.hparams_audio["conf_num_hidden_layers"]
        self.hparams_audio["num_attention_heads"] = self.hparams_audio["conf_num_attention_heads"]
        self.hparams_audio["feat_in"] = self.hparams_audio["input_feat_size"]
        self.hparams_audio["intermediate_size"] = self.hparams_audio.get("intermediate_size", 6144)

    def set_gguf_parameters(self):
        """Write projector types and parameters for both the vision and audio encoders."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()

        # vision params
        self.gguf_writer.add_clip_vision_projector_type(gguf.VisionProjectorType.GEMMA3NV)
        self.gguf_writer.add_vision_attention_layernorm_eps(self.hparams.get("layer_norm_eps", 1e-6))

        # audio params
        assert self.hparams_audio is not None
        self.gguf_writer.add_clip_audio_projector_type(gguf.VisionProjectorType.GEMMA3NA)
        self.gguf_writer.add_audio_num_mel_bins(self.hparams_audio["feat_in"])
        self.gguf_writer.add_audio_attention_layernorm_eps(1e-5)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        """Keep input projections in F16 and embeddings/stem convolutions in F32."""
        # Force quantization settings for specific tensor types
        if "input_projection" in name or "input_proj" in name:
            return gguf.GGMLQuantizationType.F16
        if ".embeddings." in name or "stem" in name:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    # Resolves double-indexed MobileNetV5 names via block_tensor_mapping.
    def custom_map(self, name: str) -> str:
        """Parses names like model.vision_tower.timm_model.blocks.1.2.suffix and applies template mapping."""
        # Step 1: Validate input parameters, operational preconditions, and state
        parts = name.split(".")
        # MobileNet blocks have at least 7 parts: model, vision_tower, timm_model, blocks, bid, sid, and suffix
        if len(parts) >= 7:
            bid, sid = parts[4], parts[5]
            suffix = ".".join(parts[6:])
            template = f"model.vision_tower.timm_model.blocks.{{bid}}.{{sid}}.{suffix}"
            if template in self.block_tensor_mapping:
                return self.block_tensor_mapping[template].format(bid=bid, sid=sid)

        raise ValueError(f"Unknown name: {name}")

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Route audio tensors through the Conformer path and vision tensors through the
        MobileNetV5 / standard mappings; skip everything else (text weights).
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        if (ConformerAudioModel.is_audio_tensor(name)):
            name = name.replace("model.audio_tower.conformer.", "conformer.layers.")
            yield from super().modify_tensors(data_torch, name, bid)

        # Gemma3n uses
        # - model.embed_vision.* for projection layers
        # - model.vision_tower.* for vision encoder
        # Skip non-vision tensors
        # Text-model weights are converted by Gemma3NModel, not here.
        if not (name.startswith("model.embed_vision.") or name.startswith("model.vision_tower.")):
            return

        if name.startswith("model.vision_tower.timm_model.blocks."):
            # (see note) handled by custom_map instead of tensor_mapping.py.
            # Double-indexed block tensors through custom logic
            yield (self.custom_map(name), data_torch)
            return
        else:
            # Route non-repeating (conv_stem, msfa, embedding, etc.) and un-catched through tensor_mapping.py
            new_name = self.map_tensor_name(name)

        # Biases/scales that are per-channel vectors must be broadcastable over
        # [N, C, H, W], so reshape to [1, C, 1, 1].
        if new_name.endswith("conv_stem.conv.bias") or new_name.endswith("layer_scale.gamma"):
            data_torch = data_torch.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # [1, C, 1, 1]

        yield from ModelBase.modify_tensors(self, data_torch, new_name, bid)


@ModelBase.register("Gemma3nForCausalLM", "Gemma3nForConditionalGeneration")
class Gemma3NModel(Gemma3Model):
    """Converter for Gemma 3n text models (AltUp, per-layer embeddings, shared KV layers)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.GEMMA3N

    # Stacked AltUp projection matrices arrive as 3 separate tensors; collect them
    # here until all three are present.
    _altup_proj: list[Tensor] = []
    _altup_unembd: list[Tensor] = []

    def __init__(self, *args, **kwargs):
        """Check the AltUp configuration is supported and allocate the matrix slots."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(*args, **kwargs)
        assert self.hparams["altup_num_inputs"] == 4, "Current conversion only supports 4 altup inputs"
        self._altup_proj = [
            torch.Tensor(), # to be replaced
            torch.Tensor(), # to be replaced
            torch.Tensor(), # to be replaced
        ]
        self._altup_unembd = [
            torch.Tensor(), # to be replaced
            torch.Tensor(), # to be replaced
            torch.Tensor(), # to be replaced
        ]

    def norm_shift(self, name: str) -> float:
        """Gemma 3n RMSNorm weights are used as-is (no +1 shift)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        del name
        return 0.0 # same value with Gemma3p5RMSNorm scale_shift on python code

    def set_vocab(self):
        """Build the vocab with the full vocab_size (including multimodal special tokens)."""
        # (see note) the key vocab_size_per_layer_input would otherwise take precedence.
        # For Gemma3n multimodal models, we need the FULL vocab_size (262400)
        # which includes special tokens from 262144-262399 for vision/audio.
        # The vocab_size_per_layer_input (262144) is only the embedding size per layer.
        # Temporarily override the hparams lookup order to prioritize vocab_size.

        # Store original vocab_size_per_layer_input if it exists
        vocab_size_per_layer_input = self.hparams.get("vocab_size_per_layer_input")

        # Temporarily remove vocab_size_per_layer_input to force using vocab_size
        if vocab_size_per_layer_input is not None:
            del self.hparams["vocab_size_per_layer_input"]

        # Call parent set_vocab which will now use vocab_size (262400)
        super().set_vocab()

        # Restore vocab_size_per_layer_input for later use
        if vocab_size_per_layer_input is not None:
            self.hparams["vocab_size_per_layer_input"] = vocab_size_per_layer_input

    def set_gguf_parameters(self):
        """Write AltUp, per-layer-input, shared-KV, sparsity and sliding-window metadata."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()
        self.gguf_writer.add_altup_active_idx(self.hparams["altup_active_idx"])
        self.gguf_writer.add_altup_num_inputs(self.hparams["altup_num_inputs"])
        self.gguf_writer.add_embedding_length_per_layer_input(self.hparams["hidden_size_per_layer_input"])
        self.gguf_writer.add_shared_kv_layers(self.hparams["num_kv_shared_layers"])

        # Each layer's activation sparsity is stored as the std-multiplier at which a
        # standard normal reaches the given quantile (inverse CDF), as the runtime expects.
        activation_sparsity_scale = []
        for s in self.hparams["activation_sparsity_pattern"]:
            # Iterate over items, evaluate loop invariant, and update progress
            normal_dist = torch.distributions.normal.Normal(0, 1)
            std_multiplier = normal_dist.icdf(torch.tensor(s, dtype=torch.float32))
            activation_sparsity_scale.append(std_multiplier.item())
        self.gguf_writer.add_activation_sparsity_scale(activation_sparsity_scale)

        sliding_window_pattern = []
        for t in self.hparams["layer_types"]:
            # Iterate over items, evaluate loop invariant, and update progress
            sliding_window_pattern.append(t == "sliding_attention")
        self.gguf_writer.add_sliding_window_pattern(sliding_window_pattern)

    def _stack_matrices(self, matrices: list[Tensor]) -> Tensor | None:
        """Stack the 3 collected matrices, or return None if any are still empty."""
        # Step 1: Validate input parameters, operational preconditions, and state
        has_all = all(m.numel() > 0 for m in matrices)
        if not has_all:
            return None
        else:
            return torch.stack(matrices, dim=0)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Give bare "*_scale" parameters a ".weight" suffix for name mapping."""
        # Step 1: Validate input parameters, operational preconditions, and state
        name, gen = item

        if name.endswith("_scale"):
            name = name + ".weight"

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Pad embeddings for multimodal token ids and merge AltUp projection matrices.

        Generator: yields nothing until all three AltUp matrices have been seen.
        """
        # TODO: implement self.prediction_coefs.weight.clamp_(...)

        # (see note) the embedding is shorter than the full vocab; vision/audio token
        # rows are zero-padded because those tokens get embeddings from the encoders.
        # Pad token embeddings for vision/audio special tokens (262144-262399)
        if "embed_tokens.weight" in name or "embed_tokens_per_layer" in name:
            # Move to CPU to avoid meta device issues during padding
            data_torch = data_torch.to(device="cpu")

            vocab_size = self.hparams.get("vocab_size", 262400)
            current_size = data_torch.shape[0]  # First dimension is vocab_size

            if current_size < vocab_size:
                # Pad with zeros for vision/audio tokens (they get embeddings from vision tower)
                padding_size = vocab_size - current_size
                tensor_type = "per-layer embeddings" if "per_layer" in name else "token embeddings"
                logger.info(f"Padding {tensor_type} shape {list(data_torch.shape)} from {current_size} to {vocab_size} (adding {padding_size} vision/audio token slots)")

                # Create padding with zeros (vision tokens won't use these embeddings)
                padding = torch.zeros((padding_size, data_torch.shape[1]), dtype=data_torch.dtype, device=data_torch.device)
                data_torch = torch.cat([data_torch, padding], dim=0)

            # Continue with normal processing
            yield from ModelBase.modify_tensors(self, data_torch, name, bid)
            return

        if "altup_unembed_projections" in name:
            data_torch = data_torch.to(device="cpu")
            # (see note) these are hidden x hidden matrices, not vocabulary-sized, so no padding.
            # altup_unembed matrices are [hidden_size, hidden_size], NOT vocab-based
            # They should NOT be padded
            if ".0." in name:
                self._altup_unembd[0] = data_torch
            elif ".1." in name:
                self._altup_unembd[1] = data_torch
            elif ".2." in name:
                self._altup_unembd[2] = data_torch
            else:
                raise ValueError(f"Unknown name: {name}")
            out = self._stack_matrices(self._altup_unembd)
            if out is not None:
                yield from ModelBase.modify_tensors(self, out, "model.altup_unembed_projections.weight", bid)
                return
            else:
                return

        # Same collect-then-stack scheme for the AltUp input projections.
        if "altup_projections" in name:
            data_torch = data_torch.to(device="cpu")
            if ".0." in name:
                self._altup_proj[0] = data_torch
            elif ".1." in name:
                self._altup_proj[1] = data_torch
            elif ".2." in name:
                self._altup_proj[2] = data_torch
            else:
                raise ValueError(f"Unknown name: {name}")
            out = self._stack_matrices(self._altup_proj)
            if out is not None:
                yield from ModelBase.modify_tensors(self, out, "model.altup_projections.weight", bid)
                return
            else:
                return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Gemma4ForConditionalGeneration", "Gemma4ForCausalLM")
class Gemma4Model(Gemma3Model):
    """Converter for Google Gemma 4 text models (extends Gemma 3 with MoE, per-layer
    head sizes and NVFP4 expert scales).
    """
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.GEMMA4

    def norm_shift(self, name: str) -> float:
        """Gemma 4 norm weights need no shift."""
        # Step 1: Validate input parameters, operational preconditions, and state
        del name # unused
        return 0.0

    def set_vocab(self):
        """Build the vocab from the HF vocab, marking the chat/tool markup tokens as visible."""
        # Step 1: Validate input parameters, operational preconditions, and state
        vocab = gguf.LlamaHfVocab(self.dir_model)
        tokens = []
        scores = []
        toktypes = []
        # Tool-call / channel markers that must always be rendered, otherwise the chat
        # parser could not see them in the generated text.
        visible_tokens = {"<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>", "<|\"|>"}

        for text, score, toktype in vocab.all_tokens():
            # Iterate over items, evaluate loop invariant, and update progress
            tokens.append(text)
            scores.append(score)
            text_str = text.decode()
            if text_str in visible_tokens:
                # always render these tokens, so that the chat parser can read them
                toktypes.append(gguf.TokenType.USER_DEFINED)
                logger.info(f"Token '{text_str}' is set to USER_DEFINED")
            else:
                toktypes.append(toktype)

        # Sanity check that no token was dropped while converting.
        assert len(tokens) == vocab.vocab_size

        self.gguf_writer.add_tokenizer_model("gemma4")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)
        self.gguf_writer.add_add_space_prefix(False)
        self.gguf_writer.add_add_bos_token(True)

    def set_gguf_parameters(self):
        """Write Gemma 4 parameters: shared KV layers, per-layer embeddings, split head
        sizes for global vs sliding layers, MoE FFN sizes and per-layer KV heads / RoPE size.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()

        num_kv_shared_layers = self.hparams["num_kv_shared_layers"]
        self.gguf_writer.add_shared_kv_layers(num_kv_shared_layers)

        # per-layer embedding is optional
        n_pl_embd = self.hparams.get("hidden_size_per_layer_input") or 0
        self.gguf_writer.add_embedding_length_per_layer_input(n_pl_embd)

        swa_layers = [t == "sliding_attention" for t in self.hparams["layer_types"]]
        self.gguf_writer.add_sliding_window_pattern(swa_layers)

        head_dim_full = self.hparams["global_head_dim"]
        head_dim_swa = self.hparams["head_dim"]
        # Global and sliding-window layers have different head sizes.
        # correct the head dim for global/swa layers
        self.gguf_writer.add_key_length(head_dim_full)
        self.gguf_writer.add_value_length(head_dim_full)
        self.gguf_writer.add_key_length_swa(head_dim_swa)
        self.gguf_writer.add_value_length_swa(head_dim_swa)

        expert_intermediate_size = self.find_hparam(["expert_intermediate_size", "moe_intermediate_size"])
        if expert_intermediate_size is not None:
            self.gguf_writer.add_expert_feed_forward_length(expert_intermediate_size)

        # (see note) layers that reuse another layer's KV get a doubled FFN width.
        # if use_double_wide_mlp is set, we need to adjust the value for kv shared layers
        use_double_wide_mlp = self.hparams.get("use_double_wide_mlp", False)
        first_kv_shared_layer_idx = self.block_count - num_kv_shared_layers
        if use_double_wide_mlp:
            n_ff = self.hparams["intermediate_size"]
            n_ff_arr = [n_ff if il < first_kv_shared_layer_idx else n_ff * 2 for il in range(self.block_count)]
            self.gguf_writer.add_feed_forward_length(n_ff_arr)

        # handle num_global_key_value_heads
        num_key_value_heads_full = self.hparams.get("num_global_key_value_heads")
        num_key_value_heads_swa = self.hparams.get("num_key_value_heads")
        if num_key_value_heads_full is not None and num_key_value_heads_swa is not None:
            value_arr = [num_key_value_heads_swa if is_swa else num_key_value_heads_full for is_swa in swa_layers]
            self.gguf_writer.add_head_count_kv(value_arr)

        # (see note) full-attention layers rotate the whole head via the "proportional"
        # scheme implemented by the freq-factors tensor; sliding layers use a partial factor.
        # handle n_rot differently for global vs swa layers
        partial_rotary_factor_swa = self.hparams.get("partial_rotary_factor", 1.0)
        n_rot_full = int(head_dim_full) # "proportional" is used, see generate_extra_tensors
        n_rot_swa = int(head_dim_swa * partial_rotary_factor_swa)
        self.gguf_writer.add_rope_dimension_count(n_rot_full)
        self.gguf_writer.add_rope_dimension_count_swa(n_rot_swa)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        """Emit RoPE frequency factors for the full-attention layers."""
        # full layer uses "proportional" rope with partial_rotary_factor=0.25
        # the expected ordering is cc000000ss000000 (c = cos, s = sin, 0 = unrotated),
        # but ggml neox only supports ccss000000000000, and we cannot rearrange the head because that will break use_alternative_attention
        # solution is to set specific freq_factors for the unrotated dims

        # IMPORTANT: this ROPE_FREQS tensor is ONLY used by the full_attention layers
        rope_params_full = self.hparams["rope_parameters"]["full_attention"]
        assert rope_params_full["rope_type"] == "proportional"
        head_dim_full = (self.hparams["global_head_dim"])
        partial_rotary_factor_full = rope_params_full["partial_rotary_factor"]
        n_rot_full = int(head_dim_full * partial_rotary_factor_full / 2)
        n_unrot_full = int(head_dim_full / 2) - n_rot_full
        # 1.0 = rotate normally; 1e30 effectively divides the frequency to zero, leaving
        # the dimension unrotated.
        values = [1.0] * n_rot_full + [1e30] * n_unrot_full
        rope_freqs_full = torch.tensor(values, dtype=torch.float32)
        yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FREQS), rope_freqs_full)

    def _generate_nvfp4_tensors(self):
        """Fold router per-expert scales into the NVFP4 weight_scale_2 of each expert."""
        # Gemma-4 stores a per-layer router.per_expert_scale ([n_expert]) that scales
        # each expert's contribution. It's mathematically equivalent to a per-expert
        # scalar on the down_proj output, which is exactly where ffn_down_exps_s is
        # applied at inference. Fold it into each expert's NVFP4 weight_scale_2 so the
        # existing NVFP4 path produces the right scales.
        # Folding is skipped for dense checkpoints (no experts).
        n_experts = self.find_hparam(["num_local_experts", "num_experts"], optional=True) or 0
        for name in [n for n in self.model_tensors if n.endswith(".router.per_expert_scale")]:
            # Iterate over items, evaluate loop invariant, and update progress
            bid_match = re.search(r"\.layers\.(\d+)\.", name)
            if bid_match is None:
                continue
            bid = bid_match.group(1)
            prefix = name[: name.index(f".layers.{bid}.") + len(f".layers.{bid}.")]
            w2_targets = [f"{prefix}experts.{e}.down_proj.weight_scale_2" for e in range(n_experts)]
            present = [w2 in self.model_tensors for w2 in w2_targets]
            if not any(present):
                continue
            # Either all or none of the experts of a layer must be NVFP4-quantised.
            assert all(present), f"layer {bid}: partial NVFP4 quantization across experts"
            r = self.model_tensors.pop(name)
            for e, w2 in enumerate(w2_targets):
                # Iterate over items, evaluate loop invariant, and update progress
                s = self.model_tensors[w2]
                # Lazy composition: the default arguments freeze s, r and the expert index at
                # definition time (avoiding the late-binding closure bug).
                self.model_tensors[w2] = lambda s=s, r=r, i=e: s() * r()[i]
        super()._generate_nvfp4_tensors()

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Add ".weight" suffixes to bare parameters so name mapping and quantisation work."""
        # Step 1: Validate input parameters, operational preconditions, and state
        name, gen = item

        if name.endswith("per_dim_scale") or name.endswith("layer_scalar"):
            name = name + ".weight"
        if ".experts." in name and not name.endswith((".weight", ".weight_scale", ".weight_scale_2", ".input_scale")):
            name += ".weight"

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Map router scale / per-expert scale tensors to their GGUF names."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if name.endswith("router.scale"):
            name = self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_INP, bid, ".scale")
            yield (name, data_torch)
            return
        if ".per_expert_scale" in name:
            # convert per-expert scale to FFN down scale
            name = self.format_tensor_name(gguf.MODEL_TENSOR.FFN_DOWN_EXP, bid, ".scale")
            yield (name, data_torch)
            return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Gemma4ForConditionalGeneration")
class Gemma4VisionAudioModel(MmprojModel):
    """Converter for Gemma 4 vision and audio encoders as an mmproj."""
    has_audio_encoder = True
    has_vision_encoder = True

    def __init__(self, *args, **kwargs):
        """Set required placeholders and remap audio hyperparameters (or disable audio)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        self.hparams_vision["image_size"] = 224 # unused, but set to avoid error

        # remap audio hparams
        if self.hparams_audio:
            self.hparams_audio["feat_in"] = self.hparams_audio.get("input_feat_size", 128)
            self.hparams_audio["intermediate_size"] = self.hparams_audio["hidden_size"] * 4
        else:
            self.has_audio_encoder = False

    def set_gguf_parameters(self):
        """Write projector types and parameters for vision and audio."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()

        # vision params
        assert self.hparams_vision is not None
        self.gguf_writer.add_clip_vision_projector_type(gguf.VisionProjectorType.GEMMA4V)
        self.gguf_writer.add_vision_attention_layernorm_eps(self.hparams_vision.get("layer_norm_eps", 1e-6))

        # audio params
        assert self.hparams_audio is not None
        self.gguf_writer.add_clip_audio_projector_type(gguf.VisionProjectorType.GEMMA4A)
        self.gguf_writer.add_audio_num_mel_bins(self.hparams_audio["feat_in"])
        self.gguf_writer.add_audio_attention_layernorm_eps(self.hparams_audio.get("layer_norm_eps", 1e-6))

    def is_audio_tensor(self, name: str) -> bool:
        """True if the tensor belongs to the audio tower or audio embedding."""
        # Step 1: Validate input parameters, operational preconditions, and state
        return "audio_tower" in name or "embed_audio" in name

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        """Keep audio conv weights and position-embedding tables in F32."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if self.is_audio_tensor(name):
            if ".conv" in name or "_conv" in name and ".weight" in name:
                return gguf.GGMLQuantizationType.F32
        if "position_embedding_table" in name:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Map vision and audio tensors to their GGUF names, with reshapes where needed."""
        # Step 1: Validate input parameters, operational preconditions, and state
        del bid # unused

        # GGUF cannot store 0-d tensors, so scalars become 1-element vectors.
        if len(data_torch.shape) == 0:
            # convert scalar tensors (input/output_mix/max) to 1D tensors
            data_torch = data_torch.unsqueeze(0)

        if self.is_audio_tensor(name):
            assert self.hparams_audio is not None
            name = name.replace("model.audio_tower.", "conformer.")
            name = name.replace(".linear.", ".")
            # These parameters are stored pre-softplus; the runtime expects the activated value.
            if name.endswith("per_dim_key_scale") or name.endswith("per_dim_scale"):
                name = name + ".weight"
                data_torch = torch.nn.functional.softplus(data_torch)
            if "lconv1d.depthwise_conv1d" in name and name.endswith(".weight"):
                assert data_torch.shape[1] == 1
                data_torch = data_torch.reshape(data_torch.shape[0], data_torch.shape[2])
            mapped_name = self.map_tensor_name(name, (".weight", ".bias", ".input_max", ".input_min", ".output_max", ".output_min"))
            yield (mapped_name, data_torch)

        else:
            name = name.replace("model.vision_tower.encoder.", "vision_model.model.")
            name = name.replace(".linear.weight", ".weight")
            if name.endswith("layer_scalar") or name.endswith("position_embedding_table"):
                name = name + ".weight"
            # Patch projection is a flattened conv kernel [n_embd, k*k*3]; restore it to
            # [n_embd, 3, k, k] so it can be used as a 2-D convolution.
            if name.endswith("patch_embedder.input_proj.weight"):
                n_embd, ksize_sq_c = data_torch.shape
                patch_size = int((ksize_sq_c // 3) ** 0.5)
                data_torch = data_torch.reshape(n_embd, patch_size, patch_size, 3)
                data_torch = data_torch.permute(0, 3, 1, 2).contiguous()
            mapped_name = self.map_tensor_name(name, (".weight", ".bias", ".input_max", ".input_min", ".output_max", ".output_min"))
            yield (mapped_name, data_torch)
