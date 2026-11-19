# ==============================================================================
# Program: llava.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for LLAVA.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

import json

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, gguf, logger

from .llama import LlamaModel


@ModelBase.register(
    "LlavaForConditionalGeneration", # pixtral
    "Mistral3ForConditionalGeneration", # mistral small 3.1
)
class LlavaVisionModel(MmprojModel):
    """Converter for the Pixtral / Mistral-3 (Llava-style) vision encoder (mmproj GGUF).

    Handles both the HF layout (model_type == "pixtral") and Mistral's native
    params.json layout, and exports the [IMG_BREAK] token embedding.
    """
    # Token id of [IMG_BREAK] (separates image rows); -1 means "not used / not found".
    img_break_tok_id = -1
    use_break_tok = True

    def __init__(self, *args, **kwargs):
        """Detect the config flavour and resolve the [IMG_BREAK] token id and norm epsilon."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(*args, **kwargs)
        if self.hparams.get("model_type") == "pixtral":
            # (see note) the value is hard-coded upstream, so default it here.
            # layer_norm_eps is not in config.json, it is hard-coded in modeling_pixtral.py
            self.hparams["layer_norm_eps"] = self.hparams.get("layer_norm_eps", 1e-5)
            if self.use_break_tok:
                self.img_break_tok_id = self.get_token_id("[IMG_BREAK]")
        elif self.is_mistral_format:
            # (see note) in the native format the eps lives in the top-level params.json.
            # hparams is already vision config here so norm_eps is only defined in global_config.
            self.hparams["norm_eps"] = self.global_config.get("norm_eps", None)
            assert self.hparams["norm_eps"] is not None, "norm_eps not found in params.json"
            if self.use_break_tok:
                self.img_break_tok_id = self.find_vparam(["image_break_token_id"])

                # params.json may ship -1 placeholders (Mistral Medium 3.5)
                # resolve the real id from the bundled tokenizer in that case
                if self.img_break_tok_id < 0:
                    self.img_break_tok_id = self.get_mistral_token_id("[IMG_BREAK]")
        else:
            raise ValueError(f"Unsupported model type: {self.hparams['model_type']}")
        logger.info(f"Image break token id: {self.img_break_tok_id}")

    def get_token_id(self, token: str) -> int:
        """Look up a token id by its text in tokenizer_config.json, then tokenizer.json."""
        # Step 1: Validate input parameters, operational preconditions, and state
        tokenizer_config_file = self.dir_model / 'tokenizer_config.json'
        with open(tokenizer_config_file, "r", encoding="utf-8") as f:
            added_tokens_decoder = json.load(f).get('added_tokens_decoder') or {}
            for id_, token_data in added_tokens_decoder.items():
                # Iterate over items, evaluate loop invariant, and update progress
                if token_data.get("content") == token:
                    return int(id_)
            # Not found among tokenizer_config added tokens: search tokenizer.json too.
            # fallthrough to tokenizer.json
        with open(self.dir_model / "tokenizer.json", "r", encoding="utf-8") as f:
            tokenizer_json = json.load(f)
            for token_data in tokenizer_json["added_tokens"]:
                # Iterate over items, evaluate loop invariant, and update progress
                if token_data["content"] == token:
                    return int(token_data["id"])
        raise ValueError(f"Token '{token}' not found in tokenizer config.")

    def get_mistral_token_id(self, token: str) -> int:
        """Look up a special token id in Mistral-native tokenizer files (tekken.json or tokenizer.json)."""
        # mistral native format ships tekken.json or a versioned spm tokenizer
        tekken_file = self.dir_model / "tekken.json"
        if tekken_file.is_file():
            with open(tekken_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data.get("special_tokens", []):
                # Iterate over items, evaluate loop invariant, and update progress
                if entry.get("token_str") == token:
                    return int(entry["rank"])
        tokenizer_json_file = self.dir_model / "tokenizer.json"
        if tokenizer_json_file.is_file():
            with open(tokenizer_json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data.get("added_tokens", []):
                # Iterate over items, evaluate loop invariant, and update progress
                if entry.get("content") == token:
                    return int(entry["id"])
        raise ValueError(f"Token '{token}' not found in mistral tokenizer files.")

    def set_gguf_parameters(self):
        """Write Pixtral projector settings (norm epsilon, activation, spatial merge size)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()
        hparams = self.hparams
        if hparams.get("model_type") == "pixtral":
            self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.PIXTRAL)
            self.gguf_writer.add_vision_attention_layernorm_eps(hparams["layer_norm_eps"])

            # hidden_act
            if hparams["hidden_act"] == "silu":
                self.gguf_writer.add_vision_use_silu(True)
            elif hparams["hidden_act"] == "gelu":
                self.gguf_writer.add_vision_use_gelu(True)
            else:
                raise ValueError(f"Unsupported hidden_act: {hparams['hidden_act']}")

            # spatial_merge_size
            if "spatial_merge_size" in self.global_config:
                self.gguf_writer.add_vision_spatial_merge_size(self.global_config["spatial_merge_size"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Un-permute Q/K for HF layouts and extract the [IMG_BREAK] embedding from the LLM."""
        # Step 1: Validate input parameters, operational preconditions, and state
        n_head = (
            self.hparams["num_attention_heads"] if not self.is_mistral_format else self.find_vparam(["num_attention_heads"])
        )
        n_kv_head = n_head

        # Tensors with these prefixes belong to the vision encoder/projector.
        valid_prefixes = (
            "multi_modal_projector.",
            "vision_tower.",
            "vision_encoder.",
            "vision_language_adapter.",
            "patch_merger.",
            "pre_mm_projector_norm",
        )

        if any(name.startswith(prefix) for prefix in valid_prefixes):
            # Native Mistral checkpoints are already un-permuted, so only HF ones need the
            # rotary un-permutation.
            # process vision tensors
            if name.endswith(("q_proj.weight", "q_proj.bias")) and not self.is_mistral_format:
                data_torch = LlamaModel.permute(data_torch, n_head, n_head)
            if name.endswith(("k_proj.weight", "k_proj.bias")) and not self.is_mistral_format:
                data_torch = LlamaModel.permute(data_torch, n_head, n_kv_head)
            yield from super().modify_tensors(data_torch, name, bid)
            return

        # The LLM's token embedding matrix is the only language-side tensor we need: one row
        # of it (the [IMG_BREAK] token) is exported with the vision model.
        embed_key = "embed_tokens.weight" if not self.is_mistral_format else "tok_embeddings.weight"
        if self.img_break_tok_id > 0 and embed_key in name:
            logger.info(f"Extracting [IMG_BREAK] token embedding from {name}")
            # for pixtral model, we need to extract the [IMG_BREAK] token embedding
            img_break_embd = data_torch[self.img_break_tok_id]
            name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.V_TOK_EMBD_IMG_BREAK]
            yield from super().modify_tensors(img_break_embd, name, bid)

        # All remaining language-model tensors are dropped from the mmproj.
        return # skip other tensors
