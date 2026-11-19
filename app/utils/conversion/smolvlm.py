# ==============================================================================
# Program: smolvlm.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for SMOLVLM.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, gguf


@ModelBase.register("Idefics3ForConditionalGeneration", "SmolVLMForConditionalGeneration")
class SmolVLMModel(MmprojModel):
    """Vision (mmproj) converter for SmolVLM / Idefics3."""
    def __init__(self, *args, **kwargs):
        """Fill SmolVLM2's missing vision config keys with the transformers defaults."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(*args, **kwargs)
        if self.hparams["model_type"] == "smolvlm_vision":
            # fix for SmolVLM2, missing some keys in config.json
            # default values are taken from transformers code
            self.hparams["hidden_size"] = self.hparams.get("hidden_size", 1152)
            self.hparams["num_attention_heads"] = self.hparams.get("num_attention_heads", 16)
            self.hparams["intermediate_size"] = self.hparams.get("intermediate_size", 3072)

    def set_gguf_parameters(self):
        """Write projector type, norm eps, pixel-shuffle scale factor and the preprocessor image size.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.IDEFICS3)
        self.gguf_writer.add_vision_attention_layernorm_eps(self.hparams.get("layer_norm_eps", 1e-5))
        self.gguf_writer.add_vision_projector_scale_factor(self.global_config.get("scale_factor", 2))
        self.gguf_writer.add_vision_use_gelu(True)

        # Add the preprocessor longest edge size
        preproc_image_size = self.preprocessor_config.get("size", {}).get("longest_edge", self.image_size)
        self.gguf_writer.add_vision_preproc_image_size(preproc_image_size)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        """Keep embedding tensors in F32."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if ".embeddings." in name:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Keep only vision tower and connector tensors."""
        # Step 1: Validate input parameters, operational preconditions, and state
        name, gen = item

        is_vision_tensor = "vision_tower" in name or "vision_model" in name or "model.connector" in name

        if not is_vision_tensor:
            return None

        return super().filter_tensors(item)
