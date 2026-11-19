# ==============================================================================
# Program: cogvlm.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for COGVLM.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, gguf

from .llama import LlamaModel


@ModelBase.register("CogVLMForCausalLM")
class CogVLMVisionModel(MmprojModel):
    """Converter for the CogVLM vision tower / projector (mmproj GGUF)."""

    def set_gguf_parameters(self):
        """Write base vision parameters, the layernorm epsilon and the CogVLM projector type."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()
        self.gguf_writer.add_vision_attention_layernorm_eps(self.hparams.get("layer_norm_eps", 1e-6))
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.COGVLM)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Keep only the vision-side tensors (prefix "model.vision.")."""
        # Step 1: Validate input parameters, operational preconditions, and state
        name, gen = item

        # Language-model weights belong in the text GGUF, so drop them here.
        if not name.startswith("model.vision."):
            return None

        return super().filter_tensors(item)


@ModelBase.register("CogVLMForCausalLM")
class CogVLMModel(LlamaModel):
    """Converter for the CogVLM language model (Llama layout with its own GGUF arch)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.COGVLM
