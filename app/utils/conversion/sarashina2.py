# ==============================================================================
# Program: sarashina2.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for SARASHINA2.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf

from .llama import LlamaModel
from .qwenvl import Qwen2VLVisionModel


@ModelBase.register("Sarashina2VisionForCausalLM")
class Sarashina2VLTextModel(LlamaModel):
    """Text converter for Sarashina2-Vision (llama architecture under an 'llm.' prefix)."""
    model_arch = gguf.MODEL_ARCH.LLAMA

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Strip the 'llm.' prefix and drop the stray top-level 'norm.' tensors."""
        # Step 1: Validate input parameters, operational preconditions, and state
        name, gen = item
        if name.startswith("llm."):
            name = name.replace("llm.", "", 1)
        elif name.startswith("norm."):
            return None
        return super().filter_tensors((name, gen))


@ModelBase.register("Sarashina2VisionForCausalLM")
class Sarashina2VLVisionModel(Qwen2VLVisionModel):
    """Vision converter for Sarashina2-Vision, which reuses the Qwen2-VL tower."""
    def __init__(self, *args, **kwargs):
        """Force model_type to qwen2_vl so the Qwen2-VL projector logic applies."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(*args, **kwargs)
        self.global_config['model_type'] = "qwen2_vl"
