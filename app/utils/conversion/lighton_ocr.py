# ==============================================================================
# Program: lighton_ocr.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for LIGHTON_OCR.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf

# LightOnOCR's vision side is a Pixtral/Llava-style encoder, so extend that converter.
from .llava import LlavaVisionModel


@ModelBase.register("LightOnOCRForConditionalGeneration")
class LightOnOCRVisionModel(LlavaVisionModel):
    """Converter for the LightOnOCR vision encoder and projector (mmproj GGUF)."""
    # Class flags that tell the Llava parent to read an HF (not Mistral-native) config and
    # not to look up an [IMG_BREAK] token (LightOnOCR has none).
    is_mistral_format = False
    use_break_tok = False

    def set_gguf_parameters(self):
        """Write the Llava vision parameters and tag the projector as LightOnOCR."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.LIGHTONOCR)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Rename LightOnOCR tensors to the names the Llava parent expects."""
        # Step 1: Validate input parameters, operational preconditions, and state
        name, gen = item

        name = name.replace("model.vision_encoder.", "vision_tower.")
        name = name.replace("model.vision_projection.", "multi_modal_projector.")

        return super().filter_tensors((name, gen))
