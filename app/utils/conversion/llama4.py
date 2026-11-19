# ==============================================================================
# Program: llama4.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for LLAMA4.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, gguf


@ModelBase.register("Llama4ForConditionalGeneration")
class Llama4VisionModel(MmprojModel):
    """Converter for the Llama 4 vision encoder and projector (mmproj GGUF)."""
    def set_gguf_parameters(self):
        """Write the Llama 4 projector type, norm epsilon, pixel-shuffle scale and GELU flag."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.LLAMA4)
        self.gguf_writer.add_vision_attention_layernorm_eps(self.hparams["norm_eps"])
        # Pixel-shuffle ratio r (e.g. 0.5) becomes an integer downscale factor 1/r.
        self.gguf_writer.add_vision_projector_scale_factor(int(1.0 / self.hparams["pixel_shuffle_ratio"]))
        # Only GELU is implemented in the runtime graph for this encoder.
        assert self.hparams["hidden_act"] == "gelu"
        self.gguf_writer.add_vision_use_gelu(True)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Keep only the vision model and projector, adding ".weight" to the bare position table."""
        # Step 1: Validate input parameters, operational preconditions, and state
        name, gen = item

        if "multi_modal_projector" not in name and "vision_model" not in name:
            return None

        if "positional_embedding_vlm" in name and ".weight" not in name:
            name += ".weight"

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Map the projector's single linear layer to the V_MMPROJ_FC tensor name."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if "multi_modal_projector.linear_1" in name:
            # despite the name with number postfix, this is a single fully connected layer
            yield (gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.V_MMPROJ_FC] + '.weight', data_torch)
        else:
            yield from super().modify_tensors(data_torch, name, bid)
