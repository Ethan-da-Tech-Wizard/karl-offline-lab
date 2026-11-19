# ==============================================================================
# Program: dotsocr.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for DOTSOCR.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, gguf


@ModelBase.register("DotsOCRForCausalLM")
class DotsOCRVisionModel(MmprojModel):
    """Converter for the dots.ocr vision tower (Qwen2.5-VL-like encoder) as an mmproj."""
    def __init__(self, *args, **kwargs):
        """Mark the image size as 0 because the encoder handles dynamic resolution."""
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        self.hparams_vision["image_size"] = 0 # dynamic resolution

    def set_gguf_parameters(self):
        """Write the projector type, pixel limits, norm epsilon, merge factor and SiLU flag."""
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.DOTSOCR)
        self.gguf_writer.add_vision_min_pixels(self.preprocessor_config["min_pixels"])
        self.gguf_writer.add_vision_max_pixels(self.preprocessor_config["max_pixels"])
        self.gguf_writer.add_vision_attention_layernorm_eps(self.find_vparam(["rms_norm_eps"]))
        self.gguf_writer.add_vision_projector_scale_factor(self.find_vparam(["spatial_merge_size"]))
        self.gguf_writer.add_vision_use_silu(True)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Keep only the vision tower and rename its MLP weights to Qwen2.5-VL naming."""
        name, gen = item

        if not name.startswith("vision_tower."):
            return None

        # renaming here avoids clashes in the shared tensor name table.
        if "vision_tower.blocks." in name and ".mlp." in name:
            # note: to avoid naming conflicts in tensor_mapping.py, we need to handle FFN renaming here
            # x = F.silu(self.fc1(x)) * self.fc3(x)
            # x = self.fc2(x)
            # fc1 -> gate, fc2 -> down, fc3 -> up
            # mapping original names to Qwen2.5 naming scheme
            name = name.replace("vision_tower.blocks.", "visual.blocks.")
            name = name.replace(".fc1", ".gate_proj")
            name = name.replace(".fc2", ".down_proj")
            name = name.replace(".fc3", ".up_proj")

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """No extra transformation: defer to the base class."""
        yield from super().modify_tensors(data_torch, name, bid)
