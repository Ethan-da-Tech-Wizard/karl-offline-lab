# ==============================================================================
# Program: pixtral.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for PIXTRAL.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

"""HuggingFace-to-GGUF conversion support for the 'pixtral' model family.

Converter classes here are generally registered with ModelBase under HF architecture names;
they write GGUF metadata (set_gguf_parameters), the vocabulary (set_vocab)
and remap or reshape tensors (modify_tensors) for llama.cpp.

Classes:
  PixtralModel - Vision converter for native Mistral-format Pixtral checkpoints (llava-style tower).
"""

from __future__ import annotations

from typing import Sequence

from .base import gguf

from .llava import LlavaVisionModel


class PixtralModel(LlavaVisionModel):
    """Vision converter for native Mistral-format Pixtral checkpoints (llava-style tower)."""
    model_name = "Pixtral"
    hf_arch = ""
    is_mistral_format = True

    def set_gguf_parameters(self):
        """Write the Pixtral projector type, SiLU activation, rope base and optional spatial merge.
        """
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.PIXTRAL)

        self.gguf_writer.add_vision_attention_layernorm_eps(
            self.find_hparam(["norm_eps"])
        )
        self.gguf_writer.add_rope_freq_base(self.find_vparam(["rope_theta"]))

        self.gguf_writer.add_vision_use_silu(True)

        # spatial_merge_size
        if self.find_vparam(["mm_projector_id"], optional=True) == "patch_merge":
            self.gguf_writer.add_vision_spatial_merge_size(
                self.find_vparam(["spatial_merge_size"])
            )

    def map_tensor_name(self, name: str, try_suffixes: Sequence[str] = (".weight", ".bias")) -> str:
        """Map the Mistral vision-language adapter weights to mm.1/mm.2; everything else uses the default mapping.
        """
        if name == "vision_language_adapter.w_in.weight":
            return "mm.1.weight"
        elif name == "vision_language_adapter.w_in.bias":
            return "mm.1.bias"
        elif name == "vision_language_adapter.w_out.weight":
            return "mm.2.weight"
        elif name == "vision_language_adapter.w_out.bias":
            return "mm.2.bias"
        return super().map_tensor_name(name, try_suffixes)
