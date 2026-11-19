# ==============================================================================
# Program: mistral3.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for MISTRAL3.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

"""HuggingFace-to-GGUF conversion support for the 'mistral3' model family.

Converter classes here are generally registered with ModelBase under HF architecture names;
they write GGUF metadata (set_gguf_parameters), the vocabulary (set_vocab)
and remap or reshape tensors (modify_tensors) for llama.cpp.

Classes:
  Mistral3Model - Converter facade for Mistral3/Ministral3/Mistral4 checkpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf

from .deepseek import DeepseekV2Model
from .llama import LlamaModel


@ModelBase.register(
    "Mistral3ForConditionalGeneration",
    "Ministral3ForCausalLM",
)
class Mistral3Model(TextModel):
    """Converter facade for Mistral3/Ministral3/Mistral4 checkpoints.

    It picks an implementation class from model_type and delegates every call to it.
    """
    class Ministral3Model(LlamaModel):
        """Ministral3: llama-based text model with YaRN rope."""
        model_arch = gguf.MODEL_ARCH.MISTRAL3

        def set_gguf_parameters(self):
            """Add YaRN log multiplier and attention temperature for ministral3 configs."""
            super().set_gguf_parameters()
            rope_params = self.rope_parameters
            if self.hparams.get("model_type") == "ministral3":
                assert rope_params, "ministral3 must have 'rope_parameters' config"
                assert rope_params["rope_type"] == "yarn", "ministral3 rope_type must be 'yarn'"
                self.gguf_writer.add_rope_scaling_yarn_log_mul(rope_params["mscale_all_dim"])
                self.gguf_writer.add_attn_temperature_scale(rope_params["llama_4_scaling_beta"])

    class Mistral4Model(DeepseekV2Model):
        """Mistral4: DeepSeek-style MoE whose experts are already stacked."""
        model_arch = gguf.MODEL_ARCH.MISTRAL4
        skip_mtp = False # model contains no MTP layers, so no need to skip
        merge_expert = False # experts are already stacked as 3D

        def modify_tensors(self, data_torch, name, bid):
            """Append '.weight' to stacked expert tensors that lack it, then use the parent mapping.
            """
            if name.endswith(".down_proj") or name.endswith(".gate_up_proj"):
                name = name + ".weight"
            yield from super().modify_tensors(data_torch, name, bid)

    model_arch = gguf.MODEL_ARCH.MISTRAL3 # unused
    # The chosen implementation; all public methods below forward to it.
    impl: TextModel

    def __init__(self, *args, **kwargs):
        """Select Mistral4 or Ministral3 as delegate based on hparams['model_type']."""
        super().__init__(*args, **kwargs)
        if self.hparams.get("model_type") == "mistral4":
            self.impl = Mistral3Model.Mistral4Model(*args, **kwargs)
        else:
            self.impl = Mistral3Model.Ministral3Model(*args, **kwargs)

    def set_vocab(self):
        """Delegate vocab handling."""
        self.impl.set_vocab()

    def set_gguf_parameters(self):
        """Delegate parameter writing."""
        self.impl.set_gguf_parameters()

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
        """Delegate tensor transformation."""
        yield from self.impl.modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        """Delegate tensor preparation."""
        self.impl.prepare_tensors()

    def write_vocab(self):
        """Delegate vocab writing."""
        self.impl.write_vocab()

    def write(self):
        """Delegate final file writing."""
        self.impl.write()
