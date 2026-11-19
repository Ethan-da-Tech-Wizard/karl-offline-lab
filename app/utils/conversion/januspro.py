# ==============================================================================
# Program: januspro.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for JANUSPRO.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, gguf

from .llama import LlamaModel


@ModelBase.register("JanusForConditionalGeneration")
class JanusProModel(LlamaModel):
    """Converter for the DeepSeek Janus-Pro language model (Llama architecture).

    Vision, aligner and image-generation tensors are dropped; they live in the mmproj.
    """
    # Janus-Pro's LLM is a plain Llama, so reuse that GGUF architecture.
    model_arch = gguf.MODEL_ARCH.LLAMA  # reuse Llama arch

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Skip vision, aligner, VQ and generation tensors."""
        # Step 1: Validate input parameters, operational preconditions, and state
        name, gen = item

        # Skip vision, aligner, and generation tensors
        # Non-text components of Janus-Pro: the vision tower, aligner, VQ image tokenizer and
        # the image-generation branch; the text GGUF needs none of them.
        skip_prefixes = (
            'model.vision_model.',
            'model.aligner.',
            'model.vqmodel.',
            'model.generation_embeddings.',
            'model.generation_aligner.',
            'model.generation_head.',
        )
        if name.startswith(skip_prefixes):
            return None

        return super().filter_tensors(item)


@ModelBase.register("JanusForConditionalGeneration")
class JanusProVisionModel(MmprojModel):
    """Converter for Janus-Pro's vision tower and aligner (mmproj GGUF)."""
    def __init__(self, *args, **kwargs):
        """Derive intermediate_size from mlp_ratio when the vision config omits it."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        if "intermediate_size" not in self.hparams_vision:
            # intermediate_size = hidden_size * mlp_ratio (rounded) when not given explicitly.
            mlp_ratio = self.hparams_vision.get("mlp_ratio")
            hidden_size = self.hparams_vision.get("hidden_size")
            if mlp_ratio is not None and hidden_size is not None:
                self.hparams_vision["intermediate_size"] = int(round(hidden_size * mlp_ratio))

    def set_gguf_parameters(self):
        """Write the Janus-Pro projector type, norm epsilon and activation."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()
        assert self.hparams_vision is not None

        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.JANUS_PRO)

        self.gguf_writer.add_vision_attention_layernorm_eps(self.hparams_vision.get("layer_norm_eps", 1e-6))

        # Only GELU and SiLU activations are supported by the runtime; others write nothing.
        hidden_act = str(self.hparams_vision.get("hidden_act", "")).lower()
        if hidden_act == "gelu":
            self.gguf_writer.add_vision_use_gelu(True)
        elif hidden_act == "silu":
            self.gguf_writer.add_vision_use_silu(True)

    def _map_aligner_tensor(self, data_torch: Tensor, name: str) -> Iterable[tuple[str, Tensor]]:
        """Map aligner tensors to projector format"""
        # Step 1: Validate input parameters, operational preconditions, and state
        # The aligner is a small MLP (fc1 then hidden layers); each Linear becomes projector
        # layer mm.<index>, with fc1 as 0 and hidden_layers.N as N + 1.
        suffix = ".bias" if name.endswith(".bias") else ".weight"

        if name.startswith("model.aligner."):
            # Strip the aligner prefix (HF with or without the "model." wrapper).
            local_name = name[len("model.aligner."):]
        elif name.startswith("aligner."):
            local_name = name[len("aligner."):]
        else:
            raise ValueError(f"Unsupported Janus aligner prefix: {name}")

        # fc1 is the first projector layer (index 0); hidden_layers.N are layers N + 1.
        if local_name.startswith("fc1."):
            mm_index = 0
        elif local_name.startswith("hidden_layers."):
            parts = local_name.split(".", 2)
            if len(parts) < 3:
                raise ValueError(f"Unexpected Janus aligner tensor name: {name}")
            mm_index = int(parts[1]) + 1
        else:
            raise ValueError(f"Unsupported Janus aligner tensor: {name}")

        tensor_name = self.format_tensor_name(gguf.MODEL_TENSOR.V_MMPROJ, mm_index, suffix=suffix)
        return [(tensor_name, data_torch)]

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Skip image-generation components (VQ model, generation embeddings/aligner/head)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        name, gen = item

        # Skip generation-related components
        # Image generation (VQ model, generation embeddings/aligner/head) is not needed for
        # image understanding; prefixes appear with and without "model.".
        skip_generation_prefixes = (
            'model.vqmodel.',
            'vqmodel.',
            'model.generation_embeddings.',
            'generation_embeddings.',
            'model.generation_aligner.',
            'generation_aligner.',
            'model.generation_head.',
            'generation_head.',
        )
        if name.startswith(skip_generation_prefixes):
            return None

        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Convert aligner and vision tensors; every other tensor is dropped."""
        # Handle aligner tensors
        if name.startswith(('model.aligner.', 'aligner.')):
            yield from self._map_aligner_tensor(data_torch, name)
            return

        # Anything not matched above (language model, generation) yields nothing.
        # Handle vision tensors
        if name.startswith(('model.vision_model.', 'vision_model.')):
            yield from super().modify_tensors(data_torch, name, bid)
            return

        return
