# ==============================================================================
# Program: kimivl.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for KIMIVL.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, gguf


@ModelBase.register("KimiVLForConditionalGeneration")
class KimiVLModel(MmprojModel):
    """Converter for the Kimi-VL (MoonViT) vision tower and projector (mmproj GGUF)."""
    def __init__(self, *args, **kwargs):
        """Set a nominal image size (the encoder is resolution-flexible)."""
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        self.hparams_vision["image_size"] = 64 * 14 # for compatibility

    def set_gguf_parameters(self):
        """Write the Kimi-VL projector type, GELU flag, scale factor and norm epsilon."""
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.KIMIVL)
        self.gguf_writer.add_vision_use_gelu(True)
        self.gguf_writer.add_vision_projector_scale_factor(2)
        # eps is the same as pytorch's default value
        assert self.hparams_vision is not None
        self.gguf_writer.add_vision_attention_layernorm_eps(self.hparams_vision.get("layer_norm_eps", 1e-5))

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Keep only the vision tower and multimodal projector tensors."""
        name, gen = item

        is_vision_tensor = "vision_tower" in name or "multi_modal_projector" in name

        if not is_vision_tensor:
            return None

        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Flatten the 3-D position embedding and split fused wqkv into Q, K, V."""
        # Merge the first two axes (rows x cols of the grid) into one position axis.
        if "pos_emb.weight" in name:
            data_torch = data_torch.view(data_torch.shape[0] * data_torch.shape[1], data_torch.shape[2])

        if "wqkv" in name:
            # Weights are [3*C, in] (split on dim 0); biases are [3*C] (split on the last axis).
            split_dim = 0 if "weight" in name else -1
            wq, wk, wv = data_torch.chunk(3, dim=split_dim)
            yield from super().modify_tensors(wq, name.replace("wqkv", "wq"), bid)
            yield from super().modify_tensors(wk, name.replace("wqkv", "wk"), bid)
            yield from super().modify_tensors(wv, name.replace("wqkv", "wv"), bid)
        else:
            yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("KimiK25ForConditionalGeneration")
# (MoonViT3d encoder: patches are merged merge_kernel_size^2 at a time.)
class KimiK25Model(MmprojModel):
    """Kimi-K2.5 with MoonViT3d vision encoder"""

    def __init__(self, *args, **kwargs):
        """Read patch/merge sizes and derive a nominal image size from the position grid."""
        super().__init__(*args, **kwargs)

        assert self.hparams_vision is not None, "Kimi-K2.5 requires vision_config in model config"

        self.merge_kernel_size = tuple(self.hparams_vision.get("merge_kernel_size", [2, 2]))
        self.patch_size = self.hparams_vision.get("patch_size", 14)

        # the encoder works at flexible resolution, so image_size is only a
        # reference for the shared base-class code.
        # Set image_size for compatibility with base class
        # Use position embedding dimensions as image_size reference
        pos_emb_h = self.hparams_vision.get("init_pos_emb_height", 64)
        self.hparams_vision["image_size"] = pos_emb_h * self.patch_size

    def set_gguf_parameters(self):
        """Write the Kimi-K2.5 projector type, position-embedding grid, projector and pixel limits."""
        # the base class already covers the generic vision keys.
        # Base class MmprojModel.set_gguf_parameters() already writes:
        # - vision_block_count, vision_head_count, vision_embedding_length
        # - vision_feed_forward_length, vision_patch_size, image_mean, image_std
        # via find_vparam() which handles the vt_* prefixed keys in Kimi-K2.5's config
        super().set_gguf_parameters()
        assert self.hparams_vision is not None

        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.KIMIK25)

        # Position embedding parameters (for interpolation)
        self.gguf_writer.add_uint32("vision.pos_emb_height", self.hparams_vision.get("init_pos_emb_height", 64))
        self.gguf_writer.add_uint32("vision.pos_emb_width", self.hparams_vision.get("init_pos_emb_width", 64))
        self.gguf_writer.add_uint32("vision.pos_emb_time", self.hparams_vision.get("init_pos_emb_time", 4))

        # Projector parameters
        self.gguf_writer.add_vision_use_gelu(self.hparams_vision.get("projector_hidden_act", "gelu") == "gelu")
        self.gguf_writer.add_vision_attention_layernorm_eps(self.hparams_vision.get("projector_ln_eps", 1e-5))
        self.gguf_writer.add_vision_projector_scale_factor(self.merge_kernel_size[0])

        # Max pixels follow the processor's patch limit; the minimum (8 patches) is a
        # reasonable floor chosen here, not read from the config.
        # Image size limits
        # Note: in_patch_limit is for images, in_patch_limit_each_frame is for video (not supported yet)
        in_patch_limit = self.preprocessor_config.get("in_patch_limit", 16384)
        min_patches = 8  # reasonable minimum
        pixels_per_patch = self.patch_size ** 2
        self.gguf_writer.add_vision_min_pixels(min_patches * pixels_per_patch)
        self.gguf_writer.add_vision_max_pixels(in_patch_limit * pixels_per_patch)

    @staticmethod
    def permute(weights: Tensor, n_head: int) -> Tensor:
        """Convert Q/K rows from interleaved-RoPE order to split-RoPE order, per head.

        Within each head, rows are viewed as (head_dim/4, 2, 2) and the two inner axes
        are swapped, which is the layout build_rope_2d expects at runtime.
        """
        out_dim, in_dim = weights.shape
        head_dim = out_dim // n_head
        w = weights.reshape(n_head, head_dim // 4, 2, 2, in_dim)
        w = w.permute(0, 2, 1, 3, 4)
        return w.reshape(out_dim, in_dim)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Keep only vision tower and projector tensors."""
        name, gen = item

        # Only process vision and projector tensors
        is_vision = any(x in name for x in ["vision_tower", "mm_projector"])

        if not is_vision:
            return None

        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Permute Q/K in fused wqkv, flatten temporal embeddings, rename projector layers."""
        assert self.hparams_vision is not None
        n_head = self.hparams_vision.get("num_attention_heads", 16)

        # V is left untouched: only Q and K are rotated.
        # Permute Q/K weights/biases from interleaved to split RoPE format
        # This allows using build_rope_2d at runtime without post-permutation.
        if "wqkv" in name:
            out_dim = data_torch.shape[0]
            qkv_dim = out_dim // 3
            head_dim = qkv_dim // n_head

            if "weight" in name:
                wq, wk, wv = data_torch[:qkv_dim, :], data_torch[qkv_dim:2 * qkv_dim, :], data_torch[2 * qkv_dim:, :]
                wq = self.permute(wq, n_head)
                wk = self.permute(wk, n_head)
                data_torch = torch.cat([wq, wk, wv], dim=0)
            elif "bias" in name:
                bq, bk, bv = data_torch[:qkv_dim], data_torch[qkv_dim:2 * qkv_dim], data_torch[2 * qkv_dim:]
                bq = bq.reshape(n_head, head_dim // 4, 2, 2).permute(0, 2, 1, 3).reshape(-1)
                bk = bk.reshape(n_head, head_dim // 4, 2, 2).permute(0, 2, 1, 3).reshape(-1)
                data_torch = torch.cat([bq, bk, bv], dim=0)

        # drop the singleton middle axis.
        # Temporal embeddings: (T, 1, C) → (T, C)
        if "pos_emb.time_weight" in name:
            T, _, C = data_torch.shape
            data_torch = data_torch.reshape(T, C)

        # the sequential indices 0 and 2 are the two Linear layers.
        # PatchMergerMLP tensor name mapping
        # proj.0.weight → proj.linear_1.weight
        # proj.2.weight → proj.linear_2.weight
        if "mm_projector.proj.0." in name:
            name = name.replace(".proj.0.", ".proj.linear_1.")
        elif "mm_projector.proj.2." in name:
            name = name.replace(".proj.2.", ".proj.linear_2.")

        yield from super().modify_tensors(data_torch, name, bid)
