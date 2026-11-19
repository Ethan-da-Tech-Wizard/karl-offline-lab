# ==============================================================================
# Program: gpt_oss.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for GPT_OSS.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("GptOssForCausalLM")
class GptOssModel(TextModel):
    """Converter for OpenAI gpt-oss (MoE with attention sinks and MXFP4 experts).

    Expert weights ship pre-quantised as MXFP4 (blocks + scales); they are
    repacked directly into GGUF MXFP4 tensors instead of being dequantised.
    """
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.GPT_OSS

    # (see note) skip the generic dequantisation pass for MXFP4 checkpoints.
    # TODO: remove once MXFP4 is supported more generally
    def dequant_model(self):
        """Leave MXFP4 weights packed; only dequantise other quantisation formats."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if self._is_mxfp4:
            return
        return super().dequant_model()

    def transform_nibble_layout(self, tensor):
        """Convert HF's MXFP4 nibble order into ggml's block layout.

        Blocks are 16 bytes (32 4-bit values). HF stores the first 16 values in the low
        nibbles and the next 16 in the high nibbles ("aaaa...bbbb..."); ggml wants
        values interleaved ("abababab..."). The nibble swaps compensate for the
        byte-wise recombination below.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        assert tensor.dtype == torch.uint8
        assert tensor.shape[-1] == 16
        # First swap low/high nibbles within each byte.
        # swap nibbles
        t_lo = tensor & 0x0F
        t_hi = tensor & 0xF0
        t_swapped = (t_lo << 4) | (t_hi >> 4)
        tensor = t_swapped
        # Split the 16 bytes into two halves of 8 and interleave their nibbles.
        # transform aaaa...bbbb... to abababab...
        blk_a, blk_b = tensor.chunk(2, dim=-1)
        # get a_
        blk_a0 = (blk_a & 0xF0).view(-1, 1)
        blk_a1 = (blk_a << 4).view(-1, 1)
        blk_a = torch.stack((blk_a0, blk_a1), dim=2).view(tensor.shape)
        # get _b
        blk_b0 = (blk_b >> 4).view(-1, 1)
        blk_b1 = (blk_b & 0x0F).view(-1, 1)
        blk_b = torch.stack((blk_b0, blk_b1), dim=2).view(tensor.shape)
        # Swap nibbles back into the final order after merging both halves.
        # swap once more
        out = blk_a | blk_b
        out_h = out & 0xF0
        out_l = out & 0x0F
        out = (out_h >> 4) | (out_l << 4)
        return out

    def repack_mxfp4(self, new_name: str, blocks: Tensor, scales: Tensor):
        """Write one expert tensor to the GGUF as raw MXFP4 data.

        Each row is [scale byte | 16 data bytes] per 32-value block; `blocks` is
        uint8 [experts, rows, n_blocks, 16] and `scales` [experts, rows, n_blocks].
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        assert blocks.dtype == torch.uint8
        assert scales.dtype == torch.uint8
        # Add a trailing axis so each block's scale byte can be concatenated in front
        # of its 16 data bytes.
        scales = scales.unsqueeze(-1)
        assert len(blocks.shape) == 4
        assert len(scales.shape) == 4
        blocks = self.transform_nibble_layout(blocks)
        new_data = torch.concat((scales, blocks), dim=-1)
        # Logical (unpacked) shape, used only for the log message: 32 values per block.
        new_shape = [new_data.shape[0], new_data.shape[1], new_data.shape[2] * 32]
        logger.info(f"Repacked {new_name} with shape {new_shape} and quantization MXFP4")
        # Raw MXFP4 data is a flat byte row per output row; the writer takes it as such.
        # flatten last dim
        new_data = new_data.view(new_data.shape[0], new_data.shape[1], new_data.shape[2] * new_data.shape[3])
        new_data = new_data.numpy()
        self.gguf_writer.add_tensor(new_name, new_data, raw_dtype=gguf.GGMLQuantizationType.MXFP4)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        """Repack all MoE expert tensors straight from the safetensors stream.

        The returned list is empty: tensors are written directly via repack_mxfp4,
        not yielded to the normal writer path.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        blocks0: Tensor = torch.zeros(1)
        blocks1: Tensor = torch.zeros(1)
        # (see note) down_proj blocks must be seen before their scales, likewise gate_up.
        # we assume that tensors are loaded in the correct order
        for name, data_torch in self.get_tensors():
            # Iterate over items, evaluate loop invariant, and update progress
            if "mlp.experts.down_proj_blocks" in name:
                blocks0 = data_torch
            elif "mlp.experts.down_proj_scales" in name:
                new_name = self.map_tensor_name(name.replace("_scales", ".weight"))
                self.repack_mxfp4(new_name, blocks0, data_torch)
            # gate and up rows are interleaved (even = gate, odd = up) in the fused tensor;
            # split them with a stride of 2.
            elif "mlp.experts.gate_up_proj_blocks" in name:
                blocks0, blocks1 = data_torch[:, ::2, :, :], data_torch[:, 1::2, :, :]
            elif "mlp.experts.gate_up_proj_scales" in name:
                scales0, scales1 = data_torch[:, ::2, :], data_torch[:, 1::2, :]
                new_name_gate = self.map_tensor_name(name.replace("gate_up_proj_scales", "gate_proj.weight"))
                new_name_up = self.map_tensor_name(name.replace("gate_up_proj_scales", "up_proj.weight"))
                self.repack_mxfp4(new_name_gate, blocks0, scales0)
                self.repack_mxfp4(new_name_up, blocks1, scales1)
        return []

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Give the attention "sinks" parameters a ".weight" suffix for name mapping."""
        # Step 1: Validate input parameters, operational preconditions, and state
        name, gen = item

        if "sinks" in name:
            name += ".weight"

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Handle non-MXFP4 fallbacks for expert tensors and split fused gate_up weights.

        MXFP4 blocks/scales are consumed in generate_extra_tensors, so they are
        dropped here (hence the bare returns).
        """
        # Rename HF parameter names to ".weight"/".bias" names and transpose
        # unquantised weights; already-repacked MXFP4 tensors are skipped.
        # correct naming for down_proj
        if "down_proj" in name:
            if name.endswith("_bias"):
                name = name.replace("down_proj_bias", "down_proj.bias")
            elif "_blocks" not in name and "_scales" not in name:
                logger.warning(f"{name} is not in MXFP4, performance may be degraded")
                name = name.replace("down_proj", "down_proj.weight")
                data_torch = data_torch.transpose(-1, -2)
            else:
                # otherwise, it should already be repacked to ggml MXFP4 format
                return

        # (see note) HF interleaves gate and up along one axis; split with stride 2.
        # split the gate_up into gate and up
        if "gate_up_proj" in name:
            if name.endswith("_bias"):
                name_up = name.replace("gate_up_proj_bias", "up_proj.bias")
                name_gate = name.replace("gate_up_proj_bias", "gate_proj.bias")
                gate_proj_bias, up_proj_bias = data_torch[..., ::2], data_torch[..., 1::2]
                yield from super().modify_tensors(gate_proj_bias, name_gate, bid)
                yield from super().modify_tensors(up_proj_bias, name_up, bid)
            elif "_blocks" not in name and "_scales" not in name:
                logger.warning(f"{name} is not in MXFP4, performance may be degraded")
                name_up = name.replace("gate_up_proj", "up_proj.weight")
                name_gate = name.replace("gate_up_proj", "gate_proj.weight")
                data_torch = data_torch.transpose(-1, -2)
                gate_proj_weight, up_proj_weight = data_torch[:, ::2, :], data_torch[:, 1::2, :]
                yield from super().modify_tensors(gate_proj_weight, name_gate, bid)
                yield from super().modify_tensors(up_proj_weight, name_up, bid)
        else:
            yield from super().modify_tensors(data_torch, name, bid)

    def set_vocab(self):
        """gpt-oss uses a GPT-2 style BPE tokenizer."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        """Write sliding-window size and the expert FFN width."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()
        self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])
        self.gguf_writer.add_expert_feed_forward_length(self.hparams["intermediate_size"])
