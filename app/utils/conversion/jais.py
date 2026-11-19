# ==============================================================================
# Program: jais.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for JAIS.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

import math

from typing import Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("Jais2ForCausalLM")
class Jais2Model(TextModel):
    """Converter for Jais 2 (Llama-like Arabic/English model)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.JAIS2

    def set_gguf_parameters(self):
        """Write base parameters plus the RoPE dimension (head size)."""
        super().set_gguf_parameters()
        hparams = self.hparams
        head_dim = hparams.get("head_dim", hparams["hidden_size"] // hparams["num_attention_heads"])
        self.gguf_writer.add_rope_dimension_count(head_dim)


@ModelBase.register("JAISLMHeadModel")
class JaisModel(TextModel):
    """Converter for Jais v1 (GPT-2 style, SwiGLU, ALiBi, muP scaling)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.JAIS

    def __init__(self, *args, **kwargs):
        """Validate the architecture assumptions and read the muP embedding/width scales."""
        super().__init__(*args, **kwargs)

        # Only the SwiGLU + ALiBi variant is supported by the runtime graph.
        # SwigLU activation
        assert self.hparams["activation_function"] == "swiglu"
        # ALiBi position embedding
        assert self.hparams["position_embedding_type"] == "alibi"

        # muP scaling: the input embeddings are multiplied by a constant; configs name
        # it either mup_embeddings_scale or embeddings_scale (one must exist).
        # Embeddings scale
        self.embeddings_scale = 1.0
        if 'mup_embeddings_scale' in self.hparams:
            self.embeddings_scale = self.hparams['mup_embeddings_scale']
        elif 'embeddings_scale' in self.hparams:
            self.embeddings_scale = self.hparams['embeddings_scale']
        else:
            assert False

        self.width_scale = 1.0
        # Output logits scale = mup_output_alpha * mup_width_scale (or an explicit width_scale).
        if 'mup_output_alpha' in self.hparams:
            assert 'mup_width_scale' in self.hparams
            self.width_scale = self.hparams['mup_output_alpha'] * self.hparams['mup_width_scale']
        elif 'width_scale' in self.hparams:
            self.width_scale = self.hparams['width_scale']
        else:
            assert False

        # Default ALiBi bias; recomputed from the slopes tensor in modify_tensors.
        self.max_alibi_bias = 8.0

    def set_vocab(self):
        """Jais uses a GPT-2 style BPE tokenizer."""
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        """Write Jais hyperparameters using GPT-2 style config key names."""
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_context_length(self.hparams["n_positions"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])
        self.gguf_writer.add_feed_forward_length(self.hparams["n_inner"])
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Drop the attention-mask buffers (".attn.bias")."""
        name, gen = item

        # we don't need these
        if name.endswith((".attn.bias")):
            return None

        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Derive the ALiBi bias from the slopes tensor, transpose Conv1D weights, and
        bake the muP scales into the embedding and output matrices.
        """
        if name.endswith(("relative_pe.slopes")):
            # the first slope is 2^(-max_bias / n), so max_bias is recovered from it
            # using the largest power of two <= n_head.
            # Calculate max ALiBi bias (this is the inverse of the ALiBi calculation)
            # Some other models has max_alibi_bias spelled out explicitly in the hyperparams,
            # but Jais's PyTorch model simply precalculates the slope values and places them
            # in relative_pes.slopes
            n_head_closest_log2 = 2 ** math.floor(math.log2(self.hparams["n_head"]))
            first_val = float(data_torch[0].item())
            self.max_alibi_bias = -round(math.log2(first_val) * n_head_closest_log2)

            return

        # Conv1D-style weights are stored transposed relative to nn.Linear.
        if name.endswith((".c_attn.weight", ".c_proj.weight", ".c_fc.weight", ".c_fc2.weight")):
            data_torch = data_torch.transpose(1, 0)

        new_name = self.map_tensor_name(name)

        if new_name == self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD):
            yield from super().modify_tensors(data_torch * self.embeddings_scale, new_name, bid)
        elif new_name == self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT):
            yield from super().modify_tensors(data_torch * self.width_scale, new_name, bid)
        else:
            yield from super().modify_tensors(data_torch, new_name, bid)

    def prepare_tensors(self):
        """Run the normal conversion, then record the ALiBi bias found along the way."""
        super().prepare_tensors()
        self.gguf_writer.add_max_alibi_bias(self.max_alibi_bias)
