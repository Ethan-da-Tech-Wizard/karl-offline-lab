# ==============================================================================
# Program: mamba.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for MAMBA.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

"""HuggingFace-to-GGUF conversion support for the Mamba state-space model family.

Converter classes here are registered with ModelBase under HF architecture names;
they write GGUF metadata (set_gguf_parameters), the vocabulary (set_vocab)
and remap or reshape tensors (modify_tensors) for llama.cpp.

Classes:
  MambaModel  - Converter for Mamba-1 models (including FalconMamba).
  Mamba2Model - Converter for Mamba-2 models (also reused as a base by hybrids
                such as FalconH1, hence the FalconH1 special case below).
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


# Registering under these HF architecture names lets the converter pick this class
# from the checkpoint's config.json 'architectures' entry.
@ModelBase.register("MambaForCausalLM", "MambaLMHeadModel", "FalconMambaForCausalLM")
class MambaModel(TextModel):
    """Converter for Mamba-1 (and FalconMamba) checkpoints."""

    model_arch = gguf.MODEL_ARCH.MAMBA

    def __init__(self, dir_model: Path, *args, **kwargs):
        """Load hparams straight from config.json instead of going through AutoConfig."""
        # Avoid using AutoConfig for hparams
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            with open(dir_model / "config.json", "r", encoding="utf-8") as f:
                hparams = json.load(f)
        super().__init__(dir_model, *args, hparams=hparams, **kwargs)

    def set_vocab(self):
        """Round the vocab size up to the padding multiple and pick a tokenizer source."""
        # Step 1: Validate input parameters, operational preconditions, and state
        vocab_size = self.hparams["vocab_size"]
        # Mamba checkpoints store an embedding table padded to a multiple of
        # pad_vocab_size_multiple, so the declared vocab must match that padded size.
        # Round vocab size to next multiple of 8
        pad_vocab = self.hparams.get("pad_vocab_size_multiple", 8)
        # pad using ceiling division
        # ref: https://stackoverflow.com/a/17511341/22827863
        vocab_size = -(vocab_size // -pad_vocab) * pad_vocab
        self.hparams["vocab_size"] = vocab_size

        # Prefer a full HF tokenizer.json (BPE), then a SentencePiece model file.
        if (self.dir_model / "tokenizer.json").is_file():
            self._set_vocab_gpt2()
        elif (self.dir_model / "tokenizer.model").is_file():
            self._set_vocab_sentencepiece()
        else:
            # Use the GPT-NeoX tokenizer when no tokenizer files are present
            self._set_vocab_builtin("gpt-neox", vocab_size)

    def set_gguf_parameters(self):
        """Write Mamba-1 SSM hyperparameters (conv kernel, inner size, state size, dt rank...)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        # Each hparam is looked up under both the HF name and the original mamba_ssm
        # name; optional ones fall back to the defaults used by the reference code.
        d_model = self.find_hparam(["hidden_size",       "d_model"])
        d_conv  = self.find_hparam(["conv_kernel",       "d_conv"],  optional=True) or 4
        d_inner = self.find_hparam(["intermediate_size", "d_inner"], optional=True) or 2 * d_model
        d_state = self.find_hparam(["state_size",        "d_state"], optional=True) or 16
        # ceiling division
        # ref: https://stackoverflow.com/a/17511341/22827863
        # ref: https://github.com/state-spaces/mamba/blob/ce59daea3a090d011d6476c6e5b97f6d58ddad8b/mamba_ssm/modules/mamba_simple.py#L58
        dt_rank      = self.find_hparam(["time_step_rank",     "dt_rank"],      optional=True) or -(d_model // -16)
        rms_norm_eps = self.find_hparam(["layer_norm_epsilon", "rms_norm_eps"], optional=True) or 1e-5
        # Only FalconMamba applies RMS norm to the B, DT and C projections; classic Mamba does not.
        use_dt_b_c_norm = False
        # For falconmamba we do apply RMS norm on B / DT and C layers
        if self.find_hparam(["model_type"], optional=True) in ("falcon_mamba",):
            use_dt_b_c_norm = True
        # The GGUF layout and llama.cpp graph assume the standard expansion factor of 2.
        # Fail early for models which don't have a block expansion factor of 2
        assert d_inner == 2 * d_model

        # llama.cpp reads these keys for every architecture even though the SSM
        # graph never uses them, so write placeholder values.
        self.gguf_writer.add_context_length(2**20) # arbitrary value; for those who use the default
        self.gguf_writer.add_embedding_length(d_model)
        self.gguf_writer.add_feed_forward_length(0) # unused, but seemingly required when loading
        self.gguf_writer.add_head_count(0) # unused, but seemingly required when loading
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_ssm_conv_kernel(d_conv)
        self.gguf_writer.add_ssm_inner_size(d_inner)
        self.gguf_writer.add_ssm_state_size(d_state)
        self.gguf_writer.add_ssm_time_step_rank(dt_rank)
        self.gguf_writer.add_layer_norm_rms_eps(rms_norm_eps)
        self.gguf_writer.add_ssm_dt_b_c_rms(use_dt_b_c_norm) # For classic Mamba we don't apply rms norm on B / DT layers
        self.gguf_writer.add_file_type(self.ftype)

    # Cached copy of token_embd.weight, used to detect tied output weights.
    _tok_embd = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Rename one tensor to its GGUF name, fixing up SSM-specific values and shapes.

        Converts A_log to A, squeezes the conv1d weight, and drops output.weight when it
        is identical to the token embedding (tied weights). Yields (name, tensor) pairs.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        output_name = self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT)
        tok_embd_name = self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD)

        # Translate the HF/mamba_ssm tensor name to the GGUF name.
        new_name = self.map_tensor_name(name)

        # The checkpoint stores A in log space; the runtime expects A = -exp(A_log).
        if name.endswith(".A_log"):
            logger.debug("A_log --> A ==> " + new_name)
            data_torch = -torch.exp(data_torch)

        # The conv1d weight has a singleton dimension that the GGUF layout drops.
        # [4 1 8192 1] -> [4 8192 1 1]
        if self.match_model_tensor_name(new_name, gguf.MODEL_TENSOR.SSM_CONV1D, bid):
            data_torch = data_torch.squeeze()

        # Tied weights: if output.weight equals the stored token embedding, skip it so
        # the loader falls back to token_embd for the output projection.
        # assuming token_embd.weight is seen before output.weight
        if self._tok_embd is not None and new_name == output_name:
            if torch.equal(self._tok_embd, data_torch):
                logger.debug(f"{output_name} is equivalent to {tok_embd_name}, omitting")
                return
        elif new_name == tok_embd_name:
            self._tok_embd = data_torch

        # Hand off to the base class for dtype handling and quantisation decisions.
        yield from super().modify_tensors(data_torch, new_name, bid)


# Mamba2ForCausalLM covers standard Mamba-2 checkpoints, including Mamba-Codestral.
@ModelBase.register("Mamba2ForCausalLM")
class Mamba2Model(TextModel):
    """Converter for Mamba-2 checkpoints (multi-head SSM with grouped B/C)."""

    model_arch = gguf.MODEL_ARCH.MAMBA2

    def __init__(self, dir_model: Path, *args, **kwargs):
        """Load hparams from config.json and cache the sizes shared by several methods."""
        # Avoid using AutoConfig for hparams
        # It wrongly assumes all Mamba2 models are Mamba-Codestral-7B-v0.1
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            with open(dir_model / "config.json", "r", encoding="utf-8") as f:
                hparams = json.load(f)
        # Some wrapped models nest the text config under llm_config; expose it as text_config.
        if "llm_config" in hparams:
            hparams["text_config"] = hparams["llm_config"]
        super().__init__(dir_model, *args, hparams=hparams, **kwargs)
        # Model width, SSM inner width (default 2x), and number of B/C groups.
        self.d_model = self.find_hparam(["hidden_size", "d_model", "dim"])
        self.d_inner = self.find_hparam(["mamba_d_ssm", "intermediate_size", "d_inner"], optional=True) or 2 * self.d_model
        self.n_group = self.find_hparam(["n_groups"], optional=True) or 1

    def set_vocab(self):
        """Round the vocab size up to the padding multiple and pick a tokenizer source."""
        # Step 1: Validate input parameters, operational preconditions, and state
        vocab_size = self.hparams["vocab_size"]
        # Mamba-2 pads to a multiple of 16 by default (Mamba-1 uses 8).
        # Round vocab size to next multiple of 16
        pad_vocab = self.hparams.get("pad_vocab_size_multiple", 16)
        # pad using ceiling division
        # ref: https://stackoverflow.com/a/17511341/22827863
        vocab_size = -(vocab_size // -pad_vocab) * pad_vocab
        self.hparams["vocab_size"] = vocab_size

        # SentencePiece is checked first here, unlike Mamba-1, which prefers tokenizer.json.
        if (self.dir_model / "tokenizer.model").is_file():
            self._set_vocab_sentencepiece()
        elif (self.dir_model / "tokenizer.model.v3").is_file():
            # Mamba-Codestral ships its SentencePiece file under a .v3 name that the
            # loader cannot find, so ask the user to rename it rather than guess.
            # mamba-codestral
            raise NotImplementedError(f"Please rename {self.dir_model / 'tokenizer.model.v3'} to {self.dir_model / 'tokenizer.model'}")
        elif (self.dir_model / "tokenizer.json").is_file():
            self._set_vocab_gpt2()
        else:
            # Use the GPT-NeoX tokenizer when no tokenizer files are present
            self._set_vocab_builtin("gpt-neox", vocab_size)

    def set_gguf_parameters(self):
        """Write Mamba-2 SSM hyperparameters, including head count and group count."""
        # Step 1: Validate input parameters, operational preconditions, and state
        d_conv  = self.find_hparam(["conv_kernel", "d_conv"],     optional=True) or 4
        d_state = self.find_hparam(["state_size",  "d_state"],    optional=True) or 128
        head_dim = self.find_hparam(["mamba_d_head", "head_dim"], optional=True) or 64

        rms_norm_eps = self.find_hparam(["layer_norm_epsilon", "rms_norm_eps"], optional=True) or 1e-5
        # Mamba-2 heads: d_inner is split into d_inner // head_dim heads (written as time-step rank).

        # Fail early for models which don't have a block expansion factor of 2
        # TODO: does this really matter?
        # skip the assertion for FalconH1 Model
        # FalconH1 reuses this class with a different expansion factor, so skip the checks there.
        if self.model_arch != gguf.MODEL_ARCH.FALCON_H1:
            assert self.d_inner == 2 * self.d_model
            assert self.d_inner % head_dim == 0

        # Placeholder values for keys llama.cpp requires but the SSM graph ignores.
        self.gguf_writer.add_context_length(2**20)  # arbitrary value; for those who use the default
        self.gguf_writer.add_embedding_length(self.d_model)
        self.gguf_writer.add_feed_forward_length(0)  # unused, but seemingly required when loading
        self.gguf_writer.add_head_count(0)  # unused, but seemingly required when loading
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_ssm_conv_kernel(d_conv)
        self.gguf_writer.add_ssm_inner_size(self.d_inner)
        self.gguf_writer.add_ssm_state_size(d_state)
        self.gguf_writer.add_ssm_time_step_rank(self.d_inner // head_dim)
        self.gguf_writer.add_ssm_group_count(self.n_group)
        self.gguf_writer.add_layer_norm_rms_eps(rms_norm_eps)
        self.gguf_writer.add_file_type(self.ftype)

    # Class-level hook: runs on every (name, loader) pair before tensors are indexed.
    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        # Step 1: Validate input parameters, operational preconditions, and state
        """Normalise Mamba-Codestral tensor names, then defer to the base filter."""
        name, gen = item

        # Codestral wraps the backbone and head under an extra 'model.' prefix; strip it.
        if name.startswith(("model.backbone", "model.lm_head")):
            # map Mamba-Codestral-7B-v0.1 tensor names to the names used by Mamba-2
            name = name.removeprefix("model.")

        # Rename dt_bias to dt_proj.bias so the standard tensor map recognises it.
        if name.endswith(".dt_bias"):
            name = name.rpartition(".dt_bias")[0] + ".dt_proj.bias"

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Rename one tensor to its GGUF name and reshape SSM tensors for llama.cpp.

        Squeezes conv1d, adds a trailing dim to A and D, reshapes the SSM norm into
        groups, and converts A_log to A. Yields exactly one (name, tensor) pair.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        # Translate the HF tensor name to the GGUF name.
        new_name = self.map_tensor_name(name)

        # conv1d weight has a singleton dim that the GGUF layout drops.
        if self.match_model_tensor_name(new_name, gguf.MODEL_TENSOR.SSM_CONV1D, bid):
            data_torch = data_torch.squeeze()
        elif any(self.match_model_tensor_name(new_name, t, bid, suffix="") for t in [
            gguf.MODEL_TENSOR.SSM_A,
            gguf.MODEL_TENSOR.SSM_D,
        ]):
            # Note: these two use suffix="" because A and D have no .weight suffix.
            # unsqueeze A to use similar shape semantics as Mamba-1
            # (D is also unsqueezed, but for more straightforward broadcast internally)
            data_torch = data_torch.reshape((*data_torch.shape, 1))
        # The SSM norm weight is stored flat; llama.cpp wants it as [n_group, d_inner/n_group].
        elif self.match_model_tensor_name(new_name, gguf.MODEL_TENSOR.SSM_NORM, bid):
            data_torch = data_torch.reshape((self.n_group, self.d_inner // self.n_group))

        # A is stored as log; the runtime expects A = -exp(A_log).
        if name.endswith(".A_log"):
            logger.debug("A_log --> A ==> " + new_name)
            data_torch = -torch.exp(data_torch)

        # Exactly one tensor is emitted; Mamba-2 has no tied-embedding dedup here.
        yield (new_name, data_torch)
