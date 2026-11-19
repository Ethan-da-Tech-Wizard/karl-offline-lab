# ==============================================================================
# Program: falcon_h1.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for FALCON_H1.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf

from .llama import LlamaModel
# Falcon-H1 is a hybrid: Mamba2 SSM blocks plus attention, so it extends
# Mamba2Model and delegates attention tensors to LlamaModel.
from .mamba import Mamba2Model


@ModelBase.register("FalconH1ForCausalLM")
class FalconH1Model(Mamba2Model):
    """Converter for TII Falcon-H1 (parallel hybrid Mamba2 + attention).

    Falcon-H1 bakes several muP-style multipliers into the HF forward pass; this
    converter multiplies them into the weights so llama.cpp can run the plain graph.
    """
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.FALCON_H1

    def __init__(self, *args, **kwargs):
        """Load Falcon-H1 specific hyperparameters and multipliers before the base init completes."""
        # Some SSM keys are stored as "mamba_<name>"; find_hparam tries both spellings.
        # Set the hparam prefixes for Falcon Mamba2
        self.hparam_prefixes = ["mamba"]

        # Initialize the base Mamba2Model
        super().__init__(*args, **kwargs)

        # Attention tensors are converted exactly like Llama's.
        # Use Llama conversion for attention
        self._transformer_model_class = LlamaModel

        # (see note) needed to reshape the Mamba2 norm weights in modify_tensors.
        # n_group and d_inner are used during reshape_tensors for mamba2
        self.n_group = self.find_hparam(["n_groups"])
        self.d_inner = self.find_hparam(["mamba_d_ssm"])
        self.d_head = self.find_hparam(["d_head"])

        # Initialize any Falcon Mamba2 specific attributes
        self.has_attention = True  # Falcon Mamba2 has attention components

        # Optional because not every Falcon-H1 config defines every multiplier.
        # Load Falcon-H1 multipliers from hyperparameters
        self.attention_in_multiplier = self.find_hparam(["attention_in_multiplier"], optional=True)
        self.attention_out_multiplier = self.find_hparam(["attention_out_multiplier"], optional=True)
        self.ssm_in_multiplier = self.find_hparam(["ssm_in_multiplier"], optional=True)
        self.ssm_out_multiplier = self.find_hparam(["ssm_out_multiplier"], optional=True)
        self.mlp_multipliers = self.find_hparam(["mlp_multipliers"], optional=True)
        self.ssm_multipliers = self.find_hparam(["ssm_multipliers"], optional=True)
        self.intermediate_size = self.find_hparam(["intermediate_size"])
        self.key_multiplier = self.find_hparam(["key_multiplier"], optional=True)

    def find_hparam(self, keys: Iterable[str], *args, **kwargs) -> Any:
        """Look up an hparam under both its plain key and its "mamba_"-prefixed variant."""
        # Step 1: Validate input parameters, operational preconditions, and state
        prefixed = []
        for pfx in self.hparam_prefixes:
            # Iterate over items, evaluate loop invariant, and update progress
            prefixed.extend(
                "_".join([pfx, k])
                for k in keys
            )
        # Plain names take priority; prefixed names are fallbacks.
        keys = list(keys) + prefixed
        return super().find_hparam(keys, *args, **kwargs)

    def set_vocab(self):
        """Falcon-H1 uses a GPT-2 style BPE tokenizer."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._set_vocab_gpt2()

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Apply Falcon-H1 multipliers to the converted tensor and reshape the SSM norm.

        Only the first tensor yielded by the parent is kept; the multiplier is chosen
        from the tensor's HF name.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        # Run the parent conversion first, then scale its (single) output tensor.
        tensors = list(super().modify_tensors(data_torch, name, bid))
        tensor = tensors[0][1]

        if "down_proj" in name:
            tensor = tensor  * self.mlp_multipliers[1]
        elif "gate_proj" in name:
            tensor = tensor * self.mlp_multipliers[0]
        elif "k_proj" in name:
            tensor = tensor * self.key_multiplier * self.attention_in_multiplier
        elif "q_proj" in name:
            tensor = tensor * self.attention_in_multiplier
        elif "v_proj" in name:
            tensor = tensor * self.attention_in_multiplier
        elif "o_proj" in name:
            tensor = tensor * self.attention_out_multiplier
        elif "out_proj" in name:
            tensor = tensor * self.ssm_out_multiplier
        # The Mamba2 input projection packs five segments [z | x | B | C | dt]; each has
        # its own multiplier (ssm_multipliers), applied slice by slice on top of the
        # overall ssm_in_multiplier.
        elif "in_proj" in name:
            tensor = tensor * self.ssm_in_multiplier
            zxbcdt_multipliers = self.hparams["ssm_multipliers"]
            intermediate_size = self.hparams["mamba_d_ssm"]
            groups_time_state_size = self.hparams["mamba_n_groups"] * self.hparams["mamba_d_state"]
            tensor[:intermediate_size, :] *= zxbcdt_multipliers[0]
            tensor[intermediate_size:2 * intermediate_size, :] *= zxbcdt_multipliers[1]
            tensor[2 * intermediate_size:2 * intermediate_size + groups_time_state_size, :] *= zxbcdt_multipliers[2]
            tensor[2 * intermediate_size + groups_time_state_size:2 * intermediate_size + 2 * groups_time_state_size, :] *= zxbcdt_multipliers[3]
            tensor[2 * intermediate_size + 2 * groups_time_state_size:, :] *= zxbcdt_multipliers[4]
        elif "lm_head" in name:
            tensor = tensor * self.hparams["lm_head_multiplier"]
        elif "embed_tokens" in name:
            tensor = tensor * self.hparams["embedding_multiplier"]
        # The grouped norm weight is stored per group: [n_group, d_inner / n_group].
        elif "mamba.norm" in name:
            tensor = tensor.reshape(self.n_group, self.d_inner // self.n_group)

        tensors = [(tensors[0][0], tensor)]
        return tensors

    def set_gguf_parameters(self):
        """Override Mamba2 defaults with attention/FFN parameters and validate the config."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()

        # Mamba2 leaves these unset or zero, so write the transformer-side values.
        ## General Params ##
        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])
        # Override some Mamba2 defaults
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_context_length(self.hparams.get("max_position_embeddings", 0))
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])

        ## Attention params ##
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"]) # Override value 0 from Mamba2
        self.gguf_writer.add_head_count_kv(self.hparams["num_key_value_heads"])
        self.gguf_writer.add_key_length(self.hparams["head_dim"])
        self.gguf_writer.add_value_length(self.hparams["head_dim"])

        # Fail early on configs the runtime cannot represent.
        ## Validation ##
        assert self.hparams.get("hidden_act") in [None, "silu"], "Only SILU activation supported"
        assert self.d_inner % self.d_head == 0, f"SSM inner size {self.d_inner} not a multiple of head dim {self.d_head}"

        # Add any other Falcon Mamba2 specific configuration
        self.gguf_writer.add_rope_freq_base(self.rope_parameters["rope_theta"])
