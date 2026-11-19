# ==============================================================================
# Program: arctic.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for ARCTIC.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

import json
import sys

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, SentencePieceTokenTypes, TextModel, gguf, logger

from .llama import LlamaModel


@ModelBase.register("ArcticForCausalLM")
class ArcticModel(TextModel):
    """Converter for Snowflake Arctic (dense + MoE residual) checkpoints.

    Uses a custom SentencePiece vocab loader and Llama-style Q/K permutation,
    then merges the per-expert Mixtral-named weights into stacked tensors.
    """
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.ARCTIC

    def set_vocab(self):
        """Build the vocabulary from tokenizer.model, then apply added-token overrides."""
        # The reason for using a custom implementation here is that the
        # snowflake-arctic-instruct model redefined tokens 31998 and 31999 from
        # tokenizer.model and used them as BOS and EOS instead of adding new tokens.
        # Imported lazily so the dependency is only needed for this architecture.
        from sentencepiece import SentencePieceProcessor

        tokenizer_path = self.dir_model / 'tokenizer.model'

        # Fail fast with a clear message; there is no fallback tokenizer format here.
        if not tokenizer_path.is_file():
            logger.error(f'Error: Missing {tokenizer_path}')
            sys.exit(1)

        # Read the whole vocabulary from the tokenizer.model file
        tokenizer = SentencePieceProcessor()
        tokenizer.LoadFromFile(str(tokenizer_path))

        # The model may declare a larger vocab than the tokenizer file defines; the
        # extra slots are padded below so tensor shapes still match.
        vocab_size = self.hparams.get('vocab_size', tokenizer.vocab_size())

        # Pre-fill every slot with an UNUSED placeholder ([PAD<i>], very low score), then
        # overwrite the ids that the tokenizer actually defines.
        tokens: list[bytes] = [f"[PAD{i}]".encode("utf-8") for i in range(vocab_size)]
        scores: list[float] = [-10000.0] * vocab_size
        toktypes: list[int] = [SentencePieceTokenTypes.UNUSED] * vocab_size

        # Classify each real piece. The order of the checks matters: a piece is only
        # treated as BYTE if it is none of unknown/control/unused.
        for token_id in range(tokenizer.vocab_size()):

            # Iterate over items, evaluate loop invariant, and update progress
            piece = tokenizer.IdToPiece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.GetScore(token_id)

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.IsUnknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.IsControl(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.IsUnused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.IsByte(token_id):
                toktype = SentencePieceTokenTypes.BYTE

            tokens[token_id] = text
            scores[token_id] = score
            toktypes[token_id] = toktype

        # tokenizer_config.json can redefine ids (here 31998/31999 become BOS/EOS), so it
        # is applied last and wins over tokenizer.model.
        # Use the added_tokens_decoder field from tokeniser_config.json as the source
        # of information about added/redefined tokens and modify them accordingly.
        tokenizer_config_file = self.dir_model / 'tokenizer_config.json'
        if tokenizer_config_file.is_file():
            with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                tokenizer_config_json = json.load(f)

                if "added_tokens_decoder" in tokenizer_config_json:
                    added_tokens_decoder = tokenizer_config_json["added_tokens_decoder"]
                    for token_id, token_json in added_tokens_decoder.items():
                        # Iterate over items, evaluate loop invariant, and update progress
                        token_id = int(token_id)
                        # Ids beyond the declared vocab cannot be stored; skip them with a debug note.
                        if token_id >= vocab_size:
                            logger.debug(f'ignore token {token_id}: id is out of range, max={vocab_size - 1}')
                            continue

                        token_content = token_json["content"]
                        # Defaults for a non-special added token; specials are adjusted just below.
                        token_type = SentencePieceTokenTypes.USER_DEFINED
                        token_score = -10000.0

                        # Map unk_token to UNKNOWN, other special tokens to CONTROL
                        # Set the score to 0.0 as in the original tokenizer.model
                        if ("special" in token_json) and token_json["special"]:
                            if token_content == tokenizer_config_json["unk_token"]:
                                token_type = SentencePieceTokenTypes.UNKNOWN
                            else:
                                token_type = SentencePieceTokenTypes.CONTROL
                            token_score = 0.0

                        logger.info(f"Setting added token {token_id} to '{token_content}' (type: {token_type}, score: {token_score:.2f})")
                        tokens[token_id] = token_content.encode("utf-8")
                        toktypes[token_id] = token_type
                        scores[token_id] = token_score

        # Write the vocabulary in the "llama" (SentencePiece/BPE-with-scores) flavour.
        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        # Pick up bos/eos/etc. ids from the model directory and write them too.
        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        """Write base parameters plus vocab size and RoPE dimension (hidden/heads)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_rope_dimension_count(hparams["hidden_size"] // hparams["num_attention_heads"])

    # Per-layer buffers for expert tensors awaiting merge (see modify_tensors).
    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Un-permute Q/K weights, and merge per-expert weights into stacked tensors.

        Generator: yields nothing while a layer's experts are still being collected.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams.get("num_key_value_heads")

        # HF stores Q/K with a rotary-friendly head layout; llama.cpp wants the
        # original layout, so undo it. K may use fewer heads (grouped-query attention).
        if name.endswith("q_proj.weight"):
            data_torch = LlamaModel.permute(data_torch, n_head, n_head)
        if name.endswith("k_proj.weight"):
            data_torch = LlamaModel.permute(data_torch, n_head, n_kv_head)

        # Expert tensors are only recognised by name here; other tensors fall through.
        # process the experts separately
        if name.find("block_sparse_moe.experts") != -1:
            n_experts = self.hparams["num_local_experts"]

            assert bid is not None

            # Experts arrive as separate per-expert tensors; buffer them per layer
            # (block id) until the whole set for that layer has been seen.
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            # 3 weights (w1/w2/w3) x n_experts tensors present: the layer is complete.
            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for wid in ["w1", "w2", "w3"]:
                    # Iterate over items, evaluate loop invariant, and update progress
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        # Iterate over items, evaluate loop invariant, and update progress
                        ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{wid}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    # Stack the per-expert 2D matrices into [n_experts, ...].
                    data_torch = torch.stack(datas, dim=0)

                    # Rename to the llama.cpp-style name so the tensor map recognises it.
                    merged_name = f"layers.{bid}.feed_forward.experts.{wid}.weight"

                    yield from super().modify_tensors(data_torch, merged_name, bid)
                return
            else:
                return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        """Run the normal conversion, then verify every expert tensor was merged."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().prepare_tensors()

        # Anything still buffered here means the checkpoint was missing expert
        # tensors, so fail loudly instead of writing a broken GGUF.
        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")
