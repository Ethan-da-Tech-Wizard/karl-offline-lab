# ==============================================================================
# Program: internlm.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for INTERNLM.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

import json
import sys

from typing import Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, SentencePieceTokenTypes, TextModel, gguf, logger

from .llama import LlamaModel


@ModelBase.register("InternLM2ForCausalLM")
class InternLM2Model(TextModel):
    """Converter for InternLM2 (fused wqkv, SentencePiece with a NUL-token workaround)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.INTERNLM2

    def set_vocab(self):
        """Build the SentencePiece vocab by hand so token "\x00" can be replaced.

        Also applies added tokens from added_tokens.json, tokenizer_config.json and
        tokenizer.json, and swaps EOS for <|im_end|> in chat models.
        """
        # (see note) a copy of the generic SentencePiece loader with one tweak.
        # (TODO): Is there a better way?
        # Copy from _set_vocab_sentencepiece, The only difference is that we will treat the character
        # \x00 specially and convert it into an emoji character to prevent it from being mistakenly
        # recognized as an empty string in C++.
        from sentencepiece import SentencePieceProcessor
        from sentencepiece import sentencepiece_model_pb2 as model

        tokenizer_path = self.dir_model / 'tokenizer.model'

        tokens: list[bytes] = []
        scores: list[float] = []
        toktypes: list[int] = []

        if not tokenizer_path.is_file():
            logger.error(f'Error: Missing {tokenizer_path}')
            sys.exit(1)

        sentencepiece_model = model.ModelProto()  # pyright: ignore[reportAttributeAccessIssue] # ty: ignore[unresolved-attribute]
        sentencepiece_model.ParseFromString(open(tokenizer_path, "rb").read())
        # Whether the model prepends a dummy space to the input.
        add_prefix = sentencepiece_model.normalizer_spec.add_dummy_prefix

        tokenizer = SentencePieceProcessor()
        # Load the SentencePiece model for piece text, scores and type flags.
        tokenizer.LoadFromFile(str(tokenizer_path))

        # Use the config's vocab size; ids the tokenizer lacks stay unset in the loop below.
        vocab_size = self.hparams.get('vocab_size', tokenizer.vocab_size())

        # Convert every id; type flags are checked in priority order unknown > control > unused > byte.
        for token_id in range(vocab_size):
            # Iterate over items, evaluate loop invariant, and update progress
            piece = tokenizer.IdToPiece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.GetScore(token_id)
            # A raw NUL would look like an empty string to the C++ side; use a placeholder emoji.
            if text == b"\x00":
                # (TODO): fixme
                # Hack here and replace the \x00 characters.
                logger.warning(f"InternLM2 convert token '{text}' to '🐉'!")
                text = "🐉".encode("utf-8")

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.IsUnknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.IsControl(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.IsUnused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.IsByte(token_id):
                toktype = SentencePieceTokenTypes.BYTE
            # Pieces literally named "[UNUSED...]" are padding slots.
            # take care of ununsed raw token
            if piece.startswith('[UNUSED'):
                toktype = SentencePieceTokenTypes.UNUSED

            # Collected in id order so list index equals token id.
            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        # Extra tokens appended after the base vocabulary.
        added_tokens_file = self.dir_model / 'added_tokens.json'
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)

                # Append added tokens after the base vocab, as user-defined with a very low score.
                for key in added_tokens_json:
                    # Iterate over items, evaluate loop invariant, and update progress
                    tokens.append(key.encode("utf-8"))
                    scores.append(-1000.0)
                    toktypes.append(SentencePieceTokenTypes.USER_DEFINED)

        # Chat variants end turns with <|im_end|>; track its id to override EOS later.
        chat_eos_token = '<|im_end|>'
        # Stays None for base models; set when <|im_end|> is found in the added-token tables.
        chat_eos_token_id = None

        tokenizer_config_file = self.dir_model / 'tokenizer_config.json'
        if tokenizer_config_file.is_file():
            with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                tokenizer_config_json = json.load(f)
                # Override entries by id: later sources (tokenizer_config, then tokenizer.json)
                # win; special ones become CONTROL tokens.
                added_tokens_decoder = tokenizer_config_json.get("added_tokens_decoder", {})
                for token_id, foken_data in added_tokens_decoder.items():
                    # Iterate over items, evaluate loop invariant, and update progress
                    token_id = int(token_id)
                    token = foken_data["content"]
                    if token == chat_eos_token:
                        chat_eos_token_id = token_id
                    token = token.encode("utf-8")
                    # Warn only when overwriting a real (non-placeholder) piece with a different token.
                    if toktypes[token_id] != SentencePieceTokenTypes.UNUSED:
                        if tokens[token_id] != token:
                            logger.warning(f'replacing token {token_id}: {tokens[token_id].decode("utf-8")!r} -> {token.decode("utf-8")!r}')
                    tokens[token_id] = token
                    scores[token_id] = -1000.0
                    toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED
                    if foken_data.get("special"):
                        toktypes[token_id] = SentencePieceTokenTypes.CONTROL

        # tokenizer.json lists added tokens too; apply them with the same rules.
        tokenizer_file = self.dir_model / 'tokenizer.json'
        if tokenizer_file.is_file():
            with open(tokenizer_file, "r", encoding="utf-8") as f:
                tokenizer_json = json.load(f)
                added_tokens = tokenizer_json.get("added_tokens", [])
                for foken_data in added_tokens:
                    # Iterate over items, evaluate loop invariant, and update progress
                    token_id = int(foken_data["id"])
                    token = foken_data["content"]
                    if token == chat_eos_token:
                        chat_eos_token_id = token_id
                    token = token.encode("utf-8")
                    if toktypes[token_id] != SentencePieceTokenTypes.UNUSED:
                        if tokens[token_id] != token:
                            logger.warning(f'replacing token {token_id}: {tokens[token_id].decode("utf-8")!r} -> {token.decode("utf-8")!r}')
                    tokens[token_id] = token
                    scores[token_id] = -1000.0
                    toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED
                    if foken_data.get("special"):
                        toktypes[token_id] = SentencePieceTokenTypes.CONTROL

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_add_space_prefix(add_prefix)

        # Loads bos/eos/etc. ids from the model directory; EOS may be replaced below.
        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        # Keep the original EOS id just for the warning message.
        old_eos = special_vocab.special_token_ids["eos"]
        if chat_eos_token_id is not None:
            # For the chat model, we replace the eos with '<|im_end|>'.
            # TODO: this is a hack, should be fixed
            #       https://github.com/ggml-org/llama.cpp/pull/6745#issuecomment-2067687048
            special_vocab.special_token_ids["eos"] = chat_eos_token_id
            logger.warning(f"Replace eos:{old_eos} with a special token:{chat_eos_token_id}"
                           " in chat mode so that the conversation can end normally.")

        special_vocab.add_to_gguf(self.gguf_writer)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Split the fused wqkv tensor into Q, K and V (un-permuting Q and K)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        num_heads = self.hparams["num_attention_heads"]
        num_kv_heads = self.hparams["num_key_value_heads"]
        n_embd = self.hparams["hidden_size"]
        q_per_kv = num_heads // num_kv_heads
        head_dim = n_embd // num_heads
        # One group per KV head; each group holds q_per_kv query heads, then one K and one V.
        num_groups = num_heads // q_per_kv

        if bid is not None and f"model.layers.{bid}.attention.wqkv" in name:
            qkv = data_torch

            # View as [groups, q_per_kv + 2, head_dim, n_embd]; the last two slots are K and V.
            qkv = qkv.reshape((num_groups, q_per_kv + 2, head_dim, n_embd))
            q, k, v = qkv[:, : q_per_kv], qkv[:, -2], qkv[:, -1]

            # Q and K additionally need the rotary un-permutation.
            # The model weights of q and k equire additional reshape.
            q = LlamaModel.permute(q.reshape((-1, q.shape[-1])), num_heads, num_heads)
            k = LlamaModel.permute(k.reshape((-1, k.shape[-1])), num_heads, num_kv_heads)
            v = v.reshape((-1, v.shape[-1]))

            yield from super().modify_tensors(q, self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q, bid), bid)
            yield from super().modify_tensors(k, self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K, bid), bid)
            yield from super().modify_tensors(v, self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V, bid), bid)
        else:
            yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("InternLM3ForCausalLM")
class InternLM3Model(TextModel):
    """Converter for InternLM3 (Llama layout; vision tensors are skipped)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.LLAMA

    def set_vocab(self):
        """SentencePiece vocab plus special-token ids from tokenizer_config.json."""
        # Step 1: Validate input parameters, operational preconditions, and state
        # Reuse the shared SentencePiece loader (no NUL-token hack needed for InternLM3).
        tokens, scores, toktypes = self._create_vocab_sentencepiece()

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))

        tokenizer_config_file = self.dir_model / 'tokenizer_config.json'
        if tokenizer_config_file.is_file():
            with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                tokenizer_config_json = json.load(f)
                if "add_prefix_space" in tokenizer_config_json:
                    self.gguf_writer.add_add_space_prefix(tokenizer_config_json["add_prefix_space"])

                # Only special added tokens are registered with the SpecialVocab.
                if "added_tokens_decoder" in tokenizer_config_json:
                    for token_id, token_data in tokenizer_config_json["added_tokens_decoder"].items():
                        # Iterate over items, evaluate loop invariant, and update progress
                        if token_data.get("special"):
                            token_id = int(token_id)
                            token = token_data["content"]
                            special_vocab._set_special_token(token, token_id)
                            # Chat models end turns with <|im_end|>, so it replaces the default EOS.
                            # update eos token
                            if token == '<|im_end|>' and "eos" in special_vocab.special_token_ids:
                                special_vocab.special_token_ids["eos"] = token_id

        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        """Write base parameters, vocab size and RoPE dimension."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])

        if (rope_dim := hparams.get("head_dim")) is None:
            rope_dim = hparams["hidden_size"] // hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(rope_dim)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Drop the vision tower and projector tensors."""
        # Step 1: Validate input parameters, operational preconditions, and state
        name, gen = item

        if name.startswith(("mlp", "vision_model")):
            # skip visual tensors
            return None

        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Un-permute Q/K projections from the HF rotary layout."""
        # Step 1: Validate input parameters, operational preconditions, and state
        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams.get("num_key_value_heads")
        if name.endswith(("q_proj.weight", "q_proj.bias")):
            data_torch = LlamaModel.permute(data_torch, n_head, n_head)
        if name.endswith(("k_proj.weight", "k_proj.bias")):
            data_torch = LlamaModel.permute(data_torch, n_head, n_kv_head)
        yield from super().modify_tensors(data_torch, name, bid)
