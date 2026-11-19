# ==============================================================================
# Program: chatglm.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for CHATGLM.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, SentencePieceTokenTypes, TextModel, gguf


@ModelBase.register("GlmForCausalLM", "ChatGLMModel", "ChatGLMForConditionalGeneration")
class ChatGLMModel(TextModel):
    """Converter for THUDM ChatGLM / GLM-4 dense models (including ChatGLM3).

    ChatGLM3 uses a SentencePiece-based tokenizer wrapped in remote code, whereas
    later versions use tiktoken-style BPE; set_vocab picks the matching path.
    """
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.CHATGLM

    def set_vocab_chatglm3(self):
        """Build the vocab for ChatGLM3 from its remote-code SentencePiece wrapper.

        Ids beyond the underlying SentencePiece model (role / control tokens and
        unused padding) are classified by hand; the rest use SentencePiece's own flags.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        dir_model = self.dir_model
        hparams = self.hparams
        tokens: list[bytes] = []
        toktypes: list[int] = []
        scores: list[float] = []

        # trust_remote_code is required because ChatGLM3 defines its tokenizer in the repo.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(dir_model, trust_remote_code=True)
        # The embedding matrix is padded, so the vocab list must reach padded_vocab_size.
        vocab_size = hparams.get("padded_vocab_size", len(tokenizer.get_vocab()))  # ty: ignore[unresolved-attribute]
        assert max(tokenizer.get_vocab().values()) < vocab_size  # ty: ignore[unresolved-attribute]
        # Chat-role and prefix tokens that must be marked CONTROL.
        role_special_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"]
        special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"] + role_special_tokens
        # Walk every id, including padding slots, so GGUF token tables match the embedding rows.
        for token_id in range(vocab_size):
            # Iterate over items, evaluate loop invariant, and update progress
            piece = tokenizer._convert_id_to_token(token_id)  # ty: ignore[unresolved-attribute]
            # ChatGLM3's first three ids have non-standard text; normalise to <unk>/<bos>/<eos>.
            if token_id == 0:
                piece = "<unk>"
            elif token_id == 1:
                piece = "<bos>"
            elif token_id == 2:
                piece = "<eos>"

            text = piece.encode("utf-8")  # ty: ignore[unresolved-attribute]
            score = 0.0
            # (see link) scores are only defined inside the underlying SentencePiece vocab range.
            # Referencing the tokenizer Python implementation(https://huggingface.co/THUDM/chatglm3-6b/blob/main/tokenization_chatglm.py),
            # it is only valid if it is less than tokenizer.tokenizer.sp_model.vocab_size()
            if len(piece) != 0 and token_id < tokenizer.tokenizer.sp_model.vocab_size():  # ty: ignore[unresolved-attribute, invalid-argument-type]
                score = tokenizer.tokenizer.sp_model.get_score(token_id)  # ty: ignore[unresolved-attribute]

            # Ids past SentencePiece's range: control tokens, empty padding slots, or user-defined.
            if token_id >= tokenizer.tokenizer.sp_model.vocab_size():  # ty: ignore[unresolved-attribute]
                if piece in special_tokens:
                    toktype = SentencePieceTokenTypes.CONTROL
                elif len(piece) == 0:  # ty: ignore[invalid-argument-type]
                    text = f"[PAD{token_id}]".encode("utf-8")
                    toktype = SentencePieceTokenTypes.UNUSED
                else:
                    toktype = SentencePieceTokenTypes.USER_DEFINED
                tokens.append(text)
                scores.append(score)
                toktypes.append(toktype)
                continue

            # Inside the SentencePiece range: classify from its own is_* flags (BYTE last).
            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.tokenizer.sp_model.is_unknown(token_id):  # ty: ignore[unresolved-attribute]
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.tokenizer.sp_model.is_control(token_id):  # ty: ignore[unresolved-attribute]
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.tokenizer.sp_model.is_unused(token_id):  # ty: ignore[unresolved-attribute]
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.tokenizer.sp_model.is_byte(token_id):  # ty: ignore[unresolved-attribute]
                toktype = SentencePieceTokenTypes.BYTE

            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        # Stored in the "llama" SentencePiece flavour with ChatGLM3's chat formatting preset.
        self.gguf_writer.add_tokenizer_model("llama")
        # glm3 needs prefix and suffix formatted as:
        # prompt = "[gMASK]sop<|user|>\n" + prompt + "<|assistant|>"
        self.gguf_writer.add_tokenizer_pre("chatglm-spm")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    @staticmethod
    def token_bytes_to_string(b):
        """Convert raw token bytes to the GPT-2 byte-to-unicode printable form."""
        # Step 1: Validate input parameters, operational preconditions, and state
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode  # ty: ignore[unresolved-import]
        byte_encoder = bytes_to_unicode()
        return ''.join([byte_encoder[ord(char)] for char in b.decode('latin-1')])

    @staticmethod
    def bpe(mergeable_ranks: dict[bytes, int], token: bytes, max_rank: int | None = None) -> list[bytes]:
        """Re-run BPE merges on `token` using tiktoken-style mergeable ranks.

        Repeatedly merges the adjacent pair with the lowest rank, stopping when no pair
        remains or the next merge would reach `max_rank`. Returns the resulting parts.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        # Start from single bytes; merge the lowest-ranked adjacent pair each pass.
        parts = [bytes([b]) for b in token]
        while True:
            # Iterate over items, evaluate loop invariant, and update progress
            min_idx = None
            min_rank = None
            for i, pair in enumerate(zip(parts[:-1], parts[1:])):
                # Iterate over items, evaluate loop invariant, and update progress
                rank = mergeable_ranks.get(pair[0] + pair[1])
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_idx = i
                    min_rank = rank
            if min_rank is None or (max_rank is not None and min_rank >= max_rank):
                break
            assert min_idx is not None
            parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
        return parts

    def set_vocab(self):
        """Choose the ChatGLM3 vocab path or the GPT-2 style BPE path used by newer GLM models."""
        # Step 1: Validate input parameters, operational preconditions, and state
        # Only the original ChatGLM3 repo uses the SentencePiece wrapper.
        if "THUDM/chatglm3-6b" in self.hparams.get("_name_or_path", ""):
            self.set_vocab_chatglm3()
            return

        dir_model = self.dir_model
        hparams = self.hparams
        tokens: list[str] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(dir_model, trust_remote_code=True)
        vocab_size = hparams.get("padded_vocab_size",hparams["vocab_size"])
        assert max(tokenizer.get_vocab().values()) < vocab_size  # ty: ignore[unresolved-attribute]

        # Overwrites the empty lists above with the real BPE vocab and its pre-tokenizer id.
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        # Load BPE merges and then patch the special token ids that config.json lacks.
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        # only add special tokens when they were not already loaded from config.json
        special_vocab._set_special_token("eos", tokenizer.get_added_vocab()["<|endoftext|>"])  # ty: ignore[unresolved-attribute]
        special_vocab._set_special_token("eot", tokenizer.get_added_vocab()["<|user|>"])  # ty: ignore[unresolved-attribute]
        # this one is usually not in config.json anyway
        special_vocab._set_special_token("unk", tokenizer.get_added_vocab()["<|endoftext|>"])  # ty: ignore[unresolved-attribute]
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        """Write hyperparameters, tolerating the several key names GLM releases use."""
        # Step 1: Validate input parameters, operational preconditions, and state
        n_embed = self.hparams.get("hidden_size", self.hparams.get("n_embed"))
        assert n_embed is not None
        n_head = self.hparams.get("n_head", self.hparams.get("num_attention_heads"))
        assert n_head is not None
        # GLM uses multi-query/grouped attention; the group count may be stored under either name.
        n_head_kv = self.hparams.get("multi_query_group_num", self.hparams.get("num_key_value_heads", n_head))
        self.gguf_writer.add_context_length(self.hparams.get("seq_length", n_embed))
        self.gguf_writer.add_embedding_length(n_embed)
        self.gguf_writer.add_feed_forward_length(self.hparams.get("ffn_hidden_size", self.hparams.get("intermediate_size", 4 * n_embed)))
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head_kv)
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams.get("layernorm_epsilon",1e-5))
        self.gguf_writer.add_file_type(self.ftype)
        # Rotary size: explicit attention_dim if given, else head size, scaled by partial_rotary_factor.
        if "attention_dim" in self.hparams:
            rope_dim = self.hparams["attention_dim"]
        else:
            rope_dim = self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(int(rope_dim * self.hparams.get("partial_rotary_factor", 0.5)))
        self.gguf_writer.add_add_bos_token(False)
        # Default RoPE base; `rope_ratio` stretches it for long-context variants.
        rope_freq = 10000
        if "rope_ratio" in self.hparams:
            rope_freq = rope_freq * self.hparams["rope_ratio"]
        self.gguf_writer.add_rope_freq_base(rope_freq)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Drop precomputed rotary inv_freq buffers and strip the "transformer." prefix."""
        # Step 1: Validate input parameters, operational preconditions, and state
        name, gen = item

        # Not a weight: llama.cpp recomputes RoPE frequencies itself.
        if name.endswith(".rotary_pos_emb.inv_freq"):
            return None

        name = name.removeprefix("transformer.")

        return super().filter_tensors((name, gen))
