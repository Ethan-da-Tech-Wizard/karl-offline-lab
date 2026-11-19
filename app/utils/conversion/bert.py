# ==============================================================================
# Program: bert.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for BERT.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

import json
import os

from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, SentencePieceTokenTypes, TextModel, gguf, logger


@ModelBase.register("BertModel", "BertForMaskedLM", "CamembertModel", "BertForSequenceClassification")
class BertModel(TextModel):
    """Converter for BERT-style encoder (embedding / classifier) models.

    Only the encoder is converted: pooler and MLM / NSP heads are dropped. Attention is
    non-causal, tokens are written as WordPiece converted to a SentencePiece-like
    "phantom space" vocab, and many other encoders (DistilBERT, RoBERTa, Nomic, ...)
    subclass this.
    """
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.BERT

    def __init__(self, *args, **kwargs):
        """Detect classifier labels (BertForSequenceClassification) in the config."""
        super().__init__(*args, **kwargs)
        self.vocab_size = None

        # AutoConfig invents default LABEL_0/LABEL_1 labels for models without a real
        # classifier head; treat those as "no classifier".
        if cls_out_labels := self.hparams.get("id2label"):
            if len(cls_out_labels) == 2 and cls_out_labels[0] == "LABEL_0":
                # Remove dummy labels added by AutoConfig
                cls_out_labels = None
        self.cls_out_labels = cls_out_labels

    def set_gguf_parameters(self):
        """Write bidirectional-attention parameters, pooling type and classifier labels."""
        super().set_gguf_parameters()
        self.gguf_writer.add_causal_attention(False)
        self._try_set_pooling_type()

        if self.cls_out_labels:
            self.gguf_writer.add_classifier_output_labels([v for k, v in sorted(self.cls_out_labels.items())])

    def set_vocab(self):
        """Write the WordPiece vocab as a "bert" tokenizer using the phantom-space convention."""
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.vocab_size = len(tokens)

        # token-type embeddings are present in the weights, so their count is recorded.
        # we need this to validate the size of the token_type embeddings
        # though currently we are passing all zeros to the token_type embeddings
        # "Sequence A" or "Sequence B"
        self.gguf_writer.add_token_type_count(self.hparams.get("type_vocab_size", 1))

        # convert to phantom space vocab
        def phantom(tok, toktype):
            """Convert a WordPiece token to llama.cpp's convention: "##x" -> "x" (continuation) and
            word-initial tokens get a leading "▁"; control tokens are left unchanged.
            """
            if toktype == gguf.TokenType.CONTROL:
                return tok
            if tok.startswith("##"):
                return tok[2:]
            return "\u2581" + tok
        assert len(tokens) == len(toktypes)
        tokens = list(map(phantom, tokens, toktypes))

        # add vocab to gguf
        self.gguf_writer.add_tokenizer_model("bert")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        # handle special tokens
        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Strip the "bert." prefix, rename gamma/beta to weight/bias, and drop unused heads."""
        name, gen = item

        if name.startswith("bert."):
            name = name[5:]

        if name.endswith(".gamma"):
            name = name[:-6] + ".weight"

        if name.endswith(".beta"):
            name = name[:-5] + ".bias"

        # the pooler and the pretraining heads (MLM, next-sentence) are not needed.
        # we are only using BERT for embeddings so we don't need the pooling layer
        if name in ("embeddings.position_ids", "pooler.dense.weight", "pooler.dense.bias"):
            return None

        if name.startswith("cls.predictions"):
            return None

        if name.startswith("cls.seq_relationship"):
            return None

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Rename the classification projection to the name the tensor map expects."""
        if self.cls_out_labels:
            # For BertForSequenceClassification (direct projection layer)
            if name == "classifier.weight":
                name = "classifier.out_proj.weight"

            if name == "classifier.bias":
                name = "classifier.out_proj.bias"

        yield from super().modify_tensors(data_torch, name, bid)

    def _xlmroberta_tokenizer_init(self) -> None:
        """Compute the position-embedding offset used by RoBERTa-style models.

        RoBERTa position ids start at pad_token_id + 1, so that many leading rows of the
        position matrix are unusable; max_position_embeddings is reduced to match.
        """
        # we need the pad_token_id to know how to chop down position_embd matrix
        if (pad_token_id := self.hparams.get("pad_token_id")) is not None:
            self._position_offset = 1 + pad_token_id
            if "max_position_embeddings" in self.hparams:
                self.hparams["max_position_embeddings"] -= self._position_offset
        else:
            self._position_offset = None

    def _xlmroberta_set_vocab(self) -> None:
        """Write the XLM-RoBERTa (Unigram/SentencePiece) vocab with its normalizer settings.

        Reads sentencepiece.bpe.model if present, otherwise reconstructs the same
        information from tokenizer.json (Jina v3 style). The charsmap (text normalisation
        table) and whitespace options are copied so the runtime normalises identically.
        """
        # forces the pure-python protobuf implementation before importing the proto module.
        # to avoid TypeError: Descriptors cannot be created directly
        # exception when importing sentencepiece_model_pb2
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        from sentencepiece import SentencePieceProcessor
        from sentencepiece import sentencepiece_model_pb2 as model

        tokenizer_path = self.dir_model / 'sentencepiece.bpe.model'

        tokenizer_json = {}
        tokenizer_config_json = {}
        if not tokenizer_path.is_file():
            tokenizer_path = self.dir_model / 'tokenizer.json'
            tokenizer_config_path = self.dir_model / 'tokenizer_config.json'

            if not tokenizer_path.is_file():
                raise FileNotFoundError(f"File not found: {tokenizer_path}")

            from base64 import b64decode
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.dir_model)

            with open(tokenizer_path, "r", encoding="utf-8") as fp:
                tokenizer_json = json.load(fp)

            if tokenizer_config_path.is_file():
                with open(tokenizer_config_path, "r", encoding="utf-8") as fp:
                    tokenizer_config_json = json.load(fp)

            add_prefix = tokenizer.add_prefix_space  # ty: ignore[unresolved-attribute]
            remove_whitespaces = tokenizer.clean_up_tokenization_spaces  # ty: ignore[unresolved-attribute]
            precompiled_charsmap = b64decode(tokenizer_json["normalizer"]["precompiled_charsmap"])

            vocab_size = max(self.hparams.get("vocab_size", 0), tokenizer.vocab_size)  # ty: ignore[unresolved-attribute]
        else:
            sentencepiece_model = model.ModelProto()  # pyright: ignore[reportAttributeAccessIssue] # ty: ignore[unresolved-attribute]
            sentencepiece_model.ParseFromString(open(tokenizer_path, "rb").read())
            assert sentencepiece_model.trainer_spec.model_type == 1  # UNIGRAM

            add_prefix = sentencepiece_model.normalizer_spec.add_dummy_prefix
            remove_whitespaces = sentencepiece_model.normalizer_spec.remove_extra_whitespaces
            precompiled_charsmap = sentencepiece_model.normalizer_spec.precompiled_charsmap

            tokenizer = SentencePieceProcessor()
            tokenizer.LoadFromFile(str(tokenizer_path))

            vocab_size = max(self.hparams.get("vocab_size", 0), tokenizer.vocab_size())

        # Pre-fill with placeholders; real pieces overwrite them below.
        tokens: list[bytes] = [f"[PAD{i}]".encode("utf-8") for i in range(vocab_size)]
        scores: list[float] = [-10000.0] * vocab_size
        toktypes: list[int] = [SentencePieceTokenTypes.UNUSED] * vocab_size

        # Two sources of pieces: the SentencePiece model itself, or the HF tokenizer.json path.
        if isinstance(tokenizer, SentencePieceProcessor):
            for token_id in range(tokenizer.vocab_size()):
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
        else:
            added_vocab = tokenizer.get_added_vocab()  # ty: ignore[unresolved-attribute]
            unk_token = tokenizer_config_json.get("unk_token")
            unk_token_id = added_vocab.get(unk_token, tokenizer_json["model"].get("unk_id", 3))  # ty: ignore[no-matching-overload]

            for token_id in range(tokenizer.vocab_size):  # ty: ignore[unresolved-attribute]
                piece = tokenizer._convert_id_to_token(token_id)  # ty: ignore[unresolved-attribute]
                if (piece := tokenizer._convert_id_to_token(token_id)) is not None:  # ty: ignore[unresolved-attribute]
                    text = piece.encode("utf-8")
                    score = tokenizer_json["model"]["vocab"][token_id][1]

                    toktype = SentencePieceTokenTypes.NORMAL
                    if token_id == unk_token_id:
                        toktype = SentencePieceTokenTypes.UNKNOWN
                    elif token_id in tokenizer.all_special_ids:  # ty: ignore[unresolved-attribute]
                        toktype = SentencePieceTokenTypes.CONTROL
                    elif token_id in added_vocab.values():
                        toktype = SentencePieceTokenTypes.USER_DEFINED
                    # No reliable way to detect this, but jina doesn't have any
                    # elif tokenizer.IsByte(token_id):
                    #     toktype = SentencePieceTokenTypes.BYTE

                    tokens[token_id] = text
                    scores[token_id] = score
                    toktypes[token_id] = toktype

        if isinstance(tokenizer, SentencePieceProcessor):
            # The HF XLM-R tokenizer numbers its first four ids <s>, <pad>, </s>, <unk>, unlike
            # the raw SentencePiece model. Replace the first three SentencePiece entries with
            # those four tokens and drop the last slot so the list length stays unchanged.
            # realign tokens (see HF tokenizer code)
            tokens = [b'<s>', b'<pad>', b'</s>', b'<unk>'] + tokens[3:-1]
            scores = [0.0, 0.0, 0.0, 0.0] + scores[3:-1]
            toktypes = [
                SentencePieceTokenTypes.CONTROL,
                SentencePieceTokenTypes.CONTROL,
                SentencePieceTokenTypes.CONTROL,
                SentencePieceTokenTypes.UNKNOWN,
            ] + toktypes[3:-1]

            if self.model_arch == gguf.MODEL_ARCH.NOMIC_BERT_MOE:
                # Nomic-embed-text-v2-moe uses a <mask> token that the sentencepiece model lacks.
                # Add mask token missing from sentencepiece.bpe.model
                tokens[250001] = b'<mask>'
                scores[250001] = 0.0
                toktypes[250001] = SentencePieceTokenTypes.CONTROL

        # The Unigram tokenizer is written under the "t5" (SentencePiece Unigram) model type.
        self.gguf_writer.add_tokenizer_model("t5")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_add_space_prefix(add_prefix)
        self.gguf_writer.add_token_type_count(self.hparams.get("type_vocab_size", 1))
        self.gguf_writer.add_remove_extra_whitespaces(remove_whitespaces)
        if precompiled_charsmap:
            self.gguf_writer.add_precompiled_charsmap(precompiled_charsmap)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)


@ModelBase.register("DistilBertModel", "DistilBertForMaskedLM", "DistilBertForSequenceClassification")
class DistilBertModel(BertModel):
    """Converter for DistilBERT (BERT without token types or pooler)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.BERT

    def set_gguf_parameters(self):
        """Set the fixed LayerNorm epsilon (not in the config), then the BERT parameters."""
        self.gguf_writer.add_layer_norm_eps(1e-12)
        logger.info("gguf: layer norm epsilon = 1e-12")
        super().set_gguf_parameters()

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Strip the "distilbert." prefix and drop the MLM head tensors."""
        name, gen = item

        if name.startswith("distilbert."):
            name = name[11:]

        # These layers act as MLM head, so we don't need them
        if name.startswith("vocab_"):
            return None

        return super().filter_tensors((name, gen))


@ModelBase.register("RobertaModel", "RobertaForSequenceClassification")
class RobertaModel(BertModel):
    """Converter for RoBERTa (BPE tokenizer, offset position embeddings)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.BERT

    def __init__(self, *args, **kwargs):
        """Compute the position offset (pad_token_id + 1) and shrink max_position_embeddings."""
        super().__init__(*args, **kwargs)

        # we need the pad_token_id to know how to chop down position_embd matrix
        if (pad_token_id := self.hparams.get("pad_token_id")) is not None:
            self._position_offset = 1 + pad_token_id
            if "max_position_embeddings" in self.hparams:
                self.hparams["max_position_embeddings"] -= self._position_offset
        else:
            self._position_offset = None

    def set_vocab(self):
        """Support BPE tokenizers for roberta models"""
        bpe_tok_path = self.dir_model / "tokenizer.json"
        if bpe_tok_path.exists():
            self._set_vocab_gpt2()

            # we need this to validate the size of the token_type embeddings
            # though currently we are passing all zeros to the token_type embeddings
            # "Sequence A" or "Sequence B"
            self.gguf_writer.add_token_type_count(self.hparams.get("type_vocab_size", 1))

        else:
            return super().set_vocab()

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Strip the "roberta." prefix used by some rerankers."""
        name, gen = item

        # if name starts with "roberta.", remove the prefix
        # e.g. https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/main
        if name.startswith("roberta."):
            name = name[8:]

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Chop the unusable leading rows off the position embedding matrix."""
        # position embeddings start at pad_token_id + 1, so just chop down the weight tensor
        if name == "embeddings.position_embeddings.weight":
            if self._position_offset is not None:
                data_torch = data_torch[self._position_offset:,:]

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("NomicBertModel")
class NomicBertModel(BertModel):
    """Converter for Nomic BERT embedding models (RoPE, SwiGLU/GELU, optional MoE).

    The constructor asserts the exact architecture the runtime graph supports, so any
    unexpected config fails early.
    """
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.BERT

    def __init__(self, dir_model: Path, ftype: gguf.LlamaFileType, fname_out: Path, **kwargs: Any):
        """Pick the (MoE or dense) architecture before init and validate the config."""
        # The architecture depends on the config (MoE or not), so the config is read before
        # ModelBase.__init__ creates the GGUF writer.
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            hparams = ModelBase.load_hparams(dir_model, False)

        self.is_moe = bool(hparams.get("moe_every_n_layers"))
        self.model_arch = gguf.MODEL_ARCH.NOMIC_BERT_MOE if self.is_moe else gguf.MODEL_ARCH.NOMIC_BERT

        super().__init__(dir_model, ftype, fname_out, hparams=hparams, **kwargs)

        self._tokenizer_is_xlmroberta = self._is_tokenizer_xlmroberta()
        if self._tokenizer_is_xlmroberta:
            self._xlmroberta_tokenizer_init()

        # The config's n_positions is inflated by RoPE scaling; use the length the model was
        # trained for (2048 for v1/v1.5, 512 for v2-MoE).
        npos, mtp = self.hparams["n_positions"], self.hparams.get("max_trained_positions", 2048)
        if npos == 8192 and mtp == 2048:
            self.hparams["n_positions"] = 2048  # nomic-embed-text v1 and v1.5 are trained for 2048 tokens.
        elif npos == 2048 and mtp == 2048:
            self.hparams["n_positions"] = 512   # nomic-embed-text-v2-moe is trained for 512 tokens.
        else:
            raise ValueError(f"unrecognized parameters: n_positions={npos}, max_trained_positions={mtp}")

        # Only the activation used by the runtime graph is accepted.
        # MoE (v2) models use gelu, the dense v1/v1.5 models use swiglu.
        assert self.hparams["activation_function"] == ("gelu" if self.is_moe else "swiglu")

        # the HF "causal" flag is ignored upstream.
        # this doesn't do anything in the HF version
        assert self.hparams["causal"] is False
        # no bias tensors unless MoE
        assert self.hparams["qkv_proj_bias"] == self.is_moe
        assert self.hparams["mlp_fc1_bias"]  == self.is_moe
        assert self.hparams["mlp_fc2_bias"]  == self.is_moe

        # norm at end of layer
        assert self.hparams["prenorm"] is False
        # standard RoPE
        assert self.hparams["rotary_emb_fraction"] == 1.0
        assert self.hparams["rotary_emb_interleaved"] is False
        assert self.hparams["rotary_emb_scale_base"] is None

    def set_vocab(self) -> None:
        """Use the XLM-RoBERTa Unigram loader if the tokenizer is Unigram, else the BERT one."""
        if self._tokenizer_is_xlmroberta:
            return self._xlmroberta_set_vocab()
        return super().set_vocab()

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Drop the experts' bias tensors (unused)."""
        name, gen = item

        # If the tensor is an experts bias tensor, skip it.
        if "mlp.experts.bias" in name:
            return None

        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: torch.Tensor, name: str, bid: int | None) -> Iterable[tuple[str, torch.Tensor]]:
        """Reshape MoE expert weights to 3-D (and transpose w2) with a ".weight" suffix."""
        n_experts = self.find_hparam(["num_local_experts", "num_experts"])
        if "mlp.experts.mlp.w1" in name:
            data_torch = data_torch.view(n_experts, self.hparams["n_inner"], self.hparams["n_embd"])
            name += ".weight"

        if "mlp.experts.mlp.w2" in name:
            data_torch = data_torch.view(n_experts, self.hparams["n_inner"], self.hparams["n_embd"])
            data_torch = data_torch.transpose(1, 2)
            name += ".weight"

        yield from super().modify_tensors(data_torch, name, bid)

    def set_gguf_parameters(self):
        """Add MoE interleave step and top-k for the MoE variant."""
        super().set_gguf_parameters()
        if self.is_moe:
            self.gguf_writer.add_moe_every_n_layers(self.hparams["moe_every_n_layers"])
            self.gguf_writer.add_expert_used_count(self.hparams["moe_top_k"])

    def _is_tokenizer_xlmroberta(self) -> bool:
        """True for Unigram (XLM-R) tokenizers, False for WordPiece; error otherwise."""
        with open(self.dir_model / "tokenizer.json") as f:
            tokenizer_json = json.load(f)
        toktyp = tokenizer_json["model"]["type"]
        if toktyp == "Unigram":
            return True
        if toktyp == "WordPiece":
            return False
        raise ValueError(f"unknown tokenizer: {toktyp}")


@ModelBase.register("NeoBERT", "NeoBERTLMHead", "NeoBERTForSequenceClassification")
class NeoBert(BertModel):
    """Converter for NeoBERT (modern BERT-style encoder with RoPE and RMSNorm)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.NEO_BERT

    def set_gguf_parameters(self):
        """Write NeoBERT's FFN size, RoPE base, RMS epsilon and CLS pooling."""
        super().set_gguf_parameters()

        # SwiGLU halves-ish the hidden width, hence the 2/3 factor.
        # NeoBERT uses 2/3 of the intermediate size as feed forward length
        self.gguf_writer.add_feed_forward_length(int(2 * self.hparams["intermediate_size"] / 3))
        self.gguf_writer.add_rope_freq_base(10000.0)  # default value for NeoBERT
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)

        f_rms_eps = self.hparams.get("norm_eps", 1e-6)  # default value for NeoBERT
        self.gguf_writer.add_layer_norm_rms_eps(f_rms_eps)
        logger.info(f"gguf: rms norm epsilon = {f_rms_eps}")

        self.gguf_writer.add_pooling_type(gguf.PoolingType.CLS) # https://huggingface.co/chandar-lab/NeoBERT#how-to-use

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Drop the LM head decoder and strip the "model." prefix."""
        name, gen = item

        if name.startswith("decoder."):
            return None

        if name.startswith("model."):
            name = name[6:]

        return super().filter_tensors((name, gen))


@ModelBase.register("EuroBertModel", "JinaEmbeddingsV5Model")
class EuroBertModel(TextModel):
    """Converter for EuroBERT / Jina Embeddings v5 (bidirectional Llama-like encoder)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.EUROBERT

    def set_vocab(self):
        """BPE vocab without an automatic BOS token."""
        self.gguf_writer.add_add_bos_token(False)
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        """Mark attention as non-causal, disable RoPE scaling and set the pooling type."""
        super().set_gguf_parameters()

        # EuroBert is bidirectional (encoder)
        self.gguf_writer.add_causal_attention(False)

        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)

        self._try_set_pooling_type()

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Strip the "model." prefix."""
        name, gen = item

        if name.startswith("model."):
            name = name[6:]

        return super().filter_tensors((name, gen))


@ModelBase.register("XLMRobertaModel", "XLMRobertaForSequenceClassification")
class XLMRobertaModel(BertModel):
    """Converter for XLM-RoBERTa, including Jina-embeddings-v3 with task LoRAs.

    For Jina v3 each task LoRA is written to its own adapter GGUF next to the main file.
    """
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.BERT
    # Class-level state: one GGUFWriter per task LoRA (name -> writer) and their names.
    _lora_files = {}
    _lora_names = []

    def __init__(self, dir_model: Path, ftype: gguf.LlamaFileType, fname_out: Path, **kwargs: Any):
        """Detect Jina v3 LoRA adaptations (switching arch) and init the position offset."""
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            hparams = ModelBase.load_hparams(dir_model, False)

        if lora_names := hparams.get("lora_adaptations"):
            self._lora_names = lora_names
            self.model_arch = gguf.MODEL_ARCH.JINA_BERT_V3

        super().__init__(dir_model, ftype, fname_out, hparams=hparams, **kwargs)
        self._xlmroberta_tokenizer_init()

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        """Create one adapter GGUF writer per LoRA task, then defer to the base implementation."""
        if self._lora_names:
            for name in self._lora_names:
                fname = self.add_prefix_to_filename(self.fname_out, f"lora-{name}-")
                self._lora_files[name] = gguf.GGUFWriter(fname, arch=gguf.MODEL_ARCH_NAMES[self.model_arch], endianess=self.endianess, use_temp_file=self.use_temp_file, dry_run=self.dry_run)

        return super().generate_extra_tensors()

    def set_type(self):
        """Mark the LoRA files as adapters of type "lora", then the main file as a model."""
        for lora_writer in self._lora_files.values():
            lora_writer.add_type(gguf.GGUFType.ADAPTER)
            lora_writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
        super().set_type()

    def set_vocab(self):
        """Write the XLM-RoBERTa Unigram vocab."""
        self._xlmroberta_set_vocab()

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Strip the "roberta." prefix and Jina v3's ".parametrizations." wrapper names."""
        name, gen = item

        # if name starts with "roberta.", remove the prefix
        # e.g. https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/main
        if name.startswith("roberta."):
            name = name[8:]

        # jina-embeddings-v3
        if ".parametrizations." in name:
            name = name.replace(".parametrizations.", ".")
            if name.endswith(".original"):
                name = name[:-9]

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Trim position embeddings and split stacked LoRA A/B tensors into per-task files."""
        # position embeddings start at pad_token_id + 1, so just chop down the weight tensor
        if name == "embeddings.position_embeddings.weight":
            if self._position_offset is not None:
                data_torch = data_torch[self._position_offset:,:]

        # Jina v3 stacks all task LoRAs in one tensor [n_loras, ...]; each slice goes to the
        # corresponding adapter writer instead of the main file.
        if name.endswith(".0.lora_A") or name.endswith(".0.lora_B"):
            if name.startswith("pooler.dense"):
                return

            num_loras = data_torch.size(0)
            assert num_loras == len(self._lora_names)

            # Split out each LoRA in their own GGUF
            for i, lora_writer in enumerate(self._lora_files.values()):
                new_name = self.map_tensor_name(name[:-9]) + name[-7:].lower()
                data = data_torch[i, :, :]
                # Embedding-like LoRA tensors have swapped roles/orientation compared to linear ones.
                # Transpose/flip token_embd/types into correct shape
                if new_name == "token_embd.weight.lora_b":
                    data = data.T
                elif new_name.startswith("token_types.weight."):
                    new_name = new_name[:-1] + ("a" if new_name[-1:] == "b" else "b")
                lora_writer.add_tensor(new_name, data.float().numpy(), raw_dtype=gguf.GGMLQuantizationType.F32)

            return

        yield from super().modify_tensors(data_torch, name, bid)

    def set_gguf_parameters(self):
        """Write base parameters, then per-LoRA alpha, task name and prompt prefix metadata."""
        super().set_gguf_parameters()

        # jina-embeddings-v3
        lora_alpha = self.hparams.get("lora_alpha")
        if lora_prompt_prefixes := self.hparams.get("task_instructions"):
            assert self._lora_files and all(lora_name in lora_prompt_prefixes for lora_name in self._lora_files.keys())
        for lora_name, lora_writer in self._lora_files.items():
            lora_writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, lora_alpha if lora_alpha is not None else 1.0)
            lora_writer.add_string(gguf.Keys.Adapter.LORA_TASK_NAME, lora_name)
            if lora_prompt_prefixes:
                lora_writer.add_string(gguf.Keys.Adapter.LORA_PROMPT_PREFIX, lora_prompt_prefixes[lora_name])

    def write(self):
        """Write the main GGUF, then finish every task-LoRA GGUF file."""
        super().write()
        for lora_writer in self._lora_files.values():
            lora_writer.write_header_to_file()
            lora_writer.write_kv_data_to_file()
            lora_writer.write_tensors_to_file(progress=True)
            lora_writer.close()


@ModelBase.register("JinaBertModel", "JinaBertForMaskedLM")
class JinaBertV2Model(BertModel):
    """Converter for Jina BERT v2 (ALiBi) with either a BERT or RoBERTa tokenizer."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.JINA_BERT_V2

    def set_vocab(self):
        """Choose the vocab loader from tokenizer_config.json's tokenizer_class."""
        tokenizer_class = 'BertTokenizer'
        with open(self.dir_model / "tokenizer_config.json", "r", encoding="utf-8") as f:
            tokenizer_class = json.load(f)['tokenizer_class']

        if tokenizer_class == 'BertTokenizer':
            super().set_vocab()
        elif tokenizer_class == 'RobertaTokenizer':
            self._set_vocab_gpt2()
            self.gguf_writer.add_token_type_count(2)
        else:
            raise NotImplementedError(f'Tokenizer {tokenizer_class} is not supported for JinaBertModel')


@ModelBase.register("ModernBertModel", "ModernBertForMaskedLM", "ModernBertForSequenceClassification")
class ModernBertModel(BertModel):
    """Converter for ModernBERT (alternating local/global attention, RoPE)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.MODERN_BERT

    def set_vocab(self):
        """BPE vocab with BOS, EOS and SEP added automatically."""
        self.gguf_writer.add_add_bos_token(True)
        self.gguf_writer.add_add_eos_token(True)
        self.gguf_writer.add_add_sep_token(True)
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        """Write the local-attention window and its global-layer pattern."""
        super().set_gguf_parameters()
        self.gguf_writer.add_sliding_window(self.hparams["local_attention"])
        if (sliding_window_pattern := self.hparams.get("global_attn_every_n_layers")) is not None:
            self.gguf_writer.add_sliding_window_pattern(sliding_window_pattern)
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Strip the "model." prefix."""
        name, gen = item

        if name.startswith("model."):
            name = name[6:]

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Rename the classification projection to the name the tensor map expects."""
        if self.cls_out_labels:
            # For BertForSequenceClassification (direct projection layer)
            if name == "classifier.weight":
                name = "classifier.out_proj.weight"

            if name == "classifier.bias":
                name = "classifier.out_proj.bias"

        yield from super().modify_tensors(data_torch, name, bid)
