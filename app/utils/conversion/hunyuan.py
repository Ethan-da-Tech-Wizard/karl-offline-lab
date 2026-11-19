# ==============================================================================
# Program: hunyuan.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for HUNYUAN.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

from __future__ import annotations

import json

from pathlib import Path
from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, TextModel, gguf, logger

# QwenModel provides the tiktoken-rank helpers (byte->string, bpe) used to rebuild
# BPE merges from HunYuan's mergeable_ranks.
from .qwen import QwenModel


@ModelBase.register("HunYuanMoEV1ForCausalLM")
class HunYuanMoEModel(TextModel):
    """Converter for Tencent Hunyuan-A13B (MoE) with NTK-aware (alpha) RoPE scaling."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.HUNYUAN_MOE

    def set_vocab(self):
        """Rebuild a GPT-2 style vocab (tokens, types, merges) from the tiktoken-style tokenizer.

        Hunyuan ships mergeable ranks rather than a merges file, so merges are
        reverse-engineered by re-running BPE per token.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)

        # 1. Get the pre-tokenizer identifier hash
        tokpre = self.get_vocab_base_pre(tokenizer)

        # A token with 2 parts after BPE-ing up to its own rank is exactly one merge rule.
        # 2. Reverse-engineer the merges list from mergeable_ranks
        merges = []
        vocab = {}
        mergeable_ranks = tokenizer.mergeable_ranks  # ty: ignore[unresolved-attribute]
        for token, rank in mergeable_ranks.items():
            # Iterate over items, evaluate loop invariant, and update progress
            vocab[QwenModel.token_bytes_to_string(token)] = rank
            if len(token) == 1:
                continue
            merged = QwenModel.bpe(mergeable_ranks, token, max_rank=rank)
            if len(merged) == 2: # todo this is an assert in Qwen, why?
                merges.append(' '.join(map(QwenModel.token_bytes_to_string, merged)))

        # Fill every id up to vocab_size; ids the tokenizer lacks become [PAD<i>] (UNUSED),
        # special tokens are CONTROL, others NORMAL.
        # 3. Generate the tokens and toktypes lists
        vocab_size = self.hparams["vocab_size"]
        assert tokenizer.vocab_size == vocab_size  # ty: ignore[unresolved-attribute]
        special_tokens = tokenizer.special_tokens  # ty: ignore[unresolved-attribute]
        reverse_vocab = {id_ : encoded_tok for encoded_tok, id_ in {**vocab, **special_tokens}.items()}
        tokens: list[str] = []
        toktypes: list[int] = []
        for i in range(vocab_size):
            # Iterate over items, evaluate loop invariant, and update progress
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
            else:
                token = reverse_vocab[i]
                tokens.append(token)
                if i in special_tokens.values():
                    toktypes.append(gguf.TokenType.CONTROL)
                else:
                    toktypes.append(gguf.TokenType.NORMAL)

        # 4. Write all vocab-related fields to the GGUF writer
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_token_merges(merges)

        # 5. Add special tokens and chat templates
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=False)
        special_vocab.add_to_gguf(self.gguf_writer)
        # (see note) config.json carries a wrong BOS id for this model.
        # FIX for BOS token: Overwrite incorrect id read from config.json
        self.gguf_writer.add_bos_token_id(127959) # <|bos|>

    def set_gguf_parameters(self):
        """Write MoE sizing (requiring identical values for every layer) and RoPE scaling."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()
        hparams = self.hparams

        self.gguf_writer.add_expert_shared_feed_forward_length(hparams["intermediate_size"])

        # The config stores these as per-layer lists; llama.cpp wants one scalar, so every
        # layer must agree (asserted).
        moe_intermediate_size = hparams["moe_intermediate_size"]
        assert all(n == moe_intermediate_size[0] for n in moe_intermediate_size)
        self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size[0])

        moe_topk = hparams["moe_topk"]
        assert all(topk == moe_topk[0] for topk in moe_topk)
        self.gguf_writer.add_expert_used_count(moe_topk[0])

        moe_shared_expert = hparams["num_shared_expert"]
        assert all(n == moe_shared_expert[0] for n in moe_shared_expert)
        self.gguf_writer.add_expert_shared_count(moe_shared_expert[0])

        # (see below) NTK-aware "alpha" scaling is converted into a larger RoPE base.
        # Rope
        if self.rope_parameters.get("rope_type") == "dynamic":
            # HunYuan uses NTK Aware Alpha based scaling. Original implementation: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
            # 1000 corresponds to a usable context length of 256k (https://github.com/Tencent-Hunyuan/Hunyuan-A13B/blob/main/report/Hunyuan_A13B_Technical_Report.pdf)
            alpha = self.rope_parameters.get("alpha", 1000)
            base = self.rope_parameters.get("rope_theta", 10000.0)
            dim = (hparams["hidden_size"] // hparams["num_attention_heads"]) # 128
            # NTK-aware formula: base * alpha^(dim / (dim - 2)).
            scaled_base = base * (alpha ** (dim / (dim - 2))) # 10000 * (1000 ** (128 / 126)) = 11158839.9251
            self.gguf_writer.add_rope_freq_base(scaled_base)
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
            self.gguf_writer.add_rope_scaling_factor(1)
            # There is no consistent way to calculate ctx from alpha, and the config is incorrectly set to 32k
            self.gguf_writer.add_rope_scaling_orig_ctx_len(256 * 1024) # 256k context length
            self.gguf_writer.add_context_length(256 * 1024) # 256k context length

            # if any of our assumptions about the values are wrong, something has changed and this may need to be updated
            # Guard: the hard-coded 256k context is only valid for the known alpha/base/dim.
            assert alpha == 1000 and base == 10000.0 and dim == 128 and self.hparams["max_position_embeddings"] in [32 * 1024, 256 * 1024] , \
                "HunYuan dynamic RoPE scaling assumptions changed, please update the logic or context length manually"

    # Per-layer buffers for expert tensors awaiting merge.
    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Drop a tied lm_head and merge routed experts into stacked tensors.

        Generator: yields nothing while a layer's experts are still being collected.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        # With tied embeddings the output matrix duplicates token_embd, so skip it.
        if name == "lm_head.weight":
            if self.hparams.get("tie_word_embeddings", False):
                logger.info("Skipping tied output layer 'lm_head.weight'")
                return

        # Routed experts: buffer per layer and merge when the layer is complete.
        if name.find("mlp.experts") != -1:
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None

            # Experts arrive as separate per-expert tensors; buffer them per layer
            # (block id) until the whole set for that layer has been seen.
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            # Layer complete once all {n_experts} experts have delivered their 3 weights
            # (gate/up/down), i.e. n_experts * 3 buffered tensors.
            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    # Iterate over items, evaluate loop invariant, and update progress
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        # Iterate over items, evaluate loop invariant, and update progress
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    # Stack the per-expert 2D matrices into [n_experts, ...].
                    data_torch = torch.stack(datas, dim=0)
                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

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
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@ModelBase.register("HunYuanDenseV1ForCausalLM")
class HunYuanModel(TextModel):
    """Converter for Tencent Hunyuan dense models (also the base for Hunyuan-VL text)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.HUNYUAN_DENSE

    def _get_eod_token_id(self) -> int | None:
        """Get the actual end-of-generation token from config (eod_token_id)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        return self.hparams.get("eod_token_id")

    def _get_eot_token_id(self) -> int | None:
        """Get the end-of-turn token from generation_config.json.
        This is the first entry in eos_token_id when it's a list."""
        # Step 1: Validate input parameters, operational preconditions, and state
        gen_cfg_path = self.dir_model / "generation_config.json"
        if gen_cfg_path.is_file():
            with open(gen_cfg_path, encoding="utf-8") as f:
                gen_cfg = json.load(f)
            eos = gen_cfg.get("eos_token_id")
            if isinstance(eos, list) and len(eos) >= 2:
                return eos[0]
        return None

    # Overwrite eos/eot ids that upstream configs get wrong, using eod_token_id and
    # generation_config.json.
    def _fix_special_tokens(self):
        """Fix EOS/EOT tokens that are incorrect in upstream configs."""
        # Step 1: Validate input parameters, operational preconditions, and state
        eod_id = self._get_eod_token_id()
        if eod_id is not None:
            self.gguf_writer.add_eos_token_id(eod_id)
        eot_id = self._get_eot_token_id()
        if eot_id is not None:
            self.gguf_writer.add_eot_token_id(eot_id)

    def set_vocab(self):
        """Use tokenizer.json when present, else rebuild the vocab from tiktoken ranks."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if (self.dir_model / "tokenizer.json").is_file():
            tokens, toktypes, tokpre = self.get_vocab_base()
            self.gguf_writer.add_tokenizer_model("gpt2")
            self.gguf_writer.add_tokenizer_pre(tokpre)
            self.gguf_writer.add_token_list(tokens)
            self.gguf_writer.add_token_types(toktypes)

            # (see note) a negative pad id would be emitted as an invalid special token.
            # Some HunYuanVL variants (e.g. OCR-style configs) have pad_token_id=-1;
            # guard SpecialVocab so it doesn't try to emit an invalid pad id.
            # None means "use SpecialVocab's defaults"; otherwise restrict to the listed types.
            token_types = None
            if (self.hparams.get("pad_token_id") or 0) < 0:
                token_types = ('bos', 'eos', 'unk', 'sep', 'cls', 'mask')
            special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True, special_token_types=token_types)
            special_vocab.add_to_gguf(self.gguf_writer)
            self._fix_special_tokens()
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)

            # Fallback path for checkpoints without tokenizer.json (tiktoken-style tokenizer).
            # 1. Get the pre-tokenizer identifier hash
            tokpre = self.get_vocab_base_pre(tokenizer)

            # 2. Reverse-engineer the merges list from mergeable_ranks
            merges = []
            vocab = {}
            mergeable_ranks = tokenizer.mergeable_ranks  # ty: ignore[unresolved-attribute]
            for token, rank in mergeable_ranks.items():
                # Iterate over items, evaluate loop invariant, and update progress
                vocab[QwenModel.token_bytes_to_string(token)] = rank
                if len(token) == 1:
                    continue
                merged = QwenModel.bpe(mergeable_ranks, token, max_rank=rank)
                if len(merged) == 2:
                    merges.append(' '.join(map(QwenModel.token_bytes_to_string, merged)))

            # 3. Generate the tokens and toktypes lists
            vocab_size = self.hparams["vocab_size"]
            assert tokenizer.vocab_size == vocab_size  # ty: ignore[unresolved-attribute]
            special_tokens = tokenizer.special_tokens  # ty: ignore[unresolved-attribute]
            reverse_vocab = {id_ : encoded_tok for encoded_tok, id_ in {**vocab, **special_tokens}.items()}
            tokens: list[str] = []
            toktypes: list[int] = []
            for i in range(vocab_size):
                # Iterate over items, evaluate loop invariant, and update progress
                if i not in reverse_vocab:
                    tokens.append(f"[PAD{i}]")
                    toktypes.append(gguf.TokenType.UNUSED)
                else:
                    token = reverse_vocab[i]
                    tokens.append(token)
                    if i in special_tokens.values():
                        toktypes.append(gguf.TokenType.CONTROL)
                    else:
                        toktypes.append(gguf.TokenType.NORMAL)

            # Merges are written explicitly, so SpecialVocab below must not load them again.
            # 4. Write all vocab-related fields to the GGUF writer
            self.gguf_writer.add_tokenizer_model("gpt2")
            self.gguf_writer.add_tokenizer_pre(tokpre)
            self.gguf_writer.add_token_list(tokens)
            self.gguf_writer.add_token_types(toktypes)
            self.gguf_writer.add_token_merges(merges)

            # 5. Add special tokens and chat templates
            special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=False)
            special_vocab.add_to_gguf(self.gguf_writer)
            # FIX for BOS token: Overwrite incorrect id read from config.json
            # Only the 7B dense model has this wrong BOS id.
            if self.hparams['hidden_size'] == 4096:
                self.gguf_writer.add_bos_token_id(127958) # only for 7b dense, fix <|bos|> token
            self._fix_special_tokens()

    def set_gguf_parameters(self):
        """Write base parameters (suppressing bogus single-expert metadata) and RoPE scaling."""
        # Some HunYuanVL variants set num_experts=1 (not real MoE);
        # prevent the parent class from emitting expert_count metadata in that case.
        # Temporarily hide num_experts=1 from the parent so it does not emit an
        # expert_count for a model that is not really MoE; restored below if it was > 1.
        saved_num_experts = self.hparams.pop("num_experts", None)
        super().set_gguf_parameters()
        if saved_num_experts is not None and saved_num_experts > 1:
            self.hparams["num_experts"] = saved_num_experts
        hparams = self.hparams

        # Rope
        # Both variants use NTK-aware alpha scaling, converted into a larger RoPE base.
        if self.rope_parameters.get("rope_type") in ("dynamic", "xdrope"):
            # HunYuan uses NTK Aware Alpha based scaling. Original implementation: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
            # 1000 corresponds to a usable context length of 256k (https://github.com/Tencent-Hunyuan/Hunyuan-A13B/blob/main/report/Hunyuan_A13B_Technical_Report.pdf)
            alpha = self.rope_parameters.get("alpha", 50)
            base = self.rope_parameters.get("rope_theta", 10000.0)
            dim = hparams["head_dim"]
            scaled_base = base * (alpha ** (dim / (dim - 2)))
            self.gguf_writer.add_rope_freq_base(scaled_base)
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
            self.gguf_writer.add_rope_scaling_factor(1)
            if self.rope_parameters.get("rope_type") == "dynamic":
                # There is no consistent way to calculate ctx from alpha, and the config is incorrectly set to 32k
                self.gguf_writer.add_rope_scaling_orig_ctx_len(256 * 1024) # 256k context length
                self.gguf_writer.add_context_length(256 * 1024) # 256k context length

                # if any of our assumptions about the values are wrong, something has changed and this may need to be updated
                assert base == 10000.0 and self.hparams["max_position_embeddings"] in [32 * 1024, 256 * 1024] , \
                    "HunYuan dynamic RoPE scaling assumptions changed, please update the logic or context length manually"

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Drop a tied lm_head; pass other tensors through."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if name == "lm_head.weight":
            if self.hparams.get("tie_word_embeddings", False):
                logger.info("Skipping tied output layer 'lm_head.weight'")
                return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("HunYuanVLForConditionalGeneration")
class HunyuanVLVisionModel(MmprojModel):
    """Converter for the Hunyuan-VL vision tower as an mmproj."""
    def __init__(self, *args, **kwargs):
        """Provide image_size (the config uses max_image_size instead)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        # HunyuanVL uses max_image_size instead of image_size
        if "image_size" not in self.hparams_vision:
            self.hparams_vision["image_size"] = self.hparams_vision.get("max_image_size", 2048)

    def set_gguf_parameters(self):
        """Write projector type, GELU flag, norm epsilon, merge size and pixel limits."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()
        assert self.hparams_vision is not None
        vcfg = self.hparams_vision
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.HUNYUANVL)
        self.gguf_writer.add_vision_use_gelu(True)
        self.gguf_writer.add_vision_attention_layernorm_eps(vcfg.get("rms_norm_eps", 1e-5))
        self.gguf_writer.add_vision_spatial_merge_size(vcfg.get("spatial_merge_size", 2))
        self.gguf_writer.add_vision_min_pixels(int(self.preprocessor_config["min_pixels"]))
        self.gguf_writer.add_vision_max_pixels(int(self.preprocessor_config["max_pixels"]))

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        """Keep only the ViT tensors (prefix "vit.")."""
        # Step 1: Validate input parameters, operational preconditions, and state
        name, gen = item

        if not name.startswith("vit."):
            return None

        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Strip the CLS row from position embeddings."""
        # strip CLS token (row 0) from position embeddings so resize_position_embeddings works
        if "position_embedding" in name:
            data_torch = data_torch[1:]  # [n_patches+1, n_embd] -> [n_patches, n_embd]
        yield from super().modify_tensors(data_torch, name, bid)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        """Store the ViT->LLM projector weights as F16/F32 (avoids BF16 problems on Metal)."""
        # force conv weights to F32 or F16 to avoid BF16 IM2COL issues on Metal
        # HunyuanVL emit the ViT -> LLM projection as mm.0/mm.2.
        if ("mm.0." in new_name or "mm.2." in new_name) and new_name.endswith(".weight"):
            return gguf.GGMLQuantizationType.F16 if self.ftype == gguf.LlamaFileType.MOSTLY_F16 else gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)


@ModelBase.register("HunYuanVLForConditionalGeneration")
class HunyuanVLTextModel(HunYuanModel):
    """Converter for the Hunyuan-VL language model (XD-RoPE)."""
    # GGUF architecture id written into the output file.
    model_arch = gguf.MODEL_ARCH.HUNYUAN_VL

    def __init__(self, dir_model: Path, *args, **kwargs):
        """No extra initialisation; accepts the model directory explicitly."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__(dir_model, *args, **kwargs)

    def set_gguf_parameters(self):
        """Write XD-RoPE metadata when the config uses it."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().set_gguf_parameters()

        # XD-RoPE metadata for the HunyuanVL;
        if self.rope_parameters.get("rope_type") != "xdrope":
            return

        self.gguf_writer.add_rope_freq_base(float(self.rope_parameters["rope_theta"]))
        self.gguf_writer.add_rope_scaling_alpha(float(self.rope_parameters["alpha"]))
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
        self.gguf_writer.add_rope_scaling_factor(float(self.rope_parameters.get("factor", 1)))

        ctx_len = int(self.hparams["max_position_embeddings"])
        self.gguf_writer.add_rope_scaling_orig_ctx_len(ctx_len)
        self.gguf_writer.add_context_length(ctx_len)

        self.gguf_writer.add_rope_dimension_sections(list(self.rope_parameters["xdrope_section"]))
