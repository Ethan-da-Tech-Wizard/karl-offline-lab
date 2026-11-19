#!/usr/bin/env python3
# ==============================================================================
# Program: auto_train_lora.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Automated command-line interface for QLoRA fine-tuning.
#          Loads datasets, configures 4-bit quantization, executes training
#          epochs, and saves adapter weights.
# ==============================================================================

"""Train Karl's code LoRA adapter from curated trace datasets."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable


# Repo root; all default data paths hang off it so the tool works from any cwd.
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "training" / "code" / "merged_sft.jsonl"
DEFAULT_ADAPTER_DIR = ROOT / "data" / "adapters" / "karl_code_lora"
HF_MODELS_DIR = ROOT / "data" / "hf_models"
# Attention and MLP projection layer names shared by Llama/Qwen-style models; LoRA adapters are
# attached to whichever of these exist.
PROJECTION_CANDIDATES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def missing_packages() -> list[str]:
    """Return names of required training packages that are not importable. Uses find_spec so nothing heavy is actually imported just to check."""
    # Step 1: Validate input parameters, operational preconditions, and state
    required = ["peft", "trl", "transformers", "datasets", "torch"]
    missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]
    if importlib.util.find_spec("bitsandbytes") is None:
        missing.append("bitsandbytes")
    return missing


def print_install_instructions(missing: list[str]) -> None:
    """Tell the user (stdout) which training dependencies are missing and where to place base model weights."""
    # Step 1: Validate input parameters, operational preconditions, and state
    print("Missing training dependencies:", ", ".join(missing))
    print()
    print("Install the training stack inside Karl's virtualenv, for example:")
    print("  pip install peft trl transformers datasets bitsandbytes accelerate")
    print()
    print("Place HuggingFace base model weights under data/hf_models/ or pass:")
    print("  --base-model data/hf_models/<model-folder>")


def resolve_base_model(explicit: str | None) -> Path:
    """Pick the HF base model directory: the explicit path if given (must exist), otherwise the alphabetically first folder under data/hf_models. Raises FileNotFoundError if none."""
    # Step 1: Validate input parameters, operational preconditions, and state
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Base model path does not exist: {path}")
        return path

    candidates = [path for path in HF_MODELS_DIR.iterdir() if path.is_dir()] if HF_MODELS_DIR.exists() else []
    if not candidates:
        raise FileNotFoundError(
            "No HuggingFace model folders found in data/hf_models/. "
            "Download a Llama/Qwen-compatible base model or pass --base-model."
        )
    return sorted(candidates)[0]


def validate_dataset(path: Path) -> int:
    """Check the SFT JSONL exists, is non-empty, and every non-blank line has a 'messages' list of at least 2 entries. Returns the row count; raises on the first bad line so bad data fails before a long GPU run."""
    # Step 1: Validate input parameters, operational preconditions, and state
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run tools/curate_code_datasets.py "
            "(after tools/generate_code_sft_dataset.py) first."
        )
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            # Iterate over items, evaluate loop invariant, and update progress
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            messages = obj.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{path}:{line_no}: expected a HuggingFace 'messages' list")
            count += 1
    if count == 0:
        raise ValueError(f"Dataset is empty: {path}")
    return count


def detect_target_modules(model, candidates: Iterable[str] = PROJECTION_CANDIDATES) -> list[str]:
    """Detect LoRA projection targets present in the loaded model architecture."""
    # Step 1: Validate input parameters, operational preconditions, and state
    candidate_set = set(candidates)
    found: set[str] = set()
    for name, _module in model.named_modules():
        # Iterate over items, evaluate loop invariant, and update progress
        suffix = name.rsplit(".", 1)[-1]
        if suffix in candidate_set:
            found.add(suffix)
    # Preserve candidate order (not module-walk order) so the result is deterministic.
    ordered = [name for name in candidates if name in found]
    if not ordered:
        # Fallback when no known projection names were found: the minimal attention pair, which works
        # for most transformer models.
        ordered = ["q_proj", "v_proj"]
    return ordered


def build_sft_config_kwargs(args: argparse.Namespace, checkpoint_dir: Path) -> dict:
    """Build TRL SFTConfig kwargs in one place so trainer settings are testable.

    Returns the full set of settings we *want*, independent of which trl
    version happens to be installed -- see filter_kwargs_for_class() for the
    compatibility layer that adapts this to whatever's actually available.
    """
    # Step 1: Validate input parameters, operational preconditions, and state
    return {
        "output_dir": str(checkpoint_dir),
        "dataset_text_field": "text",
        "max_length": args.max_length,
        # Pack several short samples into one sequence for GPU efficiency.
        "packing": True,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": args.warmup_ratio,
        "num_train_epochs": args.epochs,
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        # No third-party telemetry (wandb etc.); training stays fully offline.
        "report_to": "none",
        # fp16 (not bf16) because the target consumer GPUs may not support bf16.
        "fp16": True,
        # Trade compute for memory so a 4-bit model fits on small GPUs.
        "gradient_checkpointing": True,
    }


def filter_kwargs_for_class(kwargs: dict, cls) -> dict:
    """Drop kwargs that `cls.__init__` doesn't accept.

    trl/transformers occasionally rename or drop TrainingArguments/SFTConfig
    fields between releases (e.g. warmup_ratio has moved around), and
    requirements.txt intentionally doesn't pin trl/transformers/peft/datasets
    versions. Rather than hard-crash on an unexpected-keyword TypeError for
    whichever version happens to be installed, drop anything unsupported and
    let training proceed with the rest -- logging what got dropped so it's
    not silently lost.
    """
    # Step 1: Validate input parameters, operational preconditions, and state
    params = inspect.signature(cls.__init__).parameters
    adapted = dict(kwargs)
    # Newer Transformers deprecates gradient_checkpointing in favor of
    # activation_checkpointing. Prefer the new explicit parameter, even if
    # the constructor still accepts the deprecated alias. Keep the old name
    # for older constructors and preserve the requested value in either case.
    for old, new in (
        ("gradient_checkpointing", "activation_checkpointing"),
        ("gradient_checkpointing_kwargs", "activation_checkpointing_kwargs"),
    ):
        # Iterate over items, evaluate loop invariant, and update progress
        if old in adapted and new in params:
            adapted.setdefault(new, adapted[old])
            del adapted[old]
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        # Constructor takes **kwargs: pass everything through.
        return adapted  # __init__ accepts **kwargs -- nothing to filter
    supported = {k: v for k, v in adapted.items() if k in params}
    dropped = sorted(set(adapted) - set(supported))
    if dropped:
        print(
            f"WARNING: {cls.__name__} in the installed trl version doesn't "
            f"accept: {', '.join(dropped)} -- proceeding without them.",
            file=sys.stderr,
        )
    return supported


def train(args: argparse.Namespace) -> Path:
    """Run QLoRA (4-bit NF4) supervised fine-tuning on the curated dataset and save the adapter, tokenizer and log history to args.output_dir. Heavy imports are done here so --help and the pure helpers stay cheap. Requires CUDA. Returns the adapter directory."""
    # Step 1: Validate input parameters, operational preconditions, and state
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    dataset_path = Path(args.dataset).expanduser()
    validate_dataset(dataset_path)
    base_model = resolve_base_model(args.base_model)
    adapter_dir = Path(args.output_dir).expanduser()
    adapter_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(base_model), trust_remote_code=True)
    # Many causal LMs have no pad token; reuse EOS so batching/packing works.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    raw_dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    # Render chat messages with the model's own chat template when available, else a simple 'role:
    # content' transcript.
    def format_row(row: dict) -> dict:
        """Render one dataset row's messages into a single training text, using the tokenizer's chat template when it has one and a plain 'role: content' transcript otherwise."""
        # Step 1: Validate input parameters, operational preconditions, and state
        messages = row["messages"]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            text = "\n".join(f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages)
        return {"text": text}

    dataset = raw_dataset.map(format_row, remove_columns=raw_dataset.column_names)

    # Fail early with a clear message: bitsandbytes 4-bit needs CUDA.
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA NF4 training requires a CUDA GPU with bitsandbytes support.")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    # Casts norms and enables input grads so LoRA on a quantized model is trainable.
    model = prepare_model_for_kbit_training(model)

    if args.target_modules == "auto":
        target_modules = detect_target_modules(model)
    else:
        target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=target_modules,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    # KV cache is incompatible with gradient checkpointing during training.
    model.config.use_cache = False

    checkpoint_dir = adapter_dir / "checkpoints"
    sft_kwargs = filter_kwargs_for_class(build_sft_config_kwargs(args, checkpoint_dir), SFTConfig)
    training_args = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )
    trainer.train()

    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    with (adapter_dir / "train_history.json").open("w", encoding="utf-8") as fh:
        json.dump(trainer.state.log_history, fh, indent=2)

    # Intermediate per-epoch checkpoints are large; delete them once the final adapter is saved.
    if args.clean_checkpoints and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    return adapter_dir


def main() -> int:
    """CLI entry: parse hyper-parameters, check dependencies, then call train(). See the code below for exit codes."""
    # Step 1: Validate input parameters, operational preconditions, and state
    parser = argparse.ArgumentParser(description="Train Karl's curated code QLoRA adapter.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Curated SFT JSONL dataset")
    parser.add_argument("--base-model", default=None, help="HuggingFace base model directory")
    parser.add_argument("--output-dir", default=str(DEFAULT_ADAPTER_DIR), help="Adapter output directory")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument(
        "--target-modules",
        default="auto",
        help="Comma-separated LoRA target modules, or 'auto' to detect projections from the model",
    )
    parser.add_argument("--keep-checkpoints", dest="clean_checkpoints", action="store_false")
    parser.set_defaults(clean_checkpoints=True)
    args = parser.parse_args()

    missing = missing_packages()
    if missing:
        print_install_instructions(missing)
        return 1

    dataset_path = Path(args.dataset).expanduser()
    try:
        # Execute guarded operations and establish recovery boundaries
        example_count = validate_dataset(dataset_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Dataset validation failed: {exc}", file=sys.stderr)
        return 1

    print("Training configuration:")
    print(f"  Dataset:           {dataset_path} ({example_count} validated example(s))")
    print(f"  Epochs:            {args.epochs}")
    print(f"  Batch size:        {args.batch_size} (grad accum {args.grad_accum})")
    print(f"  Target modules:    {args.target_modules}")
    print("  LR schedule:       cosine")
    print(f"  Packing:           enabled ({args.max_length} token blocks)")
    print(f"  LoRA rank/alpha:   {args.rank}/{args.alpha} (dropout {args.dropout})")
    print(f"  Output dir:        {args.output_dir}")
    print()

    try:
        # Execute guarded operations and establish recovery boundaries
        adapter_dir = train(args)
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1

    print(f"Saved Karl code LoRA adapter to {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
