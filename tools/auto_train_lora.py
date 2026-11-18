#!/usr/bin/env python3
# Program: karl-offline-lab - auto_train_lora.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Command-line tool that validates a curated SFT dataset and trains a QLoRA (4-bit) code adapter on a local HuggingFace base model, saving the adapter and training history.
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


# Repository root, two levels above this file.
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "training" / "code" / "merged_sft.jsonl"
DEFAULT_ADAPTER_DIR = ROOT / "data" / "adapters" / "karl_code_lora"
HF_MODELS_DIR = ROOT / "data" / "hf_models"
# Names of attention and MLP projection layers that LoRA can target.
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
    # Core packages needed for training.
    """Return the names of required training packages that are not installed."""
    required = ["peft", "trl", "transformers", "datasets", "torch"]
    missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]
    # bitsandbytes is checked separately because it provides the 4-bit quantization.
    if importlib.util.find_spec("bitsandbytes") is None:
        missing.append("bitsandbytes")
    return missing


def print_install_instructions(missing: list[str]) -> None:
    """Print which training dependencies are missing and how to install them and where to put base model weights."""
    print("Missing training dependencies:", ", ".join(missing))
    print()
    print("Install the training stack inside Karl's virtualenv, for example:")
    print("  pip install peft trl transformers datasets bitsandbytes accelerate")
    print()
    print("Place HuggingFace base model weights under data/hf_models/ or pass:")
    print("  --base-model data/hf_models/<model-folder>")


def resolve_base_model(explicit: str | None) -> Path:
    """Return the base model folder: the explicit path if given (it must exist), otherwise the first folder, sorted, under the local models directory."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Base model path does not exist: {path}")
        return path

    # Every subfolder of the models directory is a possible base model.
    candidates = [path for path in HF_MODELS_DIR.iterdir() if path.is_dir()] if HF_MODELS_DIR.exists() else []
    if not candidates:
        raise FileNotFoundError(
            "No HuggingFace model folders found in data/hf_models/. "
            "Download a Llama/Qwen-compatible base model or pass --base-model."
        )
    return sorted(candidates)[0]


def validate_dataset(path: Path) -> int:
    """Check that the dataset file exists and every non-empty line is JSON with a "messages" list of at least two entries.
    Returns the number of examples and raises FileNotFoundError or ValueError otherwise.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run tools/curate_code_datasets.py "
            "(after tools/generate_code_sft_dataset.py) first."
        )
    # Count valid examples while checking each line.
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
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
    candidate_set = set(candidates)
    found: set[str] = set()
    for name, _module in model.named_modules():
        suffix = name.rsplit(".", 1)[-1]
        if suffix in candidate_set:
            found.add(suffix)
    ordered = [name for name in candidates if name in found]
    # Fall back to the standard attention query and value projections.
    if not ordered:
        ordered = ["q_proj", "v_proj"]
    return ordered


def build_sft_config_kwargs(args: argparse.Namespace, checkpoint_dir: Path) -> dict:
    """Build TRL SFTConfig kwargs in one place so trainer settings are testable.

    Returns the full set of settings we *want*, independent of which trl
    version happens to be installed -- see filter_kwargs_for_class() for the
    compatibility layer that adapts this to whatever's actually available.
    """
    return {
        "output_dir": str(checkpoint_dir),
        "dataset_text_field": "text",
        "max_length": args.max_length,
        "packing": True,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": args.warmup_ratio,
        "num_train_epochs": args.epochs,
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "report_to": "none",
        "fp16": True,
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
        if old in adapted and new in params:
            adapted.setdefault(new, adapted[old])
            del adapted[old]
    # A **kwargs constructor accepts everything, so no filtering is needed.
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return adapted  # __init__ accepts **kwargs -- nothing to filter
    # Keep only the settings the installed library version accepts.
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
    """Load the dataset and 4-bit base model, attach a LoRA adapter, train with SFTTrainer and save the adapter, tokenizer and history.
    Returns the adapter directory. Requires a CUDA GPU.
    """
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
    # Many models have no padding token, so reuse the end-of-sequence token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Pad on the right, the usual choice for training.
    tokenizer.padding_side = "right"

    raw_dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    def format_row(row: dict) -> dict:
        """Turn one row's chat messages into a single training text, using the tokenizer's chat template when it has one."""
        messages = row["messages"]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            text = "\n".join(f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages)
        return {"text": text}

    # Replace each row with a single formatted "text" field.
    dataset = raw_dataset.map(format_row, remove_columns=raw_dataset.column_names)

    # 4-bit training needs a CUDA GPU, so stop early with a clear message.
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA NF4 training requires a CUDA GPU with bitsandbytes support.")

    # Load the base model in 4-bit NF4 with double quantization to save memory.
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
    # Prepare the quantized model for adapter training.
    model = prepare_model_for_kbit_training(model)

    # Detect target layers from the model or use the comma-separated list given.
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
    # The cache is not used during training and conflicts with gradient checkpointing.
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

    # Save the trained adapter, the tokenizer and the loss history.
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    with (adapter_dir / "train_history.json").open("w", encoding="utf-8") as fh:
        json.dump(trainer.state.log_history, fh, indent=2)

    # Delete intermediate checkpoints unless the user asked to keep them.
    if args.clean_checkpoints and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    return adapter_dir


def main() -> int:
    # Define command-line options for data, model, output and training hyperparameters.
    """Parse command-line options, check dependencies and the dataset, print the configuration, run training and return a process exit code."""
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

    # Stop early with install instructions if training packages are missing.
    missing = missing_packages()
    if missing:
        print_install_instructions(missing)
        return 1

    dataset_path = Path(args.dataset).expanduser()
    try:
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
        adapter_dir = train(args)
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1

    print(f"Saved Karl code LoRA adapter to {adapter_dir}")
    return 0


# Run main() and use its return value as the exit code.
if __name__ == "__main__":
    raise SystemExit(main())
