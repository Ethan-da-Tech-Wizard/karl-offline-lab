#!/usr/bin/env python3
# ==============================================================================
# Program: auto_train.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Autonomous fine-tuning and LoRA model adaptation orchestrator.
#          Loads instruction datasets, configures QLoRA/PEFT hyperparameters,
#          executes gradient accumulation and loss tracking loops, and exports
#          adapter weights for GGUF model conversion.
# ==============================================================================

"""
Karl One-Click Auto-Train Pipeline
==================================
A unified, easy-to-understand end-to-end pipeline that:
1. Generates synthetic coding/math/reasoning tasks for a target topic using the local model.
2. Solves those tasks using the local model.
3. Verifies responses in a secure sandbox, using self-reflection loops to correct errors.
4. Creates a custom SFT dataset from the verified examples.
5. Trains a LoRA adapter on the dataset locally (using SFTTrainer).
6. Converts the trained adapter into GGUF format, ready to load in Karl.
"""

import os
import re
import sys
import json
import uuid
import time
import argparse
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("karl.auto_train")

# adapter_name is used verbatim to build filesystem paths under ADAPTERS_DIR
# (see train_adapter/convert_adapter_to_gguf below). Restrict the charset so
# a caller (e.g. the WebSocket start_auto_train RPC) can't path-traverse out
# of data/adapters/ via "../" sequences.
_SAFE_ADAPTER_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

# Project paths
ROOT_DIR = Path(__file__).resolve().parent
QUEUE_DIR = ROOT_DIR / "data" / "flywheel" / "queue"
EXECUTION_DIR = ROOT_DIR / "data" / "flywheel" / "execution"
TRAINING_DIR = ROOT_DIR / "data" / "training"
ADAPTERS_DIR = ROOT_DIR / "data" / "adapters"
HF_MODELS_DIR = ROOT_DIR / "data" / "hf_models"

# Ensure dirs exist
# Create the data directories at import time so later writes never fail on a fresh checkout.
for d in [QUEUE_DIR, EXECUTION_DIR, TRAINING_DIR, ADAPTERS_DIR, HF_MODELS_DIR]:
    # Iterate over items, evaluate loop invariant, and update progress
    d.mkdir(parents=True, exist_ok=True)

# Add ROOT_DIR to path for internal imports
# This file is run as a standalone subprocess, so the repo root must be put on sys.path to import
# app/core/data packages.
sys.path.insert(0, str(ROOT_DIR))

try:
    # Execute guarded operations and establish recovery boundaries
    # Core imports are wrapped so a broken install gives one clear log line and exit code 1 instead of
    # a traceback.
    from app.engine.model_loader import ModelLoader
    from core.cognitive_parser import parse_thought_stream
    from data.flywheel.executor_sandbox import SafePythonSandbox
    from core.interaction_loop import build_prompt
    from core.security import assert_not_privileged
except ImportError as err:
    logger.error("Failed to import Karl core modules: %s", err)
    sys.exit(1)


def generate_tasks_for_topic(topic: str, count: int, llm) -> list[dict]:
    """Generates synthetic task specs (statements + python verifiers) for a topic using LLM."""
    # Step 1: Validate input parameters, operational preconditions, and state
    logger.info("Generating %d synthetic tasks for topic: '%s'...", count, topic)

    meta_prompt = f"""You are a synthetic dataset generator. Generate a JSON list of exactly {count} distinct, high-quality, and challenging programming exercises or word problems about the topic: "{topic}".

Each task MUST be in this exact JSON format:
{{
  "category": "{topic}",
  "problem_statement": "Description of the problem...",
  "ground_truth_answer": "Expected exact answer or output...",
  "verification_type": "exact_match | unit_test",
  "verification_script": "python code string of a `def verify(response: str) -> bool:` function that returns True if correct."
}}

Example:
[
  {{
    "category": "modular_exponentiation",
    "problem_statement": "Write a Python function `mod_exp(base, exp, mod)` that calculates (base^exp) % mod efficiently.",
    "ground_truth_answer": "def mod_exp(base, exp, mod):\\n    return pow(base, exp, mod)",
    "verification_type": "unit_test",
    "verification_script": "def verify(response):\\n    import re\\n    try:\\n        # extract code\\n        code_match = re.search(r'```python\\\\s*(.*?)\\\\s*```', response, re.DOTALL)\\n        code = code_match.group(1) if code_match else response\\n        loc = {{}}\\n        exec(code, loc)\\n        fn = loc['mod_exp']\\n        return fn(2, 10, 1000) == 24 and fn(3, 5, 7) == 5\\n    except Exception:\\n        return False"
  }}
]

Return ONLY the raw JSON list starting with [ and ending with ]. Do not include any reasoning, markdown blocks, or other text outside the JSON list.
"""
    # Seed '<think>' so the distilled model starts in reasoning mode; stop on 'User:' to prevent it
    # from inventing a next turn.
    # Instruct model to reason first
    prompt = f"User: {meta_prompt}\nAssistant: <think>\n"

    try:
        # Execute guarded operations and establish recovery boundaries
        output_chunks = []
        for chunk in llm(
            prompt,
            max_tokens=4096,
            temperature=0.85,
            stop=["User:"],
            stream=True
        ):
            # Iterate over items, evaluate loop invariant, and update progress
            text = chunk["choices"][0]["text"]
            output_chunks.append(text)
            
        full_output = "".join(output_chunks)
        _, response = parse_thought_stream(full_output, start_in_thought=True)

        # Parse JSON
        response = response.strip()
        # Clean markdown code block wraps if model included them
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        # Models often add prose around the JSON; slice to the outermost list brackets before parsing.
        # Find first '[' and last ']'
        start_idx = response.find('[')
        end_idx = response.rfind(']')
        if start_idx != -1 and end_idx != -1:
            response = response[start_idx:end_idx+1]

        tasks = json.loads(response)
        if isinstance(tasks, list):
            logger.info("Successfully generated %d tasks.", len(tasks))
            # Tasks from the model have no ids; add uuids so queue/dataset entries are traceable.
            # assign IDs
            for t in tasks:
                # Iterate over items, evaluate loop invariant, and update progress
                if "id" not in t:
                    t["id"] = str(uuid.uuid4())
            return tasks
        else:
            raise ValueError("Model output is not a JSON list.")
    except Exception as e:
        # Any failure (bad JSON, model error) falls back to trivial template tasks so the pipeline can
        # still be exercised end to end.
        logger.warning("Failed to generate tasks using LLM: %s. Falling back to local templates.", e)
        return generate_fallback_tasks(topic, count)


def generate_fallback_tasks(topic: str, count: int) -> list[dict]:
    """Generates simple placeholder tasks if LLM task generation fails."""
    # Step 1: Validate input parameters, operational preconditions, and state
    logger.info("Generating fallback local template tasks...")
    tasks = []
    for i in range(count):
        # Iterate over items, evaluate loop invariant, and update progress
        tasks.append({
            "id": str(uuid.uuid4()),
            "category": topic,
            "problem_statement": f"Write a Python function `add_numbers_{i}(a, b)` that returns their sum. Test index: {i}",
            "ground_truth_answer": f"def add_numbers_{i}(a, b):\n    return a + b",
            "verification_type": "unit_test",
            "verification_script": f"""def verify(response):
    import re
    try:
        code_match = re.search(r'```python\\s*(.*?)\\s*```', response, re.DOTALL)
        code = code_match.group(1) if code_match else response
        loc = {{}}
        exec(code, loc)
        fn = loc['add_numbers_{i}']
        return fn(5, 10) == 15 and fn(-1, 1) == 0
    except Exception:
        return False
"""
        })
    return tasks


def solve_task(task: dict, llm) -> tuple[str, str]:
    """Queries LLM to solve the problem, returning (thought, response)."""
    # Step 1: Validate input parameters, operational preconditions, and state
    problem = task.get("problem_statement", "")
    logger.info("Solving task: %s...", task.get("id"))
    # Plain User/Assistant prompt with a seeded <think>; temperature 0.7 keeps some diversity between
    # attempts.
    prompt = f"User: {problem}\nAssistant: <think>\n"
    
    output_chunks = []
    for chunk in llm(
        prompt,
        max_tokens=2048,
        temperature=0.7,
        stop=["User:", "<think>"],
        stream=True
    ):
        # Iterate over items, evaluate loop invariant, and update progress
        text = chunk["choices"][0]["text"]
        output_chunks.append(text)
        
    full_output = "".join(output_chunks)
    thought, response = parse_thought_stream(full_output, start_in_thought=True)
    return thought, response


def verify_solution(task: dict, solution: str) -> tuple[bool, str]:
    """Runs verifier in a secure sandbox."""
    # Step 1: Validate input parameters, operational preconditions, and state
    script = task.get("verification_script", "")
    if not script:
        return False, "No verification script found."

    sandbox = SafePythonSandbox(cpu_timeout_sec=5.0, memory_limit_mb=256)
    # Wrapper executed after the verifier definition: passes the response in as a JSON-escaped literal
    # and maps outcome to exit codes (0 pass, 1 fail, 2 verifier crashed).
    test_code = (
        "import sys as _sys, json as _json\n"
        f"_resp = _json.loads({json.dumps(json.dumps(solution))})\n"
        "try:\n"
        "    _result = verify(_resp)\n"
        "    _sys.exit(0 if _result else 1)\n"
        "except Exception as _e:\n"
        "    print('verify() raised:', _e, file=_sys.stderr)\n"
        "    _sys.exit(2)\n"
    )
    passed, trace = sandbox.run_code(script, test_code)
    return passed, trace


def reflect_and_correct(task: dict, failed_thought: str, failed_response: str, traceback: str, llm) -> tuple[str, str, bool]:
    """Queries the model to correct its own error trace."""
    # Step 1: Validate input parameters, operational preconditions, and state
    logger.info("Failed verification. Starting self-reflection debugger loop...")
    # (Read but currently unused in the correction prompt; the model is shown only its own failure and
    # traceback.)
    ground_truth = task.get("ground_truth_answer", "")
    problem = task.get("problem_statement", "")
    
    correction_prompt = (
        f"You are correcting your own mistake.\n\n"
        f"Problem:\n{problem}\n\n"
        f"Your incorrect response was:\n{failed_response}\n\n"
        f"The verification trace gave this traceback:\n{traceback}\n\n"
        f"Correct the logic. Provide a new complete solution. Return it in a code block."
    )
    prompt_text = build_prompt(
        "You are an expert coder. Self-correct your mistakes using traceback.",
        [{"role": "user", "content": correction_prompt}],
    )
    
    output_chunks = []
    for chunk in llm(
        prompt_text,
        max_tokens=2048,
        # Low temperature for repairs: we want the minimal correct fix, not variety.
        temperature=0.2,
        stream=True
    ):
        # Iterate over items, evaluate loop invariant, and update progress
        text = chunk["choices"][0]["text"]
        output_chunks.append(text)
        
    full_output = "".join(output_chunks)
    thought, response = parse_thought_stream(full_output, start_in_thought=prompt_text.rstrip().endswith("<think>"))
    passed, trace = verify_solution(task, response)
    return thought, response, passed


def train_adapter(dataset_path: str, base_model_path: str, adapter_name: str, args):
    """Launches HuggingFace SFTTrainer on the dataset."""
    # Step 1: Validate input parameters, operational preconditions, and state
    logger.info("Initializing SFT training on base model weights: %s...", base_model_path)
    
    try:
        # Execute guarded operations and establish recovery boundaries
        import torch
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model
        from trl import SFTConfig, SFTTrainer
    except ImportError:
        logger.error("Missing SFT libraries (peft, trl, datasets, transformers). Auto-training aborted.")
        sys.exit(1)

    # Prepare dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Load base weights
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # CPU fallback uses float32 because fp16 math is unsupported/slow on CPU.
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = {"": 0} if torch.cuda.is_available() else "auto"

    # 4-bit QLoRA only when a CUDA GPU is present; otherwise full-precision weights are loaded.
    if args.qlora and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map=device_map
        )

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        # All attention and MLP projections of Llama/Qwen-style models.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    # Required for gradient checkpointing during training.
    model.config.use_cache = False

    adapter_path = ADAPTERS_DIR / adapter_name
    adapter_path.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(adapter_path / "temp_checkpoints"),
        dataset_text_field="messages",
        # Short sequence cap keeps memory low; synthetic tasks are small.
        max_length=512,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=1,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        report_to="none",
        fp16=True if torch.cuda.is_available() else False,
        gradient_checkpointing=True if torch.cuda.is_available() else False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    logger.info("SFT Trainer started. Training for %d epochs...", args.epochs)
    trainer.train()

    # Save
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    
    # Save training metrics log
    with open(adapter_path / "train_history.json", "w") as fh:
        json.dump(trainer.state.log_history, fh, indent=2)

    logger.info("Saved PEFT adapter weights to %s", adapter_path)
    
    # Keep checkpoints for user access


def convert_adapter_to_gguf(base_model_path: str, adapter_name: str):
    """Runs conversion helper to compile LoRA adapter to GGUF."""
    # Step 1: Validate input parameters, operational preconditions, and state
    adapter_path = ADAPTERS_DIR / adapter_name
    outfile = adapter_path / f"{adapter_name}.gguf"
    
    logger.info("Compiling PyTorch adapter weights into GGUF format...")
    # llama.cpp-style converter shipped in app/utils turns the PEFT adapter into a single .gguf file
    # the runtime can load.
    cmd = [
        sys.executable,
        "app/utils/convert_lora_to_gguf.py",
        "--base", base_model_path,
        "--outfile", str(outfile),
        str(adapter_path)
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    # Surface the converter's stderr in both log and exception so failures are diagnosable from the
    # RPC caller.
    if res.returncode != 0:
        logger.error("GGUF Conversion failed: %s", res.stderr)
        raise RuntimeError(f"GGUF conversion failed: {res.stderr}")
        
    logger.info("Successfully created GGUF adapter at %s", outfile)


def main():
    """CLI entry for the one-click pipeline: validate the adapter name, resolve a base model, then generate and verify synthetic cases, train a LoRA adapter and convert it to GGUF (see the numbered steps below)."""
    # This script is spawned as an independent subprocess (e.g. from the
    # start_auto_train WebSocket RPC), so main.py's startup privilege check
    # does not cover it — assert it separately. Sandbox resource limits in
    # executor_sandbox.py are not binding on root, so this is load-bearing.
    assert_not_privileged()

    parser = argparse.ArgumentParser(description="Karl End-to-End One-Click Auto-Trainer")
    parser.add_argument("--topic", type=str, required=True, help="Topic or target capability to train (e.g. 'regex')")
    parser.add_argument("--adapter_name", type=str, required=True, help="Folder/filename to save the adapter")
    parser.add_argument("--count", type=int, default=15, help="Number of synthetic cases to generate/verify")
    parser.add_argument("--epochs", type=int, default=3, help="SFT training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--qlora", action="store_true", default=True, help="Use QLoRA 4-bit SFT")
    parser.add_argument("--base_model", type=str, default=None, help="Path to HF base model folder")
    args = parser.parse_args()

    if not _SAFE_ADAPTER_NAME_RE.match(args.adapter_name):
        logger.error(
            "Invalid --adapter_name %r: must match ^[A-Za-z0-9_-]{1,64}$.",
            args.adapter_name,
        )
        sys.exit(1)

    print("====================================================")
    print("      KARL ONE-CLICK AUTO-TRAINING PIPELINE")
    print("====================================================")
    print(f"Target Behavior: {args.topic}")
    print(f"Dataset Size:    {args.count}")
    print(f"Adapter Save:    data/adapters/{args.adapter_name}")
    print("====================================================")

    # Base model: explicit --base_model, else the first folder in data/hf_models.
    # 1. Resolve base model folder
    base_model = args.base_model
    if not base_model:
        # scan data/hf_models/
        dirs = [d for d in HF_MODELS_DIR.iterdir() if d.is_dir()]
        if dirs:
            base_model = str(dirs[0])
            logger.info("Auto-selected base model weights folder: %s", base_model)
        else:
            logger.error("No base model weights found in data/hf_models/. Please download/place HF model folders there first.")
            sys.exit(1)

    # 2. Load LLM for generation/solving
    logger.info("Loading local LLM singleton...")
    llm = ModelLoader.get_instance()

    # 3. Generate Tasks
    tasks = generate_tasks_for_topic(args.topic, args.count, llm)

    # Only tasks that pass verification (directly or after one self-correction) become training data;
    # failures are discarded.
    # 4. Process Loop (Solve, Verify, Reflect)
    verified_examples = []
    for index, task in enumerate(tasks, 1):
        # Iterate over items, evaluate loop invariant, and update progress
        logger.info("--- Task %d/%d ---", index, len(tasks))
        print(f"Statement: {task['problem_statement'][:100]}...")
        
        # Solve
        thought, response = solve_task(task, llm)
        
        # Verify
        passed, trace = verify_solution(task, response)
        
        # Reflection loop
        if not passed:
            thought, response, passed = reflect_and_correct(task, thought, response, trace, llm)

        if passed:
            logger.info("Task %d PASSED verification.", index)
            # Create SFT instruction format (ChatML)
            messages = [
                {"role": "user", "content": task["problem_statement"]},
                {"role": "assistant", "content": f"<think>\n{thought}\n</think>\n{response}"}
            ]
            verified_examples.append({
                "instruction": task["problem_statement"],
                "input": "",
                "output": f"<think>\n{thought}\n</think>\n{response}",
                "messages": messages,
                "source": "auto_train",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            })
        else:
            logger.warning("Task %d FAILED verification. Discarding.", index)

    # Save dataset
    dataset_file = TRAINING_DIR / f"auto_train_{args.adapter_name}.jsonl"
    with open(dataset_file, "w", encoding="utf-8") as f:
        for ex in verified_examples:
            # Iterate over items, evaluate loop invariant, and update progress
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info("Dataset curated successfully with %d examples: %s", len(verified_examples), dataset_file)

    # SFT on fewer than two examples is meaningless; abort with a clear error.
    if len(verified_examples) < 2:
        logger.error("Not enough verified examples (need at least 2) to perform SFT. Auto-training aborted.")
        sys.exit(1)

    # 5. Run SFT Trainer
    logger.info("Releasing GPU VRAM from active GGUF inference engine...")
    # Free the inference model's VRAM before training loads the HF weights; both rarely fit on the GPU
    # together.
    ModelLoader.reset_instance()
    train_adapter(str(dataset_file), base_model, args.adapter_name, args)

    # 6. GGUF compilation
    convert_adapter_to_gguf(base_model, args.adapter_name)

    print("====================================================")
    print("  AUTO-TRAINING PIPELINE COMPLETE SUCCESSFULLY!")
    print(f"  GGUF adapter ready: data/adapters/{args.adapter_name}/{args.adapter_name}.gguf")
    print("====================================================")


if __name__ == "__main__":
    main()
