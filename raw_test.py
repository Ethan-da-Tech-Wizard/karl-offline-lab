# Program: karl-offline-lab - raw_test.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Quick manual script that loads the model and sends one hand-built ChatML prompt, printing the raw output without any UI or threading.
"""
Quick raw model test -- bypasses all UI/threading.
Run: python raw_test.py
"""
import os
# Force offline mode so nothing is fetched from the network.
os.environ["HF_HUB_OFFLINE"] = "1"

from app.engine.model_loader import ModelLoader

print("Loading model...")
llm = ModelLoader.get_instance()
print(f"Model loaded: {ModelLoader.model_name()} (n_ctx={ModelLoader.n_ctx()})\n")

# Build a minimal ChatML prompt, pre-seeding <think> so the model
# immediately enters reasoning mode instead of outputting garbage.
SYSTEM = "You are Karl, a helpful AI assistant. Answer the user's question clearly and directly."
USER   = "What is 2 + 2? Explain your answer briefly."

prompt = (
    f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
    f"<|im_start|>user\n{USER}<|im_end|>\n"
    f"<|im_start|>assistant\n<think>\n"
)

print("Prompt sent:")
print(repr(prompt))
print("\n--- RAW OUTPUT ---")

# Run one non-streaming generation with the sampling settings below.
output = llm(
    prompt,
    max_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repeat_penalty=1.1,
    stop=["<|im_end|>", "<|endoftext|>", "<|end_of_text|>"],
    echo=False,
    stream=False,
)

# Print the generated text, why generation stopped, and token usage.
raw = output["choices"][0]["text"]
finish = output["choices"][0]["finish_reason"]
print(raw)
print(f"\n--- finish_reason: {finish} ---")
print(f"--- token count: {output['usage']} ---")
