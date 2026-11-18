# Program: karl-offline-lab - download_test_model.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Downloads the default small test model (a quantized 1.5B GGUF file) into data/models with a progress bar if it is not already present.
"""Downloader for the default test model."""
import os
import requests
from tqdm import tqdm

def download_file(url, filepath):
    """Stream the file at url to filepath in 1 KiB blocks, showing a progress bar.

    Raises an HTTP error if the request fails.
    """
    # Streaming download with progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    
    with open(filepath, 'wb') as file, tqdm(
        desc=os.path.basename(filepath),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

# Script entry point: skip the download when the model file already exists.
if __name__ == "__main__":
    # Downloads DeepSeek-R1-Distill-Qwen-1.5B (Q4_K_M quantized, ~986 MB).
    # This is the default Tier-1 model for Karl — see data/model_registry.json.
    url = "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
    
    os.makedirs("data/models", exist_ok=True)
    target_path = "data/models/deepseek-r1-1.5b.gguf"
    
    if not os.path.exists(target_path):
        print(f"Downloading test model to {target_path}...")
        download_file(url, target_path)
        print("Download complete!")
    else:
        print(f"Model already exists at {target_path}")
