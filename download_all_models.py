# ==============================================================================
# Program: download_all_models.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Automated model weight fetcher and validator.
#          Downloads configured GGUF LLM models and embedding weights into the
#          local data/models directory, verifying the sha256 checksum when the
#          registry entry has a "sha256" field (entries without one are not
#          verified), and ensuring model availability for offline inference.
# ==============================================================================

import hashlib
import os
import argparse
import json
import requests
from tqdm import tqdm

def download_file(url, filepath, sha256=None):
    """Stream url to filepath in 1 MiB blocks with a tqdm progress bar. Raises on HTTP errors. If sha256 (hex digest) is given the finished download must match it or ValueError is raised and nothing is kept. Atomic: data goes to filepath + '.part' and is renamed only once complete, so the caller's exists() check never mistakes a partial file for a finished model."""
    # stream=True keeps the multi-GB GGUF out of memory; no timeout is set, so a stalled server
    # blocks until the OS gives up on the socket.
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    # content-length can be missing (chunked responses); total=0 then makes tqdm show a counter
    # instead of a percentage, which is harmless.
    block_size = 1024 * 1024 # 1 MB block size
    # A sibling name on the same directory keeps os.replace on one filesystem, which is what makes
    # the final rename atomic.
    part_path = filepath + ".part"
    digest = hashlib.sha256()

    try:
        with open(part_path, 'wb') as file, tqdm(
            desc=os.path.basename(filepath),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                if data:
                    size = file.write(data)
                    digest.update(data)
                    bar.update(size)
        if sha256 and digest.hexdigest() != sha256.strip().lower():
            raise ValueError(f"sha256 mismatch for {os.path.basename(filepath)}: expected {sha256}, got {digest.hexdigest()}")
        # os.replace is atomic on the same filesystem, so readers see all or nothing.
        os.replace(part_path, filepath)
    except BaseException:
        # Do not leave a stale partial behind on failure or Ctrl-C.
        if os.path.exists(part_path):
            os.remove(part_path)
        raise

def main():
    """CLI: pick models from data/model_registry.json by --all, --tier N, or an interactive numbered menu, then download any that are not already in data/models. Per-model failures are reported but do not stop the batch."""
    parser = argparse.ArgumentParser(description="Download Karl LLM models from the registry.")
    parser.add_argument(
        # Tiers group models by hardware class (higher tiers are bigger models); see the registry.
        "--tier", type=int, choices=[1, 2, 3, 4], help="Download all models of a specific tier (e.g. tier 2 for 7B/8B class models)."
    )
    parser.add_argument(
        "--all", action="store_true", help="Download all models in the registry."
    )
    args = parser.parse_args()

    # Registry lists each model's name, tier, filename, url and size.
    # Paths are relative to the CWD, so run this from the repo root (as the docs/setup scripts do).
    registry_path = "data/model_registry.json"
    if not os.path.exists(registry_path):
        print(f"Registry not found at {registry_path}")
        return

    with open(registry_path, "r") as f:
        registry = json.load(f)

    # Created up front so the first download does not fail on a fresh checkout.
    os.makedirs("data/models", exist_ok=True)

    to_download = []
    if args.all:
        to_download = registry
    elif args.tier is not None:
        to_download = [m for m in registry if m.get("tier") == args.tier]
    else:
        # Interactive mode is only reached when neither --all nor --tier is given, so scripted
        # callers (setup, CI) never block on input().
        # No flags given: show what is available and whether it is already on disk.
        # Interactive selection
        print("Available models in registry:")
        for idx, m in enumerate(registry, 1):
            file_path = os.path.join("data/models", m["filename"])
            status = "[Downloaded]" if os.path.exists(file_path) else "[Not Downloaded]"
            print(f"  {idx}. Tier {m['tier']}: {m['name']} ({m['min_storage_gb']} GB) {status}")
        
        try:
            choice = input("\nEnter numbers of models to download (comma separated, e.g. 2,3), or 'all': ").strip().lower()
            if choice == 'all':
                to_download = registry
            else:
                # Non-numeric tokens and out-of-range numbers are dropped silently instead of aborting, so
                # "1, x, 99" still downloads model 1.
                indices = [int(x.strip()) - 1 for x in choice.split(",") if x.strip().isdigit()]
                to_download = [registry[i] for i in indices if 0 <= i < len(registry)]
        except (KeyboardInterrupt, SystemExit):
            print("\nCancelled.")
            return
        except Exception:
            print("Invalid input.")
            return

    if not to_download:
        print("No models selected for download.")
        return

    for m in to_download:
        target = os.path.join("data/models", m["filename"])
        print(f"\nProcessing {m['name']}...")
        # Existence is a safe "already complete" test only because download_file() renames from a
        # .part file at the very end.
        if os.path.exists(target):
            print(f"Model already exists at {target}")
            continue
        
        print(f"Downloading from: {m['url']}")
        try:
            download_file(m["url"], target, m.get("sha256"))
            print(f"Successfully downloaded to {target}")
        except Exception as e:
            # One failed model must not abort the batch (e.g. --all over a flaky connection); the user
            # simply reruns and the finished files are skipped.
            print(f"Error downloading {m['name']}: {e}")

if __name__ == "__main__":
    main()
