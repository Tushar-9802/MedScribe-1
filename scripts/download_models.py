#!/usr/bin/env python3
"""
Download MedGemma and MedASR to local cache
"""

import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download, login

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

print("Downloading models (this takes time)...")

# MedGemma 4B
print("\n[1/2] MedGemma 4B...")
medgemma_path = snapshot_download(
    repo_id="google/medgemma-4b-it",
    cache_dir="./models/cache",
    local_dir="./models/medgemma",
    local_dir_use_symlinks=False
)
print(f"Saved to: {medgemma_path}")

# MedASR
print("\n[2/2] MedASR...")
medasr_path = snapshot_download(
    repo_id="google/medasr",
    cache_dir="./models/cache",
    local_dir="./models/medasr",
    local_dir_use_symlinks=False
)
print(f"Saved to: {medasr_path}")

print("\nAll models downloaded!")
