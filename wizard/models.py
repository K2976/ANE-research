"""
Model discovery, download, and management for the ANE Experiment Wizard.
Scans the HuggingFace cache for locally available models, estimates memory
usage, and can auto-download models that are not yet available.
"""

import os
import sys
import json
import glob
import shutil
from pathlib import Path

# ── ANEForge path setup ──────────────────────────────────────────────────────

ANEFORGE_PATH = "/Users/kartik/Documents/ANEForge-main"
if ANEFORGE_PATH not in sys.path:
    sys.path.insert(0, ANEFORGE_PATH)

HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")

# Known model registry — maps short names to HuggingFace IDs
KNOWN_MODELS = {
    "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "Qwen3-1.7B": "Qwen/Qwen3-1.7B",
}

# Approximate parameter counts (billions) for memory estimation
PARAM_COUNTS = {
    "meta-llama/Llama-3.2-1B-Instruct": 1.24,
    "meta-llama/Llama-3.2-3B-Instruct": 3.21,
    "meta-llama/Llama-3.2-1B": 1.24,
    "Qwen/Qwen3-0.6B": 0.6,
    "Qwen/Qwen3-1.7B": 1.7,
}


def _hf_dir_name(model_id):
    """Convert a HuggingFace model ID to the cache directory name format."""
    return f"models--{model_id.replace('/', '--')}"


def discover_local_models():
    """Scan the HuggingFace cache for all locally available models.
    Returns a list of dicts with model info."""
    models = []
    pattern = os.path.join(HF_CACHE, "models--*")

    for model_dir in sorted(glob.glob(pattern)):
        dirname = os.path.basename(model_dir)
        # Convert 'models--meta-llama--Llama-3.2-1B-Instruct' -> 'meta-llama/Llama-3.2-1B-Instruct'
        model_id = dirname.replace("models--", "", 1).replace("--", "/", 1)

        # Check if snapshots exist (model is actually downloaded)
        snapshots_dir = os.path.join(model_dir, "snapshots")
        if not os.path.isdir(snapshots_dir):
            continue

        snapshot_dirs = [
            d for d in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, d))
        ]
        if not snapshot_dirs:
            continue

        # Use the latest snapshot
        snapshot_path = os.path.join(snapshots_dir, snapshot_dirs[-1])

        # Check for model weights
        has_weights = any(
            f.endswith((".safetensors", ".bin"))
            for f in os.listdir(snapshot_path)
            if os.path.isfile(os.path.join(snapshot_path, f))
                or os.path.islink(os.path.join(snapshot_path, f))
        )

        # Check for tokenizer
        has_tokenizer = any(
            f.startswith("tokenizer") for f in os.listdir(snapshot_path)
        )

        # Estimate disk size
        disk_size_bytes = _dir_size(model_dir)

        # Get short name
        short_name = model_id.split("/")[-1] if "/" in model_id else model_id

        models.append({
            "id": model_id,
            "short_name": short_name,
            "path": snapshot_path,
            "cache_dir": model_dir,
            "has_weights": has_weights,
            "has_tokenizer": has_tokenizer,
            "disk_size_bytes": disk_size_bytes,
            "disk_size_mb": round(disk_size_bytes / (1024 * 1024), 1),
            "param_count_b": PARAM_COUNTS.get(model_id, None),
            "is_downloaded": has_weights,
            "memory_estimates": _estimate_memory(model_id),
        })

    return models


def _dir_size(path):
    """Get total size of a directory in bytes, following symlinks."""
    total = 0
    for dirpath, _, filenames in os.walk(path, followlinks=True):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _estimate_memory(model_id):
    """Estimate memory usage for different compression modes."""
    params_b = PARAM_COUNTS.get(model_id, 1.0)
    params = params_b * 1e9

    return {
        "fp16_mb": round(params * 2 / (1024 * 1024), 0),
        "int8_mb": round(params * 1 / (1024 * 1024), 0),
        "int4_mb": round(params * 0.5 / (1024 * 1024), 0),
    }


def get_model_info(model_id):
    """Get info for a specific model. Returns None if not found locally."""
    models = discover_local_models()
    for m in models:
        if m["id"] == model_id:
            return m
    return None


def download_model(model_id, on_progress=None):
    """Download a model from HuggingFace if not already available.
    Uses transformers AutoModel to trigger the download.
    on_progress(msg) is called with status updates."""
    try:
        if on_progress:
            on_progress(f"Downloading {model_id} from HuggingFace...")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        if on_progress:
            on_progress(f"Downloading tokenizer for {model_id}...")
        AutoTokenizer.from_pretrained(model_id)

        if on_progress:
            on_progress(f"Downloading model weights for {model_id}... (this may take several minutes)")
        AutoModelForCausalLM.from_pretrained(model_id)

        if on_progress:
            on_progress(f"Successfully downloaded {model_id}")

        return True, f"Model {model_id} downloaded successfully"

    except Exception as e:
        return False, f"Failed to download {model_id}: {str(e)}"


def is_model_downloaded(model_id):
    """Quick check if a model is available locally."""
    cache_dir = os.path.join(HF_CACHE, _hf_dir_name(model_id))
    if not os.path.isdir(cache_dir):
        return False

    snapshots_dir = os.path.join(cache_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return False

    for snap in os.listdir(snapshots_dir):
        snap_path = os.path.join(snapshots_dir, snap)
        if os.path.isdir(snap_path):
            for f in os.listdir(snap_path):
                if f.endswith((".safetensors", ".bin")):
                    return True
    return False


def load_model_for_inference(model_id, compress="int8"):
    """Load a model using ANEForge for ANE inference.
    Downloads if not available. Returns (model, tokenizer) tuple."""
    import aneforge as af
    from transformers import AutoTokenizer

    # Auto-download if not available
    if not is_model_downloaded(model_id):
        success, msg = download_model(model_id)
        if not success:
            raise RuntimeError(msg)

    model = af.load_llm(model_id, compress=compress)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer


def get_available_compressions():
    """Return the list of valid compression formats for ANEForge."""
    return [
        {"id": "int4", "label": "INT4 (smallest, fastest)", "bytes_per_param": 0.5},
        {"id": "int8", "label": "INT8 (recommended balance)", "bytes_per_param": 1.0},
        {"id": None, "label": "FP16 (full precision, slowest)", "bytes_per_param": 2.0},
    ]


def get_known_models_list():
    """Return all known models with their download status."""
    result = []
    for short_name, model_id in KNOWN_MODELS.items():
        downloaded = is_model_downloaded(model_id)
        result.append({
            "id": model_id,
            "short_name": short_name,
            "is_downloaded": downloaded,
            "param_count_b": PARAM_COUNTS.get(model_id),
            "memory_estimates": _estimate_memory(model_id),
        })
    return result
