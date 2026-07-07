"""
Pre-run validation for the ANE Experiment Wizard.
Performs a comprehensive validation sequence before every experiment.
"""

import os
import sys
import shutil
import importlib
from pathlib import Path

ANEFORGE_PATH = "/Users/kartik/Documents/ANEForge-main"


def validate(config):
    """Run all pre-experiment validation checks.

    Args:
        config: dict with keys:
            - model_id: str (HuggingFace model ID)
            - compression: str or None
            - prompt_source: str ('single', 'file', 'dataset')
            - prompt_file: str (path, if prompt_source is 'file')
            - prompt_dataset: str (dataset name, if prompt_source is 'dataset')
            - output_dir: str
            - max_len: int

    Returns:
        list of dicts: [{name, passed, message, fix}]
    """
    results = []

    # ── 1. Model folder exists ────────────────────────────────────────────
    model_id = config.get("model_id", "")
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    cache_name = f"models--{model_id.replace('/', '--')}"
    model_cache_dir = os.path.join(hf_cache, cache_name)

    results.append(_check(
        "Model folder exists",
        os.path.isdir(model_cache_dir),
        f"Found at {model_cache_dir}",
        f"Model {model_id} not found in HuggingFace cache",
        f"The wizard will auto-download it, or run: huggingface-cli download {model_id}"
    ))

    # ── 2. Model weights exist ────────────────────────────────────────────
    has_weights = False
    if os.path.isdir(model_cache_dir):
        snapshots = os.path.join(model_cache_dir, "snapshots")
        if os.path.isdir(snapshots):
            for snap in os.listdir(snapshots):
                snap_path = os.path.join(snapshots, snap)
                if os.path.isdir(snap_path):
                    for f in os.listdir(snap_path):
                        if f.endswith((".safetensors", ".bin")):
                            has_weights = True
                            break

    results.append(_check(
        "Model weights exist",
        has_weights,
        "Weights (.safetensors/.bin) found",
        "No model weight files found",
        "Re-download the model or check the HuggingFace cache"
    ))

    # ── 3. Tokenizer exists ───────────────────────────────────────────────
    has_tokenizer = False
    if os.path.isdir(model_cache_dir):
        snapshots = os.path.join(model_cache_dir, "snapshots")
        if os.path.isdir(snapshots):
            for snap in os.listdir(snapshots):
                snap_path = os.path.join(snapshots, snap)
                if os.path.isdir(snap_path):
                    for f in os.listdir(snap_path):
                        if f.startswith("tokenizer"):
                            has_tokenizer = True
                            break

    results.append(_check(
        "Tokenizer exists",
        has_tokenizer,
        "Tokenizer files found",
        "No tokenizer files found",
        "Re-download the model with tokenizer"
    ))

    # ── 4. Prompt file exists (if applicable) ─────────────────────────────
    prompt_source = config.get("prompt_source", "single")
    if prompt_source == "file":
        prompt_file = config.get("prompt_file", "")
        results.append(_check(
            "Prompt file exists",
            os.path.isfile(prompt_file),
            f"Found: {prompt_file}",
            f"Prompt file not found: {prompt_file}",
            "Check the file path or select a different prompt source"
        ))
    elif prompt_source == "dataset":
        prompt_dataset = config.get("prompt_dataset", "")
        dataset_path = os.path.join(os.path.dirname(__file__), "prompts", f"{prompt_dataset}_prompts.txt")
        results.append(_check(
            "Dataset exists",
            os.path.isfile(dataset_path),
            f"Found: {dataset_path}",
            f"Dataset not found: {prompt_dataset}",
            "Available datasets: coding, reasoning, general"
        ))

    # ── 5. Output directory exists ────────────────────────────────────────
    output_dir = config.get("output_dir", "./power_logs")
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = Path(os.path.dirname(__file__)) / output_dir

    results.append(_check(
        "Output directory exists",
        True,  # We'll auto-create it
        f"Output: {output_path} (will create if needed)",
        "",
        ""
    ))

    # ── 6. Enough free disk space ─────────────────────────────────────────
    disk = shutil.disk_usage(str(output_path.parent) if output_path.parent.exists() else "/")
    free_gb = disk.free / (1024 ** 3)
    results.append(_check(
        "Enough free disk space",
        free_gb > 1.0,
        f"{free_gb:.1f} GB free",
        f"Only {free_gb:.1f} GB free",
        "Free up disk space. Experiments generate CSV/JSON output files."
    ))

    # ── 7. Enough available unified memory ────────────────────────────────
    try:
        import psutil
        mem = psutil.virtual_memory()
        avail_gb = mem.available / (1024 ** 3)
        # Rough estimate: int8 model needs ~1 byte per param
        from . import models as mod
        params_b = mod.PARAM_COUNTS.get(model_id, 1.0)
        compress = config.get("compression", "int8")
        if compress == "int4":
            needed_gb = params_b * 0.5
        elif compress == "int8":
            needed_gb = params_b * 1.0
        else:
            needed_gb = params_b * 2.0
        needed_gb += 1.0  # overhead

        results.append(_check(
            "Enough available memory",
            avail_gb > needed_gb,
            f"{avail_gb:.1f} GB available (need ~{needed_gb:.1f} GB)",
            f"Only {avail_gb:.1f} GB available, need ~{needed_gb:.1f} GB",
            "Close other applications to free memory"
        ))
    except ImportError:
        results.append(_check(
            "Enough available memory",
            False,
            "",
            "psutil not installed — cannot check memory",
            "Install psutil: pip install psutil"
        ))

    # ── 8. Required Python packages ───────────────────────────────────────
    packages = {
        "transformers": "transformers",
        "psutil": "psutil",
    }
    for display_name, pkg_name in packages.items():
        found = importlib.util.find_spec(pkg_name) is not None
        results.append(_check(
            f"Package: {display_name}",
            found,
            f"{display_name} is installed",
            f"{display_name} is not installed",
            f"Run: pip install {pkg_name}"
        ))

    # ── 9. ANEForge available ─────────────────────────────────────────────
    ane_available = os.path.isdir(ANEFORGE_PATH) and os.path.isfile(
        os.path.join(ANEFORGE_PATH, "aneforge", "__init__.py")
    )
    results.append(_check(
        "ANEForge available",
        ane_available,
        f"Found at {ANEFORGE_PATH}",
        "ANEForge not found",
        "Clone ANEForge to /Users/kartik/Documents/ANEForge-main"
    ))

    # ── 10. Compression format valid ──────────────────────────────────────
    compress = config.get("compression", "int8")
    valid_compress = compress in ("int4", "int8", None, "None")
    results.append(_check(
        "Valid compression format",
        valid_compress,
        f"Compression: {compress or 'fp16'}",
        f"Invalid compression: {compress}",
        "Valid options: int4, int8, or None (fp16)"
    ))

    return results


def _check(name, passed, success_msg, fail_msg, fix):
    """Create a validation result dict."""
    return {
        "name": name,
        "passed": passed,
        "message": success_msg if passed else fail_msg,
        "fix": fix if not passed else "",
    }


def all_passed(results):
    """Check if all validations passed."""
    return all(r["passed"] for r in results)


def critical_failures(results):
    """Return only the failed validation results."""
    return [r for r in results if not r["passed"]]
