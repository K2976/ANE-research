import os
import json
import shutil
from pathlib import Path

# Base cache directory
CACHE_DIR = Path(os.path.dirname(__file__)) / ".cache"
BASELINE_DIR = CACHE_DIR / "baselines"
TOKEN_DIR = CACHE_DIR / "tokens"


def init_cache():
    """Ensure cache directories exist."""
    os.makedirs(BASELINE_DIR, exist_ok=True)
    os.makedirs(TOKEN_DIR, exist_ok=True)


# ── Baseline Cache ────────────────────────────────────────────────────────

def get_baseline_cache_key(model_id: str, dataset_name: str, max_len: int) -> str:
    # Safely replace slashes in model_id
    safe_model = model_id.replace("/", "_").replace("\\", "_")
    return f"{safe_model}_{dataset_name}_len{max_len}.json"


def load_baseline(model_id: str, dataset_name: str, max_len: int) -> dict:
    """Load cached FP16 baseline metrics if available."""
    init_cache()
    key = get_baseline_cache_key(model_id, dataset_name, max_len)
    cache_path = BASELINE_DIR / key
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def save_baseline(model_id: str, dataset_name: str, max_len: int, metrics: dict):
    """Save FP16 baseline metrics to cache."""
    init_cache()
    key = get_baseline_cache_key(model_id, dataset_name, max_len)
    cache_path = BASELINE_DIR / key
    with open(cache_path, "w") as f:
        json.dump(metrics, f, indent=2)


# ── Tokenization Cache ────────────────────────────────────────────────────

def get_token_cache_key(model_id: str, dataset_name: str) -> str:
    safe_model = model_id.replace("/", "_").replace("\\", "_")
    return f"{safe_model}_{dataset_name}_tokens.json"


def load_tokenized_dataset(model_id: str, dataset_name: str) -> list:
    """Load cached tokenized dataset (list of lists of ints)."""
    init_cache()
    key = get_token_cache_key(model_id, dataset_name)
    cache_path = TOKEN_DIR / key
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def save_tokenized_dataset(model_id: str, dataset_name: str, tokens: list):
    """Save tokenized dataset to cache."""
    init_cache()
    key = get_token_cache_key(model_id, dataset_name)
    cache_path = TOKEN_DIR / key
    with open(cache_path, "w") as f:
        json.dump(tokens, f)


# ── Cache Management ──────────────────────────────────────────────────────

def get_cache_stats() -> dict:
    """Return size and count of cached items."""
    init_cache()
    
    def get_dir_stats(d: Path):
        count = 0
        size_bytes = 0
        if d.exists():
            for f in d.glob("*.json"):
                count += 1
                size_bytes += f.stat().st_size
        return {"count": count, "size_mb": round(size_bytes / (1024 * 1024), 2)}

    return {
        "baselines": get_dir_stats(BASELINE_DIR),
        "tokens": get_dir_stats(TOKEN_DIR),
    }


def clear_cache(cache_type: str = "all"):
    """Clear specific or all caches."""
    if cache_type in ("baselines", "all"):
        if BASELINE_DIR.exists():
            shutil.rmtree(BASELINE_DIR)
        os.makedirs(BASELINE_DIR, exist_ok=True)
        
    if cache_type in ("tokens", "all"):
        if TOKEN_DIR.exists():
            shutil.rmtree(TOKEN_DIR)
        os.makedirs(TOKEN_DIR, exist_ok=True)
