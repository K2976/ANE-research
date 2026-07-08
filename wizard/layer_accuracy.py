"""
Layer Accuracy Measurement
===========================
Computes perplexity and token-level accuracy for selectively-quantized
models. Used by the Layer Profiling module to measure ΔA(i,b) —
the accuracy degradation caused by quantizing a single sub-layer.

Primary metric: Perplexity (exp of cross-entropy loss)
Secondary metric: Token-level accuracy (exact-match on next-token prediction)
"""

import os
import math
import numpy as np
from pathlib import Path


# ── Dataset Loading ───────────────────────────────────────────────────────

DATASETS_DIR = Path(os.path.dirname(__file__)) / "datasets"


def load_dataset(name: str) -> list:
    """Load an evaluation dataset by name.
    Returns a list of text strings (one per document/segment).

    Supported names:
        wikitext2  — WikiText-2 sample (bundled)
        ptb        — Penn Treebank sample (bundled)
        custom     — Custom dataset from a file path
    """
    if name == "wikitext2":
        path = DATASETS_DIR / "wikitext2_sample.txt"
    elif name == "ptb":
        path = DATASETS_DIR / "ptb_sample.txt"
    else:
        # Treat as a file path
        path = Path(name)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]

    return lines


def get_available_datasets() -> list:
    """Return list of available dataset descriptors."""
    datasets = [
        {"id": "wikitext2", "name": "WikiText-2 (sample)", "type": "bundled"},
        {"id": "ptb", "name": "Penn Treebank (sample)", "type": "bundled"},
    ]

    # Check which are actually present
    for d in datasets:
        path = DATASETS_DIR / f"{d['id']}_sample.txt"
        if d["id"] == "ptb":
            path = DATASETS_DIR / "ptb_sample.txt"
        elif d["id"] == "wikitext2":
            path = DATASETS_DIR / "wikitext2_sample.txt"
        d["available"] = path.exists()

    return datasets


# ── Perplexity Computation ────────────────────────────────────────────────

def compute_perplexity(model, tokenizer, texts: list, max_len: int = 512,
                       on_progress=None) -> dict:
    """Compute perplexity of a model on a list of texts.

    For each text:
      1. Tokenize
      2. Run prefill to get all-position hidden states
      3. Compute cross-entropy loss on next-token predictions
      4. Average across all tokens

    Returns:
        {
            "perplexity": float,
            "avg_loss": float,
            "total_tokens": int,
            "token_accuracy": float,  # fraction of correct next-token predictions
        }
    """
    total_loss = 0.0
    total_tokens = 0
    correct_predictions = 0

    for text_idx, text in enumerate(texts):
        try:
            ids = tokenizer.encode(text)
        except Exception:
            continue

        if len(ids) < 2:
            continue

        # Clamp to max_len
        if len(ids) > max_len:
            ids = ids[:max_len]

        n = len(ids)

        try:
            # Get hidden states for all positions
            hidden = model._hidden(ids)  # [n, dim]

            # Compute logits for all positions
            lm_head = np.asarray(model.w["lm_head"]).T  # [dim, vocab]
            logits = hidden @ lm_head  # [n, vocab]

            # Cross-entropy loss: compare position i's logits against token i+1
            for i in range(n - 1):
                target_id = ids[i + 1]
                token_logits = logits[i].astype(np.float64)

                # Numerically stable softmax
                max_logit = np.max(token_logits)
                log_sum_exp = max_logit + np.log(np.sum(np.exp(token_logits - max_logit)))
                loss = log_sum_exp - token_logits[target_id]

                total_loss += loss
                total_tokens += 1

                # Token accuracy: check if argmax prediction matches
                if np.argmax(token_logits) == target_id:
                    correct_predictions += 1

        except Exception as e:
            # Skip texts that cause errors (e.g., token count issues)
            if on_progress:
                on_progress(f"Skipping text {text_idx}: {str(e)[:80]}")
            continue

        if on_progress:
            on_progress(f"Evaluated {text_idx + 1}/{len(texts)} texts ({total_tokens} tokens)")

    if total_tokens == 0:
        return {
            "perplexity": float("inf"),
            "avg_loss": float("inf"),
            "total_tokens": 0,
            "token_accuracy": 0.0,
        }

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 100))  # Clamp to prevent overflow

    return {
        "perplexity": round(perplexity, 4),
        "avg_loss": round(avg_loss, 6),
        "total_tokens": total_tokens,
        "token_accuracy": round(correct_predictions / total_tokens, 4),
    }


def compute_accuracy_delta(
    baseline_model, quantized_model, tokenizer,
    texts: list, max_len: int = 512, on_progress=None
) -> dict:
    """Compute the accuracy delta between a baseline and quantized model.

    Returns:
        {
            "baseline": {perplexity, avg_loss, total_tokens, token_accuracy},
            "quantized": {perplexity, avg_loss, total_tokens, token_accuracy},
            "delta_perplexity": float,
            "accuracy_loss": float,
        }
    """
    if on_progress:
        on_progress("Computing baseline perplexity...")

    baseline = compute_perplexity(baseline_model, tokenizer, texts, max_len,
                                  on_progress=on_progress)

    if on_progress:
        on_progress("Computing quantized perplexity...")

    quantized = compute_perplexity(quantized_model, tokenizer, texts, max_len,
                                   on_progress=on_progress)

    return {
        "baseline": baseline,
        "quantized": quantized,
        "delta_perplexity": round(quantized["perplexity"] - baseline["perplexity"], 4),
        "accuracy_loss": round(baseline["token_accuracy"] - quantized["token_accuracy"], 4),
    }
