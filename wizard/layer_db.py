"""
Layer Profiling Database — Persistence Layer
=============================================
Manages the Energy and Accuracy databases for mixed-precision
quantization research. Each profiling run is stored as a directory
containing JSON metadata, CSV data tables, and summary statistics.

Database structure:
    database/
        energy/
            <run_id>/
                metadata.json
                energy_records.csv
                energy_records.json
                summary.json
        accuracy/
            <run_id>/
                metadata.json
                accuracy_records.csv
                accuracy_records.json
                summary.json
"""

import os
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional


# ── Database Paths ────────────────────────────────────────────────────────

def _db_root():
    """Root directory for all profiling databases."""
    return Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "database"


def _energy_root():
    return _db_root() / "energy"


def _accuracy_root():
    return _db_root() / "accuracy"


# ── Energy Database Schema ────────────────────────────────────────────────

ENERGY_FIELDS = [
    "experiment_id",
    "model",
    "block_number",
    "sublayer",          # attention | mlp | layernorm
    "bit_width",         # fp16 | int8 | int4 | int2
    "latency_avg_ms",
    "latency_min_ms",
    "latency_max_ms",
    "latency_std_ms",
    "tokens_per_sec",
    "avg_cpu_mw",
    "avg_gpu_mw",
    "avg_ane_mw",
    "avg_combined_mw",
    "peak_power_mw",
    "temperature",
    "energy_joules",
    "memory_peak_mb",
    "memory_avg_mb",
    "execution_device",
    "duration_sec",
    "samples_collected",
    "timestamp",
]

# ── Accuracy Database Schema ─────────────────────────────────────────────

ACCURACY_FIELDS = [
    "experiment_id",
    "model",
    "block_number",
    "sublayer",
    "bit_width",
    "dataset",
    "baseline_perplexity",
    "measured_perplexity",
    "delta_perplexity",
    "baseline_accuracy",
    "measured_accuracy",
    "accuracy_loss",
    "num_tokens_evaluated",
    "timestamp",
]


# ── Record Creation ───────────────────────────────────────────────────────

def make_energy_record(
    experiment_id: str,
    model: str,
    block_number: int,
    sublayer: str,
    bit_width: str,
    latency_stats: dict,
    power_stats: dict,
    memory_stats: dict,
    duration_sec: float,
    samples_collected: int,
    execution_device: str = "ANE",
) -> dict:
    """Create a single energy database record."""
    return {
        "experiment_id": experiment_id,
        "model": model,
        "block_number": block_number,
        "sublayer": sublayer,
        "bit_width": bit_width,
        "latency_avg_ms": latency_stats.get("avg", 0),
        "latency_min_ms": latency_stats.get("min", 0),
        "latency_max_ms": latency_stats.get("max", 0),
        "latency_std_ms": latency_stats.get("std", 0),
        "tokens_per_sec": latency_stats.get("tokens_per_sec", 0),
        "avg_cpu_mw": power_stats.get("avg_cpu_mw", 0),
        "avg_gpu_mw": power_stats.get("avg_gpu_mw", 0),
        "avg_ane_mw": power_stats.get("avg_ane_mw", 0),
        "avg_combined_mw": power_stats.get("avg_combined_mw", 0),
        "peak_power_mw": power_stats.get("peak_power_mw", 0),
        "temperature": power_stats.get("temperature", "unknown"),
        "energy_joules": power_stats.get("energy_joules", 0),
        "memory_peak_mb": memory_stats.get("peak_mb", 0),
        "memory_avg_mb": memory_stats.get("avg_mb", 0),
        "execution_device": execution_device,
        "duration_sec": round(duration_sec, 2),
        "samples_collected": samples_collected,
        "timestamp": datetime.now().isoformat(),
    }


def make_accuracy_record(
    experiment_id: str,
    model: str,
    block_number: int,
    sublayer: str,
    bit_width: str,
    dataset: str,
    baseline_perplexity: float,
    measured_perplexity: float,
    baseline_accuracy: float,
    measured_accuracy: float,
    num_tokens: int,
) -> dict:
    """Create a single accuracy database record."""
    return {
        "experiment_id": experiment_id,
        "model": model,
        "block_number": block_number,
        "sublayer": sublayer,
        "bit_width": bit_width,
        "dataset": dataset,
        "baseline_perplexity": round(baseline_perplexity, 4),
        "measured_perplexity": round(measured_perplexity, 4),
        "delta_perplexity": round(measured_perplexity - baseline_perplexity, 4),
        "baseline_accuracy": round(baseline_accuracy, 4),
        "measured_accuracy": round(measured_accuracy, 4),
        "accuracy_loss": round(baseline_accuracy - measured_accuracy, 4),
        "num_tokens_evaluated": num_tokens,
        "timestamp": datetime.now().isoformat(),
    }


# ── Save Functions ────────────────────────────────────────────────────────

def save_energy_records(run_id: str, records: list, metadata: dict):
    """Save a batch of energy records for a profiling run."""
    run_dir = _energy_root() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Save as JSON
    with open(run_dir / "energy_records.json", "w") as f:
        json.dump(records, f, indent=2, default=str)

    # Save as CSV
    with open(run_dir / "energy_records.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ENERGY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    # Save summary statistics
    summary = _compute_energy_summary(records, metadata)
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return str(run_dir)


def save_accuracy_records(run_id: str, records: list, metadata: dict):
    """Save a batch of accuracy records for a profiling run."""
    run_dir = _accuracy_root() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Save as JSON
    with open(run_dir / "accuracy_records.json", "w") as f:
        json.dump(records, f, indent=2, default=str)

    # Save as CSV
    with open(run_dir / "accuracy_records.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ACCURACY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    # Save summary
    summary = _compute_accuracy_summary(records, metadata)
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return str(run_dir)


def append_energy_record(run_id: str, record: dict):
    """Append a single energy record to an in-progress run."""
    run_dir = _energy_root() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    json_path = run_dir / "energy_records.json"
    records = []
    if json_path.exists():
        with open(json_path) as f:
            records = json.load(f)

    records.append(record)

    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, default=str)

    # Append to CSV
    csv_path = run_dir / "energy_records.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ENERGY_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def append_accuracy_record(run_id: str, record: dict):
    """Append a single accuracy record to an in-progress run."""
    run_dir = _accuracy_root() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    json_path = run_dir / "accuracy_records.json"
    records = []
    if json_path.exists():
        with open(json_path) as f:
            records = json.load(f)

    records.append(record)

    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, default=str)

    # Append to CSV
    csv_path = run_dir / "accuracy_records.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ACCURACY_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(record)


# ── Load Functions ────────────────────────────────────────────────────────

def load_energy_db(run_id: str) -> list:
    """Load all energy records for a given run."""
    json_path = _energy_root() / run_id / "energy_records.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return []


def load_accuracy_db(run_id: str) -> list:
    """Load all accuracy records for a given run."""
    json_path = _accuracy_root() / run_id / "accuracy_records.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return []


def list_profiling_runs() -> list:
    """List all profiling runs (combined energy + accuracy), newest first."""
    runs = {}

    # Scan energy runs
    energy_root = _energy_root()
    if energy_root.exists():
        for run_dir in energy_root.iterdir():
            if run_dir.is_dir():
                meta_path = run_dir / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                    meta["run_id"] = run_dir.name
                    meta["has_energy"] = True
                    meta["has_accuracy"] = False
                    runs[run_dir.name] = meta

    # Scan accuracy runs
    accuracy_root = _accuracy_root()
    if accuracy_root.exists():
        for run_dir in accuracy_root.iterdir():
            if run_dir.is_dir():
                meta_path = run_dir / "metadata.json"
                if meta_path.exists():
                    if run_dir.name in runs:
                        runs[run_dir.name]["has_accuracy"] = True
                    else:
                        with open(meta_path) as f:
                            meta = json.load(f)
                        meta["run_id"] = run_dir.name
                        meta["has_energy"] = False
                        meta["has_accuracy"] = True
                        runs[run_dir.name] = meta

    # Sort by timestamp (newest first)
    sorted_runs = sorted(
        runs.values(),
        key=lambda r: r.get("timestamp", ""),
        reverse=True
    )
    return sorted_runs


def get_profiling_run(run_id: str) -> Optional[dict]:
    """Get full data for a profiling run."""
    result = {"run_id": run_id}

    # Energy data
    energy_dir = _energy_root() / run_id
    if energy_dir.exists():
        meta_path = energy_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                result["metadata"] = json.load(f)

        summary_path = energy_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                result["energy_summary"] = json.load(f)

        result["energy_records"] = load_energy_db(run_id)

    # Accuracy data
    accuracy_dir = _accuracy_root() / run_id
    if accuracy_dir.exists():
        if "metadata" not in result:
            meta_path = accuracy_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    result["metadata"] = json.load(f)

        summary_path = accuracy_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                result["accuracy_summary"] = json.load(f)

        result["accuracy_records"] = load_accuracy_db(run_id)

    return result if "metadata" in result else None


def delete_profiling_run(run_id: str) -> bool:
    """Delete a profiling run and all its data."""
    import shutil
    deleted = False

    energy_dir = _energy_root() / run_id
    if energy_dir.exists():
        shutil.rmtree(energy_dir)
        deleted = True

    accuracy_dir = _accuracy_root() / run_id
    if accuracy_dir.exists():
        shutil.rmtree(accuracy_dir)
        deleted = True

    return deleted


# ── Summary Computation ───────────────────────────────────────────────────

def _compute_energy_summary(records: list, metadata: dict) -> dict:
    """Compute summary statistics from energy records."""
    if not records:
        return {"record_count": 0}

    def _avg(key):
        vals = [r[key] for r in records if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else 0

    def _total(key):
        vals = [r[key] for r in records if isinstance(r.get(key), (int, float))]
        return round(sum(vals), 4) if vals else 0

    # Group by sublayer type
    sublayer_types = list(set(r["sublayer"] for r in records))
    bit_widths = list(set(r["bit_width"] for r in records))
    blocks = list(set(r["block_number"] for r in records))

    return {
        "record_count": len(records),
        "model": metadata.get("model", ""),
        "blocks_profiled": len(blocks),
        "sublayer_types": sublayer_types,
        "bit_widths_tested": bit_widths,
        "avg_latency_ms": _avg("latency_avg_ms"),
        "avg_power_mw": _avg("avg_combined_mw"),
        "total_energy_joules": _total("energy_joules"),
        "avg_tokens_per_sec": _avg("tokens_per_sec"),
        "timestamp": metadata.get("timestamp", ""),
    }


def _compute_accuracy_summary(records: list, metadata: dict) -> dict:
    """Compute summary statistics from accuracy records."""
    if not records:
        return {"record_count": 0}

    def _avg(key):
        vals = [r[key] for r in records if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else 0

    sublayer_types = list(set(r["sublayer"] for r in records))
    bit_widths = list(set(r["bit_width"] for r in records))
    blocks = list(set(r["block_number"] for r in records))

    return {
        "record_count": len(records),
        "model": metadata.get("model", ""),
        "dataset": metadata.get("dataset", ""),
        "blocks_profiled": len(blocks),
        "sublayer_types": sublayer_types,
        "bit_widths_tested": bit_widths,
        "avg_delta_perplexity": _avg("delta_perplexity"),
        "avg_accuracy_loss": _avg("accuracy_loss"),
        "max_delta_perplexity": max(
            (r["delta_perplexity"] for r in records
             if isinstance(r.get("delta_perplexity"), (int, float))),
            default=0
        ),
        "timestamp": metadata.get("timestamp", ""),
    }
