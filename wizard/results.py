"""
Results management and comparison for the ANE Experiment Wizard.
Loads past experiment results, generates comparisons, and exports reports.
"""

import os
import json
import csv
import glob
from pathlib import Path
from datetime import datetime


def get_output_dir():
    """Get the default output directory."""
    return Path(os.path.dirname(__file__)) / "experiment_results"


def list_experiments(output_dir=None):
    """List all saved experiments, sorted by date (newest first).
    Returns list of dicts with experiment metadata."""
    output_dir = Path(output_dir) if output_dir else get_output_dir()
    experiments = []

    if not output_dir.exists():
        return experiments

    # Each experiment is a subdirectory with a metadata.json
    for exp_dir in sorted(output_dir.iterdir(), reverse=True):
        if not exp_dir.is_dir():
            continue

        meta_path = exp_dir / "metadata.json"
        if not meta_path.exists():
            continue

        try:
            with open(meta_path) as f:
                meta = json.load(f)
            meta["dir"] = str(exp_dir)
            meta["dir_name"] = exp_dir.name
            experiments.append(meta)
        except (json.JSONDecodeError, IOError):
            continue

    return experiments


def get_experiment(experiment_id, output_dir=None):
    """Get full experiment data by ID (directory name)."""
    output_dir = Path(output_dir) if output_dir else get_output_dir()
    exp_dir = output_dir / experiment_id

    if not exp_dir.exists():
        return None

    result = {"id": experiment_id, "dir": str(exp_dir)}

    # Load metadata
    meta_path = exp_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            result["metadata"] = json.load(f)

    # Load summary
    summary_path = exp_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            result["summary"] = json.load(f)

    # Load power samples
    csv_path = exp_dir / "power_samples.csv"
    if csv_path.exists():
        result["power_samples"] = _load_csv(csv_path)

    # Load inference results
    inference_path = exp_dir / "inference_results.json"
    if inference_path.exists():
        with open(inference_path) as f:
            result["inference"] = json.load(f)

    # Load notes
    notes_path = exp_dir / "notes.txt"
    if notes_path.exists():
        with open(notes_path) as f:
            result["notes"] = f.read()

    return result


def save_experiment(experiment_id, metadata, summary, power_samples,
                    inference_results=None, notes="", output_dir=None):
    """Save a complete experiment to disk."""
    output_dir = Path(output_dir) if output_dir else get_output_dir()
    exp_dir = output_dir / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    with open(exp_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save summary
    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save power samples as CSV
    if power_samples:
        from .metrics import CSV_FIELDS
        with open(exp_dir / "power_samples.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(power_samples)

    # Save inference results
    if inference_results:
        with open(exp_dir / "inference_results.json", "w") as f:
            json.dump(inference_results, f, indent=2)

    # Save notes
    if notes:
        with open(exp_dir / "notes.txt", "w") as f:
            f.write(notes)

    return str(exp_dir)


def save_notes(experiment_id, notes, output_dir=None):
    """Save or update notes for an experiment."""
    output_dir = Path(output_dir) if output_dir else get_output_dir()
    exp_dir = output_dir / experiment_id

    if not exp_dir.exists():
        return False

    with open(exp_dir / "notes.txt", "w") as f:
        f.write(notes)

    # Also update metadata
    meta_path = exp_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta["has_notes"] = bool(notes.strip())
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    return True


def compare_experiments(exp_id_1, exp_id_2, output_dir=None):
    """Compare two experiments and return comparison data."""
    exp1 = get_experiment(exp_id_1, output_dir)
    exp2 = get_experiment(exp_id_2, output_dir)

    if not exp1 or not exp2:
        return None

    s1 = exp1.get("summary", {})
    s2 = exp2.get("summary", {})

    # Compare key metrics
    compare_keys = [
        ("avg_cpu_mw", "Avg CPU Power (mW)"),
        ("avg_gpu_mw", "Avg GPU Power (mW)"),
        ("avg_ane_mw", "Avg ANE Power (mW)"),
        ("avg_compute_mw", "Avg Compute Power (mW)"),
        ("total_compute_joules", "Total Compute Energy (J)"),
        ("ane_active_pct", "ANE Active (%)"),
        ("compute_joules_per_sec", "Compute J/sec"),
    ]

    comparison = {
        "experiment_1": {
            "id": exp_id_1,
            "metadata": exp1.get("metadata", {}),
        },
        "experiment_2": {
            "id": exp_id_2,
            "metadata": exp2.get("metadata", {}),
        },
        "metrics": [],
    }

    for key, label in compare_keys:
        v1 = s1.get(key)
        v2 = s2.get(key)
        diff = None
        pct_change = None

        if v1 is not None and v2 is not None and v1 != 0:
            diff = round(v2 - v1, 3)
            pct_change = round(((v2 - v1) / abs(v1)) * 100, 1)

        comparison["metrics"].append({
            "key": key,
            "label": label,
            "value_1": v1,
            "value_2": v2,
            "diff": diff,
            "pct_change": pct_change,
        })

    # Include time series for charts
    if "power_samples" in exp1:
        comparison["experiment_1"]["time_series"] = _extract_time_series(exp1["power_samples"])
    if "power_samples" in exp2:
        comparison["experiment_2"]["time_series"] = _extract_time_series(exp2["power_samples"])

    return comparison


def _extract_time_series(samples):
    """Extract time series data for charting from CSV samples."""
    series = {"cpu_mw": [], "ane_mw": [], "gpu_mw": [], "compute_mw": []}
    for i, s in enumerate(samples):
        for key in series:
            try:
                series[key].append(float(s.get(key, 0)))
            except (ValueError, TypeError):
                series[key].append(0)
    return series


def _load_csv(filepath):
    """Load a CSV file into a list of dicts."""
    with open(filepath) as f:
        return list(csv.DictReader(f))


def export_report(experiment_id, format="json", output_dir=None):
    """Export an experiment report in the specified format."""
    exp = get_experiment(experiment_id, output_dir)
    if not exp:
        return None

    if format == "json":
        return json.dumps(exp, indent=2, default=str)
    elif format == "csv":
        # Return the power samples CSV content
        output_dir = Path(output_dir) if output_dir else get_output_dir()
        csv_path = output_dir / experiment_id / "power_samples.csv"
        if csv_path.exists():
            return csv_path.read_text()
        return ""

    return None
