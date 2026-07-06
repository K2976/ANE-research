"""
M4 iMac Power Profiler for LLM Energy Research
================================================
Captures: CPU, GPU, ANE, Package power
Computes: Net energy (minus idle baseline), joules per sample
Stores:   CSV with all metrics + summary JSON

NOTE: DRAM power not available on Apple M4 via public APIs.
Primary research metric is ANE + CPU net energy (compute energy).

Usage:
    sudo python3 m4_profiler.py --idle --duration 120
    sudo python3 m4_profiler.py --label attention_layer_4bit --duration 60
    sudo python3 m4_profiler.py --label mlp_layer_2bit --duration 60
"""

import subprocess
import re
import csv
import json
import time
import argparse
import signal
import sys
from datetime import datetime
from pathlib import Path

# ── M4 iMac Hardware Constants ────────────────────────────────────────────────

CHIP = "Apple_M4_base_iMac"

# Your measured idle baseline (captured 2026-06-25)
# These are subtracted from every inference run to get net model energy
IDLE_BASELINE = {
    "cpu_mw": 18.0,
    "gpu_mw":  3.4,
    "ane_mw":  0.0,
}

# ANE activation threshold — readings below this are considered noise
ANE_ACTIVE_THRESHOLD_MW = 10.0

# ── Regex Patterns ────────────────────────────────────────────────────────────

PATTERNS = {
    "cpu_mw":         re.compile(r"CPU Power:\s+([\d.]+)\s+mW"),
    "gpu_mw":         re.compile(r"GPU Power:\s+([\d.]+)\s+mW"),
    "ane_mw":         re.compile(r"ANE Power:\s+([\d.]+)\s+mW"),
    "combined_mw":    re.compile(r"Combined Power.*?:\s+([\d.]+)\s+mW"),
    "e_cluster_pct":  re.compile(r"E-Cluster HW active residency:\s+([\d.]+)%"),
    "p_cluster_pct":  re.compile(r"P-Cluster HW active residency:\s+([\d.]+)%"),
    "gpu_active_pct": re.compile(r"GPU HW active residency:\s+([\d.]+)%"),
    "gpu_freq_mhz":   re.compile(r"GPU HW active frequency:\s+([\d.]+)\s+MHz"),
    "thermal":        re.compile(r"Current pressure level:\s+(\w+)"),
}

# ── Block Parser ──────────────────────────────────────────────────────────────

def parse_block(block, interval_sec):
    """Parse one powermetrics sample block. Returns dict with all metrics."""
    sample = {"timestamp": datetime.now().isoformat()}

    for key, pattern in PATTERNS.items():
        match = pattern.search(block)
        if match:
            val = match.group(1)
            try:
                sample[key] = float(val)
            except ValueError:
                sample[key] = val  # thermal stays as string

    if "cpu_mw" not in sample:
        return sample  # incomplete block — skip

    sample["interval_sec"] = interval_sec

    # ── Net power = inference power - idle baseline ────────────────────────
    # This is the energy attributable to the MODEL only
    sample["net_cpu_mw"] = round(max(0.0, sample.get("cpu_mw", 0) - IDLE_BASELINE["cpu_mw"]), 3)
    sample["net_gpu_mw"] = round(max(0.0, sample.get("gpu_mw", 0) - IDLE_BASELINE["gpu_mw"]), 3)
    sample["net_ane_mw"] = round(max(0.0, sample.get("ane_mw", 0) - IDLE_BASELINE["ane_mw"]), 3)

    # Primary research metric: compute energy = ANE + CPU (net)
    sample["compute_mw"] = round(sample["net_ane_mw"] + sample["net_cpu_mw"], 3)

    # ── Energy = Power × Time  (mW / 1000 × sec = Joules) ────────────────
    sample["cpu_joules"]     = round(sample.get("cpu_mw", 0)  / 1000 * interval_sec, 6)
    sample["gpu_joules"]     = round(sample.get("gpu_mw", 0)  / 1000 * interval_sec, 6)
    sample["ane_joules"]     = round(sample.get("ane_mw", 0)  / 1000 * interval_sec, 6)
    sample["net_cpu_joules"] = round(sample["net_cpu_mw"]      / 1000 * interval_sec, 6)
    sample["net_gpu_joules"] = round(sample["net_gpu_mw"]      / 1000 * interval_sec, 6)
    sample["net_ane_joules"] = round(sample["net_ane_mw"]      / 1000 * interval_sec, 6)
    sample["compute_joules"] = round(sample["compute_mw"]       / 1000 * interval_sec, 6)

    if "combined_mw" in sample:
        sample["combined_joules"] = round(sample["combined_mw"] / 1000 * interval_sec, 6)

    return sample

# ── Live Display ──────────────────────────────────────────────────────────────

def print_sample(sample, n, cumulative_compute_j):
    cpu      = sample.get("cpu_mw",        0)
    gpu      = sample.get("gpu_mw",        0)
    ane      = sample.get("ane_mw",        0)
    net_cpu  = sample.get("net_cpu_mw",    0)
    net_ane  = sample.get("net_ane_mw",    0)
    compute  = sample.get("compute_mw",    0)
    therm    = sample.get("thermal",       "?")
    p_cl     = sample.get("p_cluster_pct", 0)

    ane_flag = "  *** ANE ACTIVE ***" if ane > ANE_ACTIVE_THRESHOLD_MW else ""

    print(
        f"[{n:>4}] "
        f"CPU:{cpu:>6.1f}mW (net:{net_cpu:>6.1f})  "
        f"ANE:{ane:>6.1f}mW (net:{net_ane:>6.1f})  "
        f"GPU:{gpu:>6.1f}mW  "
        f"Compute:{compute:>7.1f}mW  "
        f"CumE:{cumulative_compute_j:.4f}J  "
        f"P-cl:{p_cl:.1f}%  "
        f"Thermal:{therm}"
        f"{ane_flag}"
    )

# ── Summary Statistics ────────────────────────────────────────────────────────

def compute_summary(samples, label):
    def avg(key):
        vals = [s[key] for s in samples if key in s and isinstance(s[key], (int, float)) and s.get("ane_mw", 0) > 0]
        return round(sum(vals) / len(vals), 3) if vals else None

    def total(key):
        vals = [s[key] for s in samples if key in s and isinstance(s[key], (int, float))]
        return round(sum(vals), 6) if vals else None

    duration = round(sum(s.get("interval_sec", 0) for s in samples), 2)
    ane_active_count = sum(1 for s in samples if s.get("ane_mw", 0) > ANE_ACTIVE_THRESHOLD_MW)

    summary = {
        "label":           label,
        "chip":            CHIP,
        "sample_count":    len(samples),
        "duration_sec":    duration,
        "timestamp_start": samples[0]["timestamp"] if samples else None,
        "timestamp_end":   samples[-1]["timestamp"] if samples else None,

        "idle_baseline_used": IDLE_BASELINE,
        "dram_note": "DRAM power not exposed on Apple M4. Not included in metrics.",

        # Raw averages
        "avg_cpu_mw":      avg("cpu_mw"),
        "avg_gpu_mw":      avg("gpu_mw"),
        "avg_ane_mw":      avg("ane_mw"),
        "avg_combined_mw": avg("combined_mw"),

        # Net averages (model only, idle subtracted)
        "avg_net_cpu_mw":  avg("net_cpu_mw"),
        "avg_net_gpu_mw":  avg("net_gpu_mw"),
        "avg_net_ane_mw":  avg("net_ane_mw"),
        "avg_compute_mw":  avg("compute_mw"),   # PRIMARY METRIC

        # Total energy (joules)
        "total_cpu_joules":     total("cpu_joules"),
        "total_ane_joules":     total("ane_joules"),
        "total_net_cpu_joules": total("net_cpu_joules"),
        "total_net_ane_joules": total("net_ane_joules"),
        "total_compute_joules": total("compute_joules"),  # PRIMARY METRIC
        "total_combined_joules":total("combined_joules"),

        # ANE utilization stats
        "ane_active_samples":   ane_active_count,
        "ane_active_pct":       round(100 * ane_active_count / len(samples), 1) if samples else 0,

        # Per-second compute energy rate
        "compute_joules_per_sec": round(
            (total("compute_joules") or 0) / duration, 6
        ) if duration > 0 else None,
    }

    return summary

# ── Main Capture Loop ─────────────────────────────────────────────────────────

def capture(duration_seconds=60, interval_ms=1000, label="run", output_dir="./power_logs"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = output_dir / f"{label}_{ts}.csv"
    json_path = output_dir / f"{label}_{ts}_summary.json"

    interval_sec = interval_ms / 1000.0

    print(f"\n{'='*80}")
    print(f"  M4 iMac Power Profiler  |  {label}")
    print(f"  Duration: {duration_seconds}s   Interval: {interval_ms}ms")
    print(f"  Idle baseline: CPU={IDLE_BASELINE['cpu_mw']}mW  "
          f"GPU={IDLE_BASELINE['gpu_mw']}mW  ANE={IDLE_BASELINE['ane_mw']}mW")
    print(f"  Primary metric: compute_mw = net_ANE + net_CPU")
    print(f"  Output: {csv_path}")
    print(f"{'='*80}\n")

    cmd = [
        "sudo", "powermetrics",
        "--samplers", "cpu_power,gpu_power,thermal",
        "-i", str(interval_ms),
    ]

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True
    )

    samples              = []
    buffer               = []
    start                = time.time()
    n                    = 0
    cumulative_compute_j = 0.0

    def shutdown(sig, frame):
        print("\n\nInterrupted — saving partial data...")
        process.terminate()
        finalize(samples, label, csv_path, json_path)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    for line in process.stdout:
        buffer.append(line)

        if "Sampled system activity" in line and len(buffer) > 1:
            block  = "".join(buffer)
            sample = parse_block(block, interval_sec)

            if "cpu_mw" in sample:
                n += 1
                cumulative_compute_j += sample.get("compute_joules", 0)
                samples.append(sample)
                print_sample(sample, n, cumulative_compute_j)

            buffer = [line]

        if time.time() - start > duration_seconds:
            break

    process.terminate()
    finalize(samples, label, csv_path, json_path)
    return samples, str(csv_path), str(json_path)

# ── Save Results ──────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "timestamp", "interval_sec",
    # Raw power
    "cpu_mw", "gpu_mw", "ane_mw", "combined_mw",
    # Net power (model only)
    "net_cpu_mw", "net_gpu_mw", "net_ane_mw", "compute_mw",
    # Raw energy
    "cpu_joules", "gpu_joules", "ane_joules", "combined_joules",
    # Net energy (model only) — use these in your paper
    "net_cpu_joules", "net_gpu_joules", "net_ane_joules", "compute_joules",
    # CPU utilization
    "e_cluster_pct", "p_cluster_pct",
    # GPU
    "gpu_active_pct", "gpu_freq_mhz",
    # System health
    "thermal",
]

def finalize(samples, label, csv_path, json_path):
    if not samples:
        print("No samples captured.")
        return

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(samples)

    summary = compute_summary(samples, label)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"  RESULTS — {label}")
    print(f"{'='*80}")
    print(f"  Samples          : {summary['sample_count']}  ({summary['duration_sec']}s)")
    print(f"")
    print(f"  RAW POWER")
    print(f"    Avg CPU        : {summary['avg_cpu_mw']} mW")
    print(f"    Avg GPU        : {summary['avg_gpu_mw']} mW")
    print(f"    Avg ANE        : {summary['avg_ane_mw']} mW")
    print(f"")
    print(f"  NET POWER (model only, idle subtracted)")
    print(f"    Net CPU        : {summary['avg_net_cpu_mw']} mW")
    print(f"    Net ANE        : {summary['avg_net_ane_mw']} mW")
    print(f"    Compute (ANE+CPU): {summary['avg_compute_mw']} mW  <-- PRIMARY METRIC")
    print(f"")
    print(f"  ENERGY")
    print(f"    Total compute  : {summary['total_compute_joules']} J  <-- USE IN PAPER")
    print(f"    Compute J/sec  : {summary['compute_joules_per_sec']} J/s")
    print(f"")
    print(f"  ANE UTILIZATION")
    print(f"    ANE active     : {summary['ane_active_pct']}% of samples")
    print(f"    ANE note       : {'*** ANE WAS ACTIVE - CoreML working ***' if summary['ane_active_pct'] > 0 else 'ANE idle - model not using CoreML'}")
    print(f"")
    print(f"  Saved CSV  → {csv_path}")
    print(f"  Saved JSON → {json_path}")
    print(f"{'='*80}\n")

# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M4 iMac Power Profiler for LLM Research")
    parser.add_argument("--duration", type=int, default=60,
                        help="Capture duration in seconds (default: 60)")
    parser.add_argument("--interval", type=int, default=1000,
                        help="Sample interval in ms (default: 1000)")
    parser.add_argument("--label",    type=str, default="run",
                        help="Label e.g. 'attention_layer_4bit'")
    parser.add_argument("--output",   type=str, default="./power_logs",
                        help="Output directory (default: ./power_logs)")
    parser.add_argument("--idle",     action="store_true",
                        help="Capture idle baseline (ignores --label)")

    args = parser.parse_args()

    if args.idle:
        args.label = "idle_baseline"
        print("\nCapturing IDLE BASELINE")
        print("Make sure: Brave is closed, only Terminal is open.")
        print("Starting in 5 seconds...\n")
        time.sleep(5)

    capture(
        duration_seconds=args.duration,
        interval_ms=args.interval,
        label=args.label,
        output_dir=args.output,
    )
