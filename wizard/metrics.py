"""
Metrics collection and processing for the ANE Experiment Wizard.
Wraps macOS powermetrics and psutil to capture CPU, GPU, ANE power
and system resource usage during model inference.
"""

import subprocess
import re
import csv
import json
import time
import os
import signal
from datetime import datetime
from pathlib import Path

# ── Regex Patterns (from m4_profiler.py) ──────────────────────────────────────

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

# ── CSV Field Order ───────────────────────────────────────────────────────────

CSV_FIELDS = [
    "timestamp", "interval_sec",
    "cpu_mw", "gpu_mw", "ane_mw", "combined_mw",
    "net_cpu_mw", "net_gpu_mw", "net_ane_mw", "compute_mw",
    "cpu_joules", "gpu_joules", "ane_joules", "combined_joules",
    "net_cpu_joules", "net_gpu_joules", "net_ane_joules", "compute_joules",
    "e_cluster_pct", "p_cluster_pct",
    "gpu_active_pct", "gpu_freq_mhz",
    "thermal",
]


class MetricsCollector:
    """Collects power metrics from macOS powermetrics during an experiment."""

    def __init__(self, idle_baseline=None, ane_threshold=10.0, interval_ms=1000):
        self.idle_baseline = idle_baseline or {"cpu_mw": 18.0, "gpu_mw": 3.4, "ane_mw": 0.0}
        self.ane_threshold = ane_threshold
        self.interval_ms = interval_ms
        self.interval_sec = interval_ms / 1000.0
        self.samples = []
        self._process = None
        self._running = False

    def parse_block(self, block):
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
            return None  # incomplete block

        sample["interval_sec"] = self.interval_sec

        # Net power = inference power - idle baseline
        bl = self.idle_baseline
        sample["net_cpu_mw"] = round(max(0.0, sample.get("cpu_mw", 0) - bl["cpu_mw"]), 3)
        sample["net_gpu_mw"] = round(max(0.0, sample.get("gpu_mw", 0) - bl["gpu_mw"]), 3)
        sample["net_ane_mw"] = round(max(0.0, sample.get("ane_mw", 0) - bl["ane_mw"]), 3)
        sample["compute_mw"] = round(sample["net_ane_mw"] + sample["net_cpu_mw"], 3)

        # Energy = Power × Time (mW / 1000 × sec = Joules)
        for key_mw, key_j in [
            ("cpu_mw", "cpu_joules"), ("gpu_mw", "gpu_joules"),
            ("ane_mw", "ane_joules"), ("combined_mw", "combined_joules"),
            ("net_cpu_mw", "net_cpu_joules"), ("net_gpu_mw", "net_gpu_joules"),
            ("net_ane_mw", "net_ane_joules"), ("compute_mw", "compute_joules"),
        ]:
            if key_mw in sample:
                sample[key_j] = round(sample[key_mw] / 1000 * self.interval_sec, 6)

        return sample

    def start(self):
        """Start the powermetrics subprocess."""
        cmd = [
            "sudo", "powermetrics",
            "--samplers", "cpu_power,gpu_power,thermal",
            "-i", str(self.interval_ms),
        ]
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True
        )
        self._running = True
        self.samples = []

    def collect_one(self):
        """Read one sample from the powermetrics stream.
        Returns the sample dict, or None if no data yet."""
        if not self._process or not self._running:
            return None

        buffer = []
        for line in self._process.stdout:
            buffer.append(line)
            if "Sampled system activity" in line and len(buffer) > 1:
                block = "".join(buffer)
                sample = self.parse_block(block)
                if sample:
                    self.samples.append(sample)
                    return sample
                buffer = [line]

        return None

    def stop(self):
        """Stop the powermetrics subprocess."""
        self._running = False
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    def compute_summary(self, label="run"):
        """Compute summary statistics from collected samples.
        Averages only include samples where ANE power > 0."""
        samples = self.samples
        if not samples:
            return {"label": label, "sample_count": 0}

        def avg(key):
            vals = [s[key] for s in samples
                    if key in s and isinstance(s[key], (int, float))
                    and s.get("ane_mw", 0) > 0]
            return round(sum(vals) / len(vals), 3) if vals else None

        def total(key):
            vals = [s[key] for s in samples
                    if key in s and isinstance(s[key], (int, float))]
            return round(sum(vals), 6) if vals else None

        duration = round(sum(s.get("interval_sec", 0) for s in samples), 2)
        ane_active = sum(1 for s in samples if s.get("ane_mw", 0) > self.ane_threshold)

        return {
            "label": label,
            "chip": "Apple_M4_base_iMac",
            "sample_count": len(samples),
            "duration_sec": duration,
            "timestamp_start": samples[0]["timestamp"],
            "timestamp_end": samples[-1]["timestamp"],
            "idle_baseline_used": self.idle_baseline,

            "avg_cpu_mw": avg("cpu_mw"),
            "avg_gpu_mw": avg("gpu_mw"),
            "avg_ane_mw": avg("ane_mw"),
            "avg_combined_mw": avg("combined_mw"),
            "avg_net_cpu_mw": avg("net_cpu_mw"),
            "avg_net_gpu_mw": avg("net_gpu_mw"),
            "avg_net_ane_mw": avg("net_ane_mw"),
            "avg_compute_mw": avg("compute_mw"),

            "total_cpu_joules": total("cpu_joules"),
            "total_ane_joules": total("ane_joules"),
            "total_net_cpu_joules": total("net_cpu_joules"),
            "total_net_ane_joules": total("net_ane_joules"),
            "total_compute_joules": total("compute_joules"),
            "total_combined_joules": total("combined_joules"),

            "ane_active_samples": ane_active,
            "ane_active_pct": round(100 * ane_active / len(samples), 1) if samples else 0,
            "compute_joules_per_sec": round(
                (total("compute_joules") or 0) / duration, 6
            ) if duration > 0 else None,
        }

    def save_csv(self, filepath):
        """Save all collected samples to a CSV file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.samples)

    def save_summary(self, filepath, label="run", extra=None):
        """Save the computed summary to a JSON file."""
        summary = self.compute_summary(label)
        if extra:
            summary.update(extra)

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)

        return summary


def get_snapshot_power():
    """Take a quick one-shot power reading. Returns (cpu_mw, gpu_mw, ane_mw)."""
    try:
        cmd = ["sudo", "powermetrics", "-n", "1", "-i", "100",
               "--samplers", "cpu_power,gpu_power"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        cpu = re.search(r"CPU Power:\s+([\d.]+)\s+mW", res.stdout)
        gpu = re.search(r"GPU Power:\s+([\d.]+)\s+mW", res.stdout)
        ane = re.search(r"ANE Power:\s+([\d.]+)\s+mW", res.stdout)
        return (
            float(cpu.group(1)) if cpu else 0.0,
            float(gpu.group(1)) if gpu else 0.0,
            float(ane.group(1)) if ane else 0.0,
        )
    except Exception:
        return (0.0, 0.0, 0.0)
