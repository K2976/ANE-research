"""
Layer Profiling Report Generator
==================================
Generates publication-quality reports from profiling data in
multiple formats: CSV, JSON, Markdown, and LaTeX.
"""

import os
import json
import csv
import io
from datetime import datetime
from pathlib import Path


def generate_csv(energy_records: list, accuracy_records: list) -> str:
    """Generate a combined CSV report."""
    output = io.StringIO()

    # Energy section
    output.write("# ENERGY DATABASE E(i,b)\n")
    if energy_records:
        writer = csv.DictWriter(output, fieldnames=list(energy_records[0].keys()),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(energy_records)

    output.write("\n\n# ACCURACY DATABASE ΔA(i,b)\n")
    if accuracy_records:
        writer = csv.DictWriter(output, fieldnames=list(accuracy_records[0].keys()),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(accuracy_records)

    return output.getvalue()


def generate_json(energy_records: list, accuracy_records: list,
                  metadata: dict = None) -> str:
    """Generate a structured JSON report."""
    report = {
        "report_type": "layer_profiling",
        "generated_at": datetime.now().isoformat(),
        "metadata": metadata or {},
        "energy_database": {
            "description": "E(i,b) — Energy per sub-layer per bit-width",
            "record_count": len(energy_records),
            "records": energy_records,
        },
        "accuracy_database": {
            "description": "ΔA(i,b) — Accuracy delta per sub-layer per bit-width",
            "record_count": len(accuracy_records),
            "records": accuracy_records,
        },
    }
    return json.dumps(report, indent=2, default=str)


def generate_markdown(energy_records: list, accuracy_records: list,
                      metadata: dict = None) -> str:
    """Generate a Markdown research report."""
    meta = metadata or {}
    lines = []

    lines.append(f"# Layer Profiling Report")
    lines.append(f"")
    lines.append(f"**Model**: {meta.get('model', 'N/A')}")
    lines.append(f"**Date**: {meta.get('timestamp', datetime.now().isoformat())}")
    lines.append(f"**Dataset**: {meta.get('dataset', 'N/A')}")
    lines.append(f"**Profiling Duration**: {meta.get('duration_sec', 'N/A')}s per sub-layer")
    lines.append(f"**Hardware**: Apple M4 (ANE)")
    lines.append(f"")

    # Energy table
    lines.append("## Energy Database E(i,b)")
    lines.append("")
    if energy_records:
        lines.append("| Block | Sub-layer | Bit Width | Latency (ms) | Power (mW) | Energy (J) | Tok/s |")
        lines.append("|-------|-----------|-----------|-------------|-----------|-----------|-------|")
        for r in energy_records:
            lines.append(
                f"| {r.get('block_number', '')} "
                f"| {r.get('sublayer', '')} "
                f"| {r.get('bit_width', '')} "
                f"| {r.get('latency_avg_ms', 0):.1f} "
                f"| {r.get('avg_combined_mw', 0):.1f} "
                f"| {r.get('energy_joules', 0):.4f} "
                f"| {r.get('tokens_per_sec', 0):.1f} |"
            )
    else:
        lines.append("*No energy data collected*")

    lines.append("")

    # Accuracy table
    lines.append("## Accuracy Database ΔA(i,b)")
    lines.append("")
    if accuracy_records:
        lines.append("| Block | Sub-layer | Bit Width | Baseline PPL | Measured PPL | ΔPPL | Acc Loss |")
        lines.append("|-------|-----------|-----------|-------------|-------------|------|----------|")
        for r in accuracy_records:
            lines.append(
                f"| {r.get('block_number', '')} "
                f"| {r.get('sublayer', '')} "
                f"| {r.get('bit_width', '')} "
                f"| {r.get('baseline_perplexity', 0):.2f} "
                f"| {r.get('measured_perplexity', 0):.2f} "
                f"| {r.get('delta_perplexity', 0):.2f} "
                f"| {r.get('accuracy_loss', 0):.4f} |"
            )
    else:
        lines.append("*No accuracy data collected*")

    lines.append("")

    # Summary statistics
    lines.append("## Summary Statistics")
    lines.append("")

    if energy_records:
        avg_energy = sum(r.get("energy_joules", 0) for r in energy_records) / len(energy_records)
        avg_power = sum(r.get("avg_combined_mw", 0) for r in energy_records) / len(energy_records)
        avg_tps = sum(r.get("tokens_per_sec", 0) for r in energy_records) / len(energy_records)
        lines.append(f"- **Average Energy**: {avg_energy:.4f} J")
        lines.append(f"- **Average Power**: {avg_power:.1f} mW")
        lines.append(f"- **Average Throughput**: {avg_tps:.1f} tok/s")
        lines.append(f"- **Total Energy Records**: {len(energy_records)}")

    if accuracy_records:
        avg_delta = sum(r.get("delta_perplexity", 0) for r in accuracy_records) / len(accuracy_records)
        max_delta = max(r.get("delta_perplexity", 0) for r in accuracy_records)
        lines.append(f"- **Average ΔPPL**: {avg_delta:.4f}")
        lines.append(f"- **Max ΔPPL**: {max_delta:.4f}")
        lines.append(f"- **Total Accuracy Records**: {len(accuracy_records)}")

    lines.append("")
    lines.append("---")
    lines.append(f"*Generated by ANE Experiment Wizard — Layer Profiling Module*")

    return "\n".join(lines)


def generate_latex(energy_records: list, accuracy_records: list,
                   metadata: dict = None) -> str:
    """Generate LaTeX tables ready for \\input{} in a paper."""
    meta = metadata or {}
    lines = []

    lines.append(f"% Layer Profiling Results — {meta.get('model', 'Model')}")
    lines.append(f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"% Hardware: Apple M4 (ANE)")
    lines.append("")

    # Energy table
    if energy_records:
        lines.append("% ── Energy Database E(i,b) ──────────────────────────────────")
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append(f"\\caption{{Energy profiling results for {meta.get('model', 'model')}}}")
        lines.append(f"\\label{{tab:energy_profiling}}")
        lines.append("\\begin{tabular}{cccrrrr}")
        lines.append("\\toprule")
        lines.append("Block & Sub-layer & Bits & Latency (ms) & Power (mW) & Energy (J) & Tok/s \\\\")
        lines.append("\\midrule")

        for r in energy_records:
            lines.append(
                f"{r.get('block_number', '')} & "
                f"{r.get('sublayer', '')} & "
                f"{r.get('bit_width', '')} & "
                f"{r.get('latency_avg_ms', 0):.1f} & "
                f"{r.get('avg_combined_mw', 0):.1f} & "
                f"{r.get('energy_joules', 0):.4f} & "
                f"{r.get('tokens_per_sec', 0):.1f} \\\\"
            )

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")

    # Accuracy table
    if accuracy_records:
        lines.append("% ── Accuracy Database ΔA(i,b) ───────────────────────────────")
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append(f"\\caption{{Accuracy degradation (\\(\\Delta A\\)) for {meta.get('model', 'model')}}}")
        lines.append(f"\\label{{tab:accuracy_profiling}}")
        lines.append("\\begin{tabular}{cccrrrrr}")
        lines.append("\\toprule")
        lines.append("Block & Sub-layer & Bits & PPL\\textsubscript{base} & PPL\\textsubscript{quant} & $\\Delta$PPL & Acc Loss \\\\")
        lines.append("\\midrule")

        for r in accuracy_records:
            lines.append(
                f"{r.get('block_number', '')} & "
                f"{r.get('sublayer', '')} & "
                f"{r.get('bit_width', '')} & "
                f"{r.get('baseline_perplexity', 0):.2f} & "
                f"{r.get('measured_perplexity', 0):.2f} & "
                f"{r.get('delta_perplexity', 0):.2f} & "
                f"{r.get('accuracy_loss', 0):.4f} \\\\"
            )

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

    return "\n".join(lines)


def export_profiling_run(run_id: str, fmt: str = "json") -> str:
    """Export a profiling run in the specified format.

    Args:
        run_id: The profiling run identifier
        fmt: One of 'json', 'csv', 'markdown', 'latex'

    Returns:
        The formatted report content as a string
    """
    from . import layer_db

    run = layer_db.get_profiling_run(run_id)
    if not run:
        return ""

    energy = run.get("energy_records", [])
    accuracy = run.get("accuracy_records", [])
    metadata = run.get("metadata", {})

    if fmt == "json":
        return generate_json(energy, accuracy, metadata)
    elif fmt == "csv":
        return generate_csv(energy, accuracy)
    elif fmt == "markdown":
        return generate_markdown(energy, accuracy, metadata)
    elif fmt == "latex":
        return generate_latex(energy, accuracy, metadata)
    else:
        return generate_json(energy, accuracy, metadata)
