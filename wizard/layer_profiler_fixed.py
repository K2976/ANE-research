"""
Layer Profiling Engine
=======================
Core engine for profiling individual sub-layers (Attention, MLP, LayerNorm)
of transformer models at different bit widths. Generates the empirical
Energy database E(i,b) and Accuracy database ΔA(i,b) required for
mixed-precision quantization research (ILP / MCKP optimization).

This module is completely independent from the existing ExperimentEngine.
It reuses the existing MetricsCollector for power measurement and the
existing model loading infrastructure.
"""

import os
import sys
import gc
import copy
import time
import math
import threading
import numpy as np
from datetime import datetime
from pathlib import Path

ANEFORGE_PATH = "/Users/kartik/Documents/ANEForge-main"
if ANEFORGE_PATH not in sys.path:
    sys.path.insert(0, ANEFORGE_PATH)

from . import models as model_mgr
from . import metrics as metrics_mod
from . import layer_db
from . import layer_accuracy
from . import cache_manager

# ── Constants ─────────────────────────────────────────────────────────────
COMPILE_TIMEOUT_SEC = 90  # 90 seconds max compile time before circuit breaker

# int4/int2 produce massive ANE graphs; cap the sequence length to prevent hangs
MAX_LEN_CAPS = {
    "int4": 64,
    "int2": 64,
}


def _release_model(model):
    """Explicitly release ANE programs so we don't saturate the driver."""
    try:
        if hasattr(model, '_q_model') and hasattr(model._q_model, '_decode_chunks'):
            for chunk in model._q_model._decode_chunks:
                if hasattr(chunk, 'program'): chunk.program.release()
        if hasattr(model, '_q_model') and hasattr(model._q_model, '_prefill_chunks'):
            for chunk in model._q_model._prefill_chunks:
                if hasattr(chunk, 'program'): chunk.program.release()
        if hasattr(model, 'net') and hasattr(model.net, 'program'):
            model.net.program.release()
    except Exception as e:
        print(f"Warning: Failed to release ANE program handles: {e}")


def _warmup_with_timeout(q_model, max_len):
    """Runs ANE warmup with a strict timeout to prevent driver hang."""
    import _thread
    
    exc_list = []
    
    def target():
        try:
            q_model.warmup(max_len)
        except Exception as e:
            exc_list.append(e)
            
    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=COMPILE_TIMEOUT_SEC)
    
    if t.is_alive():
        # The compile is hanging, we must raise
        raise TimeoutError("ANE Compilation timed out")
    if exc_list:
        raise exc_list[0]

# ── Sub-layer Definitions ─────────────────────────────────────────────────

# Which weight keys belong to each sub-layer type
SUBLAYER_WEIGHTS = {
    "attention": ["wq", "wk", "wv", "wo"],
    "mlp": ["wgate", "wup", "wdown"],
    "layernorm": ["attn_norm", "mlp_norm"],
}

# All sub-layer types in profiling order
SUBLAYER_TYPES = ["attention", "mlp", "layernorm"]


# ── Quantization Simulation ──────────────────────────────────────────────

def _simulate_quantize(weight: np.ndarray, bit_width: str) -> np.ndarray:
    """Simulate quantization of a weight tensor to a target bit width.

    This applies uniform quantization to simulate the accuracy effect.
    The actual ANE compilation still uses ANEForge's native compression.

    Args:
        weight: The original FP32/FP16 weight array
        bit_width: 'fp16' | 'int8' | 'int4' | 'int2'

    Returns:
        Quantized weight array (same dtype, but values are quantized)
    """
    if bit_width == "fp16":
        return weight  # No quantization

    w = weight.astype(np.float32)
    w_min, w_max = w.min(), w.max()

    if w_max == w_min:
        return weight  # Constant tensor

    bits_map = {"int8": 8, "int4": 4, "int2": 2}
    bits = bits_map.get(bit_width, 8)
    n_levels = (1 << bits) - 1  # e.g., 255 for int8

    # Uniform quantization: scale → round → dequantize
    scale = (w_max - w_min) / n_levels
    quantized = np.round((w - w_min) / scale) * scale + w_min

    return quantized.astype(weight.dtype)


def _selectively_quantize_weights(weights: dict, block_idx: int,
                                  sublayer: str, bit_width: str) -> dict:
    """Create a deep copy of the weight dict with only one sub-layer quantized.

    All other layers remain in their original precision (FP16).

    Args:
        weights: The full model weight dict {"embed", "final_norm", "lm_head", "layers": [...]}
        block_idx: Which transformer block to quantize
        sublayer: Which sub-layer type ('attention', 'mlp', 'layernorm')
        bit_width: Target bit width

    Returns:
        New weight dict with selective quantization applied
    """
    if bit_width == "fp16":
        return weights  # Nothing to change

    # Deep copy only the target block's weights to avoid modifying originals
    new_weights = {
        "embed": weights["embed"],
        "final_norm": weights["final_norm"],
        "lm_head": weights["lm_head"],
        "layers": [],
    }

    for i, lw in enumerate(weights["layers"]):
        if i == block_idx:
            # Deep copy this block and quantize the target sub-layer
            new_lw = {}
            keys_to_quantize = set(SUBLAYER_WEIGHTS.get(sublayer, []))
            for k, v in lw.items():
                if k in keys_to_quantize:
                    new_lw[k] = _simulate_quantize(np.array(v), bit_width)
                else:
                    new_lw[k] = v
            new_weights["layers"].append(new_lw)
        else:
            new_weights["layers"].append(lw)

    return new_weights


# ── Model Introspection ──────────────────────────────────────────────────

def detect_model_structure(model_id: str) -> dict:
    """Detect the transformer block structure of a model.

    Returns:
        {
            "model_id": str,
            "n_layers": int,
            "blocks": [
                {"block_number": 0, "sublayers": ["attention", "mlp", "layernorm"]},
                ...
            ],
            "total_sublayers": int,
        }
    """
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_id)
        n_layers = getattr(config, "num_hidden_layers", 0)
    except Exception:
        # Fallback: try to load and count from weights
        n_layers = 0

    blocks = []
    for i in range(n_layers):
        blocks.append({
            "block_number": i,
            "sublayers": list(SUBLAYER_TYPES),
        })

    return {
        "model_id": model_id,
        "n_layers": n_layers,
        "blocks": blocks,
        "total_sublayers": n_layers * len(SUBLAYER_TYPES),
    }


# ── Profiling Engine ─────────────────────────────────────────────────────

class LayerProfilingEngine:
    """Manages the lifecycle of a layer profiling experiment.

    State machine: IDLE → RUNNING → PAUSED → RUNNING → COMPLETED/CANCELLED

    This engine is completely independent from ExperimentEngine.
    It reuses MetricsCollector for power measurement and the model
    loading infrastructure from models.py.
    """

    # States
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"

    def __init__(self, emit_fn=None):
        self.emit = emit_fn or (lambda event, data: None)
        self.state = self.IDLE
        self._thread = None
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially
        self._cancel_flag = False

        # Progress tracking
        self.current_block = 0
        self.current_sublayer = ""
        self.current_bitwidth = ""
        self.total_steps = 0
        self.completed_steps = 0
        self.energy_records = []
        self.accuracy_records = []
        self.start_time = None

    def _emit(self, event, data):
        try:
            self.emit(event, data)
        except Exception:
            pass

    @property
    def running(self):
        return self.state in (self.RUNNING, self.PAUSED)

    def start(self, config: dict):
        """Start a layer profiling experiment in a background thread."""
        if self.running:
            return False

        self._cancel_flag = False
        self._pause_event.set()
        self.energy_records = []
        self.accuracy_records = []
        self.completed_steps = 0

        self._thread = threading.Thread(target=self._run, args=(config,), daemon=True)
        self._thread.start()
        return True

    def pause(self):
        """Pause the profiling run."""
        if self.state == self.RUNNING:
            self.state = self.PAUSED
            self._pause_event.clear()
            self._emit("profiling_paused", {})

    def resume(self):
        """Resume a paused profiling run."""
        if self.state == self.PAUSED:
            self.state = self.RUNNING
            self._pause_event.set()
            self._emit("profiling_resumed", {})

    def cancel(self):
        """Cancel the profiling run."""
        self._cancel_flag = True
        self._pause_event.set()  # Unblock if paused

    def _check_cancelled(self):
        """Check if cancelled, emit event if so. Returns True if cancelled."""
        if self._cancel_flag:
            self.state = self.CANCELLED
            self._emit("profiling_cancelled", {})
            return True
        return False

    def _wait_if_paused(self):
        """Block if paused. Returns True if cancelled during pause."""
        self._pause_event.wait()
        return self._check_cancelled()

    # ── Main Execution Loop ───────────────────────────────────────────────

    def _run(self, config):
        """The main profiling execution sequence."""
        self.state = self.RUNNING
        self.start_time = time.time()
        run_id = None

        try:
            # ── Parse config ──────────────────────────────────────────
            model_id = config["model_id"]
            profiling_mode = config.get("profiling_mode", "entire_model")
            sublayer_filter = config.get("sublayer_type", None)  # For single mode
            bit_widths = config.get("bit_widths", ["int8"])
            if isinstance(bit_widths, str):
                bit_widths = [bit_widths]
            dataset_name = config.get("dataset", "wikitext2")
            duration_sec = config.get("duration_sec", 30)
            warmup_runs = config.get("warmup_runs", 1)
            benchmark_runs = config.get("benchmark_runs", 1)
            collect_accuracy = config.get("collect_accuracy", True)
            collect_energy = config.get("collect_energy", True)
            max_len = config.get("max_len", 512)

            # Generate run ID
            model_short = model_id.split("/")[-1]
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_id = f"profile_{model_short}_{ts}"

            # ── Detect model structure ────────────────────────────────
            self._emit("profiling_progress", {
                "step": 0, "total": 1,
                "message": "Detecting model structure...",
                "phase": "setup",
            })

            structure = detect_model_structure(model_id)
            n_layers = structure["n_layers"]

            if n_layers == 0:
                self._emit("profiling_error", {
                    "message": f"Could not detect layers for {model_id}"
                })
                self.state = self.ERROR
                return

            # ── Build profiling schedule ──────────────────────────────
            schedule = []
            if profiling_mode == "single_sublayer":
                # Only profile the selected sub-layer type across all blocks
                for bw in bit_widths:
                    for block_idx in range(n_layers):
                        schedule.append((bw, sublayer_filter, block_idx))
            else:
                # Profile all sub-layers across all blocks
                for bw in bit_widths:
                    for sl in SUBLAYER_TYPES:
                        for block_idx in range(n_layers):
                            schedule.append((bw, sl, block_idx))

            self.total_steps = len(schedule)

            self._emit("profiling_progress", {
                "step": 0, "total": self.total_steps,
                "message": f"Profiling {len(schedule)} combinations across {n_layers} blocks",
                "phase": "setup",
                "n_layers": n_layers,
                "schedule_size": len(schedule),
            })

            if self._check_cancelled():
                return

            # ── Unload any existing model ─────────────────────────────
            from .engine import ExperimentEngine
            ExperimentEngine.unload_model()
            gc.collect()

            self._emit("profiling_progress", {
                "step": 0, "total": self.total_steps,
                "message": f"Loading {model_id} weights...",
                "phase": "loading",
            })

            # ── Load raw weights (not compiled) ───────────────────────
            import aneforge as af
            from aneforge.llm import from_pretrained, LlamaPrefill, _dense_adapter
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

            # Download if needed
            if not model_mgr.is_model_downloaded(model_id):
                self._emit("profiling_progress", {
                    "step": 0, "total": self.total_steps,
                    "message": f"Downloading {model_id}...",
                    "phase": "download",
                })
                success, msg = model_mgr.download_model(model_id)
                if not success:
                    self._emit("profiling_error", {"message": msg})
                    self.state = self.ERROR
                    return

            # Load HuggingFace model to extract weights
            self._emit("profiling_progress", {
                "step": 0, "total": self.total_steps,
                "message": "Loading model weights into memory...",
                "phase": "loading",
            })

            hf_model = AutoModelForCausalLM.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            sd = {k: v.detach().float().numpy() for k, v in hf_model.state_dict().items()}
            cfg, base_weights = _dense_adapter(hf_model.config, sd)

            # Free the HF model
            del hf_model
            gc.collect()

            if self._check_cancelled():
                return

            # ── Load evaluation dataset ───────────────────────────────
            eval_texts = []
            if collect_accuracy:
                try:
                    cached_tokens = cache_manager.load_tokenized_dataset(model_id, dataset_name)
                    if cached_tokens:
                        eval_texts = cached_tokens
                        self._emit("profiling_progress", {
                            "step": 0, "total": self.total_steps,
                            "message": f"Loaded {len(eval_texts)} tokenized texts from cache",
                            "phase": "setup",
                        })
                    else:
                        raw_texts = layer_accuracy.load_dataset(dataset_name)
                        self._emit("profiling_progress", {
                            "step": 0, "total": self.total_steps,
                            "message": f"Tokenizing {len(raw_texts)} evaluation texts...",
                            "phase": "setup",
                        })
                        
                        eval_texts = []
                        for t in raw_texts:
                            try:
                                ids = tokenizer.encode(t)
                                if len(ids) >= 2:
                                    eval_texts.append(ids)
                            except Exception:
                                pass
                        
                        cache_manager.save_tokenized_dataset(model_id, dataset_name, eval_texts)
                except Exception as e:
                    self._emit("profiling_progress", {
                        "step": 0, "total": self.total_steps,
                        "message": f"Warning: Could not load dataset {dataset_name}: {e}",
                        "phase": "setup",
                    })
                    collect_accuracy = False

            # ── Compute baseline perplexity (once) ────────────────────
            baseline_metrics = None
            if collect_accuracy and eval_texts:
                baseline_metrics = cache_manager.load_baseline(model_id, dataset_name, max_len)
                if baseline_metrics:
                    self._emit("profiling_progress", {
                        "step": 0, "total": self.total_steps,
                        "message": f"Loaded cached baseline: {baseline_metrics['perplexity']:.2f}",
                        "phase": "baseline",
                    })
                else:
                    self._emit("profiling_progress", {
                        "step": 0, "total": self.total_steps,
                        "message": "Computing FP16 baseline perplexity...",
                        "phase": "baseline",
                    })

                    # Build FP16 baseline model
                    baseline_model = LlamaPrefill(cfg, base_weights, compress=None)
                    _warmup_with_timeout(baseline_model, max_len)

                    baseline_metrics = layer_accuracy.compute_perplexity(
                        baseline_model, tokenizer, eval_texts, max_len,
                        on_progress=lambda msg: self._emit("profiling_progress", {
                            "step": 0, "total": self.total_steps,
                            "message": msg, "phase": "baseline",
                        })
                    )
                    
                    cache_manager.save_baseline(model_id, dataset_name, max_len, baseline_metrics)

                    # Free baseline model
                    _release_model(baseline_model)
                    del baseline_model
                    gc.collect()

                self._emit("profiling_progress", {
                    "step": 0, "total": self.total_steps,
                    "message": f"Baseline perplexity: {baseline_metrics['perplexity']:.2f}",
                    "phase": "baseline",
                })

            if self._check_cancelled():
                return

            # ── Metadata ──────────────────────────────────────────────
            metadata = {
                "run_id": run_id,
                "model": model_id,
                "profiling_mode": profiling_mode,
                "bit_widths": bit_widths,
                "dataset": dataset_name,
                "duration_sec": duration_sec,
                "warmup_runs": warmup_runs,
                "benchmark_runs": benchmark_runs,
                "collect_energy": collect_energy,
                "collect_accuracy": collect_accuracy,
                "n_layers": n_layers,
                "schedule_size": len(schedule),
                "max_len": max_len,
                "timestamp": datetime.now().isoformat(),
                "baseline_perplexity": baseline_metrics["perplexity"] if baseline_metrics else None,
                "baseline_accuracy": baseline_metrics["token_accuracy"] if baseline_metrics else None,
                "status": "running",
            }

            # ── Profile each combination ──────────────────────────────
            idle_baseline = config.get("idle_baseline", {"cpu_mw": 18.0, "gpu_mw": 3.4, "ane_mw": 0.0})
            
            # Telemetry arrays for Smart ETA
            compile_times = []
            energy_times = []
            accuracy_times = []

            for step_idx, (bit_width, sublayer, block_idx) in enumerate(schedule):
                if self._check_cancelled():
                    break

                if self._wait_if_paused():
                    break

                self.current_block = block_idx
                self.current_sublayer = sublayer
                self.current_bitwidth = bit_width
                self.completed_steps = step_idx

                avg_compile = np.mean(compile_times) if compile_times else 15.0
                avg_energy = np.mean(energy_times) if energy_times else float(duration_sec + 5)
                avg_acc = np.mean(accuracy_times) if accuracy_times else 5.0
                
                # Dynamic ETA calculation based on what is configured to run
                eta_per_step = avg_compile
                if collect_energy: eta_per_step += avg_energy
                if collect_accuracy: eta_per_step += avg_acc
                
                remaining_sec = eta_per_step * (self.total_steps - step_idx)

                self._emit("profiling_progress", {
                    "step": step_idx + 1,
                    "total": self.total_steps,
                    "message": f"Block {block_idx} / {sublayer} / {bit_width}",
                    "phase": "profiling",
                    "block": block_idx,
                    "sublayer": sublayer,
                    "bit_width": bit_width,
                    "elapsed_sec": round(time.time() - self.start_time, 1),
                    "remaining_sec": round(remaining_sec, 1),
                    "energy_records": len(self.energy_records),
                    "accuracy_records": len(self.accuracy_records),
                    "telemetry": {
                        "avg_compile": round(avg_compile, 1),
                        "avg_energy": round(avg_energy, 1),
                        "avg_accuracy": round(avg_acc, 1)
                    }
                })

                # ── Build selectively quantized model ─────────────────
                ane_compress = "int8"  # ANEForge compilation format
                if bit_width in ("int4", "int2"):
                    ane_compress = "int4"
                elif bit_width == "fp16":
                    ane_compress = None

                q_weights = _selectively_quantize_weights(
                    base_weights, block_idx, sublayer, bit_width
                )

                q_model = LlamaPrefill(cfg, q_weights, compress=ane_compress)

                # Cap max_len for aggressive quantizations to avoid ANE graph explosion
                compile_max_len = min(max_len, MAX_LEN_CAPS.get(bit_width, max_len))

                try:
                    self._emit("profiling_progress", {
                        "step": step_idx + 1, "total": self.total_steps,
                        "message": f"Compiling ANE graph for Block {block_idx}/{sublayer}/{bit_width} (max_len={compile_max_len})...",
                        "phase": "compiling",
                        "block": block_idx, "sublayer": sublayer, "bit_width": bit_width,
                    })

                    t_comp0 = time.time()
                    _warmup_with_timeout(q_model, compile_max_len)
                    compile_times.append(time.time() - t_comp0)
                except Exception as e:
                    self._emit("profiling_progress", {
                        "step": step_idx + 1, "total": self.total_steps,
                        "message": f"Compile failed for Block {block_idx}/{sublayer}/{bit_width}: {e}",
                        "phase": "error",
                    })
                    _release_model(q_model)
                    del q_model
                    gc.collect()
                    continue

                # ── Energy profiling ──────────────────────────────────
                if collect_energy:
                    self._emit("profiling_progress", {
                        "step": step_idx + 1, "total": self.total_steps,
                        "message": f"Measuring energy for Block {block_idx}/{sublayer}/{bit_width}...",
                        "phase": "energy_profiling",
                        "block": block_idx, "sublayer": sublayer, "bit_width": bit_width,
                    })
                    
                    t_ene0 = time.time()
                    energy_result = self._profile_energy(
                        q_model, tokenizer, max_len,
                        duration_sec, warmup_runs, idle_baseline,
                        block_idx, sublayer, bit_width, step_idx
                    )
                    energy_times.append(time.time() - t_ene0)

                    energy_record = layer_db.make_energy_record(
                        experiment_id=run_id,
                        model=model_id,
                        block_number=block_idx,
                        sublayer=sublayer,
                        bit_width=bit_width,
                        latency_stats=energy_result["latency"],
                        power_stats=energy_result["power"],
                        memory_stats=energy_result.get("memory", {}),
                        duration_sec=energy_result["duration"],
                        samples_collected=energy_result["samples"],
                    )

                    self.energy_records.append(energy_record)
                    layer_db.append_energy_record(run_id, energy_record)

                    self._emit("profiling_sublayer_energy", {
                        "record": energy_record,
                        "total_records": len(self.energy_records),
                    })

                # ── Accuracy profiling ────────────────────────────────
                if collect_accuracy and eval_texts and baseline_metrics:
                    self._emit("profiling_progress", {
                        "step": step_idx + 1, "total": self.total_steps,
                        "message": f"Measuring accuracy for Block {block_idx}/{sublayer}/{bit_width}...",
                        "phase": "accuracy_evaluation",
                        "block": block_idx, "sublayer": sublayer, "bit_width": bit_width,
                    })

                    t_acc0 = time.time()
                    q_metrics = layer_accuracy.compute_perplexity(
                        q_model, tokenizer, eval_texts, max_len,
                        on_progress=lambda msg: self._emit("profiling_progress", {
                            "step": step_idx + 1, "total": self.total_steps,
                            "message": msg, "phase": "accuracy_evaluation",
                        })
                    )
                    accuracy_times.append(time.time() - t_acc0)

                    accuracy_record = layer_db.make_accuracy_record(
                        experiment_id=run_id,
                        model=model_id,
                        block_number=block_idx,
                        sublayer=sublayer,
                        bit_width=bit_width,
                        dataset=dataset_name,
                        baseline_perplexity=baseline_metrics["perplexity"],
                        measured_perplexity=q_metrics["perplexity"],
                        baseline_accuracy=baseline_metrics["token_accuracy"],
                        measured_accuracy=q_metrics["token_accuracy"],
                        num_tokens=q_metrics["total_tokens"],
                    )

                    self.accuracy_records.append(accuracy_record)
                    layer_db.append_accuracy_record(run_id, accuracy_record)

                    self._emit("profiling_sublayer_accuracy", {
                        "record": accuracy_record,
                        "total_records": len(self.accuracy_records),
                    })

                # Free the quantized model
                _release_model(q_model)
                del q_model, q_weights
                gc.collect()

                self._emit("profiling_sublayer_complete", {
                    "block": block_idx,
                    "sublayer": sublayer,
                    "bit_width": bit_width,
                    "step": step_idx + 1,
                    "total": self.total_steps,
                })

            # ── Save final databases ──────────────────────────────────
            metadata["status"] = "completed" if not self._cancel_flag else "cancelled"
            metadata["completed_at"] = datetime.now().isoformat()
            metadata["total_energy_records"] = len(self.energy_records)
            metadata["total_accuracy_records"] = len(self.accuracy_records)

            if self.energy_records:
                layer_db.save_energy_records(run_id, self.energy_records, metadata)

            if self.accuracy_records:
                layer_db.save_accuracy_records(run_id, self.accuracy_records, metadata)

            self.completed_steps = self.total_steps
            self.state = self.COMPLETED

            self._emit("profiling_completed", {
                "run_id": run_id,
                "energy_records": len(self.energy_records),
                "accuracy_records": len(self.accuracy_records),
                "metadata": metadata,
            })

        except Exception as e:
            self.state = self.ERROR
            self._emit("profiling_error", {
                "message": str(e),
                "run_id": run_id,
            })
            import traceback
            traceback.print_exc()

        finally:
            # Clean up
            from .engine import ExperimentEngine
            ExperimentEngine.unload_model()
            gc.collect()

    # ── Energy Profiling Helper ───────────────────────────────────────────

    def _profile_energy(self, model, tokenizer, max_len, duration_sec,
                        warmup_runs, idle_baseline, block_idx, sublayer,
                        bit_width, step_idx):
        """Profile energy consumption for a single model configuration.

        Runs continuous inference for `duration_sec` while collecting
        power metrics via MetricsCollector.

        Returns dict with latency, power, and memory stats.
        """
        # Warmup
        for _ in range(warmup_runs):
            try:
                dummy_ids = [tokenizer.bos_token_id or 1]
            except Exception:
                dummy_ids = [1]
            model.generate(
                dummy_ids, max_new_tokens=1, max_len=max_len,
                eos_id=tokenizer.eos_token_id, on_token=lambda t: None,
            )

        # Start metrics collection
        collector = metrics_mod.MetricsCollector(
            idle_baseline=idle_baseline,
            interval_ms=500,
        )

        # Metrics collection thread
        def collect_metrics():
            while collector._running:
                try:
                    sample = collector.collect_one()
                    if sample:
                        self._emit("profiling_metric_sample", {
                            "block": block_idx,
                            "sublayer": sublayer,
                            "bit_width": bit_width,
                            "cpu_mw": sample.get("cpu_mw", 0),
                            "gpu_mw": sample.get("gpu_mw", 0),
                            "ane_mw": sample.get("ane_mw", 0),
                            "compute_mw": sample.get("compute_mw", 0),
                            "thermal": sample.get("thermal", "?"),
                        })
                except Exception:
                    break

        metrics_thread = threading.Thread(target=collect_metrics, daemon=True)

        # Run inference loop for the specified duration
        latencies = []
        tokens_generated = 0

        prompt = "The quick brown fox jumps over the lazy dog."
        try:
            ids = tokenizer.encode(prompt)
        except Exception:
            ids = [1]

        if len(ids) >= max_len - 16:
            ids = ids[:max_len - 16]

        collector.start()
        metrics_thread.start()

        loop_start = time.time()
        while (time.time() - loop_start) < duration_sec:
            if self._cancel_flag:
                break
            if not self._pause_event.is_set():
                self._pause_event.wait()
                if self._cancel_flag:
                    break

            gen_tokens = []
            start = time.time()
            try:
                model.generate(
                    ids,
                    max_new_tokens=min(32, max_len - len(ids) - 1),
                    max_len=max_len,
                    eos_id=tokenizer.eos_token_id,
                    on_token=lambda t: gen_tokens.append(t),
                )
            except Exception:
                pass

            elapsed_ms = (time.time() - start) * 1000
            latencies.append(elapsed_ms)
            tokens_generated += len(gen_tokens)

        collector.stop()
        total_duration = time.time() - loop_start

        # Compute stats
        summary = collector.compute_summary(label=f"block{block_idx}_{sublayer}_{bit_width}")

        latency_stats = {
            "avg": round(np.mean(latencies), 2) if latencies else 0,
            "min": round(np.min(latencies), 2) if latencies else 0,
            "max": round(np.max(latencies), 2) if latencies else 0,
            "std": round(np.std(latencies), 2) if latencies else 0,
            "tokens_per_sec": round(tokens_generated / total_duration, 2) if total_duration > 0 else 0,
        }

        power_stats = {
            "avg_cpu_mw": summary.get("avg_cpu_mw", 0) or 0,
            "avg_gpu_mw": summary.get("avg_gpu_mw", 0) or 0,
            "avg_ane_mw": summary.get("avg_ane_mw", 0) or 0,
            "avg_combined_mw": summary.get("avg_combined_mw", 0) or 0,
            "peak_power_mw": max(
                (s.get("combined_mw", 0) for s in collector.samples),
                default=0
            ),
            "temperature": collector.samples[-1].get("thermal", "?") if collector.samples else "?",
            "energy_joules": summary.get("total_compute_joules", 0) or 0,
        }

        memory_stats = {"peak_mb": 0, "avg_mb": 0}
        try:
            import psutil
            proc = psutil.Process()
            memory_stats["peak_mb"] = round(proc.memory_info().rss / (1024 * 1024), 1)
        except Exception:
            pass

        return {
            "latency": latency_stats,
            "power": power_stats,
            "memory": memory_stats,
            "duration": round(total_duration, 2),
            "samples": len(collector.samples),
            "tokens_generated": tokens_generated,
        }
