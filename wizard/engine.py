"""
Experiment execution engine for the ANE Experiment Wizard.
Manages the complete experiment lifecycle: validate → load → warmup →
benchmark → collect metrics → save → report → unload.
"""

import os
import sys
import gc
import time
import json
import threading
from datetime import datetime
from pathlib import Path

ANEFORGE_PATH = "/Users/kartik/Documents/ANEForge-main"
if ANEFORGE_PATH not in sys.path:
    sys.path.insert(0, ANEFORGE_PATH)

from . import models as model_mgr
from . import validator
from . import metrics as metrics_mod
from . import results as results_mod


class ExperimentEngine:
    """Manages the lifecycle of a single experiment.
    Guarantees only one model is loaded at a time.
    Emits progress events via a callback."""

    # Class-level state: ensures only one model globally
    _current_model = None
    _current_model_id = None
    _current_tokenizer = None
    _lock = threading.Lock()

    def __init__(self, emit_fn=None):
        """
        Args:
            emit_fn: callable(event, data) for emitting WebSocket events
        """
        self.emit = emit_fn or (lambda event, data: None)
        self.running = False
        self.cancelled = False
        self._thread = None

    def _emit(self, event, data):
        """Thread-safe emit."""
        try:
            self.emit(event, data)
        except Exception:
            pass

    @classmethod
    def unload_model(cls):
        """Unload the currently loaded model and free memory."""
        with cls._lock:
            cls._current_model = None
            cls._current_model_id = None
            cls._current_tokenizer = None
            gc.collect()

    @classmethod
    def get_loaded_model(cls):
        """Return the currently loaded model ID, or None."""
        return cls._current_model_id

    def cancel(self):
        """Cancel a running experiment."""
        self.cancelled = True

    def run_experiment(self, config):
        """Run an experiment in a background thread.

        config keys:
            model_id, compression, prompt_source, prompt_text, prompt_file,
            prompt_dataset, eval_mode, metric_profile, experiment_name,
            max_len, warmup_runs, benchmark_runs, notes, output_dir
        """
        self._thread = threading.Thread(target=self._run, args=(config,), daemon=True)
        self._thread.start()
        return True

    def _run(self, config):
        """The actual experiment execution sequence."""
        self.running = True
        self.cancelled = False
        experiment_id = None

        try:
            # ── Step 0: Generate experiment name ──────────────────────────
            model_short = config["model_id"].split("/")[-1]
            compress_str = config.get("compression", "int8") or "fp16"
            eval_mode = config.get("eval_mode", "single")
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
            auto_name = f"{model_short}_{compress_str}_{eval_mode}_{ts}"
            experiment_id = config.get("experiment_name", auto_name) or auto_name
            # Sanitize for filesystem
            experiment_id = experiment_id.replace(" ", "_").replace("/", "-")

            total_steps = 13
            step = 0

            # ── Step 1: Validate ──────────────────────────────────────────
            step += 1
            self._emit("progress", {"step": step, "total": total_steps,
                                     "message": "Validating configuration..."})
            val_results = validator.validate(config)
            self._emit("validation", {"results": val_results})

            # Check for critical failures (except model folder — we'll auto-download)
            critical = [r for r in val_results
                        if not r["passed"] and r["name"] not in ("Model folder exists", "Model weights exist", "Tokenizer exists")]
            if critical:
                self._emit("error", {
                    "message": "Validation failed",
                    "details": [r["message"] + " — " + r["fix"] for r in critical]
                })
                return

            if self.cancelled:
                self._emit("cancelled", {})
                return

            # ── Step 2: Unload any current model ──────────────────────────
            step += 1
            self._emit("progress", {"step": step, "total": total_steps,
                                     "message": "Unloading any previously loaded model..."})
            ExperimentEngine.unload_model()

            # ── Step 3: Clear caches and release memory ───────────────────
            step += 1
            self._emit("progress", {"step": step, "total": total_steps,
                                     "message": "Clearing caches..."})
            gc.collect()
            time.sleep(0.5)

            # ── Step 4: Verify resources ──────────────────────────────────
            step += 1
            self._emit("progress", {"step": step, "total": total_steps,
                                     "message": "Verifying available resources..."})

            if self.cancelled:
                self._emit("cancelled", {})
                return

            # ── Step 5: Load model ────────────────────────────────────────
            step += 1
            self._emit("progress", {"step": step, "total": total_steps,
                                     "message": f"Loading {config['model_id']}..."})

            compress = config.get("compression", "int8")
            if compress == "None" or compress == "fp16":
                compress = None

            model, tokenizer = model_mgr.load_model_for_inference(
                config["model_id"], compress=compress
            )

            with ExperimentEngine._lock:
                ExperimentEngine._current_model = model
                ExperimentEngine._current_model_id = config["model_id"]
                ExperimentEngine._current_tokenizer = tokenizer

            if self.cancelled:
                ExperimentEngine.unload_model()
                self._emit("cancelled", {})
                return

            # ── Step 6: Warmup ────────────────────────────────────────────
            step += 1
            max_len = config.get("max_len", 512)
            warmup_runs = config.get("warmup_runs", 1)
            
            self._emit("progress", {"step": step, "total": total_steps,
                                     "message": f"Compiling ANE graph (context={max_len}). This may take a few minutes..."})

            model.warmup(max_len)

            if self.cancelled:
                ExperimentEngine.unload_model()
                self._emit("cancelled", {})
                return
                
            for w_run in range(warmup_runs):
                self._emit("progress", {"step": step, "total": total_steps,
                                         "message": f"Warming up ANE hardware (Run {w_run + 1}/{warmup_runs})..."})
                # Run a very short dummy generation to force the ANE into active state
                try:
                    dummy_ids = [tokenizer.bos_token_id or 1]
                except Exception:
                    dummy_ids = [1]
                
                model.generate(
                    dummy_ids,
                    max_new_tokens=1,
                    max_len=max_len,
                    eos_id=tokenizer.eos_token_id,
                    on_token=lambda t: None
                )

            # ── Step 7: Prepare prompts ───────────────────────────────────
            prompts = self._load_prompts(config)
            if not prompts:
                self._emit("error", {"message": "No prompts to evaluate"})
                ExperimentEngine.unload_model()
                return

            # ── Step 8: Start metrics collection ──────────────────────────
            step += 1
            self._emit("progress", {"step": step, "total": total_steps,
                                     "message": "Starting power metrics collection..."})

            idle_baseline = config.get("idle_baseline", {"cpu_mw": 18.0, "gpu_mw": 3.4, "ane_mw": 0.0})
            interval_ms = config.get("interval_ms", 1000)
            collector = metrics_mod.MetricsCollector(
                idle_baseline=idle_baseline,
                ane_threshold=config.get("ane_threshold", 10.0),
                interval_ms=interval_ms,
            )

            # Start metrics collection in a separate thread
            metrics_thread = threading.Thread(target=self._collect_metrics,
                                              args=(collector,), daemon=True)

            benchmark_runs = config.get("benchmark_runs", 1)
            inference_results = []

            # ── Step 9: Benchmark ─────────────────────────────────────────
            step += 1
            self._emit("progress", {"step": step, "total": total_steps,
                                     "message": f"Running benchmark ({len(prompts)} prompt(s) × {benchmark_runs} run(s))..."})

            collector.start()
            metrics_thread.start()

            total_prompts = len(prompts) * benchmark_runs
            completed = 0

            for run_idx in range(benchmark_runs):
                for prompt_idx, prompt in enumerate(prompts):
                    if self.cancelled:
                        break

                    self._emit("benchmark_progress", {
                        "run": run_idx + 1,
                        "total_runs": benchmark_runs,
                        "prompt": prompt_idx + 1,
                        "total_prompts": len(prompts),
                        "prompt_text": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    })

                    result = self._run_inference(model, tokenizer, prompt, max_len, config)
                    inference_results.append(result)
                    completed += 1

                    self._emit("benchmark_progress", {
                        "completed": completed,
                        "total": total_prompts,
                        "last_result": {
                            "tokens_generated": result["tokens_generated"],
                            "tokens_per_sec": result["tokens_per_sec"],
                            "latency_ms": result["latency_ms"],
                        }
                    })

            # Stop metrics collection
            collector.stop()
            self._emit("progress", {"step": step, "total": total_steps,
                                     "message": "Benchmark complete. Processing results..."})

            if self.cancelled:
                ExperimentEngine.unload_model()
                self._emit("cancelled", {})
                return

            # ── Step 10: Save raw metrics ─────────────────────────────────
            step += 1
            self._emit("progress", {"step": step, "total": total_steps,
                                     "message": "Saving raw metrics..."})

            output_dir = config.get("output_dir") or str(results_mod.get_output_dir())

            # Build metadata
            metadata = {
                "experiment_id": experiment_id,
                "model_id": config["model_id"],
                "compression": config.get("compression", "int8"),
                "eval_mode": config.get("eval_mode", "single"),
                "metric_profile": config.get("metric_profile", "quick"),
                "prompt_source": config.get("prompt_source", "single"),
                "max_len": max_len,
                "warmup_runs": warmup_runs,
                "benchmark_runs": benchmark_runs,
                "num_prompts": len(prompts),
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "has_notes": bool(config.get("notes", "").strip()),
            }

            # Compute inference summary
            if inference_results:
                metadata["avg_tokens_per_sec"] = round(
                    sum(r["tokens_per_sec"] for r in inference_results) / len(inference_results), 2
                )
                metadata["avg_latency_ms"] = round(
                    sum(r["latency_ms"] for r in inference_results) / len(inference_results), 1
                )
                metadata["total_tokens_generated"] = sum(r["tokens_generated"] for r in inference_results)

            # Compute power summary
            summary = collector.compute_summary(label=experiment_id)
            summary.update({
                "avg_tokens_per_sec": metadata.get("avg_tokens_per_sec"),
                "total_tokens_generated": metadata.get("total_tokens_generated"),
            })

            # Calculate energy per token
            if summary.get("total_compute_joules") and metadata.get("total_tokens_generated"):
                summary["joules_per_token"] = round(
                    summary["total_compute_joules"] / metadata["total_tokens_generated"], 6
                )

            # ── Step 11: Save everything ──────────────────────────────────
            step += 1
            self._emit("progress", {"step": step, "total": total_steps,
                                     "message": "Generating reports..."})

            results_mod.save_experiment(
                experiment_id=experiment_id,
                metadata=metadata,
                summary=summary,
                power_samples=collector.samples,
                inference_results=inference_results,
                notes=config.get("notes", ""),
                output_dir=output_dir,
            )

            # ── Step 12: Unload model ─────────────────────────────────────
            step += 1
            self._emit("progress", {"step": step, "total": total_steps,
                                     "message": "Unloading model..."})
            ExperimentEngine.unload_model()

            # ── Step 13: Done ─────────────────────────────────────────────
            step += 1
            self._emit("progress", {"step": step, "total": total_steps,
                                     "message": "Experiment complete!"})

            self._emit("completed", {
                "experiment_id": experiment_id,
                "metadata": metadata,
                "summary": summary,
            })

        except Exception as e:
            self._emit("error", {"message": str(e), "experiment_id": experiment_id})
            ExperimentEngine.unload_model()
            import traceback
            traceback.print_exc()

        finally:
            self.running = False
            # Guarantee model is unloaded
            if ExperimentEngine._current_model is not None:
                ExperimentEngine.unload_model()

    def _load_prompts(self, config):
        """Load prompts based on the configured source."""
        source = config.get("prompt_source", "single")

        if source == "single":
            text = config.get("prompt_text", "").strip()
            return [text] if text else []

        elif source == "file":
            filepath = config.get("prompt_file", "")
            if os.path.isfile(filepath):
                with open(filepath) as f:
                    return [line.strip() for line in f if line.strip()]
            return []

        elif source == "dataset":
            dataset_name = config.get("prompt_dataset", "general")
            dataset_path = os.path.join(
                os.path.dirname(__file__), "prompts",
                f"{dataset_name}_prompts.txt"
            )
            if os.path.isfile(dataset_path):
                with open(dataset_path) as f:
                    return [line.strip() for line in f if line.strip()]
            return []

        return []

    def _run_inference(self, model, tokenizer, prompt, max_len, config):
        """Run inference on a single prompt and return timing results."""
        messages = [{"role": "user", "content": prompt}]

        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            ids = tokenizer.encode(prompt_text)
        except Exception:
            ids = tokenizer.encode(prompt)

        # Clamp to max context
        if len(ids) >= max_len - 8:
            ids = ids[:max_len - 8]

        generated_tokens = []

        def on_token(t):
            generated_tokens.append(t)
            # Emit each token for live streaming
            self._emit("token", {"token": tokenizer.decode([t])})

        start_time = time.time()

        model.generate(
            ids,
            max_new_tokens=max_len - len(ids) - 1,
            max_len=max_len,
            eos_id=tokenizer.eos_token_id,
            on_token=on_token,
        )

        elapsed = time.time() - start_time
        n_tokens = len(generated_tokens)
        tps = round(n_tokens / elapsed, 2) if elapsed > 0 else 0

        output_text = tokenizer.decode(generated_tokens)

        return {
            "prompt": prompt,
            "output": output_text,
            "tokens_generated": n_tokens,
            "prompt_tokens": len(ids),
            "elapsed_sec": round(elapsed, 3),
            "tokens_per_sec": tps,
            "latency_ms": round(elapsed * 1000, 1),
            "timestamp": datetime.now().isoformat(),
        }

    def _collect_metrics(self, collector):
        """Background thread for collecting powermetrics samples."""
        while collector._running:
            try:
                sample = collector.collect_one()
                if sample:
                    self._emit("metric_sample", {
                        "cpu_mw": sample.get("cpu_mw", 0),
                        "gpu_mw": sample.get("gpu_mw", 0),
                        "ane_mw": sample.get("ane_mw", 0),
                        "compute_mw": sample.get("compute_mw", 0),
                        "thermal": sample.get("thermal", "?"),
                        "net_ane_mw": sample.get("net_ane_mw", 0),
                    })
            except Exception:
                break
