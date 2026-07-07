# Apple Neural Engine (ANE) Experiment Wizard

A comprehensive local benchmarking and power profiling suite for Large Language Models running on the Apple Neural Engine (ANE). This toolkit provides both an interactive web-based wizard and standalone CLI scripts to measure model performance and compute energy on Apple Silicon (M-series).

## Features

- **Interactive Web UI**: A beautiful, modern dashboard to configure, run, and compare experiments.
- **Hardware Power Profiling**: Direct integration with macOS `powermetrics` to read CPU, GPU, and ANE power sensors.
- **Compute Energy Calculation**: Automatically subtracts idle baseline power to isolate the Net Compute Energy (Joules/token) used by the LLM.
- **Automated Memory Safety**: Guarantees only one model is loaded into memory at a time to prevent unified memory crashes.
- **HuggingFace Integration**: Auto-detects locally cached models and can seamlessly download new models (like Llama-3.2 1B/3B or Qwen) if missing.
- **Real-Time Dashboards**: Watch power consumption charts and token generation speeds live as the model runs.

## Prerequisites

- **macOS with Apple Silicon** (M1/M2/M3/M4). Tested on M4 iMac.
- **Python 3.11** or newer.
- **ANEForge**: The Apple Neural Engine inference library must be installed locally.
- **Administrator Privileges**: Running the application requires `sudo` because reading macOS hardware power sensors via `powermetrics` is restricted to root users.

## Installation & Setup

1. **Clone ANEForge**
   Make sure you have cloned the ANEForge repository to your system. The default path expected is `/Users/kartik/Documents/ANEForge-main`. If it's located elsewhere, update the `ANEFORGE_PATH` variable in `wizard/models.py` and `standalone_scripts/run_llama_3_2_1b.py`.

2. **Clone This Repository**
   ```bash
   git clone <your-repository-url>
   cd "paper tools"
   ```

3. **Install Dependencies**
   Install the required Python packages for the web wizard:
   ```bash
   pip install flask flask-socketio transformers huggingface_hub
   ```

4. **HuggingFace Authentication**
   Some models (like Meta's Llama 3) require you to accept their license on HuggingFace before downloading. Authenticate your terminal:
   ```bash
   huggingface-cli login
   ```

## Usage

### 1. The Interactive Wizard (Recommended)

To launch the web interface, you must run it with `sudo` from your terminal. This securely grants the application permission to read the hardware power sensors.

```bash
cd "paper tools"
sudo python3.11 wizard/app.py
```

Then, open your web browser and navigate to:
**http://localhost:5050**

From the dashboard, you can click "New Experiment" to follow a guided 5-step process:
1. Select a model (and compression format like `int8`).
2. Choose your prompt dataset (Single Prompt, Coding, Reasoning, etc.).
3. Configure the context window length.
4. Select a Metric Profile (determines how many warmup runs to perform to activate the sensors).
5. Start the experiment and watch the live telemetry!

### 2. Standalone CLI Scripts

If you prefer to run benchmarks directly from the command line, the `standalone_scripts/` directory contains individual python scripts.

- **Run an LLM directly**:
  ```bash
  cd "paper tools/standalone_scripts"
  python3.11 run_llama_3_2_1b.py
  ```
  *(Note: Text generation alone does not require sudo).*

- **Run the Power Profiler independently**:
  If you want to manually profile an arbitrary process (like a web browser or a different script), you can use the profiler script:
  ```bash
  sudo python3.11 m4_profiler.py --label my_custom_test --duration 60
  ```

## Folder Structure

- `/wizard`: The core application backend (Flask, Engine, Metrics, Validation).
  - `/wizard/templates`: HTML templates for the UI.
  - `/wizard/static`: CSS styles and JavaScript logic for live charts.
- `/standalone_scripts`: Legacy, hardcoded scripts for running individual models or capturing idle baselines manually.
- `/power_logs`: Directory where all experiment results, summary JSONs, and raw CSV power metrics are saved automatically.

## Troubleshooting

- **"0.0 Joules" or Blank Charts**: If your experiment finishes but shows no power data, the server was likely started without `sudo`. Stop the server (Ctrl+C) and restart it with `sudo python3.11 wizard/app.py`.
- **Hanging on Warmup**: The Apple Neural Engine's CoreML compiler can take several minutes to compile the inference graph for large context lengths (e.g., 2048). If the progress bar says "Compiling ANE graph...", let it sit for 2-3 minutes. This is a one-time cost per context length.
- **Memory Crashes**: If you experience a `status=0x2` or `Program Inference error` from the ANE, ensure you are using `int8` or `int4` compression. `int16` / `fp16` models may exceed the unified memory limits on smaller machines.
