# inferAI

**inferAI** is a modular benchmarking and inference framework for evaluating ONNX models (text and image) across multiple hardware providers, optimization levels, and quantization settings. It is designed for extensibility, reproducibility, and easy integration with HuggingFace models and ONNX Runtime.

---

## Key points Answered

- Factory Pattern used to modularize a scalable Backend switching function (currently supports only onnx)
- System profiling and timing done using perf_counter and psutil
- Dynamic Quantization used for the sake of expediency and independence of calibration data
- Currently backend modularity is applicable to Inference
- UnitTests are generated using copilot, other AI generations are somewhat used in readme/benchmark/profiler files (other than that, AI was consulted for debug errors and creating apt inputs/interpreting outputs for ORT methods)
- Run the main.py for quick test
- Run the benchmark_multiple.py for comprehensive test and report, this test modifies the config.yaml and iterates thru different configurations and outputs result and then compiles report
- Logs are in ./logs/ folder
- I tried using quantized model for the image models but realized after getting errors that my quantizer (INT8) won't be applicable to the VIT model so for the sake of expediency(to use TensorRT quantizer) this program currently uses only "optimized" model for "image" models, however the ModelConverter does generate a quantized model for "image" models too

## Features

- **Automated benchmarking**: Sweep through providers (CPU, CUDA), optimization levels, and quantization options.
- **Flexible configuration**: YAML-based model and input configuration.
- **Model conversion**: Converts HuggingFace models to ONNX, optimizes, and quantizes them.
- **Unified inference**: Run inference for both text and image models with detailed profiling.
- **Logging and Profiling**: Logs all runs and profile results and aggregates results into CSV summaries.
- **Extensible backend system**: Easily add new backends or providers.

---

## Project Structure

```
benchmark_multiple.py         # Benchmarking script (sweeps configs, aggregates results)
main.py                      # Main entry point for model conversion and inference
requirements.txt             # Python dependencies
configs/
	config.yaml              # Model and provider configuration
inference/
	backend_factory.py       # Backend abstraction and factory
	model_converter.py       # Model conversion, optimization, quantization
	inference_manager.py     # Loads models, runs inference, manages profiling
	...
inputs/
	text_inputs.yaml         # Text inputs for benchmarking
	image_inputs.yaml        # Image inputs for benchmarking
logs/
	benchmark_summary.csv    # Aggregated benchmark results
	benchmark_runs/          # Individual run logs
models/
	...                      # Converted, optimized, and quantized models
```

---

## Setup Instructions

### 1. Install Python dependencies

It is recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Key dependencies:

- `onnxruntime`
- `transformers`
- `optimum`
- `PyYAML`
- `Pillow`
- `numpy`

### 2. Prepare configuration and input files

- Edit `configs/config.yaml` to specify which models, providers, and settings to use.
- Add your text/image inputs to `inputs/text_inputs.yaml` and `inputs/image_inputs.yaml`.

---

## Usage

### A. Run a single inference (for development)

```bash
python main.py <name>
```

- This will greet `<name>`, convert models as per config, and run inference on all inputs.
- Logs and profiler reports are saved in `logs/`.

### B. Run full benchmark sweep

```bash
python benchmark_multiple.py
```

- This script will automatically sweep through all combinations of providers, optimization levels, and quantization settings for all models defined in `benchmark_multiple.py`.
- Each run's log is saved in `logs/benchmark_runs/`.
- Aggregated results are written to `logs/benchmark_summary.csv`.

---

## Components

### 1. Model Converter (`inference/model_converter.py`)

- Converts HuggingFace models to ONNX format.
- Applies optimization and quantization as specified in config.
- Saves converted models in structured directories under `models/`.

### 2. Inference Manager (`inference/inference_manager.py`)

- Loads ONNX models and tokenizers/processors.
- Runs inference for text and image inputs.
- Uses a backend factory to abstract provider/session creation.
- Profiles inference time and logs results.

### 3. Benchmarking Script (`benchmark_multiple.py`)

- Sweeps through all combinations of providers, optimization levels, and quantization.
- Generates config files, runs main.py, parses logs, and aggregates results.
- Outputs a summary CSV for easy comparison.

### 4. Logging & Profiling

- All logs are written to `logs/inferai.log` and per-run logs in `logs/benchmark_runs/`.
- Profiler reports include mean, std, and p95 latency for each model/config.

---

## Customization

- Add new models to `models` list in `benchmark_multiple.py` and `configs/config.yaml`.
- Add new backends/providers by extending `inference/backend_factory.py`.
- Add new input samples to `inputs/`.

---
