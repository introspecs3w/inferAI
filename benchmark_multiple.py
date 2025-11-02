import yaml
import subprocess
import itertools
import os
import re
import csv
from datetime import datetime

CONFIG_PATH = "configs/config.yaml"
LOG_DIR = "logs/benchmark_runs"

os.makedirs(LOG_DIR, exist_ok=True)

# Settings to sweep through
providers = ["CPUExecutionProvider", "CUDAExecutionProvider"]
optimization_levels = [0, 1, 2, 99]
quantized_options = [True, False]

# Models to benchmark
models = [
    {"name": "distilbert-base-uncased", "type": "text", "backend": "onnx"},
    {"name": "google/vit-base-patch16-224", "type": "image", "backend": "onnx"},
]


def generate_config(provider, optimization_level, quantized):
    """Generate configuration dictionary"""
    config = {"models": []}
    for model in models:
        entry = model.copy()
        entry["provider"] = provider
        entry["optimization_level"] = optimization_level
        entry["quantized"] = quantized
        config["models"].append(entry)
    return config


def write_config(config):
    """Write new config.yaml to disk"""
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f)


def run_benchmark(config_params):
    """Run main.py for a given configuration and return parsed profiler outputs"""
    provider, opt_level, quantized = config_params
    config_name = (
        f"{provider}_opt{opt_level}_quant{'1' if quantized else '0'}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    log_file = os.path.join(LOG_DIR, f"{config_name}.log")

    print(f"=== Running benchmark: {config_name} ===")
    config = generate_config(provider, opt_level, quantized)
    write_config(config)

    with open(log_file, "w") as lf:
        subprocess.run(
            ["python", "main.py", "BenchmarkRun"],
            stdout=lf,
            stderr=subprocess.STDOUT,
        )

    print(f"→ Saved log to {log_file}")
    results = parse_profiler_report(log_file)
    return config_name, results


def parse_profiler_report(log_file):
    """Extract profiler timing data from your log output"""
    results = {}

    # with open(log_file, "r") as f:
    #     content = f.read()

    # # Your log contains lines like:
    # # "Inference completed: {'model': 'distilbert-base-uncased', 'total_time': 1.23, ...}"
    # # Let's extract JSON-like contents
    # pattern = r"Inference completed:\s*({.*?})"
    # matches = re.findall(pattern, content)
    # print(f"matches: {matches}")

    # for i, match in enumerate(matches):
    #     # Try to safely evaluate small dicts
    #     try:
    #         d = eval(match, {"__builtins__": {}})
    #         print(f"Parsed profiler entry: {d}")
    #         # Expect entries like {'model': 'distilbert-base-uncased', 'total_time': NN}
    #         model_name = d.get("model", f"run_{i}")
    #         total_time = float(d.get("total_time", d.get("time", 0)))
    #         results[model_name] = total_time
    #         print(f"results  • {model_name}: {total_time:.4f}s")
    #     except Exception:
    #         pass

    # return results


    # # Detect "Running Inference on model" lines
    # model_pattern = r"Running Inference on model\s*:\s*([^\s]+)"
    # # Detect inference completion metrics
    # report_pattern = r"Inference completed:\s*({.*?})"

    # model_matches = re.findall(model_pattern, content)
    # report_matches = re.findall(report_pattern, content)

    # # If we have an equal number of report sections and model sections, pair them
    # pairs = list(zip(model_matches, report_matches))
    # # Otherwise: fallback — one report per last seen model
    # if not pairs:
    #     pairs = [(f"run_{i}", r) for i, r in enumerate(report_matches)]

    # for model_name, report in pairs:
    #     try:
    #         d = eval(report, {"__builtins__": {}})
    #         mean_ms = float(d.get("mean_ms", 0))
    #         std_ms = float(d.get("std_ms", 0))
    #         p95_ms = float(d.get("p95_ms", 0))
    #         mean_s = mean_ms / 1000.0

    #         results[model_name] = mean_s
    #         print(
    #             f"  • {model_name}: mean={mean_ms:.3f}ms  (std={std_ms:.3f}, p95={p95_ms:.3f})"
    #         )
    #     except Exception as e:
    #         print(f"  ⚠️ Could not parse report for {model_name}: {e}")

    # return results
    
    with open(log_file, "r") as f:
        lines = f.readlines()
        
    model_pattern = re.compile(r"Running Inference on model\s*:\s*([^\s]+)")
    report_pattern = re.compile(r"Inference completed:\s*({.*?})")

    for line in lines:
        model_match = model_pattern.search(line)
        report_match = report_pattern.search(line)

        if model_match:
            # Update current model to latest
            current_model = model_match.group(1).strip()

        elif report_match:
            # Use the most recent model for this report
            if not current_model:
                current_model = "UnknownModel"

            try:
                report_dict = eval(report_match.group(1), {"__builtins__": {}})
                mean_ms = float(report_dict.get("mean_ms", 0))
                std_ms = float(report_dict.get("std_ms", 0))
                p95_ms = float(report_dict.get("p95_ms", 0))
                mean_s = mean_ms / 1000.0

                results[current_model] = mean_s
                print(
                    f"  • {current_model:<28} mean={mean_ms:.3f}ms (std={std_ms:.3f}, p95={p95_ms:.3f})"
                )
            except Exception as e:
                print(f"  ⚠️ Failed to parse report for {current_model}: {e}")

    return results


def aggregate_results(run_results):
    """Combine profiler results to show averages per config"""
    aggregated = {}
    for config_name, results in run_results.items():
        for model_name, time in results.items():
            aggregated.setdefault(model_name, []).append((config_name, time))
    return aggregated


def save_summary_csv(aggregated_results, summary_path="logs/benchmark_summary.csv"):
    with open(summary_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["model", "config_name", "time (s)"])
        for model, data_points in aggregated_results.items():
            for conf_name, t in data_points:
                writer.writerow([model, conf_name, round(t, 4)])
    print(f"✅ Summary written to {summary_path}")


def main():
    all_combos = list(itertools.product(providers, optimization_levels, quantized_options))
    run_results = {}

    for combo in all_combos:
        config_name, results = run_benchmark(combo)
        run_results[config_name] = results

    # Aggregate across runs
    aggregated = aggregate_results(run_results)

    print("\n=== Aggregated Benchmark Results ===")
    for model, entries in aggregated.items():
        mean_time = sum(t for _, t in entries) / len(entries)
        print(f"• {model:<30} avg: {mean_time:.4f}s across {len(entries)} configs")

    save_summary_csv(aggregated)


if __name__ == "__main__":
    main()