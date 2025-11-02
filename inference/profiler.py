"""profiler."""

import time
import statistics
from typing import Callable, Any, Dict, List
from inference.logger import setup_logger
import psutil
import pynvml

logger = setup_logger(__name__, log_file="logs/inferai.log")

class Profiler:
    """Basic profiler for measuring inference latency and reporting stats."""

    def __init__(self):
        self.records: List[float] = []
        pynvml.nvmlInit()   

    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics like CPU and GPU usage."""
        metrics = {}
        # CPU usage
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_used_mb = mem_info.rss / (1024**2)
        metrics["cpu_usage_percent"] = psutil.cpu_percent(interval=None)
        metrics["memory_usage_percent"] = psutil.virtual_memory().percent
        metrics["mem_used_mb"] = round(mem_used_mb, 2)

        # GPU usage
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming single GPU
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            metrics["gpu_memory_total_MB"] = mem_info.total / (1024**2)
            metrics["gpu_memory_used_MB"] = mem_info.used / (1024**2)
            metrics["gpu_memory_free_MB"] = mem_info.free / (1024**2)
            metrics["gpu_utilization_percent"] = util_info.gpu
            metrics["gpu_memory_utilization_percent"] = util_info.memory
        except Exception as e:
            logger.warning(f"Could not get GPU metrics: {e}")
            metrics["gpu_metrics_error"] = str(e)

        return metrics
    
    def profile(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        start = time.perf_counter()
        pre_metrics = self.get_metrics()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        post_metrics = self.get_metrics()
        # print(f"Inference time: {elapsed*1000:.2f} ms")
        
        # self.records.append(elapsed)
        record = {
            "elapsed_s": elapsed,
            "elapsed_ms": round(elapsed * 1000, 2),
            "before": pre_metrics,
            "after": post_metrics,
        }
        self.records.append(record)

        # Logging summary
        # logger.info(f"Inference time: {elapsed*1000:.2f} ms")
        logger_str = (
            f"Inference time: {record['elapsed_ms']} ms | "
            f"CPU: {post_metrics['cpu_usage_percent']}% | "
            f"RAM: {post_metrics['mem_used_mb']} MB"
        )
        # Add GPU summary if exists
        if "gpu" in post_metrics:
            gpu_summary = " | ".join(
                [f"GPU{i}: {g['gpu_util']}% ({g['mem_used_mb']}MB/{g['mem_total_mb']}MB)"
                 for i, g in enumerate(post_metrics['gpu'])]
            )
            logger_str += " | " + gpu_summary

        logger.info(logger_str)  # or use logger.info(logger_str)

        return result

    def report(self) -> Dict[str, float]:
        if not self.records:
            return {"mean_ms": 0.0, "std_ms": 0.0, "p95_ms": 0.0}
        
        elapsed = [r["elapsed_ms"] for r in self.records]
        return {
            "mean_ms": statistics.mean(elapsed),
            "std_ms": statistics.stdev(elapsed) if len(self.records) > 1 else 0.0,
            "p95_ms": sorted(elapsed)[int(0.95 * len(elapsed)) - 1],
        }

    def clear(self) -> None:
        self.records.clear()    
