"""profiler."""

import time
import statistics
from typing import Callable, Any, Dict, List
from inference.logger import setup_logger


logger = setup_logger(__name__, log_file="inferai.log")

class Profiler:
    """Basic profiler for measuring inference latency and reporting stats."""

    def __init__(self):
        self.records: List[float] = []

    def profile(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"Inference time: {elapsed*1000:.2f} ms")
        print(f"Inference time: {elapsed*1000:.2f} ms")
        self.records.append(elapsed)
        return result

    def report(self) -> Dict[str, float]:
        if not self.records:
            return {"mean_ms": 0.0, "std_ms": 0.0, "p95_ms": 0.0}
        return {
            "mean_ms": statistics.mean(self.records) * 1000,
            "std_ms": statistics.stdev(self.records) * 1000 if len(self.records) > 1 else 0.0,
            "p95_ms": sorted(self.records)[int(0.95 * len(self.records)) - 1] * 1000,
        }

    def clear(self) -> None:
        self.records.clear()    
        