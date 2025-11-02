"""optimizer."""
from inference.logger import setup_logger

import onnxruntime as ort
from typing import Any
import os

from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig

logger = setup_logger(__name__, log_file="logs/inferai.log")


class Optimizer:

    OPT_MAP = {
        0: ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        1: ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        2: ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        99: ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }

    def __init__(self, level: int = 99):
        self.level = level


        
    @staticmethod
    def create_session_options(level: int = 99, intra_threads: int = 1) -> ort.SessionOptions:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = Optimizer.OPT_MAP.get(level, ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)
        logger.info(f"Setting optimization level to {opts.graph_optimization_level}")
        opts.intra_op_num_threads = intra_threads
        return opts
    

    def optimize(self, model_dir: str, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        logger.info(f"Optimizing model at {model_dir} with optimization level {self.level}")
        optimizer = ORTOptimizer.from_pretrained(model_dir)
        opt_config = OptimizationConfig(optimization_level=self.level)
        optimizer.optimize(save_dir=out_dir, optimization_config=opt_config)
        return out_dir



