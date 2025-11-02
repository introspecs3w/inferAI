"""optimizer."""
import psutil
from inference.logger import setup_logger

import onnxruntime as ort
from typing import Any
import os
import optimum

from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from onnxruntime.quantization import (
    quantize_dynamic,
    QuantType,
)

logger = setup_logger(__name__, log_file="logs/inferai.log")


class Quantizer:
    """Wraps ONNX Runtime optimization and session configuration."""

    OPT_MAP = {
        0: ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        1: ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        2: ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        99: ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }

    def __init__(self, mode: str = "dynamic"):
        self.mode = mode
        self.per_channel = True
        

    def quantize(self, model_dir: str, out_dir: str, model_type: str) -> str:
        """
        Optimize an exported ONNX model directory using Optimum.
        """
        os.makedirs(out_dir, exist_ok=True)

        max_ram_gb = 8 
        available_gb = psutil.virtual_memory().available / (1024**3)
        if available_gb < max_ram_gb:
            print(f"Warning: Only {available_gb:.1f} GB RAM available. Forcing per_channel=False")
            logger.info(f"Warning: Only {available_gb:.1f} GB RAM available. Forcing per_channel=False")
            self.per_channel = False
        else:
            self.per_channel = True

        quantizer = ORTQuantizer.from_pretrained(model_dir)
      
        print(dir(AutoQuantizationConfig))
        qconfig = AutoQuantizationConfig.avx512_vnni(
        # qconfig = AutoQuantizationConfig.arm64(
        # qconfig = AutoQuantizationConfig.tensorrt(
            is_static=False, 
            per_channel=self.per_channel,
            )
        quantizer.quantize(
            save_dir=out_dir,
            quantization_config=qconfig,
        )
        return out_dir



