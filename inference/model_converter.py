"""model_converter."""

import os
from typing import Dict, Any, List
import yaml
from inference.logger import setup_logger
from inference.optimizer import Optimizer
from inference.quantizer import Quantizer

from transformers import (
    AutoConfig, 
    AutoModel, 
    AutoTokenizer,
    AutoImageProcessor,
    AutoModelForSequenceClassification,
    AutoModelForImageClassification,
)
# from transformers import AutoTokenizer
# from optimum.onnx import export
from optimum.onnxruntime import (
    ORTModel,
    ORTModelForSequenceClassification,
    ORTModelForImageClassification,
)

logger = setup_logger(__name__, log_file="logs/inferai.log")


class ModelConverter:
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = "models"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def convert_all_models(self):
        print(f"Converting models using configuration from: {self.config_path}")
        # logger.info(f"Converting models using configuration from: {self.config_path}")
        for model in self.config.get("models", []):
            self.convert_model(model)

    def convert_model(self, model: Dict[str, Any]):
        name = model["name"]
        # print(f"Converting model: {name}")
        logger.info(f"Converting model: {name}")
        path = os.path.join(self.output_dir, f"{name.replace("/", "_").lower()}")

        path = os.path.join(path, f"{model['backend']}_{model['provider']}_OPT_{model['optimization_level']}_QUANT_{model['quantized']}")
        # print(f"path: {path}")
        # logger.info(f"path: {path}")

        opt_dir = os.path.join(path, "_optimized")
        quant_dir = os.path.join(opt_dir, "_quantized")


        if(os.path.exists(path)):
            print(f"Model {name} already converted at {path}. Skipping conversion.")
            logger.info(f"Model {name} already converted at {path}. Skipping conversion.")  
            # return
        else:
            
            if model["type"] == "text":
                tokenClass = AutoTokenizer
                # modelClass = ORTModel
                # modelClass = AutoModelForSequenceClassification
                modelClass = ORTModelForSequenceClassification
            elif model["type"] == "image":
                tokenClass = AutoImageProcessor
                # modelClass = ORTModel
                # modelClass = AutoModelForImageClassification
                modelClass = ORTModelForImageClassification
            else:
                raise ValueError(f"Unsupported model type: {model['type']}")
            
            provider = model.get("provider", "CPUExecutionProvider")

            # tokenizer = self.load_tokenizer(model["tokenizer"])
            tokenizer = tokenClass.from_pretrained(name)
            # print(f"tokenClass: {name}")
            # logger.info(f"tokenClass: {name}")

            # model_instance = self.load_model(model["model_class"], model["pretrained_weights"])
            # model_instance = AutoModel.from_pretrained(name)
            # model_instance = modelClass.from_pretrained(name)
            # model_instance = modelClass.from_pretrained(name, export=True, providers=[provider])
            model_instance = modelClass.from_pretrained(name, export=True)
            # onnx_model = self.convert_to_onnx(model_instance, tokenizer, model.get("conversion_params", {}))
            # print(f"modelClass: {name}")
            # logger.info(f"modelClass: {name}")

            model_instance.save_pretrained(path)
            # print(f"model_instance.save_pretrained: {name}")
            # logger.info(f"model_instance.save_pretrained: {name}")
            tokenizer.save_pretrained(path)
            # print(f"tokenizer.save_pretrained: {name}")
            # logger.info(f"tokenizer.save_pretrained: {name}")
            print(f"Model {name} converted and saved to {path}")
            logger.info(f"Model {name} converted and saved to {path}")

        
        # print(f"Start: {name}")
        # logger.info(f"Start: {name}")

        #Optmization

        if(os.path.exists(opt_dir)):
            print(f"Model {name} already converted at {opt_dir}. Skipping Optmization.")
            logger.info(f"Model {name} already converted at {opt_dir}. Skipping Optmization.")  
            # return
        else:
            optimization_level = model.get("optimization_level", 99)
            optimizer = Optimizer(level=optimization_level)
            optimizer.optimize(path, opt_dir)
            logger.info(f"Optimized model saved at: {opt_dir}")


        #Quantization

        if(os.path.exists(quant_dir)):
            print(f"Model {name} already converted at {quant_dir}. Skipping Quantization.")
            logger.info(f"Model {name} already converted at {quant_dir}. Skipping Quantization.")  
            # return
        else:
            doQuant = model.get("quantized", False)
            if doQuant is None:
                logger.info(f"doQuant NONE")
                return
            if doQuant == False:
                logger.info(f"doQuant FALSE")
                return
            
            # quant_dir = os.path.join(path, "_quantized")
            # mode = qcfg.get("mode", "dynamic")
            quantizer = Quantizer()
            quantizer.quantize(path, quant_dir, model["type"])
            logger.info(f"Quantized model saved at: {quant_dir}")





