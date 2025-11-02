"""inference_manager."""

import os
import yaml
from typing import Dict, Any, List
import onnxruntime as ort
import numpy as np
from PIL import Image
from inference.logger import setup_logger
from inference.profiler import Profiler
from inference.optimizer import Optimizer


from transformers import (
    pipeline,
    AutoConfig, 
    AutoModel, 
    AutoTokenizer, 
    AutoImageProcessor,
    AutoFeatureExtractor
)
# from transformers import AutoTokenizer
# from optimum.onnx import export
from optimum.onnxruntime import (
    ORTModelForSequenceClassification,
    ORTModelForImageClassification
)

logger = setup_logger(__name__, log_file="inferai.log")

class InferenceManager:


    def __init__(self, config_path: str, profiler: Profiler):
        self.config_path = config_path
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.profiler = profiler
        self.models = {}
        self.model_types = {}
        self.tokenizers = {}
        self.load_models()

    def load_models(self):
        # print(f"Loading models using configuration from: {self.config_path}")
        # logger.info(f"Loading models using configuration from: {self.config_path}")
        for model in self.config.get("models", []):
            name = model["name"]
            # print(f"Loading model: {name}")
            logger.info(f"Loading model: {name}")

            model_dir_path = os.path.join("models", f"{name.replace("/", "_").lower()}")
            model_onnx_path = os.path.join(model_dir_path, "model.onnx")
            # print(f"path: {model_onnx_path}")
            # logger.info(f"path: {model_onnx_path}")

            model_opt_dir_path = os.path.join(model_dir_path, "_optimized")
            model_opt_onnx_path = os.path.join(model_opt_dir_path, "model_optimized.onnx")

            model_quant_dir_path = os.path.join(model_opt_dir_path, "_quantized")
            model_quant_onnx_path = os.path.join(model_quant_dir_path, "model_quantized.onnx")

            model_actual_path = model_onnx_path
            tokenizer_actual_path = model_dir_path

            if model["type"] == "text":
                if model.get("quantized", False):
                    # model_actual_path = model_onnx_path
                    model_actual_path = model_quant_onnx_path
                    tokenizer_actual_path = model_quant_dir_path
                else:
                    model_actual_path = model_opt_onnx_path
                    tokenizer_actual_path = model_opt_dir_path
            elif model["type"] == "image":
                # model_actual_path = model_onnx_path
                model_actual_path = model_opt_onnx_path
                tokenizer_actual_path = model_opt_dir_path
            else:
                raise ValueError(f"Unsupported model type: {model['type']}")
            

            if(not os.path.exists(model_actual_path)):
                print(f"Model {name} at {model_actual_path} doesn't exist.")
                logger.info(f"Model {name} at {model_actual_path} doesn't exist.")
                continue
            
            logger.info(f"Using model path: {model_actual_path} for model: {name}")
            logger.info(f"Using tokenizer path: {tokenizer_actual_path} for model: {name}")
            
            provider = model.get("provider", "CPUExecutionProvider")
            logger.info(f"Using provider: {provider} for model: {name}")
            optimization_level = model.get("optimization_level", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)
            # print(f"optimization_level: {optimization_level}")
            # logger.info(f"optimization_level: {optimization_level}")
            # opts = ort.SessionOptions()
            # opts.graph_optimization_level = InferenceManager.OPT_MAP.get(optimization_level, ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)
            # opts.intra_op_num_threads = model.get("intra_op_num_threads", 1)
            opts = Optimizer.create_session_options(optimization_level, 1)
            # # print(f"SessionOptions: {name}")
            # logger.info(f"SessionOptions: {name}")

            # if model["type"] == "text":
            #     # session = ort.InferenceSession(model_onnx_path, sess_options=opts, providers=[provider])
            #     if model.get("quantized", False):
            #         # session = ort.InferenceSession(model_quant_onnx_path, sess_options=opts, providers=[provider])
            #         session = ort.InferenceSession(model_quant_onnx_path, sess_options=opts, providers=[provider])
            #     else:
            #         session = ort.InferenceSession(model_opt_onnx_path, sess_options=opts, providers=[provider])
            # elif model["type"] == "image":
            #     # session = ort.InferenceSession(model_onnx_path, sess_options=opts, providers=[provider])
            #     session = ort.InferenceSession(model_opt_onnx_path, sess_options=opts, providers=[provider])
            # else:
            #     raise ValueError(f"Unsupported model type: {model['type']}")
            
            session = ort.InferenceSession(model_actual_path, sess_options=opts, providers=[provider])
            
            # print(f"ORT Session created for model: {name} with provider: {provider}")
            logger.info(f"ORT Session created for model: {name} with provider: {provider}")

            if model["type"] == "text":
                tokenClass = AutoTokenizer
            elif model["type"] == "image":
                tokenClass = AutoImageProcessor
                # tokenClass = AutoFeatureExtractor
            else:
                raise ValueError(f"Unsupported model type: {model['type']}")
            
            # tokenizer = tokenClass.from_pretrained("onnx_models")
            # tokenizer = tokenClass.from_pretrained(model_dir_path)
            # tokenizer = tokenClass.from_pretrained(model_opt_dir_path)
            # tokenizer = tokenClass.from_pretrained(model_quant_dir_path)

            tokenizer = tokenClass.from_pretrained(tokenizer_actual_path)
            # print(f"tokenizer: {name}")
            # logger.info(f"tokenizer: {name}")
        

            self.models[name] = session
            self.model_types[name] = model["type"]
            self.tokenizers[name] = tokenizer
            # self.profiler[name] = f"Profiler for {name}"
            # print(f"Model {name} loaded successfully.")
            logger.info(f"Model {name} loaded successfully.")
        
    def run_inference(self, model_name: str = "default_model", input_text_data: list[str] = None):
        print(f"Running Inference on model : {model_name} with inputs {input_text_data}" )
        logger.info(f"Running Inference on model : {model_name} with inputs {input_text_data}" )
        session = self.models.get(model_name)
        tokenizer = self.tokenizers.get(model_name)
        if session is None or tokenizer is None:
            raise ValueError(f"Model {model_name} not found.")
        
        if(self.model_types[model_name] == "text"):
            # print(f"Processing text input for model: {model_name}")
            # logger.info(f"Processing text input for model: {model_name}")

            # onnx_classifier = pipeline("text-classification", model=session, tokenizer=tokenizer, framework="onnx")

            # results = onnx_classifier(input_text_data)
            # print(f"Pipeline results: {results}")   
            # logger.info(f"Pipeline results: {results}")   

            sInputs = session.get_inputs()
            sOutputs = session.get_outputs()
            input_names  = [i.name for i in sInputs]
            output_names = [o.name for o in sOutputs]
            # print(f"Input Names: {input_names}")
            logger.info(f"Input Names: {input_names}")
            # print(f"Output Names: {output_names}")
            logger.info(f"Output Names: {output_names}")

            encoded_inputs = tokenizer(input_text_data, return_tensors="np", padding=True, truncation=True)
            # print(f"Encoded inputs: {encoded_inputs}")
            # logger.info(f"Encoded inputs: {encoded_inputs}")
            # ort_inputs = {k: v for k, v in encoded_inputs.items()}
            ort_inputs = {k: v.astype(np.int64) for k, v in encoded_inputs.items()}

            ort_inputs = {k: v for k, v in ort_inputs.items() if k in input_names}   

            # print(f"Prepared inputs for ONNX Runtime: {ort_inputs}")
            # logger.info(f"Prepared inputs for ONNX Runtime: {ort_inputs}")
            # outputs = session.run(None, ort_inputs)
            outputs = self.profiler.profile(session.run, None, ort_inputs)
            # print(f"Inference outputs: {outputs}")
            # logger.info(f"Inference outputs: {outputs}")
            # logger.info(f"Inference completed: {self.profiler.report()}")

            
            last_hidden_state = outputs[0]
            # print("BERT Output Shape:", last_hidden_state.shape)
            # logger.info("BERT Output Shape:", last_hidden_state.shape)

            logits = outputs[0] 
            probabilities = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
            predicted_class_ids = np.argmax(logits, axis=-1)
            return {
                # "logits": logits.tolist(),
                # "probabilities": probabilities.tolist(),
                "predicted_class_ids": predicted_class_ids.tolist(),
                "input_texts": input_text_data
            }

        elif(self.model_types[model_name] == "image"):
            # print(f"Processing image input for model: {model_name}")
            logger.info(f"Processing image input for model: {model_name}")
            # if not isinstance(input_text_data, Image.Image):
            #     raise ValueError("Input data must be a PIL Image for image models.")
            image = Image.open(input_text_data[0])
            
            encoded_inputs = tokenizer(images=image, return_tensors="np")
            # print(f"Encoded inputs: {encoded_inputs}")
            # logger.info(f"Encoded inputs: {encoded_inputs}")
            # ort_inputs = {k: v for k, v in encoded_inputs.items()}
            ort_inputs = {
                "pixel_values": encoded_inputs["pixel_values"].astype(np.float32)
            }

            # ort_inputs = {k: v for k, v in ort_inputs.items() if k in input_names}   
            
            # print(f"Prepared inputs for ONNX Runtime: {ort_inputs}")
            # logger.info(f"Prepared inputs for ONNX Runtime: {ort_inputs}")
            # outputs = session.run(None, ort_inputs)
            outputs = self.profiler.profile(session.run, None, ort_inputs)
            # print(f"Inference outputs: {outputs}")
            # logger.info(f"Inference outputs: {outputs}")
            # logger.info(f"Inference completed: {self.profiler.report()}")

            logits = outputs[0] 
            # print("Image Classification Logits:", logits)
            # logger.info("Image Classification Logits:", logits)
            # print("Output Shape Logits:", logits.shape)
            # logger.info("Output Shape Logits:", logits.shape)

            probabilities = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
            predicted_class_id = int(np.argmax(logits, axis=-1)[0])
            return {
                # "logits": logits.tolist()[0],
                # "probabilities": probabilities.tolist()[0],
                "predicted_class_id": predicted_class_id,
                "image_path": input_text_data[0]
            }

        
        
    