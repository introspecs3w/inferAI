
import argparse
import os

import yaml
from inference import greet
from inference.model_converter import ModelConverter
from inference.inference_manager import InferenceManager
from PIL import Image
from inference.logger import setup_logger
from inference.profiler import Profiler



def main() -> None:
    parser = argparse.ArgumentParser(prog="inferai")
    parser.add_argument("name", nargs="?", default="World", help="Name to greet")
    args = parser.parse_args()
    os.makedirs("logs", exist_ok=True)
    logger = setup_logger(__name__, log_file="logs/inferai.log")
    
    print(greet(args.name))
    logger.info(f"Greeted {args.name}")
    logger.info(f"********************************************")
    logger.info(f"********************************************")


    config_path = "configs/config.yaml"
    text_inputs_path = "inputs/text_inputs.yaml"
    image_inputs_path = "inputs/image_inputs.yaml"

    with open(text_inputs_path, "r") as f:
        text_inputs = yaml.safe_load(f)["text_inputs"]  

    with open(image_inputs_path, "r") as f:
        image_inputs = yaml.safe_load(f)["image_inputs"]  
    
    logger.info(f"********************************************")
    logger.info(f"Starting ModelConverter")
    logger.info(f"********************************************")

    converter = ModelConverter(config_path)
    converter.convert_all_models()

    logger.info(f"********************************************")
    logger.info(f"Starting InferenceManager")
    logger.info(f"********************************************")
    profiler = Profiler()
    manager = InferenceManager(config_path, profiler=profiler)
    logger.info(f"********************************************")

    profiler.clear()

    for text in text_inputs:
        output = manager.run_inference("distilbert-base-uncased", [text])
        logger.info(f"Input: {text}")
        logger.info(f"Predicted class: {output['predicted_class_ids']}")
        logger.info(f"Probabilities: {output['probabilities']}")

    logger.info(f"********************************************")
    logger.info(f"********************************************")
    logger.info(f"Inference completed: {profiler.report()}")
    logger.info(f"********************************************")
    logger.info(f"********************************************")

    print(f"********************************************")
    print(f"********************************************")
    print(f"Inference completed: {profiler.report()}")
    print(f"********************************************")
    print(f"********************************************")



    profiler.clear()

    # image = Image.open("inputs/flower.jpg")
    for image in image_inputs:
        imagePath = os.path.join("inputs", image)
        output = manager.run_inference("google/vit-base-patch16-224", [imagePath])
        logger.info(f"Input: {image}")
        logger.info(f"Predicted class: {output['predicted_class_ids']}")
        # logger.info(f"Probabilities: {output['probabilities']}")

    logger.info(f"********************************************")
    logger.info(f"********************************************")
    logger.info(f"Inference completed: {profiler.report()}")
    logger.info(f"********************************************")
    logger.info(f"********************************************")

    print(f"********************************************")
    print(f"********************************************")
    print(f"Inference completed: {profiler.report()}")
    print(f"********************************************")
    print(f"********************************************")



if __name__ == "__main__":
    main()
