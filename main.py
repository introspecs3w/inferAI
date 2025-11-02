
import argparse
from inference import greet
from inference.model_converter import ModelConverter
from inference.inference_manager import InferenceManager
from PIL import Image
from inference.logger import setup_logger
from inference.profiler import Profiler

logger = setup_logger(__name__, log_file="inferai.log")


def main() -> None:
    parser = argparse.ArgumentParser(prog="inferai")
    parser.add_argument("name", nargs="?", default="World", help="Name to greet")
    args = parser.parse_args()
    print(greet(args.name))
    logger.info(f"Greeted {args.name}")
    # print(f"********************************************")
    # print(f"********************************************")
    logger.info(f"********************************************")
    logger.info(f"********************************************")

    config_path = "configs/config.yaml"
    # print(f"Using configuration file at: {config_path}")
    
    # print(f"********************************************")
    # print(f"Starting ModelConverter")
    # print(f"********************************************")
    logger.info(f"********************************************")
    logger.info(f"Starting ModelConverter")
    logger.info(f"********************************************")

    converter = ModelConverter(config_path)
    converter.convert_all_models()

    # print(f"********************************************")
    # print(f"Starting InferenceManager")
    # print(f"********************************************")
    logger.info(f"********************************************")
    logger.info(f"Starting InferenceManager")
    logger.info(f"********************************************")
    profiler = Profiler()
    manager = InferenceManager(config_path, profiler=profiler)
    # print(f"********************************************")
    logger.info(f"********************************************")

                # text_input = ["Hello world!"]
                # # print(f"Input texts for inference: {text_input}")
                # # logger.info(f"Input texts for inference: {text_input}")
                # # manager.run_inference()
                # output1 = manager.run_inference("bert-base-uncased", text_input)
                # # print(f"output1: {output1}")
                # # print(f"********************************************")
                # logger.info(f"output1: {output1}")
                # logger.info(f"********************************************")

                # text_input = ["Testing inference."]
                # # print(f"Input texts for inference: {text_input}")
                # # logger.info(f"Input texts for inference: {text_input}")
                # # manager.run_inference()
                # output1 = manager.run_inference("bert-base-uncased", text_input)
                # # print(f"output1: {output1}")
                # # print(f"********************************************")
                # logger.info(f"output1: {output1}")
                # logger.info(f"********************************************")

                # text_input = ["Go Away!"]
                # # print(f"Input texts for inference: {text_input}")
                # # logger.info(f"Input texts for inference: {text_input}")
                # # manager.run_inference()
                # output1 = manager.run_inference("bert-base-uncased", text_input)
                # # print(f"output1: {output1}")
                # logger.info(f"output1: {output1}")
                # # print(f"********************************************")
                # # print(f"********************************************")
                # # print(f"********************************************")
                # logger.info(f"********************************************")
                # logger.info(f"********************************************")
                # logger.info(f"********************************************")

    profiler.clear()

    text_input = ["Hello world!"]
    output1 = manager.run_inference("distilbert-base-uncased", text_input)
    logger.info(f"output1: {output1}")

    text_input = ["Testing inference."]
    output1 = manager.run_inference("distilbert-base-uncased", text_input)
    logger.info(f"output1: {output1}")

    text_input = ["Go Away!"]
    output1 = manager.run_inference("distilbert-base-uncased", text_input)
    logger.info(f"output1: {output1}")

    text_input = ["Hello world!"]
    output1 = manager.run_inference("distilbert-base-uncased", text_input)
    logger.info(f"output1: {output1}")

    text_input = ["Testing inference."]
    output1 = manager.run_inference("distilbert-base-uncased", text_input)
    logger.info(f"output1: {output1}")

    text_input = ["Go Away!"]
    output1 = manager.run_inference("distilbert-base-uncased", text_input)
    logger.info(f"output1: {output1}")

    text_input = ["Hello world!"]
    output1 = manager.run_inference("distilbert-base-uncased", text_input)
    logger.info(f"output1: {output1}")

    text_input = ["Testing inference."]
    output1 = manager.run_inference("distilbert-base-uncased", text_input)
    logger.info(f"output1: {output1}")

    text_input = ["Go Away!"]
    output1 = manager.run_inference("distilbert-base-uncased", text_input)
    logger.info(f"output1: {output1}")

    logger.info(f"********************************************")
    logger.info(f"********************************************")
    logger.info(f"Inference completed: {profiler.report()}")
    logger.info(f"********************************************")
    logger.info(f"********************************************")










    profiler.clear()

    # image = Image.open("inputs/flower.jpg")
    image_path = "inputs/flower.jpg"
    output2 = manager.run_inference("google/vit-base-patch16-224", [image_path])
    logger.info(f"output2: {output2}")

    image_path = "inputs/tree.jpg"
    output2 = manager.run_inference("google/vit-base-patch16-224", [image_path])
    logger.info(f"output2: {output2}")

    image_path = "inputs/ladybug.jpg"
    output2 = manager.run_inference("google/vit-base-patch16-224", [image_path])
    logger.info(f"output2: {output2}")

    image_path = "inputs/flower.jpg"
    output2 = manager.run_inference("google/vit-base-patch16-224", [image_path])
    logger.info(f"output2: {output2}")

    image_path = "inputs/tree.jpg"
    output2 = manager.run_inference("google/vit-base-patch16-224", [image_path])
    logger.info(f"output2: {output2}")

    image_path = "inputs/ladybug.jpg"
    output2 = manager.run_inference("google/vit-base-patch16-224", [image_path])
    logger.info(f"output2: {output2}")
    
    image_path = "inputs/flower.jpg"
    output2 = manager.run_inference("google/vit-base-patch16-224", [image_path])
    logger.info(f"output2: {output2}")

    image_path = "inputs/tree.jpg"
    output2 = manager.run_inference("google/vit-base-patch16-224", [image_path])
    logger.info(f"output2: {output2}")

    image_path = "inputs/ladybug.jpg"
    output2 = manager.run_inference("google/vit-base-patch16-224", [image_path])
    logger.info(f"output2: {output2}")

    logger.info(f"********************************************")
    logger.info(f"********************************************")
    logger.info(f"Inference completed: {profiler.report()}")
    logger.info(f"********************************************")
    logger.info(f"********************************************")




if __name__ == "__main__":
    main()
