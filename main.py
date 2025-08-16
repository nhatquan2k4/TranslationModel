# main.py

import argparse
import os

from src.train import run_training
from src.evaluate import run_evaluation
from src.inference import ImageToTextInference
from src.config import MODELS_DIR

def main():
    """
    Main function to orchestrate the pipeline.
    """
    parser = argparse.ArgumentParser(description="End-to-End Image to Vietnamese Text Model")
    parser.add_argument("--train", action="store_true", help="Run the training pipeline.")
    parser.add_argument("--evaluate", action="store_true", help="Run the evaluation pipeline.")
    parser.add_argument("--inference", type=str, help="Run inference on a single image. Provide the image path.")
    parser.add_argument("--model_path", type=str, default=f"{MODELS_DIR}/image_to_viet_text_model_epoch_1", help="Path to the trained model.")
    parser.add_argument("--processor_path", type=str, default=f"{MODELS_DIR}/image_to_viet_text_processor_epoch_1", help="Path to the processor.")

    args = parser.parse_args()

    if args.train:
        print("--- Starting Training ---")
        run_training()
        print("--- Training Finished ---")

    elif args.evaluate:
        print("--- Starting Evaluation ---")
        if not os.path.exists(args.model_path) or not os.path.exists(args.processor_path):
            print(f"Error: Model or processor not found at the specified paths.")
            print("Please train a model first using the --train flag.")
            return
        run_evaluation(args.model_path, args.processor_path)
        print("--- Evaluation Finished ---")

    elif args.inference:
        print("--- Starting Inference ---")
        if not os.path.exists(args.model_path) or not os.path.exists(args.processor_path):
            print(f"Error: Model or processor not found at the specified paths.")
            print("Please train a model first using the --train flag.")
            return
        if not os.path.exists(args.inference):
            print(f"Error: Image file not found at {args.inference}")
            return
            
        inference_pipeline = ImageToTextInference(args.model_path, args.processor_path)
        predicted_text = inference_pipeline.predict(args.inference)
        
        print(f"Image Path: {args.inference}")
        print(f"Predicted Text: {predicted_text}")
        print("--- Inference Finished ---")

    else:
        print("No action specified. Use --train, --evaluate, or --inference.")
        print("Example usage:")
        print("  python main.py --train")
        print("  python main.py --evaluate --model_path <path_to_model>")
        print("  python main.py --inference <path_to_image> --model_path <path_to_model>")

if __name__ == "__main__":
    main()
