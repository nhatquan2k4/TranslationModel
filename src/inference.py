# src/inference.py

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.config import DEVICE, MODELS_DIR, BEAM_WIDTH

class ImageToTextInference:
    """
    A class for running inference with the trained ImageToVietText model.
    """
    def __init__(self, model_path, processor_path):
        """
        Args:
            model_path (str): Path to the trained model directory.
            processor_path (str): Path to the trained processor directory.
        """
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path).to(DEVICE)
        self.processor = TrOCRProcessor.from_pretrained(processor_path)
        self.model.eval()

    def predict(self, image_path):
        """
        Predicts the Vietnamese text from an input image.
        
        Args:
            image_path (str): The path to the input image file.
            
        Returns:
            str: The predicted Vietnamese text.
        """
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)

        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                num_beams=BEAM_WIDTH,
                max_length=128,
                early_stopping=True
            )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Optional: Add post-processing (e.g., spell correction) here
        
        return generated_text

if __name__ == "__main__":
    # Path to a trained model and processor
    MODEL_CHECKPOINT_PATH = f"{MODELS_DIR}/image_to_viet_text_model_epoch_1"
    PROCESSOR_CHECKPOINT_PATH = f"{MODELS_DIR}/image_to_viet_text_processor_epoch_1"
    
    # Create a dummy image for inference
    dummy_image_path = "C:/Users/nhatq/ImageToVietTextProject/data/dummy_inference_image.png"
    Image.new('RGB', (200, 50), color = 'blue').save(dummy_image_path)

    try:
        # Initialize the inference pipeline
        inference_pipeline = ImageToTextInference(MODEL_CHECKPOINT_PATH, PROCESSOR_CHECKPOINT_PATH)
        
        # Get the prediction
        predicted_text = inference_pipeline.predict(dummy_image_path)
        
        print(f"--- Inference Example ---")
        print(f"Image: {dummy_image_path}")
        print(f"Predicted Text: {predicted_text}")
        print("-------------------------")
        
    except Exception as e:
        print(f"An error occurred during inference: {e}")
        print("Please ensure you have a trained model and processor saved at the specified paths.")
