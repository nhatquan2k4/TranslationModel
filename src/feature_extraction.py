# src/feature_extraction.py

import torch
import torch.nn as nn
from transformers import ViTModel

from src.config import BACKBONE_MODEL

class FeatureExtractor(nn.Module):
    """
    The feature extraction backbone. It uses a pre-trained Vision Transformer (ViT)
    to extract high-level features from the input images.
    """
    def __init__(self, model_name=BACKBONE_MODEL, freeze=False):
        """
        Args:
            model_name (str): The name of the pre-trained ViT model from Hugging Face.
            freeze (bool): If True, freeze the backbone's weights during training.
        """
        super(FeatureExtractor, self).__init__()
        self.model = ViTModel.from_pretrained(model_name)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, pixel_values):
        """
        Forward pass through the ViT model.
        
        Args:
            pixel_values (torch.Tensor): The input images tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: The output features from the ViT model's last hidden state.
        """
        outputs = self.model(pixel_values=pixel_values)
        # We use the last hidden state as the feature representation
        return outputs.last_hidden_state

if __name__ == "__main__":
    # Example of how to use the FeatureExtractor
    # Create a dummy input tensor
    dummy_input = torch.randn(4, 3, 224, 224)  # Batch of 4 images, 3 channels, 224x224 resolution

    # Initialize the feature extractor
    try:
        feature_extractor = FeatureExtractor()
        
        # Get the features
        features = feature_extractor(dummy_input)
        
        print("Input shape:", dummy_input.shape)
        print("Output features shape:", features.shape)
        # The output shape will be (batch_size, num_patches + 1, hidden_size)
        # For ViT-base, this is (4, 197, 768)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have an internet connection and the necessary libraries installed.")

