# src/preprocess.py

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image

from src.config import IMAGE_HEIGHT, IMAGE_WIDTH, MEAN, STD

def get_train_transforms():
    """
    Returns the augmentation pipeline for the training set.
    """
    return A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, always_apply=True),
            A.Rotate(limit=5, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=MEAN, std=STD, always_apply=True),
            ToTensorV2(),
        ],
        bbox_params=None,
        keypoint_params=None,
        p=1.0,
    )

def get_val_test_transforms():
    """
    Returns the transformation pipeline for the validation and test sets.
    """
    return A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, always_apply=True),
            A.Normalize(mean=MEAN, std=STD, always_apply=True),
            ToTensorV2(),
        ],
        bbox_params=None,
        keypoint_params=None,
        p=1.0,
    )

class CustomProcessor:
    """
    A custom processor to apply transformations to images.
    This can be integrated with the dataset class if not using Hugging Face's processor.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        """
        Apply the transformations to an image.
        
        Args:
            image (PIL.Image): The input image.
            
        Returns:
            torch.Tensor: The transformed image as a tensor.
        """
        # Albumentations works with numpy arrays
        image_np = np.array(image.convert("RGB"))
        transformed = self.transforms(image=image_np)
        return transformed["image"]

if __name__ == "__main__":
    # Example of how to use the custom processor
    # Create a dummy image
    dummy_image = Image.new('RGB', (200, 50), color = 'red')

    # Get the training transforms
    train_transforms = get_train_transforms()
    
    # Create a processor
    processor = CustomProcessor(transforms=train_transforms)
    
    # Process the image
    transformed_image = processor(dummy_image)
    
    print("Original image mode:", dummy_image.mode)
    print("Transformed image shape:", transformed_image.shape)
    print("Transformed image dtype:", transformed_image.dtype)
    
    # Example for validation transforms
    val_transforms = get_val_test_transforms()
    val_processor = CustomProcessor(transforms=val_transforms)
    val_transformed_image = val_processor(dummy_image)
    
    print("\nValidation transformed image shape:", val_transformed_image.shape)
