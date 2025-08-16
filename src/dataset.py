# src/dataset.py
import os
import random
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import TrOCRProcessor
import string

from src.config import (
    SYNTHETIC_DATA_COUNT,
    SYNTHETIC_FONTS_DIR,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
)

class ImageTextDataset(Dataset):
    """
    A custom dataset class to handle image-text pairs for the OCR and translation task.
    It can load data from a directory or generate synthetic data.
    """
    def __init__(self, data_path, processor=None, synthetic=False):
        """
        Args:
            data_path (str): Path to the directory containing images and labels.
            processor (TrOCRProcessor): Processor for image transformations.
            synthetic (bool): If True, generate synthetic data.
        """
        self.data_path = data_path
        self.processor = processor
        self.synthetic = synthetic
        self.image_paths = []
        self.labels = []

        if self.synthetic:
            self._generate_synthetic_data()
        else:
            self._load_real_data()

    def _generate_synthetic_data(self):
        """
        Generates synthetic data using Pillow (không cần TextRecognitionDataGenerator).
        Tạo ảnh trắng, vẽ text tiếng Anh lên ảnh.
        """
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(SYNTHETIC_FONTS_DIR, exist_ok=True)

        english_texts = [
            "".join(random.choices(string.ascii_letters + string.digits, k=random.randint(5, 20)))
            for _ in range(SYNTHETIC_DATA_COUNT)
        ]
        vietnamese_texts = ["vi du " + text for text in english_texts]

        # Chọn font mặc định hoặc font .ttf trong thư mục SYNTHETIC_FONTS_DIR
        font_path = None
        font_files = [f.path for f in os.scandir(SYNTHETIC_FONTS_DIR) if f.name.endswith(('.ttf', '.otf'))]
        if font_files:
            font_path = font_files[0]

        for i, text in enumerate(english_texts):
            img = Image.new("RGB", (200, 50), color="white")
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype(font_path, 32) if font_path else ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()
            draw.text((10, 10), text, font=font, fill="black")
            img_path = os.path.join(self.data_path, f"synth_{i}.png")
            img.save(img_path)
            self.image_paths.append(img_path)
            self.labels.append(vietnamese_texts[i])

    def _load_real_data(self):
        """
        Loads real data from a directory. Assumes a 'labels.txt' file
        with format: 'image_name.png\tlabel_text'.
        """
        labels_file = os.path.join(self.data_path, "labels.txt")
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"labels.txt not found in {self.data_path}")

        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    img_name, label = parts
                    img_path = os.path.join(self.data_path, "images", img_name)
                    if os.path.exists(img_path):
                        self.image_paths.append(img_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")

        if self.processor:
            # The processor handles resizing, normalization, and tokenization
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
            labels = self.processor(text=label, return_tensors="pt").input_ids.squeeze(0)
            return {"pixel_values": pixel_values, "labels": labels}
        
        return {"image": image, "text": label}

def get_dataloaders(processor, batch_size):
    """
    Creates and returns the training, validation, and test dataloaders.
    """
    # For demonstration, we'll use synthetic data.
    # In a real scenario, you would have separate directories for train, val, test.
    
    # Create dummy directories for demonstration
    os.makedirs(TRAIN_DATA_PATH, exist_ok=True)
    os.makedirs(VAL_DATA_PATH, exist_ok=True)
    os.makedirs(TEST_DATA_PATH, exist_ok=True)

    full_dataset = ImageTextDataset(data_path=TRAIN_DATA_PATH, processor=processor, synthetic=True)

    # Split the dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Example of how to use the dataset and dataloaders
    # This requires the transformers library to be installed.
    # You might need to adjust the model name based on availability.
    try:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        train_loader, val_loader, test_loader = get_dataloaders(processor, batch_size=4)

        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")
        print(f"Number of test batches: {len(test_loader)}")

        # Check a sample from the dataloader
        sample = next(iter(train_loader))
        print("Sample pixel values shape:", sample["pixel_values"].shape)
        print("Sample labels shape:", sample["labels"].shape)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have an internet connection and the necessary libraries installed.")

