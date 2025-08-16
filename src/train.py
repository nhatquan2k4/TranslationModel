# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

from src.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_EPOCHS_PRETRAIN,
    NUM_EPOCHS_FINETUNE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    CLIP_GRAD_NORM,
    MODELS_DIR,
    BACKBONE_MODEL,
)
from src.dataset import get_dataloaders
from src.feature_extraction import FeatureExtractor
from src.encoder import Seq2SeqEncoder
from src.decoder import Seq2SeqDecoder

class ImageToVietTextModel(nn.Module):
    """
    The complete end-to-end model that combines the feature extractor,
    encoder, and decoder.
    """
    def __init__(self, vocab_size):
        super(ImageToVietTextModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.encoder = Seq2SeqEncoder()
        self.decoder = Seq2SeqDecoder(vocab_size=vocab_size)

    def forward(self, pixel_values, decoder_input_ids):
        features = self.feature_extractor(pixel_values)
        context_vectors = self.encoder(features)
        outputs = self.decoder(decoder_input_ids, context_vectors)
        return outputs

def train_one_epoch(model, dataloader, optimizer, criterion, device, clip_norm):
    """
    Trains the model for one epoch.
    """
    model.train()
    epoch_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Teacher forcing: use the ground truth labels as input to the decoder
        outputs = model(pixel_values, labels[:, :-1])
        
        # Reshape for loss calculation
        loss = criterion(outputs.reshape(-1, outputs.shape[-1]), labels[:, 1:].reshape(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values, labels[:, :-1])
            
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), labels[:, 1:].reshape(-1))
            
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def run_training():
    """
    The main function to run the training and evaluation loops.
    """
    # For simplicity, we use the TrOCR processor and model for a baseline.
    # A custom model would require more intricate wiring.
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    
    # Using a pre-built VisionEncoderDecoderModel for a more robust example
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
    model.to(DEVICE)

    # Get dataloaders
    train_loader, val_loader, _ = get_dataloaders(processor, BATCH_SIZE)

    # Optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=processor.tokenizer.pad_token_id)

    print("Starting fine-tuning...")
    for epoch in range(NUM_EPOCHS_FINETUNE):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, CLIP_GRAD_NORM)
        val_loss = evaluate(model, val_loader, criterion, DEVICE)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS_FINETUNE}")
        print(f"\tTrain Loss: {train_loss:.4f}")
        print(f"\tVal Loss:   {val_loss:.4f}")

        # Save the model checkpoint
        model.save_pretrained(f"{MODELS_DIR}/image_to_viet_text_model_epoch_{epoch+1}")
        processor.save_pretrained(f"{MODELS_DIR}/image_to_viet_text_processor_epoch_{epoch+1}")

    print("Training finished.")

if __name__ == "__main__":
    try:
        run_training()
    except Exception as e:
        print(f"An error occurred during training: {e}")
        print("Please ensure you have an internet connection, a CUDA-enabled GPU if using 'cuda', and the necessary libraries installed.")
