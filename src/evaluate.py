# src/evaluate.py

import torch
from tqdm import tqdm
from jiwer import cer, wer
from sacrebleu.metrics import BLEU
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.config import DEVICE, BATCH_SIZE, MODELS_DIR, EVALUATION_METRICS
from src.dataset import get_dataloaders

def calculate_metrics(predictions, references):
    """
    Calculates and returns a dictionary of evaluation metrics.
    """
    metrics = {}
    
    if "cer" in EVALUATION_METRICS:
        metrics["cer"] = cer(references, predictions)
    
    if "wer" in EVALUATION_METRICS:
        metrics["wer"] = wer(references, predictions)
        
    if "bleu" in EVALUATION_METRICS:
        # Sacrebleu expects a list of references for each prediction
        references_for_bleu = [[ref] for ref in references]
        bleu = BLEU()
        metrics["bleu"] = bleu.corpus_score(predictions, references_for_bleu).score
        
    return metrics

def run_evaluation(model_path, processor_path):
    """
    Runs the evaluation on the test set using a trained model.
    """
    # Load the trained model and processor
    model = VisionEncoderDecoderModel.from_pretrained(model_path).to(DEVICE)
    processor = TrOCRProcessor.from_pretrained(processor_path)

    # Get the test dataloader
    _, _, test_loader = get_dataloaders(processor, BATCH_SIZE)

    model.eval()
    predictions = []
    references = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on Test Set"):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"]

            # Generate predictions
            generated_ids = model.generate(pixel_values, max_length=128)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Decode reference labels
            labels[labels == -100] = processor.tokenizer.pad_token_id
            reference_text = processor.batch_decode(labels, skip_special_tokens=True)

            predictions.extend(generated_text)
            references.extend(reference_text)

    # Calculate and print the metrics
    metrics = calculate_metrics(predictions, references)
    
    print("\n--- Evaluation Results ---")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.upper()}: {metric_value:.4f}")
    print("--------------------------")
    
    # Optional: Save results to a file
    with open(f"{MODELS_DIR}/evaluation_results.txt", "w", encoding="utf-8") as f:
        for pred, ref in zip(predictions, references):
            f.write(f"Prediction: {pred}\n")
            f.write(f"Reference:  {ref}\n\n")
        f.write("\n--- Metrics ---\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name.upper()}: {metric_value:.4f}\n")

if __name__ == "__main__":
    # Path to a trained model and processor (replace with the actual path after training)
    # For demonstration, this will likely fail if a model hasn't been trained yet.
    MODEL_CHECKPOINT_PATH = f"{MODELS_DIR}/image_to_viet_text_model_epoch_1"
    PROCESSOR_CHECKPOINT_PATH = f"{MODELS_DIR}/image_to_viet_text_processor_epoch_1"
    
    try:
        run_evaluation(MODEL_CHECKPOINT_PATH, PROCESSOR_CHECKPOINT_PATH)
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        print("Please ensure you have a trained model and processor saved at the specified paths.")
