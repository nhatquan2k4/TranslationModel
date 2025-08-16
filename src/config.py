# config.py

import torch

# -- Project Paths --
PROJECT_DIR = "C:/Users/nhatq/ImageToVietTextProject"
DATA_DIR = f"{PROJECT_DIR}/data"
MODELS_DIR = f"{PROJECT_DIR}/models"
SRC_DIR = f"{PROJECT_DIR}/src"

# -- Dataset Configuration --
# Path to the training, validation, and test sets
TRAIN_DATA_PATH = f"{DATA_DIR}/train"
VAL_DATA_PATH = f"{DATA_DIR}/val"
TEST_DATA_PATH = f"{DATA_DIR}/test"

# For synthetic data generation
SYNTHETIC_DATA_COUNT = 10000
SYNTHETIC_FONTS_DIR = f"{DATA_DIR}/fonts"  # Path to a directory of .ttf or .otf fonts

# -- Preprocessing Configuration --
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 512  # Adjust based on typical aspect ratio
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# -- Model Configuration --
# Feature Extractor (Backbone)
BACKBONE_MODEL = "google/vit-base-patch16-224-in21k"  # Vision Transformer

# Encoder
ENCODER_DIM = 768  # Should match the backbone's output dimension
ENCODER_LAYERS = 6
ENCODER_HEADS = 8
ENCODER_DROPOUT = 0.1

# Decoder
DECODER_DIM = 768
DECODER_LAYERS = 6
DECODER_HEADS = 8
DECODER_DROPOUT = 0.1
VOCAB_SIZE = 10000  # Adjust based on the BPE tokenizer vocabulary
MAX_SEQ_LENGTH = 128  # Max length of the Vietnamese text sequence

# -- Tokenizer Configuration --
TOKENIZER_PATH = f"{MODELS_DIR}/bpe_tokenizer"
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]

# -- Training Configuration --
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS_PRETRAIN = 10
NUM_EPOCHS_FINETUNE = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
CLIP_GRAD_NORM = 1.0

# -- Inference Configuration --
BEAM_WIDTH = 5

# -- Evaluation Metrics --
# Choose from 'cer', 'wer', 'bleu'
EVALUATION_METRICS = ["cer", "wer", "bleu"]
