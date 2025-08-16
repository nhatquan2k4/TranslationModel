# src/encoder.py

import torch
import torch.nn as nn
import math

from src.config import ENCODER_DIM, ENCODER_LAYERS, ENCODER_HEADS, ENCODER_DROPOUT

class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as the
    embeddings, so that the two can be summed.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Seq2SeqEncoder(nn.Module):
    """
    The encoder part of the Seq2Seq model. It takes the feature map from the
    backbone and encodes it into a sequence of context vectors.
    """
    def __init__(self, d_model=ENCODER_DIM, nhead=ENCODER_HEADS, num_layers=ENCODER_LAYERS, dropout=ENCODER_DROPOUT):
        """
        Args:
            d_model (int): The number of expected features in the input.
            nhead (int): The number of heads in the multiheadattention models.
            num_layers (int): The number of sub-encoder-layers in the encoder.
            dropout (float): The dropout value.
        """
        super(Seq2SeqEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        """
        Forward pass for the encoder.
        
        Args:
            src (torch.Tensor): The sequence to the encoder of shape (B, S, E),
                                where S is the source sequence length, B is the batch size,
                                and E is the feature number.
                                
        Returns:
            torch.Tensor: The encoded sequence of shape (B, S, E).
        """
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

if __name__ == "__main__":
    # Example of how to use the Seq2SeqEncoder
    # Dummy features from the feature extractor
    # (batch_size, num_patches + 1, hidden_size) -> (4, 197, 768)
    dummy_features = torch.randn(4, 197, 768)

    # Initialize the encoder
    encoder = Seq2SeqEncoder()

    # Get the context vectors
    context_vectors = encoder(dummy_features)

    print("Input features shape:", dummy_features.shape)
    print("Output context vectors shape:", context_vectors.shape)
    # The output shape should be the same as the input shape
    assert dummy_features.shape == context_vectors.shape
