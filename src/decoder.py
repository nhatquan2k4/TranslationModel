# src/decoder.py

import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from src.config import (
    DECODER_DIM,
    DECODER_LAYERS,
    DECODER_HEADS,
    DECODER_DROPOUT,
    VOCAB_SIZE,
    MAX_SEQ_LENGTH,
    TOKENIZER_PATH,
    SPECIAL_TOKENS,
)
from src.encoder import PositionalEncoding

class Seq2SeqDecoder(nn.Module):
    """
    The decoder part of the Seq2Seq model. It generates the output sequence (Vietnamese text)
    based on the context vectors from the encoder.
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=DECODER_DIM, nhead=DECODER_HEADS, num_layers=DECODER_LAYERS, dropout=DECODER_DROPOUT):
        """
        Args:
            vocab_size (int): The size of the output vocabulary.
            d_model (int): The number of expected features in the input.
            nhead (int): The number of heads in the multiheadattention models.
            num_layers (int): The number of sub-decoder-layers in the decoder.
            dropout (float): The dropout value.
        """
        super(Seq2SeqDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Forward pass for the decoder.
        
        Args:
            tgt (torch.Tensor): The sequence to the decoder of shape (B, T),
                                where T is the target sequence length.
            memory (torch.Tensor): The sequence from the last layer of the encoder of shape (B, S, E).
            tgt_mask (torch.Tensor): The mask for the target sequence.
            memory_mask (torch.Tensor): The mask for the memory sequence.
                                
        Returns:
            torch.Tensor: The output sequence of shape (B, T, V), where V is the vocab size.
        """
        tgt = self.embedding(tgt)
        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_mask)
        output = self.fc_out(output)
        return output

def get_bpe_tokenizer(corpus_file, vocab_size=VOCAB_SIZE):
    """
    Trains a BPE tokenizer from a corpus file.
    
    Args:
        corpus_file (str): Path to the text file containing the corpus for training the tokenizer.
        vocab_size (int): The size of the vocabulary.
        
    Returns:
        tokenizers.Tokenizer: The trained BPE tokenizer.
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train([corpus_file], trainer)
    tokenizer.save(TOKENIZER_PATH)
    return tokenizer

if __name__ == "__main__":
    # Example of how to use the Seq2SeqDecoder
    # Dummy context vectors from the encoder
    # (batch_size, seq_len, dim) -> (4, 197, 768)
    dummy_context_vectors = torch.randn(4, 197, 768)

    # Dummy target sequence (e.g., during training with teacher forcing)
    # (batch_size, target_seq_len) -> (4, 50)
    dummy_target_seq = torch.randint(0, VOCAB_SIZE, (4, 50))

    # Initialize the decoder
    decoder = Seq2SeqDecoder()

    # Get the output logits
    output_logits = decoder(dummy_target_seq, dummy_context_vectors)

    print("Input context vectors shape:", dummy_context_vectors.shape)
    print("Input target sequence shape:", dummy_target_seq.shape)
    print("Output logits shape:", output_logits.shape)
    # The output shape should be (batch_size, target_seq_len, vocab_size)
    assert output_logits.shape == (4, 50, VOCAB_SIZE)

    # Example of training a tokenizer
    # Create a dummy corpus file
    dummy_corpus = "C:/Users/nhatq/ImageToVietTextProject/data/dummy_corpus.txt"
    with open(dummy_corpus, "w", encoding="utf-8") as f:
        f.write("Đây là một ví dụ về tokenizer.\n")
        f.write("Chúng ta sẽ huấn luyện một tokenizer BPE.\n")
    
    try:
        tokenizer = get_bpe_tokenizer(dummy_corpus)
        print("\nTokenizer trained and saved to:", TOKENIZER_PATH)
        encoded = tokenizer.encode("Đây là một ví dụ")
        print("Encoded example:", encoded.tokens)
        print("Decoded example:", tokenizer.decode(encoded.ids))
    except Exception as e:
        print(f"\nAn error occurred during tokenizer training: {e}")
