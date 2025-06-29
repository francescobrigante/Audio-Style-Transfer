import torch
import torch.nn as nn
from style_encoder import *
from content_encoder import ContentEncoder

NUM_INSTRUMENT_CLASSES = 2 


# Given as input style_emb (B, transformer_dim) and class_emb (2, transformer_dim) from style encoder +
# + content_emb (B, S, transformer_dim) from content encoder,
# the discriminator predicts instrument class of style_emb, content_emb and class_emb.
# The loss is computed as a cross-entropy loss for style_emb and class_emb, and a uniformity loss for content_emb.
# The goal is to have high accuracy for style_emb and class_emb, while content_emb should be random => no information about instrument class from content
class Discriminator(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        
        # simple MLP
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, NUM_INSTRUMENT_CLASSES),  # outputs logits for two classes
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)                                                # (B, 2) or (2, 2) for class_emb