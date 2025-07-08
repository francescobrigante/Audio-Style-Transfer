import torch
import torch.nn as nn
import math
from style_encoder import ResBlock, SinusoidalPositionalEncoding, initialize_weights
import torch.nn.utils as utils

# it follows the same structure as the style encoder
# we no longer use CLS token
class ContentEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        cnn_out_dim: int = 256,
        transformer_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        channels_list: list = [32, 64, 128, 256, 512, 512]
    ):
        super().__init__()
        
        # CNN with spectral normalization
        layers = []
        prev_chan_size = in_channels
        
        # number to set how many ResBlocks to downsample
        downsample_number = 100
        for idx, chan_size in enumerate(channels_list):
            downsample_boolean = idx < downsample_number
            layers.append(ResBlock(prev_chan_size, chan_size, downsample=downsample_boolean))
            prev_chan_size = chan_size

            
        # global average pooling to time-frequency dimension
        # (B*S, last_chan_size=512, 5, 10) if downsample applied to all blocks
            
        layers.append(nn.AdaptiveAvgPool2d((2, 5)))  # (B*S, last_chan_size=512, 2, 5)
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))             # (B*S, last_chan_size=512, 1, 1)
        self.cnn = nn.Sequential(*layers)
        
        # in content encoder try removing layers.append(nn.AdaptiveAvgPool2d((1, 1))) to make final output shape 512*2*5 = 5120 with view in forward
        # so we will be projecting from 512*2*5 = 5120 to out_dim = 512 keeping granularity
        
        # projection to final cnn embedding dimension
        # flat_dim = prev_chan_size * 2 * 5
        # self.proj = nn.Linear(flat_dim, cnn_out_dim)
        self.proj = nn.Linear(prev_chan_size, cnn_out_dim)
        
        # Linear projection (CNN out dim -> transformer dim) if needed
        self.input_proj = (
            nn.Linear(cnn_out_dim, transformer_dim)
            if cnn_out_dim != transformer_dim else None
        )
        
        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(transformer_dim)

        # layer norm
        self.norm = nn.LayerNorm(transformer_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=0.1,                                    # change to 0.2 if overfitting
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, 2, T, F) tensor of STFT+CQT chunks
        Returns:
            content_emb: (B, S, transformer_dim) sequence of content embeddings
        """
        B, S, C, T, F = x.shape
        
        # CNN
        x = x.view(B * S, C, T, F)                                                      # (B*S, 2, T, F)
        cnn_features = self.cnn(x)                                                      # (B*S, last_chan_size, 2, 5)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)                      # (B*S, last_chan_size*2*5)
        
        # Project to cnn_out_dim if last_chan_size != cnn_out_dim
        cnn_features = self.proj(cnn_features)                                          # (B*S, cnn_out_dim)
        seq = cnn_features.view(B, S, -1)                                               # (B, S, cnn_out_dim)
        
        # Project to transformer_dim if cnn_out_dim != transformer_dim
        if self.input_proj:
            seq = self.input_proj(seq)                                                  # (B, S, transformer_dim)
        
        # Positional encoding
        seq = self.pos_encoder(seq)                                                     # (B, S, transformer_dim)
        seq = self.norm(seq)                                                            # (B, S, transformer_dim)
        
        # Transformer encoding
        content_emb = self.transformer(seq)                                             # (B, S, transformer_dim)
        
        return content_emb