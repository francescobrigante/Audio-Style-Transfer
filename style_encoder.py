import torch
import torch.nn as nn
import math

# ---------------------------------------------
# Sinusoidal Positional Encoding Module
# ---------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 500):
        super().__init__()
        # creates position encoding matrix of shape (1, max_len, hidden_dim) to compute positions
        pe = torch.zeros(max_len, hidden_dim)                   #shape: (max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        # fill even/odd dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                                    # shape: (1, max_len, hidden_dim)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, seq_len, d_model)
        seq_len = x.size(1)
        # add positional encoding to input tensor matching sequence length
        x = x + self.pe[:, :seq_len]
        return x






# ---------------------------------------------
# Residual Block (basic block for the CNN) with optional downsampling
# Conv 3x3, BatchNorm, ReLU, Conv 3x3, BatchNorm + skip connection and final ReLU
# Input has shape (B*S, in_channels, T, F), output (B*S, out_channels, T', F')
# ---------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False):
        super().__init__()
        
        # if downsample, use stride=2 to halve spatial dimensions
        # otherwise, keep stride=1 to maintain dimensions
        stride = 2 if downsample else 1
        
        # 1st conv: may change #channels or downsample spatial dims
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=1, stride=stride
        )
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 2nd conv: keeps dimensions
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1
        )
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # shortcut: identity or 1x1 conv to match dims
        if downsample or in_channels != out_channels:
            # 1x1 conv to match output channels
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # same dimensions, use identity
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # skip connection
        out = nn.ReLU()(out)
        return out                                  # (B*S, out_channels, T', F')





# ---------------------------------------------
# Deep CNN Encoder
# Processes each STFT and CQT chunk (2, T, F) independently
# Uses 6 ResBlocks (channels 32, 64, 128, 256, 512, 512) and global pooling
# ---------------------------------------------
class DeepCNN(nn.Module):
    def __init__(self, in_channels: int, out_dim: int, 
                 channels_list: list = [32, 64, 128, 256, 512, 512]):
        super().__init__()
        
        layers = []
        prev_chan_size = in_channels
        
        # create 6 ResBlocks with increasing channel sizes
        # first 4 blocks downsample, last 2 keep spatial dimensions
        for idx, chan_size in enumerate(channels_list):
            downsample_boolean = idx < 4
            layers.append(ResBlock(prev_chan_size, chan_size, downsample=downsample_boolean))
            prev_chan_size = chan_size
            
            
        # global average pooling to (1,1) on time-frequency dimensions
        # this reduces the output to (B*S, last_chan_size, 1, 1)
        # where prev_chan_size is the last ResBlock's output channels
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.net = nn.Sequential(*layers)
        
        # projection to final embedding dimension
        self.proj = nn.Linear(prev_chan_size, out_dim)          # (B*S, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B*S, 2, T, F)
        x = self.net(x)                                         # (B*S, last_chan_size, 1, 1)
        x = x.view(x.size(0), -1)                               # (B*S, last_chan_size)
        x = self.proj(x)                                        # (B*S, out_dim)
        return x






# ---------------------------------------------
# Full model
# Uses CNN to extract meaningful features for each batch and temporal frame
# Then add positional encoding and pass through transformer to get style embeddings
# Input shape: (B, S, 2, T, F) where B=batch size, S=sections per audio, 2=C=channels (Real+Imaginary), T = time, F = frequency
# Intermediate output shape: (B, transformer_dim)
# Final output shape (2, transformer_dim) for 2 instrument classes, representing style embeddings
# ---------------------------------------------
class StyleEncoder(nn.Module):
    ##############
    ## -------> Se il modello non performa bene, prova cnn_out_dim=512, transformer_dim=512, num_heads=8, num_layers=6 <-----------
    ##############
    def __init__(
        self,
        in_channels: int = 2,
        cnn_out_dim: int = 256,         # final output dim of CNN
        transformer_dim: int = 256,     # transformer embedding dimension
        num_heads: int = 4,
        num_layers: int = 4,
        use_cls: bool = True
    ):
        super().__init__()
        
        self.use_cls = use_cls
        
        # CNN feature extractor for each batch and temporal frame
        # Input shape: (B*S, 2, T, F) -> Output shape: (B*S, cnn_out_dim)
        self.cnn = DeepCNN(in_channels, cnn_out_dim)
        
        # Linear projection (CNN out dim -> transformer dim) if needed
        self.input_proj = (
            nn.Linear(cnn_out_dim, transformer_dim)
            if cnn_out_dim != transformer_dim else None
        )
        
        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(transformer_dim)
        
        # Transformer encoder single layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=0.1,
            batch_first=True                        # flag that tells first dimension is batch size B
        )
        # Stack multiple transformer encoder layers
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # Learnable CLS token if used
        if use_cls:
            self.cls_token = nn.Parameter(
                torch.randn(1, 1, transformer_dim)
            )

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, S, 2, T, F) tensor of STFT+CQT chunks
            labels: (B,) tensor with instrument IDs (0 or 1 since 2 classes)
            
        Returns:
            style_emb: (B, d) for each audio in the batch
            class_emb: (2, d) for each class (0 and 1) if labels provided
        """
        
        B, S, C, T, F = x.shape
        
        # Merge batch and sections
        x = x.view(B * S, C, T, F)                              # (B*S, 2, T, F)
        cnn_features = self.cnn(x)                              # (B*S, cnn_out_dim)
        # Reshape sequence to (B, S, cnn_out_dim)
        seq = cnn_features.view(B, S, -1)                       # (B, S, cnn_out_dim)
        
        # project to transformer dimension if needed
        if self.input_proj:
            seq = self.input_proj(seq)                          # (B, S, d)
            
        # Add CLS token at the beginning of the temporal sequences
        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)              # expand from (1, 1, d) to (B, 1, d)
            seq = torch.cat([cls, seq], dim=1)                  # (B, S+1, d)
            
        # Add positional encoding
        seq = self.pos_encoder(seq)                             # (B, S+1, d)
        
        # Transformer encoding
        encoded = self.transformer(seq)                         # (B, S+1, d)
        
        # Pooling across sections
        # If using CLS token, take it as the style embedding
        # Otherwise, average across all sections
        if self.use_cls:
            style_emb = encoded[:, 0, :]                        # (B, d)
        else:
            style_emb = encoded.mean(dim=1)                     # (B, d)
            
        # Aggregate by class if labels provided
        if labels is not None:
            class_ids = labels.unique()
            class_ids, _ = torch.sort(class_ids)                # ensure ascending order (0, 1) -> important for Discriminator
            class_embs = []
            
            for cid in class_ids:
                mask = labels == cid
                emb = style_emb[mask].mean(dim=0) if mask.any() else torch.zeros_like(style_emb[0])
                class_embs.append(emb)
                
            class_emb = torch.stack(class_embs, dim=0)           # -> (2, d)
            
        else:
            class_emb = None
            
        return style_emb, class_emb
    

# initialization that was originally used
# however it was constantly causing grad explosion so I switched to a more conservative one 
def initialize_weights(model):
    """
    Initializes weights of the model according to the following scheme:
    - Kaiming He for nn.Conv2d with ReLU
    - Xavier for linear layers and transformer layers
    - Gaussian with mean=0, std=0.02 for the CLS token
    - Gaussian with mean=1, std=0 for nn.BatchNorm2d
    """
    for name, module in model.named_modules():
        
        if isinstance(module, nn.Conv2d):
            # Kaiming He initialization for Conv2d layers
            weight = getattr(module, 'weight_orig', module.weight)
            nn.init.kaiming_normal_(weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
        # InstanceNorm added for content encoder
        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.InstanceNorm2d):
            # batchnorm with mean=1, std=0
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, 1.0)
                
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
            
        elif isinstance(module, nn.Linear):
            # xavier
            nn.init.xavier_normal_(module.weight, gain=0.2)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
        elif isinstance(module, nn.TransformerEncoderLayer):
            # xavier
            nn.init.xavier_normal_(module.self_attn.in_proj_weight, gain=0.2)
            nn.init.xavier_normal_(module.self_attn.out_proj.weight, gain=0.2)
            nn.init.xavier_normal_(module.linear1.weight, gain=0.2)
            nn.init.xavier_normal_(module.linear2.weight, gain=0.2)
            nn.init.constant_(module.self_attn.in_proj_bias, 0)
            nn.init.constant_(module.self_attn.out_proj.bias, 0)
            nn.init.constant_(module.linear1.bias, 0)
            nn.init.constant_(module.linear2.bias, 0)
            
        elif isinstance(module, nn.Parameter) and 'cls_token' in name:
            # CLS tokesn with Gaussian initialization
            nn.init.normal_(module, mean=0.0, std=0.02)