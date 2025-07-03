import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TransformerDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=4, dim_feedforward=1024, S=4, F_prime=8, T_prime=16):
        super().__init__()
        self.S = S
        self.d_model = d_model
        self.F_prime = F_prime
        self.T_prime = T_prime

        # CNN encoder per estrarre feature da STFT
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),                    # [B, 16, 287, 513]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),         # [B, 32, 144, 257]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),         # [B, 64, 72, 129]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),        # [B, 128, 36, 65]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((F_prime, T_prime))                      # [B, 128, F_prime, T_prime]
        )
        self.feature_projection = nn.Linear(128 * F_prime * T_prime, d_model)  # [B, 128*F_prime*T_prime] → [B, d_model]

        # CNN decoder per generare STFT
        self.output_projection = nn.Linear(d_model, 128 * F_prime * T_prime)   # [B, d_model] → [B, 128*F_prime*T_prime]
        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 64, 2*F_prime, 2*T_prime]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # [B, 32, 4*F_prime, 4*T_prime]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),   # [B, 16, 8*F_prime, 8*T_prime]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),    # [B, 8, 16*F_prime, 16*T_prime]
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),                      # [B, 4, 32*F_prime, 32*T_prime]
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, kernel_size=4, stride=2, padding=1),                      # [B, 2, 64*F_prime, 64*T_prime]
            nn.Upsample(size=(287, 513), mode='bilinear', align_corners=False)                 # [B, 2, 287, 513]
        )

        # Proiezione content e class embeddings
        self.content_proj = nn.Linear(d_model, d_model)
        self.class_proj = nn.Linear(d_model, d_model)

        # Positional encoding
        self.pos_emb = nn.Parameter(torch.randn(S, d_model))  # [S, d_model]

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def encode_input(self, x):
        """
        x: [B, 2, TIME, FREQ]
        returns: [B, d_model]
        """
        features = self.conv_encoder(x)  # [B, 128, F_prime, T_prime]
        features = features.flatten(1)   # [B, 128*F_prime*T_prime]
        return self.feature_projection(features)  # [B, d_model]

    def generate_output(self, decoder_outputs):
        """
        decoder_outputs: [B, S, d_model]
        returns: [B, S, 2, 287, 513]
        """
        B, S, D = decoder_outputs.shape
        out = self.output_projection(decoder_outputs)  # [B, S, 128*F_prime*T_prime]
        out = out.view(B * S, 128, self.F_prime, self.T_prime)  # [B*S, 128, F_prime, T_prime]
        out = self.conv_decoder(out)     # [B*S, 2, 287, 513]
        return out.view(B, S, 2, 287, 513)

    def forward(self, content_emb, class_emb, y=None):
        B, S, D = content_emb.shape
        
        content_memory = self.content_proj(content_emb)                                 # (B, S, D)
        class_memory = self.class_proj(class_emb).unsqueeze(1).expand(-1, S, -1)        # (B, S, D)
        
        
        memory = torch.cat([content_memory, class_memory], dim=1)                       # (B, 2*S, D)

        if self.training and y is not None:
            # Teacher forcing
            y = y.view(B * self.S, 2, 287, 513)
            y_feat = self.conv_encoder(y)                     # [B*S, 128, F_prime, T_prime]
            y_feat = y_feat.flatten(1)                        # [B*S, 128*F_prime*T_prime]
            y_emb = self.feature_projection(y_feat)           # [B*S, d_model]
            y_emb = y_emb.view(B, self.S, self.d_model)       # [B, S, d_model]

            pos = self.pos_emb.unsqueeze(0).expand(B, -1, -1) # [B, S, d_model]
            tgt = y_emb + pos                                 # [B, S, d_model]

            out = self.transformer_decoder(tgt=tgt, memory=memory)  # [B, S, d_model]
        else:
            # Autoregressive inference
            generated = []
            prev = torch.zeros(B, 1, self.d_model, device=device)
            for t in range(self.S):
                tgt = prev + self.pos_emb[t].view(1, 1, -1)  # [B, 1, d_model]
                out = self.transformer_decoder(tgt=tgt, memory=memory)  # [B, 1, d_model]
                generated.append(out)
                prev = out
            out = torch.cat(generated, dim=1)  # [B, S, d_model]

        return self.generate_output(out)  # [B, S, 2, 287, 513]

# Loss
def compute_loss(output, target, S=4):
    """
    Loss function migliorata con coerenza di fase
    
    Args:
        output: [B, S, 2, 287, 513] - STFT predetto
        target: [B, S, 2, 287, 513] - STFT target
        S: numero di sequenze
    
    Returns:
        total_loss: loss scalare
    """
    mse_loss = mag_loss = phase_loss = 0
    
    # 1. MSE Loss e Magnitude Loss per ogni sequenza
    for s in range(S):
        # MSE: confronta direttamente STFT
        mse_loss += F.mse_loss(output[:, s, :, :, :], target[:, s, :, :, :])
        
        # Magnitude Loss: confronta magnitudini
        mag_output = torch.sqrt(output[:, s, 0, :, :]**2 + output[:, s, 1, :, :]**2)
        mag_target = torch.sqrt(target[:, s, 0, :, :]**2 + target[:, s, 1, :, :]**2)
        mag_loss += F.mse_loss(mag_output, mag_target)
    
    # 2. Phase Consistency Loss: coerenza di fase tra sequenze consecutive
    if S > 1:
        phase_output = torch.atan2(output[:, :, 1, :, :], output[:, :, 0, :, :])  # [B, S, 287, 513]
        phase_target = torch.atan2(target[:, :, 1, :, :], target[:, :, 0, :, :])
        
        for s in range(S-1):
            # Differenze di fase tra sequenze consecutive
            phase_diff_out = phase_output[:, s+1] - phase_output[:, s]
            phase_diff_tgt = phase_target[:, s+1] - phase_target[:, s]
            
            # Wrap delle differenze di fase in [-π, π]
            phase_diff_out = torch.remainder(phase_diff_out + np.pi, 2*np.pi) - np.pi
            phase_diff_tgt = torch.remainder(phase_diff_tgt + np.pi, 2*np.pi) - np.pi
            
            phase_loss += F.mse_loss(phase_diff_out, phase_diff_tgt)
    
    # Loss totale con pesi bilanciati
    total_loss = 0.4 * mse_loss + 0.4 * mag_loss + 0.2 * phase_loss
    return total_loss
    
if __name__ == "__main__":
    B, S, TIME, FREQ = 4, 4, 287, 513
    d_model = 256
    F_prime = 8
    T_prime = 16

    model = TransformerDecoder(d_model=d_model, S=S, F_prime=F_prime, T_prime=T_prime)
    model.train()

    # Dummy input
    y_true = torch.randn(B, S, 2, TIME, FREQ)
    content_emb = torch.randn(B, d_model)
    class_emb = torch.randn(B, d_model)

    # Teacher forcing
    y_pred = model(content_emb, class_emb, y_true)
    print("Output shape (teacher forcing):", y_pred.shape)
    assert y_pred.shape == (B, S, 2, TIME, FREQ)

    # Eval mode: autoregressivo
    model.eval()
    y_autoreg = model(content_emb, class_emb)
    print("Output shape (autoregressive):", y_autoreg.shape)
    assert y_autoreg.shape == (B, S, 2, TIME, FREQ)

    # Test loss e parametri
    loss = compute_loss(y_autoreg, y_true)
    print(f"Loss: {loss.item():.4f}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    if torch.cuda.is_available():
        print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
