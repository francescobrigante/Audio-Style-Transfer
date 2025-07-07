import torch
import torch.nn as nn
import torch.nn.functional as F
from style_encoder import SinusoidalPositionalEncoding


class Decoder(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.stft_dim = 2 * 287 * 513  # Flattened STFT

        # Linear encoder/decoder in luogo delle CNN
        self.stft_to_embedding = nn.Linear(self.stft_dim, d_model)
        self.embedding_to_stft = nn.Linear(d_model, self.stft_dim)

        # Proiezioni per content e class
        self.content_proj = nn.Linear(d_model, d_model)
        self.class_proj = nn.Linear(d_model, d_model)

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Start token
        self.start_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Norm e dropout
        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param, gain=0.2)
                else:
                    nn.init.zeros_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def encode_input(self, x):
        B, S, C, H, W = x.shape
        x = x.view(B * S, -1)
        embeddings = self.stft_to_embedding(x)  # [B*S, d_model]
        return embeddings.view(B, S, self.d_model)

    def generate_output(self, decoder_outputs):
        B, S, D = decoder_outputs.shape
        decoder_outputs = self.output_norm(decoder_outputs)
        stft_flat = self.embedding_to_stft(decoder_outputs)  # [B, S, 2*287*513]
        return stft_flat.view(B, S, 2, 287, 513)

    def create_causal_mask(self, seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def prepare_memory(self, content_emb, class_emb):
        B, S, D = content_emb.shape
        content_proj = self.content_proj(content_emb)
        class_proj = self.class_proj(class_emb).unsqueeze(1).expand(-1, S, -1)
        memory = torch.cat([content_proj, class_proj], dim=1)
        return self.dropout(memory)

    def forward_training(self, y, memory):
        B, S, C, H, W = y.shape
        device = y.device

        y_emb = self.encode_input(y)  # [B, S, d_model]

        start_token = self.start_token.expand(B, 1, -1)
        y_shifted = torch.cat([start_token, y_emb[:, :-1, :]], dim=1)  # [B, S, d_model]

        y_shifted = self.pos_encoding(y_shifted)
        y_shifted = self.input_norm(y_shifted)

        causal_mask = self.create_causal_mask(S, device=device)

        decoder_output = self.transformer_decoder(
            tgt=y_shifted,
            memory=memory,
            tgt_mask=causal_mask
        )

        return self.generate_output(decoder_output)

    def forward_inference(self, memory, target_length=None):
        B = memory.size(0)
        S = memory.size(1) // 2 if target_length is None else target_length
        device = memory.device

        generated = self.start_token.expand(B, 1, -1)
        outputs = []

        for t in range(S):
            current = self.pos_encoding(generated)
            causal_mask = self.create_causal_mask(current.size(1), device=device)

            decoder_output = self.transformer_decoder(
                tgt=current,
                memory=memory,
                tgt_mask=causal_mask
            )

            next_token = decoder_output[:, -1:, :]
            outputs.append(next_token)
            generated = torch.cat([generated, next_token], dim=1)

        decoder_output = torch.cat(outputs, dim=1)
        return self.generate_output(decoder_output)

    def forward(self, content_emb, class_emb, y=None, target_length=None):
        memory = self.prepare_memory(content_emb, class_emb)

        if self.training and y is not None:
            return self.forward_training(y, memory)
        else:
            return self.forward_inference(memory, target_length)

      

      
def compute_comprehensive_loss(output, target, lambda_temporal=0.3, lambda_phase=0.2, 
                              lambda_spectral=0.1):
    """
    Loss function comprensiva per audio style transfer - CORRETTA
    
    Args:
        output: [B, S, 2, 287, 513] - STFT predetto
        target: [B, S, 2, 287, 513] - STFT target
        lambda_temporal: peso per consistenza temporale
        lambda_phase: peso per coerenza di fase
        lambda_spectral: peso per consistenza spettrale
        lambda_consistency: peso per consistenza magnitude-phase
    
    Returns:
        dict: dizionario con loss totale e componenti
    """
    B, S, _, Freq, T = output.shape
    
    # ================== LOSS BASE ==================
    # MSE loss su tutto
    mse_loss = F.mse_loss(output, target)
    
    # ================== MAGNITUDE E PHASE LOSS ==================
    # Calcola magnitude e phase
    mag_output = torch.sqrt(output[:, :, 0]**2 + output[:, :, 1]**2 + 1e-8)  # Aggiunta stabilità numerica
    mag_target = torch.sqrt(target[:, :, 0]**2 + target[:, :, 1]**2 + 1e-8)
    mag_loss = F.mse_loss(mag_output, mag_target)
    
    # Loss di fase (più stabile con atan2)
    phase_output = torch.atan2(output[:, :, 1], output[:, :, 0])
    phase_target = torch.atan2(target[:, :, 1], target[:, :, 0])
    
    # Gestisci il wrapping della fase
    phase_diff = phase_output - phase_target
    phase_diff = torch.remainder(phase_diff + np.pi, 2*np.pi) - np.pi
    phase_loss = F.mse_loss(phase_diff, torch.zeros_like(phase_diff))
    
    # ================== TEMPORAL CONSISTENCY LOSS ==================
    # Versione migliorata senza ciclo for
    if S > 1:
      temp_diff_out = output[:, 1:] - output[:, :-1]  # shape: [B, S-1, 2, Freq, T]
      temp_diff_tgt = target[:, 1:] - target[:, :-1]
      temporal_loss = F.mse_loss(temp_diff_out, temp_diff_tgt)
    else:
      temporal_loss = torch.tensor(0.0, device=output.device)
    
    # ================== SPECTRAL CONSISTENCY LOSS ==================
    # Consistenza lungo la dimensione delle frequenze
    spectral_loss = torch.tensor(0.0, device=output.device)
    if Freq > 1:
        # Gradiente spettrale
        spectral_grad_out = output[:, :, :, 1:, :] - output[:, :, :, :-1, :]
        spectral_grad_tgt = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        spectral_loss = F.mse_loss(spectral_grad_out, spectral_grad_tgt)

    
    # ================== LOSS TOTALE ==================
    total_loss = (
        mse_loss +
        0.5 * mag_loss +
        lambda_phase * phase_loss +
        lambda_temporal * temporal_loss +
        lambda_spectral * spectral_loss
    )
    
    return {
        'total_loss': total_loss,
        'mse_loss': mse_loss,
        'mag_loss': mag_loss,
        'phase_loss': phase_loss,
        'temporal_loss': temporal_loss,
        'spectral_loss': spectral_loss
    }
