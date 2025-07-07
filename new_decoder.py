import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from style_encoder import SinusoidalPositionalEncoding
from torch.nn.utils import spectral_norm

class Decoder(nn.Module):
    """
    Decoder completamente dinamico senza parametro S fisso.
    Adatta automaticamente la lunghezza della sequenza agli input.
    """
    
    def __init__(self, d_model=256, nhead=4, num_layers=4, dim_feedforward=1024, 
                 dropout=0.1, max_seq_len=1000):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Dimensioni intermedie per preservare struttura 2D
        self.F_compressed = 32  # Frequenza compressa
        self.T_compressed = 16  # Tempo compresso
        self.feature_dim = 64   # Canali delle feature
        
        # ================== ENCODER CNN  ==================
        self.conv_encoder = nn.Sequential(

            spectral_norm(nn.Conv2d(2, 16, kernel_size=3, padding=1)),                    # [B, 16, 287, 513]
            nn.BatchNorm2d(16),
            # nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(),

            spectral_norm(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)),         # [B, 32, 144, 257]
            nn.BatchNorm2d(32),
            # nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            
            # Secondo blocco: compressione controllata
            spectral_norm(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)),         # [B, 64, 72, 129]
            nn.BatchNorm2d(64),
            # nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),

            spectral_norm(nn.Conv2d(64, self.feature_dim, kernel_size=3, stride=2, padding=1)),  # [B, 64, 36, 65]
            nn.BatchNorm2d(self.feature_dim),
            # nn.InstanceNorm2d(self.feature_dim, affine=True),
            nn.ReLU(),
            
            # Compressione finale controllata (mantiene proporzioni)
            nn.AdaptiveAvgPool2d((self.F_compressed, self.T_compressed))   # [B, 64, 32, 16]
        )
        
        # Proiezione dei canali
        self.spatial_projection = nn.Sequential(
            spectral_norm(nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1)),
            nn.BatchNorm2d(self.feature_dim),
            # nn.InstanceNorm2d(self.feature_dim, affine=True),
            nn.ReLU(),

            spectral_norm(nn.Conv2d(self.feature_dim, 1, kernel_size=1)),  # [B, 1, 32, 16]
        )
        
        # Proiezione finale a d_model
        self.feature_to_sequence = nn.Linear(self.F_compressed * self.T_compressed, d_model)
        
        # ================== DECODER CNN  ==================
        self.sequence_to_feature = nn.Linear(d_model, self.F_compressed * self.T_compressed)
        
        self.conv_decoder = nn.Sequential(
            # Primo upsampling
            spectral_norm(nn.ConvTranspose2d(1, self.feature_dim, kernel_size=3, stride=2, padding=1, output_padding=1)),  # [B, 64, 64, 32]
            nn.BatchNorm2d(self.feature_dim),
            # nn.InstanceNorm2d(self.feature_dim, affine=True),
            nn.ReLU(),
            
            # Secondo upsampling
            spectral_norm(nn.ConvTranspose2d(self.feature_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1)),  # [B, 32, 128, 64]
            nn.BatchNorm2d(32),
            # nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            
            # Terzo upsampling
            spectral_norm(nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)),  # [B, 16, 256, 128]
            nn.BatchNorm2d(16),
            # nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(),
            
            # Quarto upsampling
            spectral_norm(nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1)),   # [B, 8, 512, 256]
            nn.BatchNorm2d(8),
            # nn.InstanceNorm2d(8, affine=True),
            nn.ReLU(),
            
            # Output finale (reale + immaginario)
            spectral_norm(nn.ConvTranspose2d(8, 2, kernel_size=3, padding=1)),  # [B, 2, 512, 256]
            
            # Upsampling finale alle dimensioni originali
            nn.Upsample(size=(287, 513), mode='bilinear', align_corners=False)  # [B, 2, 287, 513]
        )
        
        # ================== TRANSFORMER COMPONENTS ==================
        # Proiezioni per content e class embeddings
        self.content_proj = nn.Linear(d_model, d_model)
        self.class_proj = nn.Linear(d_model, d_model)
        
        # Positional encoding sinusoidale
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        
        # Transformer decoder con attenzione causale
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm per stabilità
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Token speciale per iniziare la generazione
        self.start_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Layer norm
        self.input_norm  = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        
        # Dropout per regolarizzazione
        self.dropout = nn.Dropout(dropout)
        
        # Inizializzazione dei pesi
        self._init_weights()
    
    def _init_weights(self):
        """Inizializzazione dei pesi migliorata"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param, gain=0.2)
                else:
                    nn.init.zeros_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def encode_input(self, x):
        """
        Codifica l'input STFT preservando la struttura spaziale
        
        Args:
            x: [B, S, 2, 287, 513] - STFT complesso
            
        Returns:
            [B * S, d_model] - Embedding dell'input
        """
        # Estrai feature preservando la struttura
        features = self.conv_encoder(x)  # [B, 64, 32, 16]
        
        # Proiezione spaziale
        spatial_features = self.spatial_projection(features)  # [B, 1, 32, 16]
        
        # Converti in sequenza mantenendo info spaziale
        B, C, F, T = spatial_features.shape
        spatial_flat = spatial_features.view(B, F * T)  # [B, 32*16]
        
        # Proiezione finale
        embedding = self.feature_to_sequence(spatial_flat)  # [B, d_model]
        
        return embedding
    
    def generate_output(self, decoder_outputs):
        """
        Genera l'output STFT dalle rappresentazioni del decoder
        
        Args:
            decoder_outputs: [B, S, d_model] - Output del transformer
            
        Returns:
            [B, S, 2, 287, 513] - STFT complesso ricostruito
        """
        B, S, D = decoder_outputs.shape
        
        # Normalizza l'output
        decoder_outputs = self.output_norm(decoder_outputs)
        
        # Converti in feature spaziali
        spatial_features = self.sequence_to_feature(decoder_outputs)  # [B, S, F*T]
        spatial_features = spatial_features.view(B * S, 1, self.F_compressed, self.T_compressed)  # [B*S, 1, F, T]
        
        # Ricostruisci STFT
        reconstructed = self.conv_decoder(spatial_features)  # [B*S, 2, 287, 513]
        
        # Reshape finale usando le dimensioni corrette
        return reconstructed.view(B, S, 2, 287, 513)
    
    def create_causal_mask(self, seq_len):
        """
        Crea maschera causale per autoregressione
        
        Args:
            seq_len: lunghezza della sequenza
            
        Returns:
            Maschera causale [seq_len, seq_len]
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask
    
    def prepare_memory(self, content_emb, class_emb):
        """
        Prepara la memoria per cross-attention
        
        Args:
            content_emb: [B, S_content, d_model]
            class_emb: [B, d_model]
            
        Returns:
            [B, 2*S_content, d_model] - Memoria per cross-attention
        """
        B, S_content, D = content_emb.shape
        
        # Proietta gli embeddings
        content_memory = self.content_proj(content_emb)  # [B, S_content, D]
        class_memory = self.class_proj(class_emb).unsqueeze(1).expand(-1, S_content, -1)  # [B, S_content, D]
        
        # Combina content e class come memoria per cross-attention
        memory = torch.cat([content_memory, class_memory], dim=1)  # [B, 2*S_content, D]
        memory = self.dropout(memory)
        
        return memory
    
    def forward_training(self, y, memory):
        """
        Forward pass per training con teacher forcing
        
        Args:
            y: [B, S_target, 2, 287, 513] - Target STFT
            memory: [B, 2*S_content, d_model] - Memoria per cross-attention
            
        Returns:
            [B, S_target, 2, 287, 513] - STFT predetto
        """
        B_y, S_y, C_y, H_y, W_y = y.shape
        device = y.device
        
        # Codifica la sequenza target        
        # Estrai embeddings per ogni frame in modo più efficiente
        y_flat = y.view(B_y * S_y, C_y, H_y, W_y)  # [B*S_target, 2, 287, 513]
        y_embeddings = self.encode_input(y_flat)    # [B*S_target, d_model]
        y_emb_seq = y_embeddings.view(B_y, S_y, self.d_model)  # [B, S_target, d_model]
        
        
        # Aggiungi positional encoding
        y_emb_seq = self.pos_encoding(y_emb_seq)
        y_emb_seq = self.input_norm(y_emb_seq)  # [B, S_target, d_model]
        
        # Applica maschera causale
        causal_mask = self.create_causal_mask(S_y).to(device)
        
        # Transformer decoder con teacher forcing
        decoder_output = self.transformer_decoder(
            tgt=y_emb_seq,
            memory=memory,
            tgt_mask=causal_mask
        )  # [B, S_y, d_model]
        
        return self.generate_output(decoder_output)
    
    def forward_inference(self, memory, target_length=None):
        """
        Forward pass per inference autoregressivo
        
        Args:
            memory: [B, 2*S_content, d_model] - Memoria per cross-attention
            target_length: Lunghezza target (opzionale)
            
        Returns:
            [B, target_length, 2, 287, 513] - STFT predetto
        """
        B = memory.size(0)
        device = memory.device
        
        # Se non specificato, usa una lunghezza di default
        if target_length is None:
            target_length = memory.size(1) // 2  # Usa S_content come default
        
        # Inizializza con start token
        generated_sequence = self.start_token.expand(B, -1, -1)  # [B, 1, d_model]
        generated_outputs = []
        
        for t in range(target_length):
            # Applica positional encoding
            current_seq = self.pos_encoding(generated_sequence)
            
            # Crea maschera causale per la sequenza corrente
            current_len = generated_sequence.size(1)
            causal_mask = self.create_causal_mask(current_len).to(device)
            
            # Transformer decoder step
            decoder_output = self.transformer_decoder(
                tgt=current_seq,
                memory=memory,
                tgt_mask=causal_mask
            )  # [B, current_len, d_model]
            
            # Prendi solo l'ultimo output
            next_token = decoder_output[:, -1:, :]  # [B, 1, d_model]
            generated_outputs.append(next_token)
            
            # Aggiorna la sequenza per il prossimo step
            generated_sequence = torch.cat([generated_sequence, next_token], dim=1)
        
        # Combina tutti gli output generati
        decoder_output = torch.cat(generated_outputs, dim=1)  # [B, target_length, d_model]
        
        return self.generate_output(decoder_output)
    
    def forward(self, content_emb, class_emb, y=None, target_length=None):
        """
        Forward pass del decoder completamente dinamico
        
        Args:
            content_emb: [B, S_content, d_model] - Embeddings di contenuto (sequenza temporale)
            class_emb: [B, d_model] - Embedding di classe/stile
            y: [B, S_target, 2, 287, 513] - Target STFT (solo durante training)
            target_length: Lunghezza desiderata per inference (opzionale)
            
        Returns:
            [B, S, 2, 287, 513] - STFT predetto (S dinamico)
        """
        # Prepara memoria per cross-attention
        memory = self.prepare_memory(content_emb, class_emb)
        
        # Training con teacher forcing
        if self.training and y is not None:
            if len(y.shape) != 5:
                raise ValueError(f"Expected y to have shape [B, S, 2, 287, 513], got {y.shape}")
            return self.forward_training(y, memory)
        
        # Inference autoregressivo
        else:
            return self.forward_inference(memory, target_length)


def compute_comprehensive_loss(output, target, lambda_temporal=0.3, lambda_phase=0.2, 
                              lambda_spectral=0.1, lambda_consistency=0.1):
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
    temporal_loss = torch.tensor(0.0, device=output.device)
    if S > 1:
        # Consistenza temporale dei frame
        for s in range(S-1):
            # Differenza temporale nell'output
            temp_diff_out = output[:, s+1] - output[:, s]
            temp_diff_tgt = target[:, s+1] - target[:, s]
            temporal_loss += F.mse_loss(temp_diff_out, temp_diff_tgt)
        temporal_loss /= (S-1)
    
    # ================== SPECTRAL CONSISTENCY LOSS ==================
    # Consistenza lungo la dimensione delle frequenze
    spectral_loss = torch.tensor(0.0, device=output.device)
    if Freq > 1:
        # Gradiente spettrale
        spectral_grad_out = output[:, :, :, 1:, :] - output[:, :, :, :-1, :]
        spectral_grad_tgt = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        spectral_loss = F.mse_loss(spectral_grad_out, spectral_grad_tgt)
    
    # ================== MAGNITUDE-PHASE CONSISTENCY LOSS ==================
    # Assicura che magnitude e phase siano coerenti con la rappresentazione complessa
    reconstructed_real = mag_output * torch.cos(phase_output)
    reconstructed_imag = mag_output * torch.sin(phase_output)
    
    consistency_loss = (
        F.mse_loss(reconstructed_real, output[:, :, 0]) +
        F.mse_loss(reconstructed_imag, output[:, :, 1])
    )
    
    # ================== LOSS TOTALE ==================
    total_loss = (
        mse_loss +
        0.5 * mag_loss +
        lambda_phase * phase_loss +
        lambda_temporal * temporal_loss +
        lambda_spectral * spectral_loss +
        lambda_consistency * consistency_loss
    )
    
    return {
        'total_loss': total_loss,
        'mse_loss': mse_loss,
        'mag_loss': mag_loss,
        'phase_loss': phase_loss,
        'temporal_loss': temporal_loss,
        'spectral_loss': spectral_loss,
        'consistency_loss': consistency_loss
    }

