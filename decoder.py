import torch
import torch.nn as nn

class DummyAutoregressiveDecoder(nn.Module):
    """
    Un decoder autoregressivo semplificato.
    Per ogni sezione temporale 's' in input:
    - Prende l'embedding di contenuto per quella sezione.
    - Prende l'embedding di stile (costante per tutte le sezioni).
    - Prende l'output della sezione precedente (o un token di inizio).
    - Usa un GRU per generare la predizione per la sezione corrente.
    """
    def __init__(self, content_dim=256, style_dim=256, output_dims=(2, 287, 513), hidden_dim=512):
        super().__init__()
        self.output_dims = output_dims
        self.output_flat_size = output_dims[0] * output_dims[1] * output_dims[2]
        
        # Token di inizio per la prima predizione
        self.start_token = nn.Parameter(torch.randn(1, 1, self.output_flat_size))
        
        # Il GRU è il cuore autoregressivo
        self.gru = nn.GRU(
            input_size=content_dim + style_dim + self.output_flat_size,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Layer lineare per proiettare l'output del GRU alla dimensione dello spettrogramma
        self.proj = nn.Linear(hidden_dim, self.output_flat_size)

    def forward(self, content_emb, style_emb, target_stft=None, teacher_forcing_ratio=0.5):
        """
        Args:
            content_emb (torch.Tensor): (B, S, content_dim)
            style_emb (torch.Tensor): (B, style_dim)
            target_stft (torch.Tensor): (B, S, 2, T, F) - Spettrogramma reale per teacher forcing.
            teacher_forcing_ratio (float): Probabilità di usare il ground truth.
        
        Returns:
            torch.Tensor: Spettrogramma ricostruito (B, S, 2, T, F)
        """
        B, S, _ = content_emb.shape
        device = content_emb.device
        
        # Espandi l'embedding di stile per ogni sezione
        style_emb_expanded = style_emb.unsqueeze(1).expand(-1, S, -1) # (B, S, style_dim)
        
        # Inizializza l'input per la prima sezione
        prev_output = self.start_token.expand(B, -1, -1) # (B, 1, flat_size)
        
        hidden = None
        outputs = []
        
        use_teacher_forcing = True if self.training and torch.rand(1).item() < teacher_forcing_ratio else False

        for s in range(S):
            # Concatena gli input per il GRU
            gru_input = torch.cat([
                content_emb[:, s:s+1, :],    # (B, 1, content_dim)
                style_emb_expanded[:, s:s+1, :], # (B, 1, style_dim)
                prev_output                  # (B, 1, flat_size)
            ], dim=-1)
            
            # Passaggio nel GRU
            gru_output, hidden = self.gru(gru_input, hidden) # gru_output: (B, 1, hidden_dim)
            
            # Proiezione all'output
            predicted_flat = self.proj(gru_output) # (B, 1, flat_size)
            outputs.append(predicted_flat)
            
            # Prepara l'input per il prossimo step
            if use_teacher_forcing and target_stft is not None:
                # Usa il ground truth
                prev_output = target_stft[:, s:s+1, ...].reshape(B, 1, -1)
            else:
                # Usa la predizione corrente
                prev_output = predicted_flat

        # Concatena gli output di tutte le sezioni
        reconstructed_flat = torch.cat(outputs, dim=1) # (B, S, flat_size)
        
        # Riformatta alla dimensione dello spettrogramma
        C, T, F = self.output_dims
        reconstructed_spec = reconstructed_flat.view(B, S, C, T, F)
        
        return reconstructed_spec