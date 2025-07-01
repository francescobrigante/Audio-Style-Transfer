import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

# ---------------------------------------------
# LightweightDecoder
# Genera STFT predetti in modo autoregressivo per la predizione audio
# Input:
#   - y_prev: [B, S, 2, 287, 513] (STFT precedente, B=batch size, S=4 sequenze)
#   - class_emb: [256] (embedding di stile)
#   - content_emb: [B, S, 256] (embedding di contenuto per ogni sequenza)
#   - hidden: Stato GRU (None o [1, B, 128])
# Output:
#   - output: [B, 2, S, 287, 513] (STFT predetto)
#   - hidden: [1, B, 128] (stato aggiornato della GRU)
# ---------------------------------------------
class LightweightDecoder(nn.Module):
    def __init__(self, d=256, d_hidden=128, num_channels=16, S=4):
        super().__init__()
        self.S = S  # Numero di sequenze temporali (default: 4)
        # Convoluzione iniziale per estrarre feature da STFT
        self.conv_in = nn.Conv2d(
            in_channels=2,  # Canali input: reale + immaginario
            out_channels=num_channels,  # Canali output: 16
            kernel_size=3, stride=2, padding=1
        )  # Output: [B, 16, 144, 257]
        self.ln1 = nn.LayerNorm([num_channels, 144, 257])  # Normalizzazione delle feature
        self.linear_in = nn.Linear(num_channels * 144 * 257, d)  # Proiezione a d=256
        # Meccanismo di attenzione per integrare contenuto e stile
        self.attention = nn.MultiheadAttention(embed_dim=d, num_heads=2, batch_first=True)
        # GRU per modellare dipendenze temporali tra sequenze
        self.gru = nn.GRU(input_size=d, hidden_size=d_hidden, num_layers=1, batch_first=True)
        self.linear_out = nn.Linear(d_hidden, num_channels * 144 * 257)  # Proiezione inversa
        # Deconvoluzione per ricostruire STFT
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                num_channels, 8, kernel_size=3, stride=2, padding=1, output_padding=0
            ),  # [B, 8, 287, 513]
            nn.ReLU(),
            nn.Conv2d(8, 2, kernel_size=1, stride=1, padding=0)  # [B, 2, 287, 513]
        )
        self.dropout = nn.Dropout(0.1)  # Regolarizzazione

    def forward(self, y_prev, class_emb, content_emb, hidden=None):
        batch_size = content_emb.size(0)  # Dimensione del batch
        outputs = []
        with autocast('cuda'):  # Usa mixed precision per efficienza su GPU
            # Processa ogni sequenza s in modo autoregressivo
            for s in range(self.S):
                # Estrae feature da STFT della sequenza s
                x = F.relu(self.ln1(self.conv_in(y_prev[:, s, :, :, :])))  # [B, 16, 144, 257]
                x = x.view(x.size(0), -1)  # Appiattisce: [B, 16*144*257]
                x = self.dropout(self.linear_in(x))  # Proiezione: [B, 256]
                # Combina contenuto e stile con attenzione
                context = torch.stack(
                    [content_emb[:, s, :], class_emb.expand(batch_size, -1)], dim=1
                )  # [B, 2, 256] (contenuto + stile)
                attn_output, _ = self.attention(x.unsqueeze(1), context, context)  # [B, 1, 256]
                attn_output = self.dropout(attn_output.squeeze(1))  # [B, 256]
                # GRU per dipendenze temporali
                gru_output, hidden = self.gru(attn_output.unsqueeze(1), hidden)  # [B, 1, 128]
                gru_output = gru_output.squeeze(1)  # [B, 128]
                # Ricostruisce STFT
                out = self.linear_out(gru_output)  # [B, 16*144*257]
                out = out.view(-1, num_channels, 144, 257)  # [B, 16, 144, 257]
                out = self.deconv(out)  # [B, 2, 287, 513]
                outputs.append(out.unsqueeze(2))  # [B, 2, 1, 287, 513]
                y_prev[:, s, :, :, :] = out.detach()  # Aggiorna y_prev per il prossimo passo
            output = torch.cat(outputs, dim=2)  # [B, 2, 4, 287, 513]
            # Verifica dimensioni per debug
            if (y_prev.shape[0] == output.shape[0] and
                y_prev.shape[1] == output.shape[2] and
                y_prev.shape[2] == output.shape[1] and
                y_prev.shape[3] == output.shape[3] and
                y_prev.shape[4] == output.shape[4]):
                print("Dimensioni corrette")
            else:
                print(f"Problema dimensioni: input {y_prev.shape}, output {output.shape}")
        return output, hidden  # [B, 2, 4, 287, 513], [1, B, 128]

# ---------------------------------------------
# Funzione di perdita
# Combina MSE, Magnitude Loss e Spectral Convergence Loss per valutare l'STFT predetto
# Input: output [B, 2, S, 287, 513], target [B, 2, S, 287, 513]
# Output: Loss scalare
# ---------------------------------------------
def compute_loss(output, target, S=4):
    # MSE: confronta direttamente STFT predetto e target
    mse_loss = sum(F.mse_loss(output[:, :, s, :, :], target[:, :, s, :, :]) for s in range(S))
    # Magnitude Loss: confronta magnitudini dello STFT
    mag_output = torch.sqrt(output[:, 0, :, :, :]**2 + output[:, 1, :, :, :]**2)
    mag_target = torch.sqrt(target[:, 0, :, :, :]**2 + target[:, 1, :, :, :]**2)
    mag_loss = sum(F.mse_loss(mag_output[:, s, :, :], mag_target[:, s, :, :]) for s in range(S))
    # Spectral Convergence Loss: differenza normalizzata delle magnitudini
    sc_loss = sum(
        torch.norm(mag_output[:, s, :, :] - mag_target[:, s, :, :], p='fro') /
        torch.norm(mag_target[:, s, :, :], p='fro') for s in range(S)
    )
    # Loss totale: combinazione pesata
    total_loss = 0.45 * mse_loss + 0.45 * mag_loss + 0.1 * sc_loss
    return total_loss

if __name__ == "__main__":
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    batch_size = 4
    d = 256
    d_hidden = 128
    num_channels = 16
    S = 4

    decoder = LightweightDecoder(d=d, d_hidden=d_hidden, num_channels=num_channels, S=S).cuda()
    y_prev = torch.randn(batch_size, S, 2, 287, 513).cuda()
    class_emb = torch.randn(d).cuda()
    content_emb = torch.randn(batch_size, S, d).cuda()
    target = torch.randn(batch_size, 2, S, 287, 513).cuda()
    hidden = None

    output, hidden = decoder(y_prev, class_emb, content_emb, hidden)
    print(f"Output shape: {output.shape}")
    loss = compute_loss(output, target, S=4)
    print(f"Loss: {loss.item()}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
